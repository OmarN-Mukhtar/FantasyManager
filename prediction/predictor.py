import pandas as pd
import numpy as np
import xgboost as xgb
import json
import requests


BOOTSTRAP_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'
FIXTURES_URL = 'https://fantasy.premierleague.com/api/fixtures/?future=1'
N_FIXTURES = 5


def weighted_total(preds_and_fdrs):
    """Discounted next-5 total: closer games and easier opponents weigh more.

    ponytail: two knobs — 0.8/GW time decay, ±0.2 per FDR step from neutral 3.
    """
    return sum(
        pred * 0.8 ** i * (1 + (3 - fdr) / 5)
        for i, (pred, fdr) in enumerate(preds_and_fdrs)
    )

ROLLING_WINDOWS = [1, 3, 5, 7]
TRAIN_LAST_N_GWS = 25
BASE_ROLLING_FEATURES = [
    'bps',
    'ict_index',
    'minutes',
    'total_points',
    'selected',
    'transfers_balance',
    'now_cost',
    'goals_scored',
    'assists',
    'clean_sheets',
    'goals_conceded',
    'expected_goals',
    'expected_goal_involvements',
    'expected_assists',
    'expected_goals_conceded',
    'form',
]


class PlayerPredictor:
    
    def __init__(self, data_path='data/cleaned_merged_seasons.csv'):
        self.data_path = data_path
        self.full_df = None
        self.df = None
        self.model = None
        self.feature_cols = []
        self.predictions = {}
        self.player_name_to_id = {}
        self.team_of_player = {}
        self.upcoming_by_team = {}
        
    def load_data(self):

        self.df = pd.read_csv(self.data_path, low_memory=False)
        self.full_df = self.df.copy()
        return True
    
    def train_model(self):
        self.df = self.df.sort_values(['name', 'season', 'GW']).copy()
        self.df = self.df.groupby('name', group_keys=False).tail(TRAIN_LAST_N_GWS).copy()

        grouped = self.df.groupby('name', sort=False)
        self.feature_cols = []
        for col in BASE_ROLLING_FEATURES:
            for window in ROLLING_WINDOWS:
                feature_name = f"{col}_rolling_{window}"
                self.df[feature_name] = grouped[col].transform(
                    lambda s: s.rolling(window=window, min_periods=1).mean()
                )
                self.feature_cols.append(feature_name)

        # Keep forward predictions independent from defensive stats.
        is_fwd = self.df['position'].astype(str).str.upper().eq('FWD')
        if 'element_type' in self.df.columns:
            is_fwd = is_fwd | (pd.to_numeric(self.df['element_type'], errors='coerce') == 4)
        for base_col in ['goals_conceded', 'clean_sheets']:
            for window in ROLLING_WINDOWS:
                self.df.loc[is_fwd, f"{base_col}_rolling_{window}"] = 0.0

        # total_points is already per-gameweek points in this dataset.
        self.df['target_next_gw_points'] = grouped['total_points'].shift(-1)
        self.df['team_id_feature'] = pd.to_numeric(self.df.get('team_id', 0), errors='coerce').fillna(0)
        self.df['opponent_team_feature'] = pd.to_numeric(self.df.get('opponent_team', 0), errors='coerce').fillna(0)
        self.df['is_home_feature'] = pd.to_numeric(self.df.get('was_home', 0), errors='coerce').fillna(0)
        self.feature_cols.extend(['team_id_feature', 'opponent_team_feature', 'is_home_feature'])

        df_train = self.df[self.feature_cols + ['target_next_gw_points']].dropna()

        X = df_train[self.feature_cols].astype(np.float32)
        y = df_train['target_next_gw_points'].astype(np.float32)
        
        # Train model
        self.model = xgb.XGBRegressor(
            max_depth=6, learning_rate=0.1, n_estimators=150,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        self.model.fit(X, y)
        
        return True

    def _fetch_upcoming_fixtures(self):
        """Fetch each team's next N_FIXTURES fixtures with opponent and difficulty."""
        response = requests.get(BOOTSTRAP_URL, timeout=15)
        response.raise_for_status()
        bootstrap = response.json()

        self.team_of_player = {int(p['id']): int(p['team']) for p in bootstrap.get('elements', [])}
        team_names = {int(t['id']): t['short_name'] for t in bootstrap.get('teams', [])}

        fx_response = requests.get(FIXTURES_URL, timeout=15)
        fx_response.raise_for_status()
        fixtures = sorted(
            fx_response.json(),
            key=lambda f: (f.get('event') or 999, f.get('kickoff_time') or '')
        )

        self.upcoming_by_team = {}
        for fx in fixtures:
            if fx.get('event') is None:
                continue  # unscheduled (postponed) fixture
            for team, opp, is_home, fdr in (
                (fx['team_h'], fx['team_a'], True, fx.get('team_h_difficulty')),
                (fx['team_a'], fx['team_h'], False, fx.get('team_a_difficulty')),
            ):
                team_fixtures = self.upcoming_by_team.setdefault(int(team), [])
                if len(team_fixtures) < N_FIXTURES:
                    team_fixtures.append({
                        'gw': int(fx['event']),
                        'opponent_id': int(opp),
                        'opponent': team_names.get(int(opp), 'UNK'),
                        'is_home': is_home,
                        'fdr': int(fdr or 3),
                    })

    def predict_player(self, player_name, player_id=None, sentiment_score=0.0):

        # Get player's most recent record
        player_data = self.df[self.df['name'] == player_name]

        if len(player_data) == 0:
            return None

        latest = player_data.iloc[-1]
        latest_season = latest['season']

        # Start from model features computed during training.
        feature_row = latest[self.feature_cols].copy()

        if player_id is None:
            player_id = self.player_name_to_id.get(str(player_name))
        team_id = self.team_of_player.get(int(player_id)) if player_id else None
        if team_id is None:
            team_id = int(pd.to_numeric(latest.get('team_id', 0), errors='coerce') or 0)
        upcoming = self.upcoming_by_team.get(team_id, [])
        if not upcoming:
            # ponytail: no scheduled fixtures (season end) — one prediction from frozen features
            upcoming = [{
                'gw': None,
                'opponent_id': int(feature_row['opponent_team_feature']),
                'opponent': '',
                'is_home': bool(feature_row['is_home_feature']),
                'fdr': 3,
            }]

        sentiment_score = pd.to_numeric(sentiment_score, errors='coerce')
        if pd.isna(sentiment_score):
            sentiment_score = 0.0

        per_fixture = []
        for i, fx in enumerate(upcoming):
            feature_row['team_id_feature'] = float(team_id)
            feature_row['opponent_team_feature'] = float(fx['opponent_id'])
            feature_row['is_home_feature'] = 1.0 if fx['is_home'] else 0.0

            X_pred = feature_row.values.reshape(1, -1).astype(np.float32)
            X_pred = np.nan_to_num(X_pred, nan=0.0)
            pred = float(self.model.predict(X_pred)[0])

            if i == 0:
                # ep_next and news only cover the next GW; later fixtures are model-only.
                pred += float(sentiment_score)
                ep_next = pd.to_numeric(latest.get('ep_next', np.nan), errors='coerce')
                if not pd.isna(ep_next):
                    pred = 0.5 * pred + 0.5 * float(ep_next)

            per_fixture.append({
                'gw': fx['gw'],
                'opponent': fx['opponent'],
                'home': fx['is_home'],
                'fdr': fx['fdr'],
                'predicted_points': round(pred, 2),
            })

        next_5 = [f for f in per_fixture if f['gw'] is not None]
        predicted_next_gw = per_fixture[0]['predicted_points']
        weighted = weighted_total(
            [(f['predicted_points'], f['fdr']) for f in per_fixture]
        )

        position = str(latest.get('position', '')).upper()

        season_points = self.full_df[
            (self.full_df['name'] == player_name) &
            (self.full_df['season'] == latest_season)
        ]['total_points'].sum()
        
        # Get team and cost info
        team = str(latest.get('team', latest.get('team_id', 'Unknown')))
        now_cost = pd.to_numeric(latest.get('now_cost', 0), errors='coerce')
        
        return {
            'player_name': str(player_name),
            'position': position,
            'team': team,
            'now_cost': round(float(now_cost), 1) if now_cost > 0 else 0.0,  # CSV now_cost is already in £M
            'predicted_next_gw_points': round(predicted_next_gw, 2),
            'predicted_next_5_weighted': round(weighted, 2),
            'next_5': next_5,
            'current_season_points': round(float(season_points), 2),
        }

    def predict_all_current_players(self, current_players_list):
        self._fetch_upcoming_fixtures()

        sentiment_map = {}
        sentiment_df = pd.read_csv('data/sentiment_analysis.csv')
        sentiment_map = dict(zip(sentiment_df['player_name'], sentiment_df['sentiment_score']))

        predictions_list = []
        for player in current_players_list:
            player_name = player.get('name') or player.get('full_name')
            player_id = player.get('id') or player.get('element')
            if player_name and player_id:
                self.player_name_to_id[str(player_name)] = int(player_id)
            sentiment_score = float(sentiment_map.get(str(player_name), 0.0))
            pred = self.predict_player(player_name, player_id=player_id, sentiment_score=sentiment_score)
            if pred:
                predictions_list.append(pred)
        
        self.predictions = {p['player_name']: p for p in predictions_list}
        return predictions_list
    
    def save_predictions(self):
        with open('data/predictions.json', 'w') as f:
            json.dump(self.predictions, f, indent=2, default=float)

        # CSV keeps flat columns; per-fixture detail becomes a compact string.
        rows = []
        for p in self.predictions.values():
            row = {k: v for k, v in p.items() if k != 'next_5'}
            row['next_5_fixtures'] = ', '.join(
                f"{f['opponent']} ({'H' if f['home'] else 'A'}) {f['predicted_points']}"
                for f in p['next_5']
            )
            rows.append(row)
        pd.DataFrame(rows).to_csv('data/predictions.csv', index=False)
        

def main():
    """Main execution function."""
    predictor = PlayerPredictor()
    
    predictor.load_data()
    predictor.train_model()
    
    qualified_players_df = pd.read_csv('data/players.csv')
    current_players = qualified_players_df.to_dict('records')
    
    predictions = predictor.predict_all_current_players(current_players)
    
    predictor.save_predictions()
    

if __name__ == "__main__":
    main()

