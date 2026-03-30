import pandas as pd
import numpy as np
import xgboost as xgb
import json
import requests


MODEL_CONFIG = {
    'xgb': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 150,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }
}

BOOTSTRAP_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'
FIXTURES_URL = 'https://fantasy.premierleague.com/api/fixtures/?event={event_id}'

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
        self.next_fixture_by_player_id = {}
        
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
        self.model = xgb.XGBRegressor(**MODEL_CONFIG['xgb'])
        self.model.fit(X, y)
        
        return True

    def _fetch_next_gameweek_fixture_map(self):
        """Fetch next GW fixtures and build player->(team,opponent) map."""
        response = requests.get(BOOTSTRAP_URL, timeout=15)
        response.raise_for_status()
        bootstrap = response.json()

        next_event = next((e for e in bootstrap['events'] if e.get('is_next')), None)
        if next_event is None:
            next_event = next((e for e in bootstrap['events'] if e.get('is_current')), None)
        if next_event is None:
            self.next_fixture_by_player_id = {}
            return

        next_gw = int(next_event['id'])
        player_team_map = {int(p['id']): int(p['team']) for p in bootstrap.get('elements', [])}

        fx_response = requests.get(FIXTURES_URL.format(event_id=next_gw), timeout=15)
        fx_response.raise_for_status()
        fixtures = fx_response.json()

        fixture_context_by_team = {}
        for fx in fixtures:
            home = int(fx.get('team_h', 0))
            away = int(fx.get('team_a', 0))
            if home and away:
                fixture_context_by_team[home] = {'opponent_team_feature': away, 'is_home_feature': 1.0}
                fixture_context_by_team[away] = {'opponent_team_feature': home, 'is_home_feature': 0.0}

        self.next_fixture_by_player_id = {
            player_id: {
                'team_id_feature': team_id,
                'opponent_team_feature': fixture_context_by_team.get(team_id, {}).get('opponent_team_feature', 0),
                'is_home_feature': fixture_context_by_team.get(team_id, {}).get('is_home_feature', 0.0),
            }
            for player_id, team_id in player_team_map.items()
        }
    
    def predict_player(self, player_name, player_id=None, sentiment_score=-0.05):

        # Get player's most recent record
        player_data = self.df[self.df['name'] == player_name]
        
        if len(player_data) == 0:
            return None
        
        latest = player_data.iloc[-1]
        latest_season = latest['season']

        # Start from model features computed during training.
        feature_row = latest[self.feature_cols].copy()

        # Override team/opponent with next GW fixture context if available.
        if player_id is None:
            player_id = self.player_name_to_id.get(str(player_name))
        fixture_context = self.next_fixture_by_player_id.get(int(player_id), {}) if player_id else {}
        if fixture_context:
            feature_row['team_id_feature'] = float(fixture_context.get('team_id_feature', 0))
            feature_row['opponent_team_feature'] = float(fixture_context.get('opponent_team_feature', 0))
            feature_row['is_home_feature'] = float(fixture_context.get('is_home_feature', 0.0))
        
        X_pred = feature_row.values.reshape(1, -1).astype(np.float32)
        X_pred = np.nan_to_num(X_pred, nan=0.0)
        
        # Predict next gameweek points
        xgb_pred = float(self.model.predict(X_pred)[0])
        sentiment_score = pd.to_numeric(sentiment_score, errors='coerce')
        if pd.isna(sentiment_score):
            sentiment_score = -0.05
        xgb_pred += float(sentiment_score)
        ep_next = pd.to_numeric(latest.get('ep_next', np.nan), errors='coerce')
        if pd.isna(ep_next):
            ep_next = xgb_pred
        predicted_next_gw = 0.5 * xgb_pred + 0.5 * float(ep_next)

        position = str(latest.get('position', '')).upper()
        multiplier_map = {'GK': 1.5, 'DEF': 1.75, 'MID': 2.0, 'FWD': 2.0}
        predicted_next_gw *= multiplier_map.get(position, 1.0)

        season_points = self.full_df[
            (self.full_df['name'] == player_name) &
            (self.full_df['season'] == latest_season)
        ]['total_points'].sum()
        
        return {
            'player_name': str(player_name),
            'position': position,
            'predicted_next_gw_points': round(predicted_next_gw, 2),
            'current_season_points': round(float(season_points), 2),
        }
    
    def predict_all_current_players(self, current_players_list):
        self._fetch_next_gameweek_fixture_map()

        sentiment_map = {}
        sentiment_df = pd.read_csv('data/sentiment_analysis.csv')
        sentiment_map = dict(zip(sentiment_df['player_name'], sentiment_df['sentiment_score']))

        predictions_list = []
        for player in current_players_list:
            player_name = player.get('name') or player.get('full_name')
            player_id = player.get('id') or player.get('element')
            if player_name and player_id:
                self.player_name_to_id[str(player_name)] = int(player_id)
            sentiment_score = float(sentiment_map.get(str(player_name), -0.05))
            pred = self.predict_player(player_name, player_id=player_id, sentiment_score=sentiment_score)
            if pred:
                predictions_list.append(pred)
        
        self.predictions = {p['player_name']: p for p in predictions_list}
        return predictions_list
    
    def save_predictions(self):
        with open('data/predictions.json', 'w') as f:
            json.dump(self.predictions, f, indent=2, default=float)
        
        predictions_df = pd.DataFrame(self.predictions.values())
        predictions_df.to_csv('data/predictions.csv', index=False)
        

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

