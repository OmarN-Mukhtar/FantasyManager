import pandas as pd
import numpy as np
import xgboost as xgb
import json
import requests


BOOTSTRAP_URL = 'https://fantasy.premierleague.com/api/bootstrap-static/'
FIXTURES_URL = 'https://fantasy.premierleague.com/api/fixtures/?future=1'
# Cap, not a target — a season is 38 GWs; /fixtures/?future=1 just returns whatever's left,
# which shrinks as the season progresses. The model isn't trained "per horizon": it's a
# single-step regressor conditioned on (current rolling stats, opponent, venue, fdr), so it
# generalizes to any future fixture the same way, near or far — no special training needed.
MAX_FIXTURES_AHEAD = 38
N_FIXTURES_SUMMARY = 5  # how many fixtures feed the headline predicted_next_5_weighted stat


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

MIN_EXPERIENCE_SEASON = '2023-24'  # career-games count only looks back this far
EXPERIENCE_FULL_GWS = 7  # cold-start blend weight reaches 0 once a player has this many recent games
# ponytail: single per-GW decay knob. 0.99 -> ~69 GW half-life (~1.8 seasons), i.e. very slow.
# Retune down (e.g. 0.97) if predictions feel too slow to adapt to current form.
RECENCY_DECAY = 0.99

# Raw (non-rolled) context features, plus the subset usable for a player with zero history.
# ep_next/sentiment are NOT trained features — they're post-hoc corrections applied to the
# model's output at prediction time (see predict_player), not baked into what it learns.
# gameweeks_ahead is a trained feature (not a separate model per horizon): one model sees
# how far out the fixture is and can learn that e.g. recent minutes matter more for GW+1
# than GW+8. Trees don't extrapolate past what they've seen, so MAX_TRAINED_HORIZON caps
# both what we train on and what we feed at inference (see predict_player).
CONTEXT_FEATURES = [
    'team_id_feature', 'opponent_team_feature', 'is_home_feature',
    'fdr_feature', 'now_cost_feature', 'player_id_feature', 'gameweeks_ahead',
]
COLD_START_FEATURES = [
    'now_cost_feature', 'team_id_feature', 'opponent_team_feature',
    'is_home_feature', 'gameweeks_ahead',
]
MAX_TRAINED_HORIZON = 10
TRAIN_HORIZONS = list(range(1, MAX_TRAINED_HORIZON + 1))

# ponytail: post-hoc corrections (ep_next, sentiment) fade with distance the same way
# weighted_total does, rather than applying at full strength or not at all.
POST_HOC_DECAY = 0.8
# Sentiment score is -1..1 but ~65% of players sit at exactly 0 (no recent news), which is
# already "no effect" since it's additive. This just keeps a real +/-1 from swinging a
# prediction by a whole point — small nudge, not a dominant signal.
SENTIMENT_WEIGHT = 0.3

BASE_ROLLING_FEATURES = [
    'bps',
    'ict_index',
    'minutes',
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
# total_points dropped from BASE_ROLLING_FEATURES (rolling mean smooths away per-gameweek
# signal) in favor of explicit per-GW lags — see POINTS_LAGS in _build_features.
POINTS_LAGS = 5


class PlayerPredictor:
    
    def __init__(self, data_path='data/cleaned_merged_seasons.csv'):
        self.data_path = data_path
        self.full_df = None
        self.df = None
        self.model = None
        self.cold_model = None
        self.feature_cols = []
        self.predictions = {}
        self.player_name_to_id = {}
        self.player_stable_id = {}
        self.team_of_player = {}
        self.cost_of_player = {}
        self.ep_next_of_player = {}
        self.position_of_player = {}
        self.team_name_of_id = {}
        self.team_name_to_id = {}
        self.upcoming_by_team = {}
        
    def load_data(self):

        self.df = pd.read_csv(self.data_path, low_memory=False)
        self.full_df = self.df.copy()
        return True
    
    def _build_features(self):
        """Feature engineering only — no training. Split out so evaluate_horizons.py can
        reuse it without duplicating this logic."""
        self._fetch_upcoming_fixtures()  # need live team-name-to-id map before building features

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

        # Individual (non-averaged) points scored in each of the last POINTS_LAGS gameweeks —
        # points_gw_minus_1 is the GW right before "today", points_gw_minus_2 is two GWs
        # before, etc. Kept as separate lags rather than a rolling mean.
        for lag in range(1, POINTS_LAGS + 1):
            feature_name = f"points_gw_minus_{lag}"
            self.df[feature_name] = grouped['total_points'].shift(lag - 1)
            self.feature_cols.append(feature_name)

        # Keep forward predictions independent from defensive stats.
        is_fwd = self.df['position'].astype(str).str.upper().eq('FWD')
        if 'element_type' in self.df.columns:
            is_fwd = is_fwd | (pd.to_numeric(self.df['element_type'], errors='coerce') == 4)
        for base_col in ['goals_conceded', 'clean_sheets']:
            for window in ROLLING_WINDOWS:
                self.df.loc[is_fwd, f"{base_col}_rolling_{window}"] = 0.0

        # FPL's numeric team ids are reassigned every season (e.g. Liverpool has been
        # 9, 10, 11, and 12 in different seasons), so raw team_id/opponent_team columns
        # alias different clubs onto the same number across rows. Map club *names* onto
        # the live bootstrap's id space instead, extending it for clubs not currently in
        # the league, so ids stay a stable club identity consistent with what predict_player
        # sets at inference time.
        club_names = set(self.df['team'].dropna()) | set(self.df['opp_team_name'].dropna())
        next_id = max(self.team_name_to_id.values(), default=20) + 1
        for club in sorted(club_names - self.team_name_to_id.keys()):
            self.team_name_to_id[club] = next_id
            next_id += 1

        # Each row's own opponent/venue/fdr — this is that row's fixture, i.e. the context for
        # whichever gameweek THIS row's stats came from. Horizon-shifted in _stack_horizons to
        # align with the target gameweek instead, which is what a model must train on.
        self.df['team_id_feature'] = self.df['team'].map(self.team_name_to_id).fillna(0)
        self.df['opponent_team_feature'] = self.df['opp_team_name'].map(self.team_name_to_id).fillna(0)
        self.df['is_home_feature'] = pd.to_numeric(self.df.get('was_home', 0), errors='coerce').fillna(0)

        # fdr history only exists from whenever update_current_season.py started recording it
        # (see fdr_for_fixture there); older rows have no 'fdr' column value at all, so treat
        # missing as neutral difficulty rather than a wrong signal.
        fdr_col = self.df['fdr'] if 'fdr' in self.df.columns else pd.Series(np.nan, index=self.df.index)
        self.df['fdr_feature'] = pd.to_numeric(fdr_col, errors='coerce').fillna(3)
        self.df['now_cost_feature'] = pd.to_numeric(self.df.get('now_cost', 0), errors='coerce').fillna(0)

        # FPL's own player element ids are reassigned every season too (same trap as team ids),
        # so build a name-based stable id the same way, letting the model learn a per-player
        # fixed effect (e.g. penalty takers) on top of the rolling stats.
        all_names = sorted(set(self.df['name'].dropna()))
        self.player_stable_id = {name: i + 1 for i, name in enumerate(all_names)}
        self.df['player_id_feature'] = self.df['name'].map(self.player_stable_id).fillna(0)

        self.df['gameweeks_ahead'] = 0.0  # placeholder; real values only exist in the stacked training frame
        self.feature_cols.extend(CONTEXT_FEATURES)

        # Recency weighting: older rows count for less, decaying very slowly per gameweek so a
        # game from a year ago still carries meaningful weight (see RECENCY_DECAY).
        season_start_year = pd.to_numeric(self.df['season'].astype(str).str.slice(0, 4), errors='coerce')
        gw_num = pd.to_numeric(self.df['GW'], errors='coerce')
        self.df['season_ordinal'] = season_start_year * 38 + gw_num

        return self.df

    def _stack_horizons(self, df, horizons):
        """Build one training row per (real row, horizon h): rolling stats stay as they were
        'today', but the target and fixture context (opponent/venue/fdr) are pulled from h
        gameweeks ahead — the actual future fixture, not today's. gameweeks_ahead is fed in as
        a feature so a single model can learn horizon-dependent patterns instead of needing a
        separate model per h."""
        grouped = df.groupby('name', sort=False)
        base_cols = [c for c in self.feature_cols if c != 'gameweeks_ahead']

        parts = []
        for h in horizons:
            part = df[base_cols + ['season_ordinal']].copy()
            part['target'] = grouped['total_points'].shift(-h)
            part['opponent_team_feature'] = grouped['opponent_team_feature'].shift(-h)
            part['is_home_feature'] = grouped['is_home_feature'].shift(-h)
            part['fdr_feature'] = grouped['fdr_feature'].shift(-h)
            part['gameweeks_ahead'] = float(h)
            parts.append(part)

        stacked = pd.concat(parts, ignore_index=True)
        return stacked[self.feature_cols + ['target', 'season_ordinal']].dropna()

    def train_model(self):
        self._build_features()
        df_train = self._stack_horizons(self.df, TRAIN_HORIZONS)

        X = df_train[self.feature_cols].astype(np.float32)
        y = df_train['target'].astype(np.float32)
        max_ordinal = df_train['season_ordinal'].max()
        sample_weight = RECENCY_DECAY ** (max_ordinal - df_train['season_ordinal'])

        # Full-featured model: every feature, for players with enough recent history.
        # Hyperparams from Optuna tuning in notebooks/model_exploration.ipynb (100 trials,
        # best XGBoost MAE=0.834 on the GW33-37 2025-26 holdout).
        self.model = xgb.XGBRegressor(
            max_depth=4, learning_rate=0.108, n_estimators=290,
            subsample=0.977, colsample_bytree=0.626, min_child_weight=4,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        self.model.fit(X, y, sample_weight=sample_weight)

        # Cold-start model: price/team/opponent/venue/horizon only, for players who don't yet
        # have enough recent history for the rolling features to mean much.
        self.cold_model = xgb.XGBRegressor(
            max_depth=4, learning_rate=0.1, n_estimators=150,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        self.cold_model.fit(X[COLD_START_FEATURES], y, sample_weight=sample_weight)

        return True

    @staticmethod
    def _get_position_name(element_type):
        return {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}.get(element_type, 'Unknown')

    def _career_games_count(self, player_name):
        """Rows this player has from MIN_EXPERIENCE_SEASON onward, used for cold-start blending."""
        hist = self.full_df[
            (self.full_df['name'] == player_name) &
            (self.full_df['season'] >= MIN_EXPERIENCE_SEASON)
        ]
        return len(hist)

    def _fetch_upcoming_fixtures(self):
        """Fetch each team's remaining fixtures (up to MAX_FIXTURES_AHEAD) with opponent and difficulty."""
        if self.team_name_of_id:
            return  # already fetched this run

        response = requests.get(BOOTSTRAP_URL, timeout=15)
        response.raise_for_status()
        bootstrap = response.json()

        self.team_of_player = {int(p['id']): int(p['team']) for p in bootstrap.get('elements', [])}
        self.cost_of_player = {int(p['id']): p['now_cost'] / 10 for p in bootstrap.get('elements', [])}
        self.ep_next_of_player = {
            int(p['id']): float(p['ep_next']) for p in bootstrap.get('elements', [])
            if p.get('ep_next') not in (None, '')
        }
        self.position_of_player = {
            int(p['id']): self._get_position_name(int(p['element_type']))
            for p in bootstrap.get('elements', [])
        }
        team_names = {int(t['id']): t['short_name'] for t in bootstrap.get('teams', [])}
        self.team_name_of_id = {int(t['id']): t['name'] for t in bootstrap.get('teams', [])}
        self.team_name_to_id = {name: id_ for id_, name in self.team_name_of_id.items()}

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
                if len(team_fixtures) < MAX_FIXTURES_AHEAD:
                    team_fixtures.append({
                        'gw': int(fx['event']),
                        'opponent_id': int(opp),
                        'opponent': team_names.get(int(opp), 'UNK'),
                        'is_home': is_home,
                        'fdr': int(fdr or 3),
                    })

    def predict_player(self, player_name, player_id=None, sentiment_score=0.0):

        if player_id is None:
            player_id = self.player_name_to_id.get(str(player_name))
        player_id_int = int(player_id) if player_id else None

        player_data = self.df[self.df['name'] == player_name]
        has_history = len(player_data) > 0
        latest = player_data.iloc[-1] if has_history else None
        latest_season = latest['season'] if has_history else None

        # Cold-start blend weight: 1.0 at 0 recent games, fading to 0 by EXPERIENCE_FULL_GWS.
        games_count = self._career_games_count(player_name)
        cold_weight = max(0.0, min(1.0, (EXPERIENCE_FULL_GWS - games_count) / EXPERIENCE_FULL_GWS))
        full_weight = 1.0 - cold_weight

        team_id = self.team_of_player.get(player_id_int) if player_id_int else None
        if team_id is None and has_history:
            team_id = int(pd.to_numeric(latest.get('team_id', 0), errors='coerce') or 0)
        if team_id is None:
            return None  # no live team and no history at all — nothing to predict from

        now_cost_live = self.cost_of_player.get(player_id_int) if player_id_int else None
        ep_next_live = self.ep_next_of_player.get(player_id_int) if player_id_int else None

        if has_history:
            # Start from model features computed during training, refreshed with live price.
            feature_row = latest[self.feature_cols].copy()
            if now_cost_live is not None:
                feature_row['now_cost_feature'] = now_cost_live
        else:
            # Brand-new player: no rolling stats exist yet, so only the cold-start features
            # (which get cold_weight == 1.0 here) carry real information.
            feature_row = pd.Series(0.0, index=self.feature_cols)
            feature_row['now_cost_feature'] = now_cost_live if now_cost_live is not None else 0.0
            # Never collapse unseen players onto a shared id 0 - mint each one a fresh unique id.
            if player_name not in self.player_stable_id:
                self.player_stable_id[player_name] = max(self.player_stable_id.values(), default=0) + 1
            feature_row['player_id_feature'] = self.player_stable_id[player_name]

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
            feature_row['fdr_feature'] = float(fx.get('fdr', 3))
            # Cap at MAX_TRAINED_HORIZON: trees don't extrapolate past what they've seen, so
            # feeding a bigger number than that just wastes the signal, not corrupts it.
            feature_row['gameweeks_ahead'] = float(min(i + 1, MAX_TRAINED_HORIZON))

            X_full = feature_row[self.feature_cols].values.reshape(1, -1).astype(np.float32)
            X_full = np.nan_to_num(X_full, nan=0.0)
            full_pred = float(self.model.predict(X_full)[0])

            X_cold = feature_row[COLD_START_FEATURES].values.reshape(1, -1).astype(np.float32)
            X_cold = np.nan_to_num(X_cold, nan=0.0)
            cold_pred = float(self.cold_model.predict(X_cold)[0])

            pred = full_weight * full_pred + cold_weight * cold_pred

            # Post-hoc corrections: ep_next and sentiment are never trained features, just
            # adjustments layered on top of the model's own prediction. Both fade with distance
            # (same 0.8/GW shape as weighted_total) instead of applying fully at GW+1 and not
            # at all afterward.
            fixture_decay = POST_HOC_DECAY ** i
            pred += fixture_decay * SENTIMENT_WEIGHT * float(sentiment_score)
            if ep_next_live is not None:
                ep_next_weight = 0.5 * fixture_decay
                pred = (1 - ep_next_weight) * pred + ep_next_weight * float(ep_next_live)

            per_fixture.append({
                'gw': fx['gw'],
                'opponent': fx['opponent'],
                'home': fx['is_home'],
                'fdr': fx['fdr'],
                'predicted_points': round(pred, 2),
            })

        season_fixtures = [f for f in per_fixture if f['gw'] is not None]
        next_5 = season_fixtures[:N_FIXTURES_SUMMARY]
        predicted_next_gw = per_fixture[0]['predicted_points']
        # Headline "next 5" stat stays scoped to the near term (what squad_optimizer.py
        # actually acts on); season_fixtures below carries the full remaining-season list.
        weighted = weighted_total(
            [(f['predicted_points'], f['fdr']) for f in next_5]
        )

        if has_history:
            position = str(latest.get('position', '')).upper()
            season_points = self.full_df[
                (self.full_df['name'] == player_name) &
                (self.full_df['season'] == latest_season)
            ]['total_points'].sum()
        else:
            position = self.position_of_player.get(player_id_int, '')
            season_points = 0.0

        # Get team and cost info. Prefer live bootstrap values (handles summer transfers
        # and in-season price changes); historical CSV values can be stale.
        team = self.team_name_of_id.get(team_id, str(latest.get('team', latest.get('team_id', 'Unknown'))) if has_history else 'Unknown')
        now_cost = now_cost_live
        if now_cost is None:
            now_cost = pd.to_numeric(latest.get('now_cost', 0), errors='coerce') if has_history else 0.0

        return {
            'player_name': str(player_name),
            'position': position,
            'team': team,
            'now_cost': round(float(now_cost), 1) if now_cost > 0 else 0.0,  # CSV now_cost is already in £M
            'predicted_next_gw_points': round(predicted_next_gw, 2),
            'predicted_next_5_weighted': round(weighted, 2),
            'next_5': next_5,
            'season_fixtures': season_fixtures,
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
            row = {k: v for k, v in p.items() if k not in ('next_5', 'season_fixtures')}
            row['next_5_fixtures'] = ', '.join(
                f"{f['opponent']} ({'H' if f['home'] else 'A'}) {f['predicted_points']}"
                for f in p['next_5']
            )
            row['season_fixtures'] = ', '.join(
                f"GW{f['gw']} {f['opponent']} ({'H' if f['home'] else 'A'}) {f['predicted_points']}"
                for f in p['season_fixtures']
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

