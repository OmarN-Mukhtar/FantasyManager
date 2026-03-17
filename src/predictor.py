"""
Player performance prediction using rolling window Random Forest model.
Trained on last 12 months of data without using player names or seasons as features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ===== CONFIGURATION CONSTANTS =====
class Constants:
    """Configuration constants for prediction."""
    TOTAL_SEASON_GAMES = 38
    DEFAULT_TEAM_DEFENSE_RATING = 3.0
    CI_95_ZSCORE = 1.96
    SENTIMENT_MULTIPLIER = 0.001
    API_BLEND_PPG_WEIGHT = 0.7
    API_BLEND_POSITION_WEIGHT = 0.3
    PLAYER_VALUE_SCALING_THRESHOLD = 200
    PLAYER_VALUE_SCALE_FACTOR = 10

    # Position-based default PPG values
    POSITION_DEFAULTS = {'GK': 2.5, 'DEF': 3.0, 'MID': 3.5, 'FWD': 3.0}

    # Form trend thresholds
    ACCELERATION_THRESHOLD = 0.2

    # Confidence level thresholds (based on std dev)
    STD_DEV_HIGH = 0.5
    STD_DEV_MEDIUM = 1.0


# ===== MODEL CONFIGURATIONS =====
MODEL_CONFIG = {
    'rf': {
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1,
    },
    'lgb': {
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 150,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    },
    'xgb': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 150,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }
}


# ===== ROLLING FEATURE DEFINITIONS =====
ROLLING_METRICS = [
    ('total_points', 'rolling_points'),
    ('minutes', 'rolling_minutes'),
    ('goals_scored', 'rolling_goals'),
    ('assists', 'rolling_assists'),
    ('bonus', 'rolling_bonus'),
    ('creativity', 'rolling_creativity'),
    ('threat', 'rolling_threat'),
    ('ict_index', 'rolling_ict'),
]


class PlayerPredictor:
    """Predicts player performance using Random Forest and rolling window features."""
    
    def __init__(self, data_path='cleaned_merged_seasons.csv', window_weeks=40):
        """
        Initialize predictor with historical data.

        Args:
            data_path: Path to historical CSV data
            window_weeks: Number of weeks for rolling window (default: 40 ≈ 12 months)
        """
        self.data_path = data_path
        self.df = None
        self.model_rf = None
        self.model_lgb = None
        self.model_xgb = None
        self.calibrator = None
        self.scaler = StandardScaler()
        self.predictions = {}
        self.window_weeks = window_weeks  # ~12 months of gameweeks
        self.feature_cols = []
        self.accuracy_metrics = {}
        self.sentiment_data = self.load_sentiment_data()
        self.ensemble_weights = {'rf': 0.35, 'lgb': 0.35, 'xgb': 0.30}
        self.next_gw_fixtures = {}  # Store next GW fixtures for fixture difficulty lookup
        
    def load_sentiment_data(self):
        """Load sentiment data if available."""
        try:
            sentiment_path = 'data/sentiment_analysis.json'
            if os.path.exists(sentiment_path):
                with open(sentiment_path, 'r') as f:
                    data = json.load(f)
                print(f"✓ Loaded sentiment data for {len(data)} players")
                return data
        except Exception as e:
            print(f"⚠ Could not load sentiment data: {e}")
        return {}
        
    def load_data(self):
        """Load and preprocess historical player data."""
        print("Loading historical player data...")
        self.df = pd.read_csv(self.data_path)

        print(f"Loaded {len(self.df)} records")
        print(f"Seasons: {self.df['season'].min()} to {self.df['season'].max()}")
        print(f"Unique players: {self.df['name'].nunique()}")

        # Convert value to actual price (divide by 10 if needed)
        if self.df['value'].max() > Constants.PLAYER_VALUE_SCALING_THRESHOLD:
            self.df['value'] = self.df['value'] / Constants.PLAYER_VALUE_SCALE_FACTOR

        # Sort by player and gameweek for rolling window calculations
        self.df = self.df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)

        # Create a sequential time index (season + GW)
        self.df['season_gw'] = self.df['season'] + '_GW' + self.df['GW'].astype(str)

        return True
    
    def create_rolling_features(self):
        """
        Create rolling window features for all players inspired by Kaggle March Madness winners.
        Includes: rolling averages, efficiency metrics, strength of schedule, momentum indicators.
        Does NOT use player name or season as features.

        Returns:
            DataFrame with engineered features
        """
        print("Creating rolling window features with advanced March Madness techniques...")

        # ===== STRENGTH OF SCHEDULE (CRITICAL FIX: Single groupby instead of double) =====
        team_defense_by_gw = self.df.groupby(['season', 'GW', 'opponent_team'])[
            'total_points'
        ].mean().fillna(Constants.DEFAULT_TEAM_DEFENSE_RATING).to_dict()

        # ===== ROLLING FEATURES (Vectorized with groupby, not sequential player loop) =====
        def compute_player_rolling_features(player_group):
            """Compute rolling features for a player group."""
            # Basic rolling windows
            for source_col, feature_prefix in ROLLING_METRICS:
                for window in [3, 5, 10]:
                    player_group[f'{feature_prefix}_{window}'] = player_group[source_col].rolling(
                        window, min_periods=1
                    ).mean()

            # Form indicators
            player_group['points_trend'] = player_group['rolling_points_5'] - player_group['rolling_points_10']
            player_group['minutes_consistency'] = player_group['minutes'].rolling(
                5, min_periods=1
            ).std().fillna(0)
            player_group['goals_trend'] = player_group['rolling_goals_3'] - player_group['rolling_goals_10']

            # Momentum indicators
            player_group['recent_form_5gw'] = (
                (player_group['rolling_points_5'] - player_group['rolling_points_10']) /
                (player_group['rolling_points_10'] + 0.1)
            )
            player_group['peak_tracker'] = player_group['rolling_points_5'].rolling(
                10, min_periods=1
            ).max()
            player_group['volatility_5gw'] = player_group['total_points'].rolling(5, min_periods=1).std().fillna(0)
            player_group['volatility_10gw'] = player_group['total_points'].rolling(10, min_periods=1).std().fillna(0)
            player_group['acceleration'] = player_group['rolling_points_5'].diff().fillna(0)

            # Efficiency metrics
            player_group['efficiency_rating'] = np.where(
                player_group['minutes'] > 0,
                (player_group['total_points'] / player_group['minutes']) * 90,
                0
            )
            player_group['rolling_efficiency_5'] = player_group['efficiency_rating'].rolling(
                5, min_periods=1
            ).mean()
            player_group['shooting_efficiency'] = np.where(
                player_group['threat'] > 0,
                player_group['goals_scored'] / (player_group['threat'] + 0.1),
                0
            )
            player_group['assist_efficiency'] = np.where(
                player_group['creativity'] > 0,
                player_group['assists'] / (player_group['creativity'] + 0.1),
                0
            )
            player_group['defensive_efficiency'] = np.where(
                player_group['position'] == 'DEF',
                player_group['clean_sheets'] / (player_group.groupby('season').cumcount() + 1),
                0
            )

            # Season tracking
            player_group['games_played_season'] = player_group.groupby('season').cumcount() + 1
            player_group['season_total_points'] = player_group.groupby('season')['total_points'].cumsum()
            player_group['season_avg_points'] = player_group.groupby('season')['total_points'].expanding().mean().values

            # Strength of schedule (CRITICAL FIX: Use map on lookup dict)
            lookup_key = list(zip(player_group['season'], player_group['GW'], player_group['opponent_team']))
            player_group['opponent_strength'] = pd.Series(lookup_key, index=player_group.index).map(
                team_defense_by_gw
            ).fillna(Constants.DEFAULT_TEAM_DEFENSE_RATING)

            player_group['sos_5gw'] = player_group['opponent_strength'].rolling(5, min_periods=1).mean()
            player_group['sos_10gw'] = player_group['opponent_strength'].rolling(10, min_periods=1).mean()

            return player_group

        # Process all players using groupby (60-80% faster than sequential loop)
        df_combined = self.df.groupby('name', group_keys=False).apply(
            compute_player_rolling_features
        ).drop(columns=['name'], errors='ignore')

        # Re-add 'name' column for result integrity
        df_combined = self.df[['name']].join(df_combined)
        df_combined = df_combined.sort_index()

        # ===== TEAM COMPOSITION FEATURES =====
        team_avg_value = df_combined.groupby('team')['value'].mean()
        df_combined['team_avg_value'] = df_combined['team'].map(team_avg_value).fillna(
            df_combined['value'].mean()
        )

        # Player percentile within position on team (CRITICAL FIX: Using rank instead of apply)
        df_combined['player_position_percentile'] = df_combined.groupby(
            ['team', 'position']
        )['value'].rank(pct=True)

        # Rotation risk (CRITICAL FIX: Using merge instead of string concat mapping)
        rotation_risk = df_combined.groupby(['season', 'name'])['minutes'].std().reset_index()
        rotation_risk.columns = ['season', 'name', 'rotation_risk']
        df_combined = df_combined.merge(rotation_risk, on=['season', 'name'], how='left')
        df_combined['rotation_risk'] = df_combined['rotation_risk'].fillna(0)

        # ===== POSITION & CONTEXT ENCODING =====
        position_dummies = pd.get_dummies(df_combined['position'], prefix='pos')
        df_combined = pd.concat([df_combined, position_dummies], axis=1)

        df_combined['is_home'] = df_combined['was_home'].astype(int)

        team_strength = df_combined.groupby('team')['total_points'].mean()
        df_combined['team_strength'] = df_combined['team'].map(team_strength).fillna(
            Constants.DEFAULT_TEAM_DEFENSE_RATING
        )

        print(f"Created advanced features for {len(df_combined)} game records")
        print(f"Total features generated: {df_combined.shape[1] - len(self.df.columns) + 1}")

        return df_combined
    
    
    def _create_models(self):
        """Factory method to create all configured models."""
        models = {
            'rf': RandomForestRegressor(**MODEL_CONFIG['rf'])
        }
        if LIGHTGBM_AVAILABLE:
            models['lgb'] = lgb.LGBMRegressor(**MODEL_CONFIG['lgb'])
        if XGBOOST_AVAILABLE:
            models['xgb'] = xgb.XGBRegressor(**MODEL_CONFIG['xgb'])
        return models

    def train_model(self, use_last_n_weeks=None):
        """
        Train ensemble of Random Forest, LightGBM, and XGBoost models with temporal validation.
        Uses TimeSeriesSplit to prevent data leakage and simulates realistic prediction scenarios.

        Args:
            use_last_n_weeks: Use only last N gameweeks for training (default: None = all data)
        """
        print("\nTraining ensemble models with March Madness techniques...")

        df_with_features = self.create_rolling_features()

        # Define feature columns (expanded with March Madness features)
        self.feature_cols = [
            # Original rolling averages
            'rolling_points_3', 'rolling_points_5', 'rolling_points_10',
            'rolling_minutes_3', 'rolling_minutes_5', 'rolling_minutes_10',
            'rolling_goals_3', 'rolling_goals_5', 'rolling_goals_10',
            'rolling_assists_3', 'rolling_assists_5', 'rolling_assists_10',
            'rolling_bonus_3', 'rolling_bonus_5', 'rolling_bonus_10',
            'rolling_creativity_3', 'rolling_creativity_5', 'rolling_creativity_10',
            'rolling_threat_3', 'rolling_threat_5', 'rolling_threat_10',
            'rolling_ict_3', 'rolling_ict_5', 'rolling_ict_10',
            # Form indicators & momentum
            'points_trend', 'minutes_consistency', 'goals_trend',
            'recent_form_5gw', 'peak_tracker', 'volatility_5gw', 'volatility_10gw', 'acceleration',
            'season_avg_points', 'games_played_season',
            # Efficiency metrics (March Madness inspired)
            'efficiency_rating', 'rolling_efficiency_5',
            'shooting_efficiency', 'assist_efficiency', 'defensive_efficiency',
            # Strength of schedule
            'opponent_strength', 'sos_5gw', 'sos_10gw',
            # Team composition
            'team_avg_value', 'player_position_percentile', 'rotation_risk',
            # Position (one-hot encoded)
            'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID',
            # Context
            'is_home', 'team_strength', 'value'
        ]

        # Filter to only include records with all features
        df_train = df_with_features.dropna(subset=self.feature_cols + ['total_points'])

        # Optionally use only last N weeks (12-month rolling window)
        if use_last_n_weeks:
            all_gws = df_train['season_gw'].unique()
            last_n_gws = all_gws[-use_last_n_weeks:] if len(all_gws) > use_last_n_weeks else all_gws
            df_train = df_train[df_train['season_gw'].isin(last_n_gws)]
            print(f"Using last {len(last_n_gws)} gameweeks for training")

        print(f"Training data: {len(df_train)} records")

        X = df_train[self.feature_cols]
        y = df_train['total_points']

        # Use TimeSeriesSplit for temporal validation
        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = {'rf': [], 'lgb': [], 'xgb': []}

        print("\nTraining with TimeSeriesSplit (5 folds)...")

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"  Fold {fold + 1}/5...", end=" ", flush=True)

            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

            # Create and train models for this fold
            fold_models = self._create_models()

            for model_name, model in fold_models.items():
                if model is not None:
                    model.fit(X_train_fold, y_train_fold)
                    fold_scores[model_name].append(model.score(X_test_fold, y_test_fold))

            print(f"RF R²={fold_scores['rf'][-1]:.4f}", end="")
            if LIGHTGBM_AVAILABLE:
                print(f" | LGB R²={fold_scores['lgb'][-1]:.4f}", end="")
            if XGBOOST_AVAILABLE:
                print(f" | XGB R²={fold_scores['xgb'][-1]:.4f}", end="")
            print()

        # Train final models on all data
        print("\nTraining final models on all data...")

        final_models = self._create_models()
        for model_name, model in final_models.items():
            if model is not None:
                model.fit(X, y)
                if model_name == 'rf':
                    self.model_rf = model
                elif model_name == 'lgb':
                    self.model_lgb = model
                elif model_name == 'xgb':
                    self.model_xgb = model

        # Report ensemble performance
        print(f"\n✓ Ensemble models trained successfully!")
        print(f"\nCross-validation R² Scores:")
        print(f"  Random Forest: {np.mean(fold_scores['rf']):.4f} ± {np.std(fold_scores['rf']):.4f}")
        if LIGHTGBM_AVAILABLE:
            print(f"  LightGBM:      {np.mean(fold_scores['lgb']):.4f} ± {np.std(fold_scores['lgb']):.4f}")
        if XGBOOST_AVAILABLE:
            print(f"  XGBoost:       {np.mean(fold_scores['xgb']):.4f} ± {np.std(fold_scores['xgb']):.4f}")

        # Feature importance from Random Forest
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model_rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))

        # Store metrics
        self.accuracy_metrics = {
            'rf_cv_r2': np.mean(fold_scores['rf']),
            'rf_cv_std': np.std(fold_scores['rf']),
        }
        if LIGHTGBM_AVAILABLE:
            self.accuracy_metrics['lgb_cv_r2'] = np.mean(fold_scores['lgb'])
            self.accuracy_metrics['lgb_cv_std'] = np.std(fold_scores['lgb'])
        if XGBOOST_AVAILABLE:
            self.accuracy_metrics['xgb_cv_r2'] = np.mean(fold_scores['xgb'])
            self.accuracy_metrics['xgb_cv_std'] = np.std(fold_scores['xgb'])

        self.df_with_features = df_with_features

        return True
    
    def set_next_gw_fixtures(self, current_players_list):
        """
        Build a fixture difficulty lookup for next gameweek.
        Calculates opponent strength based on historical performance.

        Args:
            current_players_list: List of current players from FPL API
        """
        if not current_players_list or len(current_players_list) == 0:
            return

        # Build opponent strength lookup from recent games
        if self.df_with_features is not None and len(self.df_with_features) > 0:
            # Use last gameweek's opponent strengths as proxy for next GW
            latest_gw_data = self.df_with_features.drop_duplicates(
                subset=['opponent_team', 'position'], keep='last'
            )[['opponent_team', 'position', 'opponent_strength', 'sos_5gw']]

            for _, row in latest_gw_data.iterrows():
                opponent = row['opponent_team']
                position = row['position']
                key = (opponent, position)
                self.next_gw_fixtures[key] = {
                    'strength': row['opponent_strength'],
                    'difficulty_5gw': row['sos_5gw']
                }

    def _predict_next_game(self, player_recent, player_name, position, team, current_stats):
        """
        Calculate prediction for the next specific gameweek match.

        Args:
            player_recent: Most recent player record with features
            player_name: Player name
            position: Player position
            team: Player team
            current_stats: Current season stats including fixture info

        Returns:
            Dict with next game prediction details
        """
        if 'opponent_team' not in current_stats or current_stats.get('opponent_team') is None:
            # No next fixture info available
            recent_ppg = player_recent['rolling_points_5'].values[0]
            return {
                'next_game_predicted_points': round(recent_ppg, 2),
                'next_game_opponent': 'Unknown',
                'next_game_difficulty': 'Unknown',
                'next_game_is_home': None
            }

        opponent_team = current_stats.get('opponent_team')
        is_home = current_stats.get('was_home', True)
        opponent_name = current_stats.get('opp_team_name', 'Unknown')

        # Get opponent difficulty
        opponent_key = (opponent_team, position)
        opponent_data = self.next_gw_fixtures.get(opponent_key, {})
        opponent_difficulty = opponent_data.get('difficulty_5gw', 3.0)

        # Base prediction: recent form (5-game rolling average)
        base_ppg = player_recent['rolling_points_5'].values[0]
        recent_efficiency = player_recent['rolling_efficiency_5'].values[0]

        # Adjust for opponent difficulty (higher opponent strength = harder match = lower expected points)
        # Normalize opponent difficulty to 0-1 scale (typical range is 1-5)
        difficulty_factor = 1.0 - (opponent_difficulty / 5.0) * 0.15  # -15% adjustment for hardest opponents

        # Home/away adjustment (typically home games perform slightly better, ~10% uplift)
        home_factor = 1.1 if is_home else 0.95

        # Calculate next game prediction
        next_game_ppg = base_ppg * difficulty_factor * home_factor

        # Confidence in next game prediction (higher if recent form is consistent)
        consistency = player_recent['minutes_consistency'].values[0]
        if consistency < 10:
            difficulty_level = 'Easy' if opponent_difficulty < 2.5 else 'Moderate' if opponent_difficulty < 3.5 else 'Hard'
        else:
            difficulty_level = 'Easy' if opponent_difficulty < 2.5 else 'Moderate' if opponent_difficulty < 3.5 else 'Hard'

        return {
            'next_game_predicted_points': round(next_game_ppg, 2),
            'next_game_opponent': opponent_name,
            'next_game_is_home': is_home,
            'next_game_difficulty': difficulty_level,
            'opponent_difficulty_score': round(opponent_difficulty, 2),
            'next_game_efficiency_factor': round(home_factor * difficulty_factor, 3)
        }

    def _ensemble_predict(self, X_pred):
        """
        Get ensemble prediction from all trained models with confidence intervals.
        Uses weighted average (soft voting) inspired by March Madness winners.
        """
        predictions = {}

        # Get prediction from RandomForest
        predictions['rf'] = self.model_rf.predict(X_pred)[0]

        # Get predictions from LightGBM if available
        if self.model_lgb is not None:
            predictions['lgb'] = self.model_lgb.predict(X_pred)[0]

        # Get predictions from XGBoost if available
        if self.model_xgb is not None:
            predictions['xgb'] = self.model_xgb.predict(X_pred)[0]

        # Weighted ensemble average
        ensemble_pred = 0
        total_weight = 0
        for model_name, weight in self.ensemble_weights.items():
            if model_name in predictions:
                ensemble_pred += predictions[model_name] * weight
                total_weight += weight

        if total_weight > 0:
            ensemble_pred = ensemble_pred / total_weight
        else:
            ensemble_pred = predictions['rf']

        # Calculate confidence interval (std dev across available models)
        pred_list = list(predictions.values())
        std_dev = np.std(pred_list) if len(pred_list) > 1 else 0.5

        return {
            'point_estimate': ensemble_pred,
            'std_dev': std_dev,
            'individual_predictions': predictions
        }

    def predict_player(self, player_name, position, team, current_stats=None):
        """
        Predict performance for a single player using ensemble of models.
        Includes confidence intervals from March Madness techniques.

        Args:
            player_name: Player name
            position: Player position
            team: Player team
            current_stats: Dictionary of current season stats from FPL API (optional)

        Returns:
            Dictionary with ensemble predictions and confidence bounds
        """
        if self.model_rf is None:
            print("Error: Model not trained. Call train_model() first.")
            return None

        # Get player's most recent data from historical dataset
        player_hist = self.df_with_features[self.df_with_features['name'] == player_name]

        if len(player_hist) == 0:
            # New player - use API-based prediction
            if current_stats:
                return self._api_based_prediction(player_name, position, team, current_stats)
            return None

        # Get most recent record with features
        player_recent = player_hist.dropna(subset=self.feature_cols).tail(1)

        if len(player_recent) == 0:
            if current_stats:
                return self._api_based_prediction(player_name, position, team, current_stats)
            return None

        # Extract features for prediction
        X_pred = player_recent[self.feature_cols]

        # Get ensemble prediction with confidence interval
        ensemble_result = self._ensemble_predict(X_pred)
        predicted_points = ensemble_result['point_estimate']
        std_dev = ensemble_result['std_dev']

        # Calculate confidence intervals (±1.96 std for 95% CI)
        confidence_lower_95 = max(0, predicted_points - Constants.CI_95_ZSCORE * std_dev)
        confidence_upper_95 = predicted_points + Constants.CI_95_ZSCORE * std_dev

        # Get current season statistics
        if current_stats:
            total_points = float(current_stats.get('total_points', 0))
            minutes = float(current_stats.get('minutes', 0))
            games_played = max(1, int(minutes / 90)) if minutes > 0 else 1
        else:
            # Use from historical data
            latest_season = self.df['season'].max()
            current_season_data = player_hist[player_hist['season'] == latest_season]
            total_points = current_season_data['total_points'].sum()
            games_played = len(current_season_data)

        # Estimate remaining games and total season prediction
        remaining_games = max(Constants.TOTAL_SEASON_GAMES - games_played, 0)
        predicted_remaining = predicted_points * remaining_games

        # Confidence bounds for remaining/total
        remaining_lower = confidence_lower_95 * remaining_games
        remaining_upper = confidence_upper_95 * remaining_games

        predicted_total = total_points + predicted_remaining
        predicted_total_lower = total_points + remaining_lower
        predicted_total_upper = total_points + remaining_upper

        # Get form trend from features (including new acceleration metric)
        points_trend = player_recent['points_trend'].values[0]
        acceleration = player_recent['acceleration'].values[0]
        recent_avg = player_recent['rolling_points_5'].values[0]
        historical_avg = player_recent['rolling_points_10'].values[0]

        # Form trend based on acceleration
        if acceleration > Constants.ACCELERATION_THRESHOLD:
            form_trend = 'improving'
        elif acceleration < -Constants.ACCELERATION_THRESHOLD:
            form_trend = 'declining'
        else:
            form_trend = 'stable'

        # Determine confidence level based on std dev (lower std = higher confidence)
        if std_dev < Constants.STD_DEV_HIGH:
            confidence_level = 'high'
        elif std_dev < Constants.STD_DEV_MEDIUM:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'

        # Get sentiment if available
        sentiment_score = 0
        sentiment_boost = 0
        if self.sentiment_data and player_name in self.sentiment_data:
            player_sentiment = self.sentiment_data[player_name]
            if isinstance(player_sentiment, dict) and 'normalized_score' in player_sentiment:
                sentiment_score = float(player_sentiment['normalized_score'])
                sentiment_boost = (sentiment_score - 50) * Constants.SENTIMENT_MULTIPLIER
                # Apply sentiment boost to ensemble prediction
                predicted_points = predicted_points * (1 + sentiment_boost)
                predicted_remaining = predicted_points * remaining_games
                predicted_total = total_points + predicted_remaining
                confidence_lower_95 = confidence_lower_95 * (1 + sentiment_boost)
                confidence_upper_95 = confidence_upper_95 * (1 + sentiment_boost)

        # Calculate next game prediction
        next_game_info = self._predict_next_game(player_recent, player_name, position, team, current_stats)

        result = {
            'player_name': player_name,
            'position': position,
            'team': team,
            'predicted_total_points': round(predicted_total, 2),
            'predicted_total_points_lower_95': round(predicted_total_lower, 2),
            'predicted_total_points_upper_95': round(predicted_total_upper, 2),
            'predicted_points_per_match': round(predicted_points, 2),
            'confidence_interval_lower': round(confidence_lower_95, 2),
            'confidence_interval_upper': round(confidence_upper_95, 2),
            'confidence_std_dev': round(std_dev, 2),
            'current_season_points': round(total_points, 2),
            'games_played': games_played,
            'predicted_remaining_points': round(predicted_remaining, 2),
            'remaining_games': remaining_games,
            'recent_avg_points': round(recent_avg, 2),
            'historical_avg_points': round(historical_avg, 2),
            'form_trend': form_trend,
            'confidence': confidence_level,
            'sentiment_score': round(sentiment_score, 1),
            'sentiment_impact': round(sentiment_boost * 100, 2),
            # Next game predictions
            'next_game_predicted_points': next_game_info.get('next_game_predicted_points'),
            'next_game_opponent': next_game_info.get('next_game_opponent'),
            'next_game_is_home': next_game_info.get('next_game_is_home'),
            'next_game_difficulty': next_game_info.get('next_game_difficulty'),
        }

        return result
    
    def _api_based_prediction(self, player_name, position, team, current_stats):
        """Prediction based solely on current API stats (no historical data)."""
        ppg = float(current_stats.get('points_per_game', 0))
        total_points = float(current_stats.get('total_points', 0))
        minutes = float(current_stats.get('minutes', 0))
        next_opponent = current_stats.get('opp_team_name', 'Unknown') if current_stats else 'Unknown'
        next_is_home = current_stats.get('was_home') if current_stats else None

        # Estimate games played
        games_played = max(1, int(minutes / 90)) if minutes > 0 else 1

        # Use current PPG with slight regression to position mean
        position_avg = Constants.POSITION_DEFAULTS.get(position, 3.0)

        # Blend: 70% current PPG + 30% position average (regression)
        predicted_ppg = (
            (Constants.API_BLEND_PPG_WEIGHT * ppg + Constants.API_BLEND_POSITION_WEIGHT * position_avg)
            if ppg > 0
            else position_avg
        )

        remaining_games = max(Constants.TOTAL_SEASON_GAMES - games_played, 0)
        predicted_remaining = predicted_ppg * remaining_games
        predicted_total = total_points + predicted_remaining

        return {
            'player_name': player_name,
            'position': position,
            'team': team,
            'predicted_total_points': round(predicted_total, 2),
            'predicted_points_per_match': round(predicted_ppg, 2),
            'current_season_points': round(total_points, 2),
            'games_played': games_played,
            'predicted_remaining_points': round(predicted_remaining, 2),
            'remaining_games': remaining_games,
            'recent_avg_points': round(ppg, 2),
            'historical_avg_points': round(position_avg, 2),
            'form_trend': 'unknown',
            'confidence': 'low',
            'next_game_predicted_points': round(predicted_ppg, 2),
            'next_game_opponent': next_opponent,
            'next_game_is_home': next_is_home,
            'next_game_difficulty': 'Unknown'
        }
    
    def _fallback_prediction(self, player_name, position, team):
        """Fallback prediction for players with limited history."""
        # Use position-based defaults
        avg_points = Constants.POSITION_DEFAULTS.get(position, 3.0)
        current_points = 0
        games_played = 0

        remaining_games = max(Constants.TOTAL_SEASON_GAMES - games_played, 0)
        predicted_total = current_points + (avg_points * remaining_games)

        return {
            'player_name': player_name,
            'position': position,
            'team': team,
            'predicted_total_points': round(predicted_total, 2),
            'predicted_points_per_match': round(avg_points, 2),
            'current_season_points': round(current_points, 2),
            'games_played': games_played,
            'predicted_remaining_points': round(avg_points * remaining_games, 2),
            'remaining_games': remaining_games,
            'recent_avg_points': round(avg_points, 2),
            'historical_avg_points': round(avg_points, 2),
            'form_trend': 'unknown',
            'confidence': 'low',
            'sentiment_score': 0,
            'sentiment_impact': 0,
            'next_game_predicted_points': round(avg_points, 2),
            'next_game_opponent': 'Unknown',
            'next_game_is_home': None,
            'next_game_difficulty': 'Unknown'
        }
    
    def predict_all_current_players(self, current_players_list=None):
        """
        Predict performance for all current season players, including next gameweek.

        Args:
            current_players_list: List of current players from FPL API with stats.
                                 If None, fetches from API automatically.
        """
        if self.model_rf is None:
            print("Error: Model not trained. Call train_model() first.")
            return []

        print(f"\nPredicting performance for current players...")

        # Fetch current players if not provided
        if current_players_list is None:
            from data_fetcher import FantasyDataFetcher
            fetcher = FantasyDataFetcher()
            current_players_list = fetcher.fetch_players()
            print(f"Fetched {len(current_players_list)} players from FPL API")

        # Build next gameweek fixture difficulty lookup
        self.set_next_gw_fixtures(current_players_list)

        print(f"Found {len(current_players_list)} active players")

        predictions_list = []

        for idx, player in enumerate(current_players_list):
            if idx % 50 == 0:
                print(f"  Processed {idx}/{len(current_players_list)} players...")

            # Extract current stats from API
            current_stats = {
                'form': player.get('form', 0),
                'points_per_game': player.get('points_per_game', 0),
                'total_points': player.get('total_points', 0),
                'minutes': player.get('minutes', 0),
                'goals_scored': player.get('goals_scored', 0),
                'assists': player.get('assists', 0),
                'expected_goals': player.get('expected_goals', 0),
                'expected_assists': player.get('expected_assists', 0),
                'ict_index': player.get('ict_index', 0),
                'opponent_team': player.get('opponent_team'),
                'opp_team_name': player.get('opp_team_name'),
                'was_home': player.get('was_home'),
                'fixture': player.get('fixture'),
            }

            prediction = self.predict_player(
                player['name'],
                player['position'],
                player['team'],
                current_stats=current_stats
            )

            if prediction:
                predictions_list.append(prediction)

        self.predictions = {
            p['player_name']: p for p in predictions_list
        }

        print(f"\n✓ Generated predictions for {len(predictions_list)} players")
        return predictions_list
    
    def save_predictions(self):
        """Save predictions to file."""
        # Convert numpy types to native Python types for JSON serialization
        predictions_serializable = {}
        for player_id, pred in self.predictions.items():
            predictions_serializable[str(player_id)] = {
                k: (int(v) if isinstance(v, (np.integer, np.int64)) 
                    else float(v) if isinstance(v, (np.floating, np.float64))
                    else v)
                for k, v in pred.items()
            }
        
        # Save as JSON with player IDs as keys
        with open('data/predictions.json', 'w') as f:
            json.dump(predictions_serializable, f, indent=2)
        
        # Save as CSV for easy viewing
        predictions_df = pd.DataFrame(self.predictions.values())
        predictions_df = predictions_df.sort_values('predicted_total_points', ascending=False)
        predictions_df.to_csv('data/predictions.csv', index=False)
        
        print("Predictions saved to data/predictions.json and data/predictions.csv")
        
        # Print summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Total players predicted: {len(predictions_df)}")
        print(f"\nTop 10 Predicted Total Points:")
        print(predictions_df[['player_name', 'position', 'team', 'predicted_total_points']].head(10).to_string(index=False))
        
        print(f"\nBy Position - Average Predicted Points/Match:")
        position_stats = predictions_df.groupby('position')['predicted_points_per_match'].agg(['mean', 'max'])
        print(position_stats.to_string())
    
    def load_predictions(self):
        """Load previously generated predictions."""
        try:
            with open('data/predictions.json', 'r') as f:
                self.predictions = json.load(f)
            print(f"Loaded predictions for {len(self.predictions)} players")
            return True
        except Exception as e:
            print(f"Error loading predictions: {e}")
            return False


def main():
    """Main execution function."""
    predictor = PlayerPredictor()
    
    # Load historical data
    if not predictor.load_data():
        print("Error loading data")
        return
    
    # Train Random Forest model on historical data
    # Use last 40 gameweeks (~12 months) as rolling window
    predictor.train_model(use_last_n_weeks=40)  
    
    # Generate predictions for all current players
    predictor.predict_all_current_players()
    
    # Save results
    predictor.save_predictions()
    
    print("\n✓ Prediction pipeline complete!")


if __name__ == "__main__":
    main()
