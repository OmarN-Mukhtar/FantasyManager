"""
Player performance prediction using rolling window Random Forest model.
Trained on last 12 months of data without using player names or seasons as features.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


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
        self.model = None
        self.scaler = StandardScaler()
        self.predictions = {}
        self.window_weeks = window_weeks  # ~12 months of gameweeks
        self.feature_cols = []
        self.accuracy_metrics = {}
        self.sentiment_data = self.load_sentiment_data()
        
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
        if self.df['value'].max() > 200:
            self.df['value'] = self.df['value'] / 10
        
        # Sort by player and gameweek for rolling window calculations
        self.df = self.df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)
        
        # Create a sequential time index (season + GW)
        self.df['season_gw'] = self.df['season'] + '_GW' + self.df['GW'].astype(str)
        
        return True
    
    def create_rolling_features(self):
        """
        Create rolling window features for all players.
        Does NOT use player name or season as features.
        
        Returns:
            DataFrame with engineered features
        """
        print("Creating rolling window features...")
        
        df_features = []
        
        # Process each player separately to create rolling features
        for player_name in self.df['name'].unique():
            player_df = self.df[self.df['name'] == player_name].copy()
            
            if len(player_df) < 5:  # Skip players with insufficient data
                continue
            
            # Rolling windows (3, 5, 10 games)
            for window in [3, 5, 10]:
                player_df[f'rolling_points_{window}'] = player_df['total_points'].rolling(window, min_periods=1).mean()
                player_df[f'rolling_minutes_{window}'] = player_df['minutes'].rolling(window, min_periods=1).mean()
                player_df[f'rolling_goals_{window}'] = player_df['goals_scored'].rolling(window, min_periods=1).mean()
                player_df[f'rolling_assists_{window}'] = player_df['assists'].rolling(window, min_periods=1).mean()
                player_df[f'rolling_bonus_{window}'] = player_df['bonus'].rolling(window, min_periods=1).mean()
                player_df[f'rolling_creativity_{window}'] = player_df['creativity'].rolling(window, min_periods=1).mean()
                player_df[f'rolling_threat_{window}'] = player_df['threat'].rolling(window, min_periods=1).mean()
                player_df[f'rolling_ict_{window}'] = player_df['ict_index'].rolling(window, min_periods=1).mean()
            
            # Form indicators
            player_df['points_trend'] = player_df['rolling_points_5'] - player_df['rolling_points_10']
            player_df['minutes_consistency'] = player_df['minutes'].rolling(5, min_periods=1).std().fillna(0)
            player_df['goals_trend'] = player_df['rolling_goals_3'] - player_df['rolling_goals_10']
            
            # Games played counter (reset each season)
            player_df['games_played_season'] = player_df.groupby('season').cumcount() + 1
            
            # Cumulative season stats
            player_df['season_total_points'] = player_df.groupby('season')['total_points'].cumsum()
            player_df['season_avg_points'] = player_df.groupby('season')['total_points'].expanding().mean().values
            
            df_features.append(player_df)
        
        # Combine all players
        df_combined = pd.concat(df_features, ignore_index=True)
        
        # Position encoding (one-hot style)
        position_dummies = pd.get_dummies(df_combined['position'], prefix='pos')
        df_combined = pd.concat([df_combined, position_dummies], axis=1)
        
        # Home/away
        df_combined['is_home'] = df_combined['was_home'].astype(int)
        
        # Team strength indicator (average points per game by team) - recalculated each window
        team_strength = df_combined.groupby('team')['total_points'].mean()
        df_combined['team_strength'] = df_combined['team'].map(team_strength).fillna(3.0)
        
        print(f"Created features for {len(df_combined)} game records")
        
        return df_combined
    
    
    def train_model(self, use_last_n_weeks=None):
        """
        Train Random Forest model on historical data with rolling window features.
        Uses last N weeks of data for training (simulating 12-month rolling window).
        
        Args:
            use_last_n_weeks: Use only last N gameweeks for training (default: None = all data)
        """
        print("\nTraining Random Forest model...")
        
        # Create features
        df_with_features = self.create_rolling_features()
        
        # Define feature columns (excluding player name, season, and target)
        self.feature_cols = [
            # Rolling averages
            'rolling_points_3', 'rolling_points_5', 'rolling_points_10',
            'rolling_minutes_3', 'rolling_minutes_5', 'rolling_minutes_10',
            'rolling_goals_3', 'rolling_goals_5', 'rolling_goals_10',
            'rolling_assists_3', 'rolling_assists_5', 'rolling_assists_10',
            'rolling_bonus_3', 'rolling_bonus_5', 'rolling_bonus_10',
            'rolling_creativity_3', 'rolling_creativity_5', 'rolling_creativity_10',
            'rolling_threat_3', 'rolling_threat_5', 'rolling_threat_10',
            'rolling_ict_3', 'rolling_ict_5', 'rolling_ict_10',
            # Form indicators
            'points_trend', 'minutes_consistency', 'goals_trend',
            'season_avg_points', 'games_played_season',
            # Position (one-hot encoded)
            'pos_DEF', 'pos_FWD', 'pos_GK', 'pos_MID',
            # Context
            'is_home', 'team_strength', 'value'
        ]
        
        # Filter to only include records with all features
        df_train = df_with_features.dropna(subset=self.feature_cols + ['total_points'])
        
        # Optionally use only last N weeks (12-month rolling window)
        if use_last_n_weeks:
            # Get unique season_gw combinations and take last N
            all_gws = df_train['season_gw'].unique()
            last_n_gws = all_gws[-use_last_n_weeks:] if len(all_gws) > use_last_n_weeks else all_gws
            df_train = df_train[df_train['season_gw'].isin(last_n_gws)]
            print(f"Using last {len(last_n_gws)} gameweeks for training")
        
        print(f"Training data: {len(df_train)} records")
        
        # Prepare features and target
        X = df_train[self.feature_cols]
        y = df_train['total_points']
        
        # Train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print("Training Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        y_pred = self.model.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        
        self.accuracy_metrics = {
            'train_r2': train_score,
            'test_r2': test_score,
            'mae': mae,
            'rmse': rmse
        }
        
        print(f"\n✓ Model trained successfully!")
        print(f"  Train R²: {train_score:.4f}")
        print(f"  Test R²: {test_score:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Store the feature dataframe for predictions
        self.df_with_features = df_with_features
        
        return True
    
    def predict_player(self, player_name, position, team, current_stats=None):
        """
        Predict performance for a single player using the trained Random Forest model.
        
        Args:
            player_name: Player name
            position: Player position
            team: Player team
            current_stats: Dictionary of current season stats from FPL API (optional)
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
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
        
        # Predict next gameweek performance
        predicted_points = self.model.predict(X_pred)[0]
        
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
        remaining_games = max(38 - games_played, 0)
        predicted_remaining = predicted_points * remaining_games
        predicted_total = total_points + predicted_remaining
        
        # Get form trend from features
        points_trend = player_recent['points_trend'].values[0]
        recent_avg = player_recent['rolling_points_5'].values[0]
        historical_avg = player_recent['rolling_points_10'].values[0]
        
        if points_trend > 0.5:
            form_trend = 'improving'
        elif points_trend < -0.5:
            form_trend = 'declining'
        else:
            form_trend = 'stable'
        
        # Get sentiment if available
        sentiment_score = 0
        sentiment_boost = 0
        if self.sentiment_data and player_name in self.sentiment_data:
            player_sentiment = self.sentiment_data[player_name]
            if isinstance(player_sentiment, dict) and 'normalized_score' in player_sentiment:
                sentiment_score = float(player_sentiment['normalized_score'])
                sentiment_boost = (sentiment_score - 50) * 0.001
                predicted_points = predicted_points * (1 + sentiment_boost)
                predicted_remaining = predicted_points * remaining_games
                predicted_total = total_points + predicted_remaining
        
        return {
            'player_name': player_name,
            'position': position,
            'team': team,
            'predicted_total_points': round(predicted_total, 2),
            'predicted_points_per_match': round(predicted_points, 2),
            'current_season_points': round(total_points, 2),
            'games_played': games_played,
            'predicted_remaining_points': round(predicted_remaining, 2),
            'remaining_games': remaining_games,
            'recent_avg_points': round(recent_avg, 2),
            'historical_avg_points': round(historical_avg, 2),
            'form_trend': form_trend,
            'confidence': 'high',
            'sentiment_score': round(sentiment_score, 1),
            'sentiment_impact': round(sentiment_boost * 100, 2)
        }
    
    def _api_based_prediction(self, player_name, position, team, current_stats):
        """Prediction based solely on current API stats (no historical data)."""
        ppg = float(current_stats.get('points_per_game', 0))
        total_points = float(current_stats.get('total_points', 0))
        minutes = float(current_stats.get('minutes', 0))
        
        # Estimate games played
        games_played = max(1, int(minutes / 90)) if minutes > 0 else 1
        
        # Use current PPG with slight regression to position mean
        position_defaults = {'GK': 2.5, 'DEF': 3.0, 'MID': 3.5, 'FWD': 3.0}
        position_avg = position_defaults.get(position, 3.0)
        
        # Blend: 70% current PPG + 30% position average (regression)
        predicted_ppg = (0.7 * ppg + 0.3 * position_avg) if ppg > 0 else position_avg
        
        remaining_games = max(38 - games_played, 0)
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
            'confidence': 'low'
        }
    
    def _fallback_prediction(self, player_name, position, team):
        """Fallback prediction for players with limited history."""
        # Use position-based defaults
        position_defaults = {'GK': 2.5, 'DEF': 3.0, 'MID': 3.5, 'FWD': 3.0}
        avg_points = position_defaults.get(position, 3.0)
        current_points = 0
        games_played = 0
        
        remaining_games = max(38 - games_played, 0)
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
            'sentiment_impact': 0
        }
    
    def predict_all_current_players(self, current_players_list=None):
        """
        Predict performance for all current season players.
        
        Args:
            current_players_list: List of current players from FPL API with stats.
                                 If None, fetches from API automatically.
        """
        if self.model is None:
            print("Error: Model not trained. Call train_model() first.")
            return []
        
        print(f"\nPredicting performance for current players...")
        
        # Fetch current players if not provided
        if current_players_list is None:
            from data_fetcher import FantasyDataFetcher
            fetcher = FantasyDataFetcher()
            current_players_list = fetcher.fetch_players()
            print(f"Fetched {len(current_players_list)} players from FPL API")
        
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
