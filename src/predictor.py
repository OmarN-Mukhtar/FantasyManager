"""
Player performance prediction using rolling window models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import json
import warnings
warnings.filterwarnings('ignore')


class PlayerPredictor:
    """Predicts player performance using historical data."""
    
    def __init__(self, data_path='cleaned_merged_seasons.csv', current_season=None):
        """
        Initialize predictor with historical data.
        
        Args:
            data_path: Path to historical CSV data
            current_season: Current season (e.g., '2025-26'). Auto-detected if None.
        """
        self.data_path = data_path
        self.df = None
        self.predictions = {}
        self.current_season = current_season
        self.accuracy_metrics = {}
        
    def load_data(self):
        """Load and preprocess historical player data."""
        print("Loading historical player data...")
        self.df = pd.read_csv(self.data_path)
        
        # If current season not provided, use latest in dataset + 1 year
        if self.current_season is None:
            latest_season = self.df['season_x'].max()
            # Increment by 1 year (e.g., 2023-24 -> 2024-25)
            year_start = int(latest_season.split('-')[0])
            year_end = int(latest_season.split('-')[1])
            self.current_season = f"{year_start + 1}-{str(year_end + 1).zfill(2)}"
            print(f"Auto-detected next season: {self.current_season}")
        
        print(f"Historical data: {len(self.df)} records from {self.df['season_x'].min()} to {self.df['season_x'].max()}")
        print(f"Predicting for season: {self.current_season}")
        
        # Convert value to actual price (divide by 10 if needed)
        if self.df['value'].max() > 200:
            self.df['value'] = self.df['value'] / 10
        
        return True
    
    def create_features(self, player_df):
        """
        Create rolling window features for a player.
        
        Args:
            player_df: DataFrame for a single player, sorted by GW
            
        Returns:
            DataFrame with engineered features
        """
        df = player_df.copy().sort_values('GW')
        
        # Rolling windows (3, 5, 10 games)
        for window in [3, 5, 10]:
            df[f'rolling_points_{window}'] = df['total_points'].rolling(window, min_periods=1).mean()
            df[f'rolling_minutes_{window}'] = df['minutes'].rolling(window, min_periods=1).mean()
            df[f'rolling_goals_{window}'] = df['goals_scored'].rolling(window, min_periods=1).mean()
            df[f'rolling_assists_{window}'] = df['assists'].rolling(window, min_periods=1).mean()
            df[f'rolling_bonus_{window}'] = df['bonus'].rolling(window, min_periods=1).mean()
        
        # Cumulative season stats
        df['season_total_points'] = df.groupby('season_x')['total_points'].cumsum()
        df['season_avg_points'] = df.groupby('season_x')['total_points'].expanding().mean().values
        df['games_played'] = df.groupby('season_x').cumcount() + 1
        
        # Form indicators
        df['points_trend'] = df['rolling_points_5'] - df['rolling_points_10']
        df['minutes_consistency'] = df['minutes'].rolling(5, min_periods=1).std().fillna(0)
        
        # Position encoding
        position_map = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
        df['position_encoded'] = df['position'].map(position_map).fillna(2)
        
        # Home/away
        df['is_home'] = df['was_home'].astype(int)
        
        return df
    
    def predict_player(self, player_name, position, team, current_stats=None):
        """
        Predict performance for a single player using historical data + current form.
        
        Args:
            player_name: Player name
            position: Player position
            team: Player team
            current_stats: Dictionary of current season stats from FPL API (optional).
                          Keys: form, points_per_game, total_points, minutes, etc.
            
        Returns:
            Dictionary with predictions
        """
        # Get player historical data
        player_df = self.df[self.df['name'] == player_name].copy()
        
        # If we have current stats from API but no historical data, use fallback
        if len(player_df) < 5 and current_stats:
            return self._api_based_prediction(player_name, position, team, current_stats)
        
        if len(player_df) < 10:  # Need minimum history
            return None
        
        # Create features
        player_df = self.create_features(player_df)
        
        # Define feature columns
        feature_cols = [
            'rolling_points_3', 'rolling_points_5', 'rolling_points_10',
            'rolling_minutes_3', 'rolling_minutes_5', 'rolling_minutes_10',
            'rolling_goals_3', 'rolling_goals_5', 'rolling_goals_10',
            'rolling_assists_3', 'rolling_assists_5', 'rolling_assists_10',
            'rolling_bonus_3', 'rolling_bonus_5', 'rolling_bonus_10',
            'season_avg_points', 'points_trend', 'minutes_consistency',
            'position_encoded', 'is_home', 'value'
        ]
        
        # Prepare training data (all but current season)
        train_df = player_df[player_df['season_x'] != self.current_season].copy()
        current_season_df = player_df[player_df['season_x'] == self.current_season].copy()
        
        if len(train_df) < 5:  # Not enough training data
            return self._fallback_prediction(player_name, position, team, current_season_df)
        
        # Remove NaN
        train_df = train_df.dropna(subset=feature_cols + ['total_points'])
        
        if len(train_df) < 5:
            return self._fallback_prediction(player_name, position, team, current_season_df)
        
        # Train model
        X_train = train_df[feature_cols]
        y_train = train_df['total_points']
        
        # Use Random Forest (free, no API needed)
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predict for current season
        if len(current_season_df) > 0:
            current_season_df = current_season_df.dropna(subset=feature_cols)
            if len(current_season_df) > 0:
                X_current = current_season_df[feature_cols]
                predictions = model.predict(X_current)
                predicted_avg = predictions.mean()
            else:
                predicted_avg = y_train.mean()
        else:
            predicted_avg = y_train.mean()
        
        # Calculate statistics
        recent_games = player_df.tail(10)
        recent_avg = recent_games['total_points'].mean()
        historical_avg = train_df['total_points'].mean()
        
        # Estimate remaining games (assume 38 total gameweeks)
        # If we have current stats from API, use actual games played
        if current_stats and current_stats.get('total_points', 0) > 0:
            # Calculate games played from minutes (90 min = 1 full game)
            minutes_played = float(current_stats.get('minutes', 0))
            games_played_current = max(1, int(minutes_played / 90))
            current_points = float(current_stats.get('total_points', 0))
            
            # Blend predictions: 60% current form + 40% historical model
            current_ppg = float(current_stats.get('points_per_game', 0))
            predicted_avg_blended = (0.6 * current_ppg) + (0.4 * predicted_avg)
        else:
            games_played_current = len(current_season_df)
            current_points = current_season_df['total_points'].sum() if len(current_season_df) > 0 else 0
            predicted_avg_blended = predicted_avg
        
        remaining_games = max(38 - games_played_current, 0)
        
        # Predict total points for season
        predicted_remaining = predicted_avg_blended * remaining_games
        predicted_total = current_points + predicted_remaining
        
        # Determine form trend
        if recent_avg > historical_avg * 1.1:
            form_trend = 'improving'
        elif recent_avg < historical_avg * 0.9:
            form_trend = 'declining'
        else:
            form_trend = 'stable'
        
        return {
            'player_name': player_name,
            'position': position,
            'team': team,
            'predicted_total_points': round(predicted_total, 2),
            'predicted_points_per_match': round(predicted_avg_blended, 2),
            'current_season_points': round(current_points, 2),
            'games_played': games_played_current,
            'predicted_remaining_points': round(predicted_remaining, 2),
            'remaining_games': remaining_games,
            'recent_avg_points': round(recent_avg, 2),
            'historical_avg_points': round(historical_avg, 2),
            'form_trend': form_trend,
            'confidence': 'high' if len(train_df) > 20 else 'medium'
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
    
    def _fallback_prediction(self, player_name, position, team, current_season_df):
        """Fallback prediction for players with limited history."""
        if len(current_season_df) > 0:
            avg_points = current_season_df['total_points'].mean()
            current_points = current_season_df['total_points'].sum()
            games_played = len(current_season_df)
        else:
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
            'confidence': 'low'
        }
    
    def predict_all_current_players(self, current_players_list=None):
        """
        Predict performance for all current season players.
        
        Args:
            current_players_list: List of current players from FPL API with stats.
                                 If None, fetches from API automatically.
        """
        print(f"\nPredicting performance for {self.current_season} season...")
        
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
    
    # Generate predictions
    predictor.predict_all_current_players()
    
    # Save results
    predictor.save_predictions()
    
    print("\n✓ Prediction pipeline complete!")


if __name__ == "__main__":
    main()
