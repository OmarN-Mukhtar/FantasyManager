"""
Quick script to test the prediction system with current player data from FPL API.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_fetcher import FantasyDataFetcher
from predictor import PlayerPredictor
import pandas as pd
import json


def merge_current_with_historical():
    """Use current FPL API data with historical trends to predict performance."""
    print("="*70)
    print("FANTASY PL PREDICTION - CURRENT SEASON")
    print("="*70)
    
    # Fetch current players with all their stats
    print("\n[1/3] Fetching current player data from FPL API...")
    fetcher = FantasyDataFetcher()
    current_players = fetcher.fetch_players()
    
    current_season = fetcher.current_season
    current_gameweek = fetcher.current_gameweek
    
    print(f"  Found {len(current_players)} current players")
    print(f"  Season: {current_season}")
    print(f"  Current Gameweek: {current_gameweek}")
    
    # Load historical data with detected season
    print("\n[2/3] Loading historical data...")
    predictor = PlayerPredictor(current_season=current_season)
    if not predictor.load_data():
        print("Error: cleaned_merged_seasons.csv not found!")
        return
    
    # Run predictions
    print(f"\n[3/3] Generating predictions for {current_season} season...")
    predictor.predict_all_current_players()
    predictor.save_predictions()
    
    # Show top predictions
    print("\n" + "="*70)
    print(f"TOP 20 PREDICTED SCORERS FOR {current_season}")
    print("="*70)
    
    pred_df = pd.DataFrame(predictor.predictions.values())
    top_20 = pred_df.nlargest(20, 'predicted_total_points')
    
    print(f"\n{'Rank':<6}{'Player':<25}{'Pos':<6}{'Team':<20}{'Pred Pts':<10}")
    print("-"*70)
    for idx, (_, row) in enumerate(top_20.iterrows(), 1):
        print(f"{idx:<6}{row['player_name']:<25}{row['position']:<6}{row['team']:<20}{row['predicted_total_points']:<10.1f}")
    
    print(f"\n✓ Predictions saved to data/predictions.json")
    print(f"  Season: {current_season} | Gameweek: {current_gameweek}")
    print("  Run 'streamlit run src/dashboard.py' to view interactive dashboard")


if __name__ == "__main__":
    merge_current_with_historical()
