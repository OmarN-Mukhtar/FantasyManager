"""
Fetches player data from Fantasy Premier League API.
"""

import requests
import json
import os
from typing import List, Dict
import pandas as pd


class FantasyDataFetcher:
    """Handles fetching player data only."""
    
    API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    def __init__(self):
        self.players_data = []
        self.teams = {}
        
    def fetch_players(self) -> List[Dict]:
        """Fetch all players from the Fantasy Premier League API."""
        print("Fetching player data from Fantasy Premier League API...")
        
        try:
            response = requests.get(self.API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()

            players = data['elements']
            self.teams = {team['id']: team['name'] for team in data['teams']}

            self.players_data = []
            for player in players:
                cost = player['now_cost'] / 10

                player_info = {
                    'id': player['id'],
                    'element': player['id'],
                    'name': f"{player['first_name']} {player['second_name']}",
                    'first_name': player['first_name'],
                    'second_name': player['second_name'],
                    'web_name': player['web_name'],
                    'team': self.teams.get(player['team'], 'Unknown'),
                    'team_id': player['team'],
                    'position': self._get_position_name(player['element_type']),
                    'element_type': player['element_type'],
                    'now_cost': cost,
                    'total_points': player.get('total_points', 0),
                    'form': float(player.get('form', 0) or 0),
                    'points_per_game': float(player.get('points_per_game', 0) or 0),
                    'selected_by_percent': float(player.get('selected_by_percent', 0) or 0),
                }
                self.players_data.append(player_info)
            
            print(f"Successfully fetched {len(self.players_data)} players")
            return self.players_data
            
        except Exception as e:
            print(f"Error fetching player data: {e}")
            return []
    
    def _get_position_name(self, position_id: int) -> str:
        """Convert position ID to name."""
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return positions.get(position_id, 'Unknown')
    
    def save_data(self):
        """Save fetched player data to files."""
        os.makedirs('data', exist_ok=True)
        with open('data/players.json', 'w') as f:
            json.dump(self.players_data, f, indent=2)
        pd.DataFrame(self.players_data).to_csv('data/players.csv', index=False)
        print(f"\nData saved to 'data/' directory")


def main():
    """Main execution function."""
    fetcher = FantasyDataFetcher()
    fetcher.fetch_players()
    fetcher.save_data()
    print("\n✓ Player fetching complete!")


if __name__ == "__main__":
    main()
