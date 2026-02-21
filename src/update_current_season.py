"""
Fetch current season gameweek-by-gameweek data and append to cleaned_merged_seasons.csv
"""

import requests
import pandas as pd
import time
from datetime import datetime
import os


class CurrentSeasonUpdater:
    """Fetches current season data and appends to historical CSV."""
    
    API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    PLAYER_DETAIL_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    
    def __init__(self, csv_path='cleaned_merged_seasons.csv'):
        self.csv_path = csv_path
        self.teams = {}
        self.current_season = None
        self.current_gameweek = None
        self.new_records = []
        
    def fetch_bootstrap_data(self):
        """Fetch basic data including teams and current gameweek."""
        print("Fetching bootstrap data...")
        try:
            response = requests.get(self.API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Get teams
            self.teams = {team['id']: team['name'] for team in data['teams']}
            
            # Get current season and gameweek
            events = data['events']
            current_event = next((e for e in events if e['is_current']), None)
            
            if current_event:
                self.current_gameweek = current_event['id']
            else:
                # Find last finished gameweek
                finished_events = [e for e in events if e['finished']]
                self.current_gameweek = finished_events[-1]['id'] if finished_events else 1
            
            # Determine season
            current_year = datetime.now().year
            current_month = datetime.now().month
            if current_month >= 8:
                self.current_season = f"{current_year}-{str(current_year + 1)[-2:]}"
            else:
                self.current_season = f"{current_year - 1}-{str(current_year)[-2:]}"
            
            print(f"Current Season: {self.current_season}")
            print(f"Current/Last Gameweek: {self.current_gameweek}")
            print(f"Teams loaded: {len(self.teams)}")
            
            return data['elements']  # Return players list
            
        except Exception as e:
            print(f"Error fetching bootstrap data: {e}")
            return []
    
    def get_position_name(self, position_id):
        """Convert position ID to name."""
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return positions.get(position_id, 'Unknown')
    
    def fetch_player_gameweek_data(self, player_id, player_name, position, team_id):
        """Fetch gameweek-by-gameweek data for a specific player with retry logic."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                url = self.PLAYER_DETAIL_URL.format(player_id=player_id)
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                # Get current season history
                history = data.get('history', [])
                
                if not history:
                    return []
                
                records = []
                team_name = self.teams.get(team_id, 'Unknown')
                
                for gw in history:
                    # Only include current season data
                    record = {
                        'season': self.current_season,
                        'name': player_name,
                        'position': position,
                        'team': team_name,
                        'assists': gw.get('assists', 0),
                        'bonus': gw.get('bonus', 0),
                        'bps': gw.get('bps', 0),
                        'clean_sheets': gw.get('clean_sheets', 0),
                        'creativity': float(gw.get('creativity', 0)),
                        'element': player_id,
                        'fixture': gw.get('fixture', 0),
                        'goals_conceded': gw.get('goals_conceded', 0),
                        'goals_scored': gw.get('goals_scored', 0),
                        'ict_index': float(gw.get('ict_index', 0)),
                        'influence': float(gw.get('influence', 0)),
                        'kickoff_time': gw.get('kickoff_time', ''),
                        'minutes': gw.get('minutes', 0),
                        'opponent_team': gw.get('opponent_team', 0),
                        'opp_team_name': self.teams.get(gw.get('opponent_team', 0), 'Unknown'),
                        'own_goals': gw.get('own_goals', 0),
                        'penalties_missed': gw.get('penalties_missed', 0),
                        'penalties_saved': gw.get('penalties_saved', 0),
                        'red_cards': gw.get('red_cards', 0),
                        'round': gw.get('round', 0),
                        'saves': gw.get('saves', 0),
                        'selected': gw.get('selected', 0),
                        'team_a_score': gw.get('team_a_score', 0),
                        'team_h_score': gw.get('team_h_score', 0),
                        'threat': float(gw.get('threat', 0)),
                        'total_points': gw.get('total_points', 0),
                        'transfers_balance': gw.get('transfers_balance', 0),
                        'transfers_in': gw.get('transfers_in', 0),
                        'transfers_out': gw.get('transfers_out', 0),
                        'value': gw.get('value', 0) / 10,  # Convert to actual price
                        'was_home': gw.get('was_home', False),
                        'yellow_cards': gw.get('yellow_cards', 0),
                        'GW': gw.get('round', 0)
                    }
                    records.append(record)
                
                return records
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 503 and attempt < max_retries - 1:
                    # Rate limited, wait and retry
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                else:
                    # Other error or max retries reached
                    return []
            except Exception as e:
                return []
        
        return []
    
    def load_existing_data(self):
        """Load existing CSV to check for duplicates."""
        if os.path.exists(self.csv_path):
            print(f"\nLoading existing data from {self.csv_path}...")
            df = pd.read_csv(self.csv_path)
            print(f"Existing records: {len(df)}")
            return df
        else:
            print(f"No existing file found at {self.csv_path}")
            return pd.DataFrame()
    
    def fetch_all_current_season_data(self):
        """Fetch gameweek data for all players in current season."""
        players = self.fetch_bootstrap_data()
        
        if not players:
            print("No players found!")
            return
        
        print(f"\nFetching gameweek data for {len(players)} players...")
        print("This may take a while due to API rate limiting...")
        print("Progress will be saved periodically.\n")
        
        # Load existing data to check for duplicates
        existing_df = self.load_existing_data()
        
        # Get existing season-player-gameweek combinations
        if not existing_df.empty:
            existing_keys = set(
                existing_df[existing_df['season'] == self.current_season]
                .apply(lambda row: f"{row['name']}_{row['GW']}", axis=1)
            )
            print(f"Found {len(existing_keys)} existing records for {self.current_season}\n")
        else:
            existing_keys = set()
        
        all_records = []
        new_count = 0
        skip_count = 0
        error_count = 0
        batch_size = 20  # Save after every 20 players
        
        for idx, player in enumerate(players):
            player_id = player['id']
            player_name = f"{player['first_name']} {player['second_name']}"
            position = self.get_position_name(player['element_type'])
            team_id = player['team']
            
            # Fetch gameweek data
            records = self.fetch_player_gameweek_data(player_id, player_name, position, team_id)
            
            if not records:
                error_count += 1
            
            # Filter out duplicates
            for record in records:
                key = f"{record['name']}_{record['GW']}"
                if key not in existing_keys:
                    all_records.append(record)
                    existing_keys.add(key)
                    new_count += 1
                else:
                    skip_count += 1
            
            # Progress update and batch save
            if (idx + 1) % batch_size == 0:
                print(f"  Progress: {idx + 1}/{len(players)} players | "
                      f"New: {new_count} | Skipped: {skip_count} | Errors: {error_count}")
                
                # Save batch to CSV immediately
                if all_records:
                    self._append_batch_to_csv(all_records)
                    all_records = []  # Clear after saving
                    new_count = 0
                    skip_count = 0
            
            # Longer delay to avoid rate limiting
            time.sleep(0.15)
        
        # Save any remaining records
        if all_records:
            self._append_batch_to_csv(all_records)
        
        self.new_records = []  # Already saved in batches
        print(f"\n✓ Fetching complete!")
        print(f"  Total errors: {error_count}")
        
        return []
    
    def _append_batch_to_csv(self, records):
        """Append a batch of records to CSV immediately."""
        if not records:
            return
        
        new_df = pd.DataFrame(records)
        
        # Ensure column order matches existing CSV
        if os.path.exists(self.csv_path):
            existing_df = pd.read_csv(self.csv_path, nrows=1)
            existing_cols = existing_df.columns.tolist()
            new_df = new_df[existing_cols]
        
        # Append to CSV
        new_df.to_csv(
            self.csv_path,
            mode='a',
            header=False,  # Don't write header when appending
            index=False
        )
        print(f"    ✓ Saved batch of {len(records)} records to CSV")
    
    def _save_intermediate_progress(self, records):
        """Save intermediate progress to avoid data loss."""
        if not records:
            return
        
        temp_file = f"{self.csv_path}.temp"
        temp_df = pd.DataFrame(records)
        
        # Ensure column order matches existing CSV
        if os.path.exists(self.csv_path):
            existing_df = pd.read_csv(self.csv_path, nrows=1)
            existing_cols = existing_df.columns.tolist()
            temp_df = temp_df[existing_cols]
        
        temp_df.to_csv(temp_file, index=False)
    
    
    def append_to_csv(self):
        """This method is no longer needed as we save in batches."""
        if self.new_records:
            print(f"\nAppending final {len(self.new_records)} records...")
            self._append_batch_to_csv(self.new_records)
        else:
            print("\nAll data already saved in batches during fetch.")
    
    def update(self):
        """Main update function."""
        print("="*60)
        print("CURRENT SEASON DATA UPDATER")
        print("="*60)
        
        # Fetch current season data
        self.fetch_all_current_season_data()
        
        # Append to CSV
        self.append_to_csv()
        
        print("\n" + "="*60)
        print("✓ Update complete!")
        print("="*60)


def main():
    """Main execution function."""
    updater = CurrentSeasonUpdater()
    updater.update()


if __name__ == "__main__":
    main()
