"""
Fetch current season gameweek-by-gameweek data and append to data/cleaned_merged_seasons.csv
"""

import requests
import pandas as pd
import time
from datetime import datetime
import os
from typing import Dict, List


DEFAULT_CLEANED_COLUMNS = [
    'season', 'current_season', 'current_gameweek', 'name', 'first_name',
    'second_name', 'web_name', 'position', 'element_type', 'team', 'team_id',
    'element', 'id', 'assists', 'bonus', 'bps', 'clean_sheets', 'creativity',
    'expected_assists', 'expected_goal_involvements', 'expected_goals',
    'expected_goals_conceded', 'fixture', 'form', 'goals_conceded',
    'goals_scored', 'ict_index', 'influence', 'kickoff_time', 'minutes',
    'now_cost', 'ep_next', 'opponent_team', 'opp_team_name', 'own_goals',
    'penalties_missed', 'penalties_saved', 'points_per_game', 'red_cards',
    'round', 'saves', 'selected', 'selected_by_percent', 'starts', 'status',
    'team_a_score', 'team_h_score', 'threat', 'total_points',
    'transfers_balance', 'transfers_in', 'transfers_out', 'value', 'was_home',
    'yellow_cards', 'chance_of_playing_next_round',
    'chance_of_playing_this_round', 'GW'
]

# Map cleaned_merged column names to keys returned by FPL's element-summary history or bootstrap.
API_FIELD_RENAMES = {
    'GW': 'round',
    'now_cost': 'now_cost',  # Bootstrap field
    'selected_by_percent': 'selected_by_percent',  # Bootstrap field
}

INT_COLUMNS = {
    'assists', 'bonus', 'bps', 'clean_sheets', 'element_type', 'fixture',
    'goals_conceded', 'goals_scored', 'id', 'minutes', 'opponent_team',
    'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 'round',
    'saves', 'selected', 'starts', 'team_a_score', 'team_h_score', 'team_id',
    'total_points', 'transfers_balance', 'transfers_in', 'transfers_out',
    'yellow_cards', 'GW', 'element', 'current_gameweek'
}

FLOAT_COLUMNS = {
    'chance_of_playing_next_round', 'chance_of_playing_this_round',
    'creativity', 'expected_assists', 'expected_goal_involvements',
    'expected_goals', 'expected_goals_conceded', 'form', 'ict_index',
    'influence', 'now_cost', 'ep_next', 'points_per_game', 'selected_by_percent',
    'threat', 'value'
}

BOOL_COLUMNS = {'was_home'}

STRING_COLUMNS = {
    'current_season', 'first_name', 'kickoff_time', 'name', 'opp_team_name',
    'position', 'second_name', 'season', 'status', 'team', 'web_name'
}


class CurrentSeasonUpdater:
    """Fetches current season data and appends to historical CSV."""
    
    API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    PLAYER_DETAIL_URL = "https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    
    def __init__(self, csv_path='data/cleaned_merged_seasons.csv'):
        self.csv_path = csv_path
        self.teams = {}
        self.current_season = None
        self.current_gameweek = None
        self.last_finished_gameweek = None
        self.new_records = []
        self.output_columns = DEFAULT_CLEANED_COLUMNS.copy()
        self.player_count = 0
        
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
            next_event = next((e for e in events if e['is_next']), None)
            finished_events = [e for e in events if e['finished']]
            self.last_finished_gameweek = finished_events[-1]['id'] if finished_events else 0
            
            if current_event:
                self.current_gameweek = current_event['id']
            elif next_event:
                # No current GW means we are between gameweeks; use last finished.
                self.current_gameweek = self.last_finished_gameweek or max(next_event['id'] - 1, 1)
            else:
                self.current_gameweek = self.last_finished_gameweek or 1
            
            # Determine season
            current_year = datetime.now().year
            current_month = datetime.now().month
            if current_month >= 8:
                self.current_season = f"{current_year}-{str(current_year + 1)[-2:]}"
            else:
                self.current_season = f"{current_year - 1}-{str(current_year)[-2:]}"
            
            print(f"Current Season: {self.current_season}")
            print(f"Current/Last Gameweek: {self.current_gameweek}")
            print(f"Last Finished Gameweek: {self.last_finished_gameweek}")
            print(f"Teams loaded: {len(self.teams)}")
            self.player_count = len(data['elements'])
            
            return data['elements']  # Return players list
            
        except Exception as e:
            print(f"Error fetching bootstrap data: {e}")
            return []
    
    def get_position_name(self, position_id):
        """Convert position ID to name."""
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return positions.get(position_id, 'Unknown')
    
    def fetch_player_gameweek_data(self, player_obj):
        """Fetch gameweek-by-gameweek data for a specific player with retry logic."""
        player_id = player_obj['id']
        player_name = f"{player_obj['first_name']} {player_obj['second_name']}"
        position = self.get_position_name(player_obj['element_type'])
        team_id = player_obj['team']
        
        max_retries = 3
        retry_delay = 2

        def _to_float(value, default=0.0):
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        def _to_int(value, default=0):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default
        
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
                    # Initialize all target columns first, then fill from API values.
                    record = {col: None for col in self.output_columns}
                    
                    # Set fixed values from player object
                    record['season'] = self.current_season
                    record['current_season'] = self.current_season
                    record['current_gameweek'] = self.current_gameweek
                    record['name'] = player_name
                    record['first_name'] = player_obj.get('first_name', '')
                    record['second_name'] = player_obj.get('second_name', '')
                    record['web_name'] = player_obj.get('web_name', '')
                    record['position'] = position
                    record['element_type'] = player_obj.get('element_type', '')
                    record['team'] = team_name
                    record['team_id'] = team_id
                    record['element'] = player_id
                    record['id'] = player_obj.get('id', player_id)
                    
                    # Bootstrap player attributes (from current state, not per-GW specific)
                    record['now_cost'] = _to_float(player_obj.get('now_cost', 0.0), 0.0) / 10
                    record['ep_next'] = _to_float(player_obj.get('ep_next', 0.0), 0.0)
                    record['form'] = _to_float(player_obj.get('form', 0.0), 0.0)
                    record['points_per_game'] = _to_float(player_obj.get('points_per_game', 0.0), 0.0)
                    selected_pct = player_obj.get('selected_by_percent', '0')
                    try:
                        record['selected_by_percent'] = float(str(selected_pct).rstrip('%'))
                    except (TypeError, ValueError):
                        record['selected_by_percent'] = 0.0
                    record['starts'] = _to_int(player_obj.get('starts', 0), 0)
                    record['status'] = player_obj.get('status', 'u')
                    record['expected_goals'] = _to_float(player_obj.get('expected_goals', 0.0), 0.0)
                    record['expected_assists'] = _to_float(player_obj.get('expected_assists', 0.0), 0.0)
                    record['expected_goal_involvements'] = _to_float(player_obj.get('expected_goal_involvements', 0.0), 0.0)
                    record['expected_goals_conceded'] = _to_float(player_obj.get('expected_goals_conceded', 0.0), 0.0)
                    record['chance_of_playing_next_round'] = _to_float(player_obj.get('chance_of_playing_next_round', 100.0), 100.0)
                    record['chance_of_playing_this_round'] = _to_float(player_obj.get('chance_of_playing_this_round', 100.0), 100.0)

                    # Gameweek history fields
                    for col in self.output_columns:
                        source_key = API_FIELD_RENAMES.get(col, col)
                        if source_key in gw:
                            record[col] = gw.get(source_key)

                    # Cleaned CSV expects resolved opponent name and price in real units.
                    opponent_team_id = _to_int(record.get('opponent_team', 0), 0)
                    record['opp_team_name'] = self.teams.get(opponent_team_id, 'Unknown')

                    record['creativity'] = _to_float(record.get('creativity', 0.0), 0.0)
                    record['ict_index'] = _to_float(record.get('ict_index', 0.0), 0.0)
                    record['influence'] = _to_float(record.get('influence', 0.0), 0.0)
                    record['threat'] = _to_float(record.get('threat', 0.0), 0.0)

                    # API returns value as tenths (e.g. 55 -> 5.5)
                    record['value'] = _to_float(record.get('value', 0.0), 0.0) / 10

                    # Keep GW synchronized with round for compatibility.
                    gw_round = _to_int(record.get('round', 0), 0)
                    record['round'] = gw_round
                    record['GW'] = gw_round

                    # Fill default values for any missing numeric columns.
                    for numeric_col in [
                        'assists', 'bonus', 'bps', 'clean_sheets', 'fixture', 'goals_conceded',
                        'goals_scored', 'minutes', 'opponent_team', 'own_goals',
                        'penalties_missed', 'penalties_saved', 'red_cards', 'saves',
                        'selected', 'team_a_score', 'team_h_score', 'total_points',
                        'transfers_balance', 'transfers_in', 'transfers_out', 'yellow_cards'
                    ]:
                        if record.get(numeric_col) is None:
                            record[numeric_col] = _to_int(record.get(numeric_col, 0), 0)

                    if record.get('kickoff_time') is None:
                        record['kickoff_time'] = ''
                    if record.get('was_home') is None:
                        record['was_home'] = False

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

    def _to_float(self, value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _to_int(self, value, default=0):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _to_bool(self, value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {'1', 'true', 't', 'yes', 'y'}
        return default

    def _normalize_record(self, record: Dict) -> Dict:
        """Normalize types and fill missing values using schema defaults."""
        normalized = {}

        for col in self.output_columns:
            value = record.get(col)

            if col in INT_COLUMNS:
                normalized[col] = self._to_int(value, 0)
            elif col in FLOAT_COLUMNS:
                default = 100.0 if col in {
                    'chance_of_playing_next_round', 'chance_of_playing_this_round'
                } else 0.0
                normalized[col] = self._to_float(value, default)
            elif col in BOOL_COLUMNS:
                normalized[col] = self._to_bool(value, False)
            elif col in STRING_COLUMNS:
                normalized[col] = '' if value is None else str(value)
            else:
                normalized[col] = value

        # Keep GW and round synchronized.
        gw_round = self._to_int(normalized.get('round'), 0)
        normalized['round'] = gw_round
        normalized['GW'] = gw_round

        # Resolve opponent team name if absent.
        if not normalized.get('opp_team_name'):
            normalized['opp_team_name'] = self.teams.get(
                self._to_int(normalized.get('opponent_team'), 0),
                'Unknown'
            )

        # `value` from history is in tenths. `now_cost` already converted from bootstrap below.
        normalized['value'] = self._to_float(normalized.get('value'), 0.0) / 10

        return normalized

    def fetch_player_gameweek_data_for_range(self, player_obj: Dict, min_gw: int, max_gw: int) -> List[Dict]:
        """Fetch and normalize player gameweek rows in a bounded GW range."""
        player_id = player_obj['id']
        player_name = f"{player_obj['first_name']} {player_obj['second_name']}"
        position = self.get_position_name(player_obj['element_type'])
        team_id = player_obj['team']
        team_name = self.teams.get(team_id, 'Unknown')

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                url = self.PLAYER_DETAIL_URL.format(player_id=player_id)
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                history = data.get('history', [])

                if not history:
                    return []

                filtered_history = [
                    gw for gw in history
                    if min_gw <= self._to_int(gw.get('round'), 0) <= max_gw
                ]

                if not filtered_history:
                    return []

                records = []
                for gw in filtered_history:
                    record = {col: None for col in self.output_columns}

                    record['season'] = self.current_season
                    record['current_season'] = self.current_season
                    record['current_gameweek'] = self.current_gameweek
                    record['name'] = player_name
                    record['first_name'] = player_obj.get('first_name', '')
                    record['second_name'] = player_obj.get('second_name', '')
                    record['web_name'] = player_obj.get('web_name', '')
                    record['position'] = position
                    record['element_type'] = player_obj.get('element_type', 0)
                    record['team'] = team_name
                    record['team_id'] = team_id
                    record['element'] = player_id
                    record['id'] = player_id

                    # Bootstrap attributes used to fill missing values when history lacks a field.
                    record['now_cost'] = self._to_float(player_obj.get('now_cost', 0.0), 0.0) / 10
                    record['ep_next'] = self._to_float(player_obj.get('ep_next', 0.0), 0.0)
                    record['form'] = self._to_float(player_obj.get('form', 0.0), 0.0)
                    record['points_per_game'] = self._to_float(player_obj.get('points_per_game', 0.0), 0.0)
                    selected_pct = str(player_obj.get('selected_by_percent', '0')).rstrip('%')
                    record['selected_by_percent'] = self._to_float(selected_pct, 0.0)
                    record['starts'] = self._to_int(player_obj.get('starts', 0), 0)
                    record['status'] = player_obj.get('status', 'u')
                    record['expected_goals'] = self._to_float(player_obj.get('expected_goals', 0.0), 0.0)
                    record['expected_assists'] = self._to_float(player_obj.get('expected_assists', 0.0), 0.0)
                    record['expected_goal_involvements'] = self._to_float(player_obj.get('expected_goal_involvements', 0.0), 0.0)
                    record['expected_goals_conceded'] = self._to_float(player_obj.get('expected_goals_conceded', 0.0), 0.0)
                    record['chance_of_playing_next_round'] = self._to_float(player_obj.get('chance_of_playing_next_round', 100.0), 100.0)
                    record['chance_of_playing_this_round'] = self._to_float(player_obj.get('chance_of_playing_this_round', 100.0), 100.0)

                    # Fill gameweek-specific fields from element-summary history.
                    for col in self.output_columns:
                        source_key = API_FIELD_RENAMES.get(col, col)
                        if source_key in gw:
                            record[col] = gw.get(source_key)

                    normalized = self._normalize_record(record)
                    records.append(normalized)

                return records

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 503 and attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                return None
            except Exception:
                return None

        return None
    
    def load_existing_data(self):
        """Load existing CSV to check for duplicates."""
        # Always use DEFAULT_CLEANED_COLUMNS for new records
        self.output_columns = DEFAULT_CLEANED_COLUMNS.copy()
        
        if os.path.exists(self.csv_path):
            print(f"\nLoading existing data from {self.csv_path}...")
            df = pd.read_csv(self.csv_path, low_memory=False)
            print(f"Existing records: {len(df)}")
            
            # Check if CSV has missing columns from updated schema
            missing_cols = set(self.output_columns) - set(df.columns)
            if missing_cols:
                print(f"⚠ CSV missing {len(missing_cols)} columns: {sorted(missing_cols)}")
                print("Adding missing columns to CSV...")
                for col in missing_cols:
                    df[col] = None
                # Reorder to match DEFAULT_CLEANED_COLUMNS
                df = df[self.output_columns]
                df.to_csv(self.csv_path, index=False)
                print(f"✓ CSV migrated to new schema with {len(self.output_columns)} columns")
            
            return df
        else:
            print(f"No existing file found at {self.csv_path}")
            return pd.DataFrame(columns=self.output_columns)

    def _calculate_update_window(self, existing_df: pd.DataFrame):
        """Determine GW window for weekly updates and light backfill."""
        if existing_df.empty:
            return 1, self.current_gameweek

        season_df = existing_df[existing_df['season'] == self.current_season].copy()
        if season_df.empty:
            return 1, self.current_gameweek

        season_df['GW_num'] = pd.to_numeric(season_df['GW'], errors='coerce').fillna(0).astype(int)
        max_existing_gw = int(season_df['GW_num'].max())

        # Re-fetch last 2 GWs to catch post-deadline/live corrections and bonus recalculations.
        min_gw = max(1, min(max_existing_gw, self.current_gameweek) - 1)
        return min_gw, self.current_gameweek

    def _upsert_current_season_window(self, existing_df: pd.DataFrame, new_records: List[Dict], min_gw: int, max_gw: int):
        """Replace current-season rows in [min_gw, max_gw] and append fresh rows."""
        if existing_df.empty:
            base_df = pd.DataFrame(columns=self.output_columns)
        else:
            base_df = existing_df.copy()

        # Enforce expected schema before merge.
        for col in self.output_columns:
            if col not in base_df.columns:
                base_df[col] = None
        base_df = base_df[self.output_columns]

        if not base_df.empty:
            base_df['GW_num'] = pd.to_numeric(base_df['GW'], errors='coerce').fillna(0).astype(int)
            keep_mask = ~(
                (base_df['season'] == self.current_season) &
                (base_df['GW_num'] >= min_gw) &
                (base_df['GW_num'] <= max_gw)
            )
            base_df = base_df[keep_mask].drop(columns=['GW_num'])

        if new_records:
            new_df = pd.DataFrame(new_records)
            for col in self.output_columns:
                if col not in new_df.columns:
                    new_df[col] = None
            new_df = new_df[self.output_columns]
            final_df = pd.concat([base_df, new_df], ignore_index=True)
        else:
            final_df = base_df

        final_df.to_csv(self.csv_path, index=False)
        print(f"\n✓ Saved updated CSV to {self.csv_path}")
        print(f"  Rows written: {len(final_df)}")
    
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
        min_gw, max_gw = self._calculate_update_window(existing_df)

        print(
            f"Weekly update window for {self.current_season}: "
            f"GW{min_gw} to GW{max_gw}"
        )
        
        all_records = []
        new_count = 0
        error_count = 0
        batch_size = 20
        
        for idx, player in enumerate(players):
            records = self.fetch_player_gameweek_data_for_range(player, min_gw, max_gw)
            
            if records is None:
                error_count += 1
                records = []

            all_records.extend(records)
            new_count += len(records)
            
            if (idx + 1) % batch_size == 0:
                print(f"  Progress: {idx + 1}/{len(players)} players | "
                      f"Collected Rows: {new_count} | Errors: {error_count}")
            
            # Longer delay to avoid rate limiting
            time.sleep(0.15)

        self._upsert_current_season_window(existing_df, all_records, min_gw, max_gw)
        self.new_records = []
        print(f"\n✓ Fetching complete!")
        print(f"  Total rows collected: {len(all_records)}")
        print(f"  Total errors: {error_count}")
        
        return []
    
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
        """Kept for compatibility with previous script flow."""
        print("\nData is already persisted via upsert in fetch step.")
    
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
