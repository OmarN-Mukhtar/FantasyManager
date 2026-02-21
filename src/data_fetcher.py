"""
Fetches player data from Fantasy Premier League API and collects news articles.
"""

import requests
import json
import time
from datetime import datetime, timedelta
import feedparser
import os
from typing import List, Dict
import pandas as pd


class FantasyDataFetcher:
    """Handles fetching player data and news articles."""
    
    API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
    
    def __init__(self):
        self.players_data = []
        self.news_data = {}
        self.current_gameweek = None
        self.current_season = None
        self.events = []
        self.teams = {}
        self.team_fixtures = {}  # Store next fixture for each team
        
    def fetch_players(self) -> List[Dict]:
        """Fetch all players from the Fantasy Premier League API with detailed stats."""
        print("Fetching player data from Fantasy Premier League API...")
        
        try:
            response = requests.get(self.API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract season and gameweek information
            self.events = data['events']
            current_event = next((e for e in self.events if e['is_current']), None)
            if current_event:
                self.current_gameweek = current_event['id']
                # Parse season from event name (e.g., "Gameweek 1" in 2025/26)
                # Use current year to determine season
                current_year = datetime.now().year
                current_month = datetime.now().month
                # If we're past August, season is current_year/next_year
                if current_month >= 8:
                    self.current_season = f"{current_year}-{str(current_year + 1)[-2:]}"
                else:
                    self.current_season = f"{current_year - 1}-{str(current_year)[-2:]}"
            else:
                # Season hasn't started, use next season
                current_year = datetime.now().year
                self.current_season = f"{current_year}-{str(current_year + 1)[-2:]}"
                self.current_gameweek = 0
            
            print(f"Current Season: {self.current_season}, Gameweek: {self.current_gameweek}")
            
            # Extract team information
            players = data['elements']
            self.teams = {team['id']: team['name'] for team in data['teams']}
            
            # Fetch upcoming fixtures
            self._fetch_team_fixtures()
            
            self.players_data = []
            for player in players:
                cost = player['now_cost'] / 10
                fixture = self.team_fixtures.get(player['team'], {})
                
                player_info = {
                    'id': player['id'], 'element': player['id'],
                    'name': f"{player['first_name']} {player['second_name']}",
                    'first_name': player['first_name'], 'second_name': player['second_name'], 'web_name': player['web_name'],
                    'team': self.teams[player['team']], 'team_id': player['team'],
                    'position': self._get_position_name(player['element_type']), 'element_type': player['element_type'],
                    'total_points': player['total_points'], 'now_cost': cost, 'value': cost,
                    'form': float(player['form'] or 0), 'points_per_game': float(player['points_per_game'] or 0),
                    'selected_by_percent': float(player['selected_by_percent'] or 0),
                    'minutes': player['minutes'], 'goals_scored': player['goals_scored'], 'assists': player['assists'],
                    'clean_sheets': player['clean_sheets'], 'goals_conceded': player['goals_conceded'],
                    'own_goals': player['own_goals'], 'penalties_saved': player['penalties_saved'],
                    'penalties_missed': player['penalties_missed'], 'yellow_cards': player['yellow_cards'],
                    'red_cards': player['red_cards'], 'saves': player['saves'], 'bonus': player['bonus'], 'bps': player['bps'],
                    'influence': float(player['influence'] or 0), 'creativity': float(player['creativity'] or 0),
                    'threat': float(player['threat'] or 0), 'ict_index': float(player['ict_index'] or 0),
                    'starts': player.get('starts', 0),
                    'expected_goals': float(player.get('expected_goals', 0)),
                    'expected_assists': float(player.get('expected_assists', 0)),
                    'expected_goal_involvements': float(player.get('expected_goal_involvements', 0)),
                    'expected_goals_conceded': float(player.get('expected_goals_conceded', 0)),
                    'chance_of_playing_next_round': player.get('chance_of_playing_next_round'),
                    'chance_of_playing_this_round': player.get('chance_of_playing_this_round'),
                    'status': player.get('status', 'a'),
                    'transfers_in': player.get('transfers_in', 0), 'transfers_out': player.get('transfers_out', 0),
                    'transfers_in_event': player.get('transfers_in_event', 0), 'transfers_out_event': player.get('transfers_out_event', 0),
                    'transfers_balance': player.get('transfers_in', 0) - player.get('transfers_out', 0),
                    'selected': player.get('selected_by', 0),
                    'current_gameweek': self.current_gameweek, 'GW': self.current_gameweek, 'round': self.current_gameweek,
                    'current_season': self.current_season, 'season': self.current_season,
                    'fixture': fixture.get('fixture_id'), 'opponent_team': fixture.get('opponent_team'),
                    'opp_team_name': fixture.get('opponent_name'), 'was_home': fixture.get('is_home'),
                    'kickoff_time': fixture.get('kickoff_time')
                }
                self.players_data.append(player_info)
            
            print(f"Successfully fetched {len(self.players_data)} players")
            return self.players_data
            
        except Exception as e:
            print(f"Error fetching player data: {e}")
            return []
    
    def _fetch_team_fixtures(self):
        """Fetch upcoming fixtures for all teams."""
        try:
            print("Fetching fixture data...")
            response = requests.get(self.FIXTURES_URL, timeout=10)
            response.raise_for_status()
            fixtures = response.json()
            
            for fixture in fixtures:
                if not fixture.get('finished', False) and not fixture.get('started', False):
                    fid, team_h, team_a, kickoff = fixture['id'], fixture['team_h'], fixture['team_a'], fixture.get('kickoff_time', '')
                    
                    if team_h not in self.team_fixtures:
                        self.team_fixtures[team_h] = {'fixture_id': fid, 'opponent_team': team_a, 
                                                       'opponent_name': self.teams.get(team_a, 'Unknown'), 
                                                       'is_home': True, 'kickoff_time': kickoff}
                    if team_a not in self.team_fixtures:
                        self.team_fixtures[team_a] = {'fixture_id': fid, 'opponent_team': team_h,
                                                       'opponent_name': self.teams.get(team_h, 'Unknown'),
                                                       'is_home': False, 'kickoff_time': kickoff}
            
            print(f"Loaded fixtures for {len(self.team_fixtures)} teams")
        except Exception as e:
            print(f"Warning: Could not fetch fixtures: {e}")
            self.team_fixtures = {}
    
    def _get_position_name(self, position_id: int) -> str:
        """Convert position ID to name."""
        positions = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
        return positions.get(position_id, 'Unknown')
    
    def _get_search_name(self, player: Dict) -> str:
        """Get search name for news: first_name + web_name, or full name if web_name equals first_name."""
        first, second, web = player.get('first_name', ''), player.get('second_name', ''), player.get('web_name', '')
        return f"{first} {second}" if web.lower() == first.lower() else f"{first} {web}".strip()
    
    def fetch_player_news(self, player_name: str, max_articles: int = 50, days_back: int = 7) -> List[Dict]:
        """Fetch news headlines for a player using Google News RSS (last N days only)."""
        rss_url = f"https://news.google.com/rss/search?q={player_name.replace(' ', '+')}+Premier+League&hl=en-GB&gl=GB&ceid=GB:en"
        try:
            feed = feedparser.parse(rss_url)
            cutoff = datetime.now() - timedelta(days=days_back)
            articles = []
            
            for entry in feed.entries[:max_articles]:
                try:
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        if datetime(*entry.published_parsed[:6]) < cutoff:
                            continue
                except:
                    pass
                
                articles.append({
                    'title': entry.title, 'link': entry.link, 'published': entry.get('published', ''),
                    'source': entry.get('source', {}).get('title', 'Unknown'), 'summary': entry.get('summary', '')
                })
            
            if articles:
                print(f"  Found {len(articles)} recent articles")
            return articles
        except Exception as e:
            print(f"  Error fetching news for {player_name}: {e}")
            return []
    
    def fetch_all_player_news(self, limit: int = 150, days_back: int = 7):
        """Fetch news for top players by ownership (last N days only)."""
        if not self.players_data:
            self.fetch_players()
        
        players = sorted(self.players_data, key=lambda p: p.get('selected_by_percent', 0), reverse=True)[:limit]
        print(f"\nFetching news for top {len(players)} players (last {days_back} days)...")
        
        for i, player in enumerate(players, 1):
            print(f"[{i}/{len(players)}] {player['name']} ({player['selected_by_percent']}% owned)", end=" ")
            articles = self.fetch_player_news(self._get_search_name(player), max_articles=50, days_back=days_back)
            self.news_data[player['id']] = {
                'player_name': player['name'], 'search_name': self._get_search_name(player),
                'web_name': player['web_name'], 'team': player['team'],
                'articles': articles, 'article_count': len(articles)
            }
            if i % 20 == 0:
                self.save_data()
            time.sleep(0.3)
    
    def save_data(self):
        """Save fetched data to files."""
        os.makedirs('data', exist_ok=True)
        with open('data/players.json', 'w') as f:
            json.dump(self.players_data, f, indent=2)
        with open('data/news.json', 'w') as f:
            json.dump(self.news_data, f, indent=2)
        pd.DataFrame(self.players_data).to_csv('data/players.csv', index=False)
        print(f"\nData saved to 'data/' directory")
    
    def load_data(self):
        """Load previously fetched data."""
        try:
            with open('data/players.json', 'r') as f:
                self.players_data = json.load(f)
            with open('data/news.json', 'r') as f:
                self.news_data = json.load(f)
            print("Data loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False


def main():
    """Main execution function."""
    fetcher = FantasyDataFetcher()
    fetcher.fetch_players()
    fetcher.save_data()
    print("\n" + "="*50 + "\nFetching recent news for top players...\n" + "="*50)
    fetcher.fetch_all_player_news(limit=400, days_back=7)
    fetcher.save_data()
    print("\n✓ Data fetching complete!")


if __name__ == "__main__":
    main()
