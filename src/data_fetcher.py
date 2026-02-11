"""
Fetches player data from Fantasy Premier League API and collects news articles.
"""

import requests
import json
import time
from datetime import datetime
import feedparser
import os
from typing import List, Dict
import pandas as pd


class FantasyDataFetcher:
    """Handles fetching player data and news articles."""
    
    API_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    
    def __init__(self):
        self.players_data = []
        self.news_data = {}
        self.current_gameweek = None
        self.current_season = None
        self.events = []
        self.teams = {}
        
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
            
            self.players_data = []
            for player in players:
                player_info = {
                    'id': player['id'],
                    'name': f"{player['first_name']} {player['second_name']}",
                    'first_name': player['first_name'],
                    'second_name': player['second_name'],
                    'web_name': player['web_name'],
                    'team': self.teams[player['team']],
                    'team_id': player['team'],
                    'position': self._get_position_name(player['element_type']),
                    'element_type': player['element_type'],
                    # Current season stats
                    'total_points': player['total_points'],
                    'now_cost': player['now_cost'] / 10,
                    'form': float(player['form']) if player['form'] else 0.0,
                    'points_per_game': float(player['points_per_game']) if player['points_per_game'] else 0.0,
                    'selected_by_percent': float(player['selected_by_percent']) if player['selected_by_percent'] else 0.0,
                    # Detailed stats for feature engineering
                    'minutes': player['minutes'],
                    'goals_scored': player['goals_scored'],
                    'assists': player['assists'],
                    'clean_sheets': player['clean_sheets'],
                    'goals_conceded': player['goals_conceded'],
                    'own_goals': player['own_goals'],
                    'penalties_saved': player['penalties_saved'],
                    'penalties_missed': player['penalties_missed'],
                    'yellow_cards': player['yellow_cards'],
                    'red_cards': player['red_cards'],
                    'saves': player['saves'],
                    'bonus': player['bonus'],
                    'bps': player['bps'],
                    'influence': float(player['influence']) if player['influence'] else 0.0,
                    'creativity': float(player['creativity']) if player['creativity'] else 0.0,
                    'threat': float(player['threat']) if player['threat'] else 0.0,
                    'ict_index': float(player['ict_index']) if player['ict_index'] else 0.0,
                    # Performance indicators
                    'starts': player.get('starts', 0),
                    'expected_goals': float(player.get('expected_goals', 0)),
                    'expected_assists': float(player.get('expected_assists', 0)),
                    'expected_goal_involvements': float(player.get('expected_goal_involvements', 0)),
                    'expected_goals_conceded': float(player.get('expected_goals_conceded', 0)),
                    # Availability
                    'chance_of_playing_next_round': player.get('chance_of_playing_next_round'),
                    'chance_of_playing_this_round': player.get('chance_of_playing_this_round'),
                    'status': player.get('status', 'a'),
                    # Meta
                    'current_gameweek': self.current_gameweek,
                    'current_season': self.current_season
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
    
    def _get_search_name(self, player: Dict) -> str:
        """
        Get the search name for news fetching.
        Uses first_name + web_name, unless web_name equals first_name,
        then uses first_name + second_name.
        
        Args:
            player: Player dictionary with first_name, second_name, web_name
            
        Returns:
            Search name string
        """
        first_name = player.get('first_name', '')
        second_name = player.get('second_name', '')
        web_name = player.get('web_name', '')
        
        # If web_name is the same as first_name, use full name
        if web_name.lower() == first_name.lower():
            search_name = f"{first_name} {second_name}"
        else:
            # Otherwise use first_name + web_name
            search_name = f"{first_name} {web_name}"
        
        return search_name.strip()
    
    def fetch_player_news(self, player_name: str, max_articles: int = 100) -> List[Dict]:
        """
        Fetch news headlines for a specific player using Google News RSS.
        
        Args:
            player_name: Name of the player
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of news articles with title, link, published date, and summary
        """
        print(f"Fetching news for {player_name}...")
        
        # Google News RSS feed URL
        query = f"{player_name} Premier League"
        rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-GB&gl=GB&ceid=GB:en"
        
        try:
            feed = feedparser.parse(rss_url)
            articles = []
            
            for entry in feed.entries[:max_articles]:
                article_data = {
                    'title': entry.title,
                    'link': entry.link,
                    'published': entry.get('published', ''),
                    'source': entry.get('source', {}).get('title', 'Unknown'),
                    'summary': entry.get('summary', '')
                }
                
                articles.append(article_data)
                time.sleep(0.5)  # Be respectful to servers
            
            print(f"  Found {len(articles)} articles for {player_name}")
            return articles
            
        except Exception as e:
            print(f"  Error fetching news for {player_name}: {e}")
            return []
    
    def fetch_all_player_news(self, limit: int = None):
        """
        Fetch news for all players.
        
        Args:
            limit: Limit number of players to process (for testing)
        """
        if not self.players_data:
            self.fetch_players()
        
        players_to_process = self.players_data[:limit] if limit else self.players_data
        
        for i, player in enumerate(players_to_process, 1):
            print(f"\n[{i}/{len(players_to_process)}]", end=" ")
            
            # Use intelligent search name
            search_name = self._get_search_name(player)
            print(f"Searching for: {search_name} (web_name: {player['web_name']})")
            
            articles = self.fetch_player_news(search_name)
            self.news_data[player['id']] = {
                'player_name': player['name'],
                'search_name': search_name,
                'web_name': player['web_name'],
                'team': player['team'],
                'articles': articles,
                'article_count': len(articles)
            }
            
            # Save progress periodically
            if i % 10 == 0:
                self.save_data()
    
    def save_data(self):
        """Save fetched data to files."""
        os.makedirs('data', exist_ok=True)
        
        # Save players data
        with open('data/players.json', 'w') as f:
            json.dump(self.players_data, f, indent=2)
        
        # Save news data
        with open('data/news.json', 'w') as f:
            json.dump(self.news_data, f, indent=2)
        
        # Save as CSV for easy viewing
        df = pd.DataFrame(self.players_data)
        df.to_csv('data/players.csv', index=False)
        
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
    
    # Fetch player data
    fetcher.fetch_players()
    fetcher.save_data()
    
    # Fetch news for all players (or limit for testing)
    # Use limit=5 for testing, remove limit for production
    print("\n" + "="*50)
    print("Fetching news articles for players...")
    print("="*50)
    fetcher.fetch_all_player_news(limit=10)  # Start with 10 players for testing
    
    fetcher.save_data()
    print("\n✓ Data fetching complete!")


if __name__ == "__main__":
    main()
