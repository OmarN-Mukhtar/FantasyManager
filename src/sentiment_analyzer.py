"""
Performs sentiment analysis on player news articles.
"""

import json
import pandas as pd
from textblob import TextBlob
from typing import Dict, List
import numpy as np
from datetime import datetime


class SentimentAnalyzer:
    """Analyzes sentiment of news articles for players."""
    
    def __init__(self):
        self.players_data = []
        self.news_data = {}
        self.sentiment_scores = {}
        
    def load_data(self):
        """Load player and news data."""
        try:
            with open('data/players.json', 'r') as f:
                self.players_data = json.load(f)
            
            with open('data/news.json', 'r') as f:
                self.news_data = json.load(f)
            
            print(f"Loaded {len(self.players_data)} players and news for {len(self.news_data)} players")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def analyze_text_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a text using TextBlob.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment polarity score (-1 to 1)
        """
        if not text:
            return 0.0
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def analyze_article_sentiment(self, article: Dict) -> Dict:
        """
        Analyze sentiment of a single article.
        
        Args:
            article: Article dictionary with title and content
            
        Returns:
            Dictionary with sentiment scores
        """
        title_sentiment = self.analyze_text_sentiment(article.get('title', ''))
        content_sentiment = self.analyze_text_sentiment(article.get('content', ''))
        
        # Weight content more heavily than title
        combined_sentiment = (title_sentiment * 0.3 + content_sentiment * 0.7)
        
        return {
            'title': article.get('title', ''),
            'published': article.get('published', ''),
            'title_sentiment': title_sentiment,
            'content_sentiment': content_sentiment,
            'combined_sentiment': combined_sentiment
        }
    
    def analyze_player_sentiment(self, player_id: str, min_articles: int = 50) -> Dict:
        """
        Analyze overall sentiment for a player.
        
        Args:
            player_id: Player ID
            min_articles: Minimum number of articles required for scoring
            
        Returns:
            Dictionary with player sentiment analysis
        """
        if player_id not in self.news_data:
            return None
        
        news = self.news_data[player_id]
        articles = news.get('articles', [])
        
        if len(articles) < min_articles:
            return {
                'player_id': player_id,
                'player_name': news['player_name'],
                'team': news['team'],
                'article_count': len(articles),
                'has_enough_articles': False,
                'sentiment_score': None,
                'normalized_score': None
            }
        
        # Analyze each article
        article_sentiments = []
        for article in articles:
            sentiment = self.analyze_article_sentiment(article)
            article_sentiments.append(sentiment['combined_sentiment'])
        
        # Calculate statistics
        avg_sentiment = np.mean(article_sentiments)
        std_sentiment = np.std(article_sentiments)
        median_sentiment = np.median(article_sentiments)
        
        # Normalize to 0-100 scale
        # TextBlob polarity ranges from -1 (negative) to 1 (positive)
        # We'll map this to 0-100 where 50 is neutral
        normalized_score = (avg_sentiment + 1) * 50
        
        return {
            'player_id': player_id,
            'player_name': news['player_name'],
            'team': news['team'],
            'article_count': len(articles),
            'has_enough_articles': True,
            'avg_sentiment': avg_sentiment,
            'median_sentiment': median_sentiment,
            'std_sentiment': std_sentiment,
            'normalized_score': round(normalized_score, 2),
            'sentiment_distribution': {
                'positive': sum(1 for s in article_sentiments if s > 0.1),
                'neutral': sum(1 for s in article_sentiments if -0.1 <= s <= 0.1),
                'negative': sum(1 for s in article_sentiments if s < -0.1)
            }
        }
    
    def analyze_all_players(self, min_articles: int = 50):
        """
        Analyze sentiment for all players.
        
        Args:
            min_articles: Minimum number of articles required for scoring
        """
        print(f"\nAnalyzing sentiment for players with at least {min_articles} articles...")
        
        results = []
        qualified_count = 0
        
        for player_id in self.news_data.keys():
            result = self.analyze_player_sentiment(player_id, min_articles)
            if result:
                results.append(result)
                if result['has_enough_articles']:
                    qualified_count += 1
                    print(f"✓ {result['player_name']} ({result['team']}): "
                          f"Score {result['normalized_score']}/100 "
                          f"({result['article_count']} articles)")
        
        self.sentiment_scores = results
        print(f"\n{qualified_count}/{len(results)} players have enough articles for sentiment scoring")
        
        return results
    
    def save_results(self):
        """Save sentiment analysis results."""
        # Save detailed results
        with open('data/sentiment_analysis.json', 'w') as f:
            json.dump(self.sentiment_scores, f, indent=2)
        
        # Create summary DataFrame
        df = pd.DataFrame(self.sentiment_scores)
        
        # Merge with player data for complete information
        players_df = pd.DataFrame(self.players_data)
        
        # Convert player_id to string for merging
        df['player_id'] = df['player_id'].astype(str)
        players_df['id'] = players_df['id'].astype(str)
        
        # Merge dataframes
        complete_df = players_df.merge(
            df[['player_id', 'article_count', 'has_enough_articles', 'normalized_score']],
            left_on='id',
            right_on='player_id',
            how='left'
        )
        
        # Save complete dataset
        complete_df.to_csv('data/players_with_sentiment.csv', index=False)
        
        # Save only qualified players (with sentiment scores)
        qualified_df = complete_df[complete_df['has_enough_articles'] == True]
        qualified_df = qualified_df.sort_values('normalized_score', ascending=False)
        qualified_df.to_csv('data/qualified_players.csv', index=False)
        
        print(f"\n✓ Results saved to 'data/' directory")
        print(f"  - sentiment_analysis.json: Detailed analysis")
        print(f"  - players_with_sentiment.csv: All players with sentiment data")
        print(f"  - qualified_players.csv: Players with {qualified_df.shape[0]} or more articles")


def main():
    """Main execution function."""
    analyzer = SentimentAnalyzer()
    
    # Load data
    if not analyzer.load_data():
        print("Please run data_fetcher.py first to fetch player data and news")
        return
    
    # Analyze sentiment
    analyzer.analyze_all_players(min_articles=50)
    
    # Save results
    analyzer.save_results()
    
    print("\n✓ Sentiment analysis complete!")


if __name__ == "__main__":
    main()
