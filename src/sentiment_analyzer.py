"""
Performs sentiment analysis on player news articles using BERT.
"""

import json
import pandas as pd
from transformers import pipeline
from typing import Dict, List
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Analyzes sentiment of news articles for players using BERT."""
    
    def __init__(self):
        self.players_data = []
        self.news_data = {}
        self.sentiment_scores = {}
        
        # Initialize BERT sentiment analysis pipeline
        print("Loading BERT sentiment model (distilbert-base-uncased-finetuned-sst-2-english)...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Use CPU (-1), change to 0 for GPU
        )
        print("✓ BERT model loaded successfully")
        
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
        """Analyze sentiment using BERT. Returns polarity score (-1 to 1)."""
        if not text or len(text.strip()) == 0:
            return 0.0
        try:
            result = self.sentiment_pipeline(text[:2000])[0]  # Truncate to BERT's max length
            score = result['score']
            polarity = (score - 0.5) * 2 if result['label'] == 'POSITIVE' else -((score - 0.5) * 2)
            return polarity
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def analyze_article_sentiment(self, article: Dict) -> Dict:
        """Analyze sentiment of a single article (title + summary)."""
        title_sent = self.analyze_text_sentiment(article.get('title', ''))
        summary_sent = self.analyze_text_sentiment(article.get('summary', ''))
        return {
            'title': article.get('title', ''), 'published': article.get('published', ''),
            'title_sentiment': title_sent, 'summary_sentiment': summary_sent,
            'combined_sentiment': title_sent * 0.3 + summary_sent * 0.7
        }
    
    def analyze_player_sentiment(self, player_id: str, min_articles: int = 50) -> Dict:
        """Analyze overall sentiment for a player based on their news articles."""
        if player_id not in self.news_data:
            return None
        
        news = self.news_data[player_id]
        articles = news.get('articles', [])
        
        if len(articles) < min_articles:
            return {'player_id': player_id, 'player_name': news['player_name'], 'team': news['team'],
                    'article_count': len(articles), 'has_enough_articles': False,
                    'sentiment_score': None, 'normalized_score': None}
        
        sentiments = [self.analyze_article_sentiment(a)['combined_sentiment'] for a in articles]
        avg_sent = np.mean(sentiments)
        
        return {
            'player_id': player_id, 'player_name': news['player_name'], 'team': news['team'],
            'article_count': len(articles), 'has_enough_articles': True,
            'avg_sentiment': avg_sent, 'median_sentiment': np.median(sentiments),
            'std_sentiment': np.std(sentiments), 'normalized_score': round((avg_sent + 1) * 50, 2),
            'sentiment_distribution': {
                'positive': sum(1 for s in sentiments if s > 0.1),
                'neutral': sum(1 for s in sentiments if -0.1 <= s <= 0.1),
                'negative': sum(1 for s in sentiments if s < -0.1)
            }
        }
    
    def analyze_all_players(self, min_articles: int = 50):
        """Analyze sentiment for all players with news articles."""
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
                          f"Score {result['normalized_score']}/100 ({result['article_count']} articles)")
        
        self.sentiment_scores = results
        print(f"\n{qualified_count}/{len(results)} players have enough articles for sentiment scoring")
        return results
    
    def save_results(self):
        """Save sentiment analysis results."""
        with open('data/sentiment_analysis.json', 'w') as f:
            json.dump(self.sentiment_scores, f, indent=2)
        
        df = pd.DataFrame(self.sentiment_scores)
        players_df = pd.DataFrame(self.players_data)
        df['player_id'] = df['player_id'].astype(str)
        players_df['id'] = players_df['id'].astype(str)
        
        complete_df = players_df.merge(
            df[['player_id', 'article_count', 'has_enough_articles', 'normalized_score']],
            left_on='id', right_on='player_id', how='left'
        )
        complete_df.to_csv('data/players_with_sentiment.csv', index=False)
        
        qualified_df = complete_df[complete_df['has_enough_articles'] == True].sort_values('normalized_score', ascending=False)
        qualified_df.to_csv('data/qualified_players.csv', index=False)
        
        print(f"\n✓ Results saved to 'data/' directory")
        print(f"  - sentiment_analysis.json: Detailed analysis")
        print(f"  - players_with_sentiment.csv: All players with sentiment data")
        print(f"  - qualified_players.csv: Players with {qualified_df.shape[0]} or more articles")


def main():
    """Main execution function."""
    analyzer = SentimentAnalyzer()
    if not analyzer.load_data():
        print("Please run data_fetcher.py first to fetch player data and news")
        return
    analyzer.analyze_all_players(min_articles=50)
    analyzer.save_results()
    print("\n✓ Sentiment analysis complete!")


if __name__ == "__main__":
    main()
