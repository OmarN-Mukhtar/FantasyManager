import json
import pandas as pd
from transformers import pipeline
from typing import Dict, List
from datetime import datetime, timedelta
import feedparser
import warnings
warnings.filterwarnings('ignore')

# ponytail: 3-day half-life over the 7-day window; raise if news stays relevant longer
HALF_LIFE_DAYS = 3


def recency_weight(age_days):
    """Weight for an article's sentiment: halves every HALF_LIFE_DAYS."""
    return 0.5 ** (max(age_days, 0) / HALF_LIFE_DAYS)


class SentimentAnalyzer:
    """Fetches player news and computes simple BERT sentiment scores."""
    
    def __init__(self):
        self.players_data = []
        self.news_data = {}
        self.sentiment_scores = []
        
        # Initialize BERT sentiment analysis pipeline
        print("Loading BERT sentiment model (distilbert-base-uncased-finetuned-sst-2-english)...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # Use CPU (-1), change to 0 for GPU
        )
        print("✓ BERT model loaded successfully")
        
    def load_data(self):
        """Load player data."""
        try:
            with open('data/players.json', 'r') as f:
                self.players_data = json.load(f)
            print(f"Loaded {len(self.players_data)} players")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _search_name(self, player: Dict) -> str:
        """Build search name from first_name + web_name."""
        first = str(player.get('first_name', '')).strip()
        web = str(player.get('web_name', '')).strip()
        return f"{first} {web}".strip()

    def fetch_player_news(self, search_name: str, max_articles: int = 30, days_back: int = 7) -> List[Dict]:
        """Fetch recent Google News RSS articles for a player search term."""
        rss_url = (
            f"https://news.google.com/rss/search?q={search_name.replace(' ', '+')}+Premier+League"
            "&hl=en-GB&gl=GB&ceid=GB:en"
        )

        feed = feedparser.parse(rss_url)
        cutoff = datetime.now() - timedelta(days=days_back)
        articles = []

        for entry in feed.entries[:max_articles]:
            age_days = float(days_back)  # undated articles get minimum weight
            try:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    published_dt = datetime(*entry.published_parsed[:6])
                    if published_dt < cutoff:
                        continue
                    age_days = (datetime.now() - published_dt).total_seconds() / 86400
            except Exception:
                pass

            articles.append({
                'title': entry.get('title', ''),
                'summary': entry.get('summary', ''),
                'published': entry.get('published', ''),
                'link': entry.get('link', ''),
                'age_days': age_days,
            })

        return articles
    
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
    
    def analyze_all_players(self, days_back: int = 7):
        """Fetch player news and compute one sentiment score per player."""
        print("\nFetching headlines and scoring sentiment...")

        results = []
        self.news_data = {}

        for i, player in enumerate(self.players_data, 1):
            player_name = player.get('name') or f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()
            search_name = self._search_name(player)
            articles = self.fetch_player_news(search_name, max_articles=30, days_back=days_back)

            # ponytail: headlines only — only titles are saved/shown; halves BERT calls
            scored = [
                (self.analyze_text_sentiment(a.get('title', '')), recency_weight(a.get('age_days', days_back)))
                for a in articles
            ]
            total_weight = sum(w for _, w in scored)
            sentiment_score = (
                sum(s * w for s, w in scored) / total_weight if total_weight else 0.0
            )

            results.append({
                'player_name': str(player_name),
                'sentiment_score': round(sentiment_score, 4),
            })

            self.news_data[str(player_name)] = {
                'search_name': search_name,
                'headlines': [a.get('title', '') for a in articles],
            }

            if i % 25 == 0:
                print(f"  Processed {i}/{len(self.players_data)} players")

        self.sentiment_scores = results
        return results
    
    def save_results(self):
        """Save only sentiment_analysis outputs and fetched headlines."""
        with open('data/news.json', 'w') as f:
            json.dump(self.news_data, f, indent=2)

        with open('data/sentiment_analysis.json', 'w') as f:
            json.dump(self.sentiment_scores, f, indent=2)

        pd.DataFrame(self.sentiment_scores).to_csv('data/sentiment_analysis.csv', index=False)

        print("\n✓ Results saved")
        print("  - data/news.json")
        print("  - data/sentiment_analysis.json")
        print("  - data/sentiment_analysis.csv")


def main():
    """Main execution function."""
    analyzer = SentimentAnalyzer()
    if not analyzer.load_data():
        print("Please run data_fetcher.py first to fetch player data and news")
        return
    analyzer.analyze_all_players(days_back=7)
    analyzer.save_results()
    print("\n✓ Sentiment analysis complete!")


if __name__ == "__main__":
    main()