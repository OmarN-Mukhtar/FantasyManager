"""
Main script to run the complete Fantasy PL analysis pipeline.
"""

import sys
import argparse
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_fetcher import FantasyDataFetcher
from sentiment_analyzer import SentimentAnalyzer
from predictor import PlayerPredictor
from rag_system import PlayerNewsRAG


def run_pipeline(fetch_limit=None, min_articles=50, skip_news=False):
    """
    Run the complete analysis pipeline.
    
    Args:
        fetch_limit: Limit number of players to fetch news for (None = all)
        min_articles: Minimum articles required for sentiment scoring
        skip_news: Skip news fetching (use existing data)
    """
    print("\n" + "="*70)
    print("FANTASY PREMIER LEAGUE ANALYSIS PIPELINE")
    print("="*70)
    
    if not skip_news:
        # Step 1: Fetch player data and news
        print("\n[STEP 1/5] Fetching player data and news...")
        print("-"*70)
        fetcher = FantasyDataFetcher()
        fetcher.fetch_players()
        
        if fetch_limit:
            print(f"Fetching news for {fetch_limit} players (testing mode)...")
        else:
            print("Fetching news for all players (this may take a while)...")
        
        fetcher.fetch_all_player_news(limit=fetch_limit)
        fetcher.save_data()
        
        print("\n✓ Step 1 complete: Player data and news fetched")
    else:
        print("\n[STEP 1/5] Skipping news fetch - using existing data")
    
    # Step 2: Sentiment analysis
    print("\n[STEP 2/5] Analyzing sentiment...")
    print("-"*70)
    analyzer = SentimentAnalyzer()
    analyzer.load_data()
    analyzer.analyze_all_players(min_articles=min_articles)
    analyzer.save_results()
    
    print("\n✓ Step 2 complete: Sentiment analysis done")
    
    # Step 3: Performance predictions
    print("\n[STEP 3/5] Predicting player performance...")
    print("-"*70)
    predictor = PlayerPredictor()
    predictor.load_data()
    predictor.predict_all_current_players()
    predictor.save_predictions()
    
    print("\n✓ Step 3 complete: Predictions generated")
    
    # Step 4: Create RAG system
    print("\n[STEP 4/5] Creating FAISS vector database...")
    print("-"*70)
    rag = PlayerNewsRAG()
    rag.load_data()
    rag.create_vector_db()
    
    print("\n✓ Step 4 complete: RAG system ready")
    
    # Step 5: Summary
    print("\n[STEP 5/5] Pipeline Summary")
    print("-"*70)
    
    if not skip_news:
        print(f"✓ Players processed: {len(fetcher.players_data)}")
        print(f"✓ Players with news: {len(fetcher.news_data)}")
        total_articles = sum(len(news.get('articles', [])) for news in fetcher.news_data.values())
        print(f"✓ Total articles collected: {total_articles}")
    
    qualified = sum(1 for s in analyzer.sentiment_scores if s.get('has_enough_articles'))
    print(f"✓ Players with sentiment scores: {qualified}")
    
    print(f"✓ Players with predictions: {len(predictor.predictions)}")
    
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. View data in 'data/' directory")
    print("2. Launch dashboard: streamlit run src/dashboard.py")
    print("3. Query RAG system: python src/rag_system.py")
    print("\n")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Fantasy Premier League Analysis Pipeline"
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode (process only 10 players)'
    )
    
    parser.add_argument(
        '--skip-news',
        action='store_true',
        help='Skip news fetching step (use existing data)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of players to process'
    )
    
    parser.add_argument(
        '--min-articles',
        type=int,
        default=50,
        help='Minimum articles required for sentiment scoring (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Create data directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('vector_db', exist_ok=True)
    
    # Set limit based on test mode
    limit = 10 if args.test else args.limit
    
    if args.test:
        print("\n⚠️  Running in TEST MODE (10 players only)")
        print("Remove --test flag to process all players\n")
    
    # Run the pipeline
    run_pipeline(fetch_limit=limit, min_articles=args.min_articles, skip_news=args.skip_news)


if __name__ == "__main__":
    main()
