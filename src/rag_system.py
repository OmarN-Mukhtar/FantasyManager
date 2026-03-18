"""
RAG (Retrieval Augmented Generation) system for player news using FAISS.
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import pickle
from datetime import datetime
from pathlib import Path


# Set up project paths for compatibility with different working directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
VECTOR_DB_DIR = PROJECT_ROOT / 'vector_db'


class PlayerNewsRAG:
    """RAG system for querying player news using FAISS."""

    def __init__(self, persist_directory: str = None):
        """Initialize the RAG system with FAISS."""
        if persist_directory is None:
            persist_directory = str(VECTOR_DB_DIR)

        self.persist_directory = persist_directory
        self.data_dir = DATA_DIR
        os.makedirs(persist_directory, exist_ok=True)

        # Use sentence transformers for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2

        self.index = None
        self.documents = []
        self.metadatas = []
        self.players_data = []
        self.news_data = {}
        self.sentiment_lookup = {}
        self.sentiment_name_lookup = {}
        self.predictions_data = {}
        self.data_signature = {}

    def _safe_file_stats(self, path: Path) -> Dict:
        """Return lightweight file stats for cache invalidation checks."""
        try:
            stats = path.stat()
            return {
                'exists': True,
                'size': int(stats.st_size),
                'mtime': int(stats.st_mtime)
            }
        except Exception:
            return {
                'exists': False,
                'size': 0,
                'mtime': 0
            }

    def _get_data_signature(self) -> Dict:
        """Create a compact signature of source data to detect stale indexes."""
        players_file = self.data_dir / 'players.json'
        news_file = self.data_dir / 'news.json'
        sentiment_file = self.data_dir / 'sentiment_analysis.json'
        predictions_file = self.data_dir / 'predictions.json'

        total_articles = 0
        if isinstance(self.news_data, dict):
            for item in self.news_data.values():
                total_articles += len(item.get('articles', []))

        return {
            'files': {
                'players': self._safe_file_stats(players_file),
                'news': self._safe_file_stats(news_file),
                'sentiment': self._safe_file_stats(sentiment_file),
                'predictions': self._safe_file_stats(predictions_file),
            },
            'counts': {
                'players': len(self.players_data),
                'news_players': len(self.news_data) if isinstance(self.news_data, dict) else 0,
                'predictions': len(self.predictions_data) if isinstance(self.predictions_data, dict) else 0,
                'total_articles': total_articles,
            }
        }
    
    def load_data(self) -> bool:
        """Load player data, news, sentiment, and prediction data."""
        try:
            # Use absolute paths for reliability on different working directories
            players_file = self.data_dir / 'players.json'
            news_file = self.data_dir / 'news.json'
            sentiment_file = self.data_dir / 'sentiment_analysis.json'
            predictions_file = self.data_dir / 'predictions.json'

            self.players_data = []
            self.news_data = {}
            self.predictions_data = {}

            if players_file.exists():
                with open(players_file, 'r') as f:
                    self.players_data = json.load(f)
            else:
                print("players.json missing, fetching current players from FPL API...")
                try:
                    from data_fetcher import FantasyDataFetcher
                    fetcher = FantasyDataFetcher()
                    self.players_data = fetcher.fetch_players() or []
                    if self.players_data:
                        os.makedirs(self.data_dir, exist_ok=True)
                        with open(players_file, 'w') as f:
                            json.dump(self.players_data, f, indent=2)
                except Exception as e:
                    print(f"Could not fetch players data: {e}")

            if news_file.exists():
                with open(news_file, 'r') as f:
                    self.news_data = json.load(f)
            elif self.players_data:
                print("news.json missing, fetching recent news for top players...")
                try:
                    from data_fetcher import FantasyDataFetcher
                    fetcher = FantasyDataFetcher()
                    fetcher.players_data = self.players_data
                    fetcher.fetch_all_player_news(limit=40, days_back=7)
                    self.news_data = fetcher.news_data
                    if self.news_data:
                        os.makedirs(self.data_dir, exist_ok=True)
                        with open(news_file, 'w') as f:
                            json.dump(self.news_data, f, indent=2)
                except Exception as e:
                    print(f"Could not fetch news data: {e}")

            # Load sentiment data if available
            try:
                with open(sentiment_file, 'r') as f:
                    self.sentiment_data = json.load(f)
                    self.sentiment_lookup = {
                        str(s['player_id']): s for s in self.sentiment_data
                    }
                    self.sentiment_name_lookup = {
                        str(s.get('player_name', '')).lower(): s for s in self.sentiment_data
                    }
            except:
                self.sentiment_lookup = {}
                self.sentiment_name_lookup = {}

            # Load predictions if available
            try:
                if predictions_file.exists():
                    with open(predictions_file, 'r') as f:
                        self.predictions_data = json.load(f)
                    print(f"Loaded predictions for {len(self.predictions_data)} players")
                else:
                    self.predictions_data = {}
                    print("No predictions data found")
            except:
                self.predictions_data = {}
                print("No predictions data found")

            self.data_signature = self._get_data_signature()

            total_articles = 0
            if isinstance(self.news_data, dict):
                total_articles = sum(len(v.get('articles', [])) for v in self.news_data.values())

            has_minimum_data = (len(self.news_data) > 0 or len(self.predictions_data) > 0)
            print(
                f"Loaded {len(self.players_data)} players, "
                f"{len(self.news_data)} news entries, "
                f"{len(self.predictions_data)} predictions, "
                f"{total_articles} articles"
            )
            return has_minimum_data
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_vector_db(self):
        """Create FAISS vector database from news articles and predictions."""
        print("Creating FAISS vector database...")
        
        # Prepare documents
        self.documents = []
        self.metadatas = []
        
        # Add news articles
        print("Processing news articles...")
        for player_id, news in self.news_data.items():
            player_name = news['player_name']
            team = news['team']
            articles = news.get('articles', [])
            
            # Get sentiment and prediction data
            sentiment_score = None
            if str(player_id) in self.sentiment_lookup:
                sentiment_score = self.sentiment_lookup[str(player_id)].get('normalized_score')
            if sentiment_score is None and player_name.lower() in self.sentiment_name_lookup:
                sentiment_score = self.sentiment_name_lookup[player_name.lower()].get('normalized_score')
            
            predicted_points = None
            if str(player_id) in self.predictions_data:
                predicted_points = self.predictions_data[str(player_id)].get('predicted_total_points')
            
            for article in articles:
                # Combine title and summary
                summary = article.get('summary', '')
                doc_text = f"{article['title']}\n\n{summary}" if summary else article['title']
                
                self.documents.append(doc_text)
                self.metadatas.append({
                    'type': 'news',
                    'player_id': str(player_id),
                    'player_name': player_name,
                    'team': team,
                    'title': article['title'],
                    'published': article.get('published', ''),
                    'source': article.get('source', 'Unknown'),
                    'link': article.get('link', ''),
                    'sentiment_score': sentiment_score if sentiment_score is not None else 50.0,
                    'predicted_points': predicted_points if predicted_points else 0.0
                })
        
        # Add prediction summaries as documents
        print("Processing prediction data...")
        for player_id, pred_data in self.predictions_data.items():
            player_name = pred_data.get('player_name', 'Unknown')
            team = pred_data.get('team', 'Unknown')
            next_game_points = pred_data.get('next_game_predicted_points', 0)
            next_game_opponent = pred_data.get('next_game_opponent', 'Unknown')
            next_game_difficulty = pred_data.get('next_game_difficulty', 'Unknown')
            
            # Create summary document
            summary = f"""Player Analysis: {player_name} ({team})
Predicted Total Points: {pred_data.get('predicted_total_points', 0):.1f}
Predicted Points per Match: {pred_data.get('predicted_points_per_match', 0):.2f}
Next Gameweek Prediction: {next_game_points:.2f}
Next Opponent: {next_game_opponent}
Fixture Difficulty: {next_game_difficulty}
Form Trend: {pred_data.get('form_trend', 'stable')}
Recent Performance: {pred_data.get('recent_avg_points', 0):.1f} points/match
"""
            
            self.documents.append(summary)
            self.metadatas.append({
                'type': 'prediction',
                'player_id': str(player_id),
                'player_name': player_name,
                'team': team,
                'predicted_points': pred_data.get('predicted_total_points', 0),
                'predicted_ppm': pred_data.get('predicted_points_per_match', 0),
                'form_trend': pred_data.get('form_trend', 'stable'),
                'next_game_predicted_points': next_game_points,
                'next_game_opponent': next_game_opponent,
                'next_game_difficulty': next_game_difficulty
            })
        
        print(f"Total documents: {len(self.documents)}")
        
        # Generate embeddings
        if len(self.documents) == 0:
            raise ValueError("No news or prediction documents available to index")

        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            self.documents,
            show_progress_bar=True,
            batch_size=32
        )
        
        # Create FAISS index
        print("Creating FAISS index...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Save index and metadata
        self._save_index()
        
        print(f"✓ Vector database created with {len(self.documents)} documents")
    
    def _save_index(self):
        """Save FAISS index and metadata."""
        faiss.write_index(self.index, f"{self.persist_directory}/index.faiss")
        
        with open(f"{self.persist_directory}/metadata.pkl", 'wb') as f:
            pickle.dump({
                'metadatas': self.metadatas,
                'documents': self.documents,
                'data_signature': self.data_signature or self._get_data_signature()
            }, f)
        
        print(f"Index saved to {self.persist_directory}/")
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        try:
            self.index = faiss.read_index(f"{self.persist_directory}/index.faiss")
            
            with open(f"{self.persist_directory}/metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.metadatas = data['metadatas']
                self.documents = data['documents']
                self.data_signature = data.get('data_signature', {})
            
            print(f"Index loaded from {self.persist_directory}/")
            return True
        except Exception as e:
            print(f"Could not load index: {e}")
            return False

    def is_index_stale(self) -> bool:
        """Check whether index metadata no longer matches the current source data."""
        current_signature = self._get_data_signature()
        return self.data_signature != current_signature

    def get_index_stats(self) -> Dict:
        """Return quick index health metrics for diagnostics."""
        prediction_docs = sum(1 for m in self.metadatas if m.get('type') == 'prediction')
        news_docs = sum(1 for m in self.metadatas if m.get('type') == 'news')
        return {
            'total_docs': len(self.metadatas),
            'news_docs': news_docs,
            'prediction_docs': prediction_docs
        }
    
    def query(self, query_text: str, n_results: int = 5, player_filter: str = None, 
              doc_type_filter: str = None) -> List[Dict]:
        """
        Query the RAG system using FAISS.
        
        Args:
            query_text: Natural language query
            n_results: Number of results to return
            player_filter: Optional player name to filter results
            doc_type_filter: Filter by document type ('news' or 'prediction')
            
        Returns:
            List of relevant documents with metadata
        """
        if self.index is None:
            if not self._load_index():
                raise ValueError("No index available. Run create_vector_db() first.")

        if len(self.metadatas) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])
        
        # Search in FAISS - get more results for filtering
        search_k = min(max(n_results * 10, n_results), len(self.metadatas))
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Process results with filtering
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.metadatas):
                continue
            
            metadata = self.metadatas[idx]
            
            # Apply filters
            if player_filter and player_filter.lower() not in metadata['player_name'].lower():
                continue
            
            if doc_type_filter and metadata['type'] != doc_type_filter:
                continue
            
            result = {
                'player_name': metadata['player_name'],
                'team': metadata['team'],
                'type': metadata['type'],
                'content': self.documents[idx],
                'distance': float(distance)
            }
            
            if metadata['type'] == 'news':
                result.update({
                    'title': metadata.get('title', ''),
                    'published': metadata.get('published', ''),
                    'source': metadata.get('source', 'Unknown'),
                    'link': metadata.get('link', ''),
                    'sentiment_score': metadata.get('sentiment_score', 0)
                })
            else:
                result.update({
                    'predicted_points': metadata.get('predicted_points', 0),
                    'predicted_ppm': metadata.get('predicted_ppm', 0),
                    'form_trend': metadata.get('form_trend', 'stable'),
                    'next_game_predicted_points': metadata.get('next_game_predicted_points', 0),
                    'next_game_opponent': metadata.get('next_game_opponent', 'Unknown'),
                    'next_game_difficulty': metadata.get('next_game_difficulty', 'Unknown')
                })
            
            results.append(result)
            
            if len(results) >= n_results:
                break
        
        return results
    
    def answer_question(self, question: str, n_context: int = 5) -> str:
        """
        Answer a question using RAG with FPL rules context.
        
        Args:
            question: Natural language question
            n_context: Number of context documents to retrieve
            
        Returns:
            Context for LLM
        """
        # Load FPL rules
        try:
            from fpl_rules import FPL_RULES
        except ImportError:
            FPL_RULES = ""
        
        # Retrieve relevant context
        results = self.query(question, n_results=n_context)
        
        # Build context from results
        context_parts = [FPL_RULES, "\n## Recent Player Analysis:\n"]
        
        for i, result in enumerate(results, 1):
            if result['type'] == 'news':
                context_parts.append(
                    f"\n{i}. NEWS - {result['player_name']} ({result['team']}):\n"
                    f"   Title: {result['title']}\n"
                    f"   Sentiment: {result.get('sentiment_score', 0):.1f}/100\n"
                    f"   Content: {result['content'][:400]}...\n"
                )
            else:
                context_parts.append(
                    f"\n{i}. PREDICTION - {result['player_name']} ({result['team']}):\n"
                    f"   {result['content']}\n"
                )
        
        context = "\n".join(context_parts)
        
        return f"""
Based on FPL rules and player analysis:

{context}

Question: {question}

To get an AI-generated answer, integrate with an LLM API (OpenAI, Anthropic, etc.)
The context above includes FPL rules and relevant player information.
"""
    
    def get_player_summary(self, player_name: str) -> Dict:
        """Get a comprehensive summary for a specific player."""
        # Find player in data
        player_info = None
        for player in self.players_data:
            if player['name'].lower() == player_name.lower():
                player_info = player
                break
        
        if not player_info:
            return {"error": f"Player '{player_name}' not found"}
        
        # Get sentiment data
        sentiment_info = self.sentiment_lookup.get(str(player_info['id']), {})
        
        # Get prediction data
        prediction_info = self.predictions_data.get(str(player_info['id']), {})
        
        # Query recent articles
        recent_articles = self.query(
            f"{player_name} recent performance",
            n_results=10,
            player_filter=player_name,
            doc_type_filter='news'
        )
        
        return {
            'player': player_info,
            'sentiment': sentiment_info,
            'predictions': prediction_info,
            'recent_articles': recent_articles[:5]
        }


def main():
    """Main execution function."""
    rag = PlayerNewsRAG()
    
    # Load data
    if not rag.load_data():
        print("Please run data_fetcher.py first")
        return
    
    # Create vector database
    rag.create_vector_db()
    
    # Example queries
    print("\n" + "="*60)
    print("Example RAG Queries with FAISS")
    print("="*60)
    
    # Example 1: General query
    print("\n1. Query: 'Which players are predicted to score well?'")
    results = rag.query("high predicted points good form", n_results=5)
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. {result['player_name']} ({result['team']}) - Type: {result['type']}")
        if result['type'] == 'prediction':
            print(f"     Predicted Points: {result.get('predicted_points', 0):.1f}")
        else:
            print(f"     Sentiment: {result.get('sentiment_score', 0):.1f}/100")
    
    # Example 2: Player-specific query
    if rag.players_data:
        example_player = rag.players_data[0]['name']
        print(f"\n2. Player Summary: {example_player}")
        summary = rag.get_player_summary(example_player)
        if 'error' not in summary:
            print(f"   Position: {summary['player']['position']}")
            print(f"   Team: {summary['player']['team']}")
            if summary.get('predictions'):
                print(f"   Predicted Points: {summary['predictions'].get('predicted_total_points', 'N/A')}")
    
    print("\n✓ FAISS RAG system ready!")
    print("\nFeatures:")
    print("- Fast vector search with FAISS")
    print("- Integrated predictions and sentiment")
    print("- FPL rules aware")
    print("- No API limits or costs")


if __name__ == "__main__":
    main()
