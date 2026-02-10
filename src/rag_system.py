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


class PlayerNewsRAG:
    """RAG system for querying player news using FAISS."""
    
    def __init__(self, persist_directory: str = "vector_db"):
        """Initialize the RAG system with FAISS."""
        self.persist_directory = persist_directory
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
        self.predictions_data = {}
    
    def load_data(self) -> bool:
        """Load player data, news, sentiment, and prediction data."""
        try:
            with open('data/players.json', 'r') as f:
                self.players_data = json.load(f)
            
            with open('data/news.json', 'r') as f:
                self.news_data = json.load(f)
            
            # Load sentiment data if available
            try:
                with open('data/sentiment_analysis.json', 'r') as f:
                    self.sentiment_data = json.load(f)
                    self.sentiment_lookup = {
                        str(s['player_id']): s for s in self.sentiment_data
                    }
            except:
                self.sentiment_lookup = {}
            
            # Load predictions if available
            try:
                with open('data/predictions.json', 'r') as f:
                    self.predictions_data = json.load(f)
                print(f"Loaded predictions for {len(self.predictions_data)} players")
            except:
                self.predictions_data = {}
                print("No predictions data found")
            
            print(f"Loaded {len(self.players_data)} players, {len(self.news_data)} player news entries")
            return True
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
            
            predicted_points = None
            if str(player_id) in self.predictions_data:
                predicted_points = self.predictions_data[str(player_id)].get('predicted_total_points')
            
            for article in articles:
                # Combine title and content
                doc_text = f"{article['title']}\n\n{article['content']}"
                
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
                    'sentiment_score': sentiment_score if sentiment_score else 0.0,
                    'predicted_points': predicted_points if predicted_points else 0.0
                })
        
        # Add prediction summaries as documents
        print("Processing prediction data...")
        for player_id, pred_data in self.predictions_data.items():
            player_name = pred_data.get('player_name', 'Unknown')
            team = pred_data.get('team', 'Unknown')
            
            # Create summary document
            summary = f"""Player Analysis: {player_name} ({team})
Predicted Total Points: {pred_data.get('predicted_total_points', 0):.1f}
Predicted Points per Match: {pred_data.get('predicted_points_per_match', 0):.2f}
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
                'form_trend': pred_data.get('form_trend', 'stable')
            })
        
        print(f"Total documents: {len(self.documents)}")
        
        # Generate embeddings
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
                'documents': self.documents
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
            
            print(f"Index loaded from {self.persist_directory}/")
            return True
        except Exception as e:
            print(f"Could not load index: {e}")
            return False
    
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
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text])
        
        # Search in FAISS - get more results for filtering
        search_k = n_results * 10
        distances, indices = self.index.search(query_embedding.astype('float32'), search_k)
        
        # Process results with filtering
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(self.metadatas):
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
                    'form_trend': metadata.get('form_trend', 'stable')
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
