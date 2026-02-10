# Fantasy Premier League Manager

An intelligent fantasy football assistant that analyzes player performance using news sentiment analysis, ML-powered predictions, and FPL-aware RAG recommendations.

## Features

1. **Player Data Fetching**: Automatically retrieves current Premier League player data
2. **News Aggregation**: Collects recent news articles for each player via RSS
3. **Sentiment Analysis**: Generates normalized sentiment scores (0-100) for players with sufficient news coverage
4. **Performance Prediction**: ML models predict player points using rolling window analysis of historical data
5. **Interactive Dashboard**: Multi-tab visualization with predictions, sentiment, and analytics
6. **FPL-Aware RAG System**: Query player news and predictions using natural language with FPL rules knowledge
7. **100% Free**: No API costs - uses FAISS, scikit-learn, and free data sources

## Installation

```bash
# Create conda environment
conda create -n fantasy_manager python=3.10 -y
conda activate fantasy_manager

# Install dependencies
pip install -r requirements.txt

# Download NLP data
python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
python -m textblob.download_corpora
```

## Quick Start

```bash
# Run complete pipeline (test mode - 10 players)
python run_pipeline.py --test

# Run full pipeline (all ~600 players - takes 1-2 hours)
python run_pipeline.py

# Skip news fetching if you already have data
python run_pipeline.py --skip-news

# Launch interactive dashboard
streamlit run src/dashboard.py
```

## Pipeline Stages

### 1. Data Fetching (`data_fetcher.py`)
- Fetches ~600 current Premier League players from FPL API
- Collects news articles via Google News RSS
- Saves to `data/players.json` and `data/news.json`

### 2. Sentiment Analysis (`sentiment_analyzer.py`)
- Analyzes news sentiment using TextBlob NLP
- Normalizes scores to 0-100 scale
- Requires 50+ articles for scoring
- Saves to `data/sentiment_analysis.json`

### 3. Performance Prediction (`predictor.py`)
- Loads historical player data from `cleaned_merged_seasons.csv`
- Creates rolling window features (3, 5, 10 game averages)
- Trains Random Forest models per player
- Predicts total season points and points per match
- Saves to `data/predictions.json`

### 4. RAG System (`rag_system.py`)
- Creates FAISS vector index from news and predictions
- Embeds FPL rules knowledge
- Enables natural language queries
- Saves to `vector_db/`

### 5. Dashboard (`dashboard.py`)
Five interactive tabs:
- **Player Table**: Sortable/filterable player data
- **Predictions**: ML predictions with visualizations
- **Analytics**: Sentiment analysis charts
- **RAG Search**: Natural language Q&A
- **About**: Documentation

## Project Structure

```
FantasyManager/
├── src/
│   ├── data_fetcher.py        # Fetch players & news
│   ├── sentiment_analyzer.py  # Sentiment analysis
│   ├── predictor.py           # ML predictions
│   ├── rag_system.py          # FAISS RAG system
│   ├── dashboard.py           # Streamlit UI
│   ├── fpl_rules.py           # FPL rules & validation
│   └── llm_integration.py     # LLM integration template
├── data/                      # Generated data
├── vector_db/                 # FAISS index
├── cleaned_merged_seasons.csv # Historical data
├── run_pipeline.py            # Main pipeline
└── requirements.txt           # Dependencies
```

## Technology Stack

- **Data**: Fantasy Premier League API, Google News RSS
- **NLP**: TextBlob for sentiment, sentence-transformers for embeddings
- **ML**: scikit-learn Random Forest (no cloud APIs needed)
- **Vector DB**: FAISS (free, local, fast)
- **Dashboard**: Streamlit with Plotly charts
- **All Free**: No API costs or rate limits

## FPL Rules Integration

The RAG system understands:
- Squad requirements (2 GK, 5 DEF, 5 MID, 3 FWD)
- £100M budget constraint
- Max 3 players per team
- Formation rules for starting XI
- Points scoring system by position
- Captain/Vice-Captain mechanics

## Usage Examples

```bash
# Generate only predictions (if you have existing data)
python src/predictor.py

# Rebuild RAG index only
python src/rag_system.py

# Run pipeline without news fetch
python run_pipeline.py --skip-news
```

## Dashboard Features

### Predictions Tab
- Top predicted scorers
- Points distribution by position
- Value analysis (price vs predicted performance)
- Form trend indicators

### RAG Search Tab
- Filter by player or document type (news/predictions)
- Natural language queries
- FPL-aware suggestions
- Example: "Which defenders under £5M are predicted to score well?"

## Data Files

After running the pipeline:
- `data/players.json` - Current FPL players
- `data/news.json` - News articles per player
- `data/sentiment_analysis.json` - Sentiment scores
- `data/predictions.json` - ML predictions
- `data/players_with_sentiment.csv` - Complete dataset
- `vector_db/index.faiss` - FAISS vector index

## Command Line Options

```bash
python run_pipeline.py --help

Options:
  --test                Process only 10 players
  --limit N            Process only N players
  --min-articles N     Minimum articles for sentiment (default: 50)
  --skip-news          Skip news fetching, use existing data
```

## Requirements

- Python 3.10+
- ~2GB RAM for FAISS index
- ~500MB disk space for data
- Historical data file: `cleaned_merged_seasons.csv`

## Contributing

This is an educational project demonstrating:
- ML for sports analytics
- Sentiment analysis at scale
- RAG system implementation
- Interactive data dashboards
- Cost-free ML/NLP pipeline

## Disclaimer

This tool is for informational and educational purposes only. Always do your own research before making fantasy football decisions!

## License

MIT License - Feel free to use and modify
