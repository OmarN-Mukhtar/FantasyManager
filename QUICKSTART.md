# Fantasy Premier League Manager - Quick Start Guide

## Overview

This project analyzes Premier League Fantasy Football players using:
- Real-time player data from the official FPL API
- News aggregation from Google News
- Sentiment analysis (NLP)
- Interactive dashboard
- RAG (Retrieval Augmented Generation) system for Q&A

## Installation

1. **Clone or navigate to the project:**
   ```bash
   cd /Users/omar/Desktop/Career/Projects/FantasyManager
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run Complete Pipeline (Recommended)

Run everything in one command:

```bash
# Test mode (10 players only - quick test)
python run_pipeline.py --test

# Full mode (all players - takes 1-2 hours)
python run_pipeline.py
```

### Option 2: Run Steps Individually

1. **Fetch player data and news:**
   ```bash
   python src/data_fetcher.py
   ```

2. **Analyze sentiment:**
   ```bash
   python src/sentiment_analyzer.py
   ```

3. **Create RAG database:**
   ```bash
   python src/rag_system.py
   ```

### Launch the Dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Dashboard Features

### 1. Player Table Tab
- View all players with sentiment scores
- Sort by: sentiment, points, price, form, etc.
- Filter by team, position, price range
- See article counts and sentiment ratings

### 2. Analytics Tab
- Sentiment distribution by team
- Sentiment vs points correlation
- Top/bottom performers by sentiment
- Position-based analysis
- Value analysis (price vs sentiment)

### 3. RAG Search Tab
- Natural language search through player news
- Filter by specific players
- View relevant articles with sentiment scores
- Example queries:
  - "Which players are in good form?"
  - "Who has injury concerns?"
  - "Best value midfielders?"

### 4. About Tab
- Project documentation
- Feature descriptions
- Sentiment score explanation

## Project Structure

```
FantasyManager/
├── src/
│   ├── data_fetcher.py        # Fetches player data and news
│   ├── sentiment_analyzer.py  # Analyzes sentiment
│   ├── rag_system.py          # RAG implementation
│   ├── dashboard.py           # Streamlit dashboard
│   └── llm_integration.py     # Optional LLM integration
├── data/                      # Generated data files
│   ├── players.json
│   ├── news.json
│   ├── sentiment_analysis.json
│   └── players_with_sentiment.csv
├── chroma_db/                 # Vector database
├── run_pipeline.py            # Main pipeline script
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

## Data Files Generated

After running the pipeline, you'll find:

- `data/players.json` - All FPL player data
- `data/players.csv` - Players in CSV format
- `data/news.json` - News articles for each player
- `data/sentiment_analysis.json` - Detailed sentiment analysis
- `data/players_with_sentiment.csv` - Complete dataset for dashboard
- `data/qualified_players.csv` - Players with 50+ articles
- `chroma_db/` - Vector database for RAG system

## Understanding Sentiment Scores

Sentiment scores range from 0-100:

- **0-40**: Negative (injuries, poor form, criticism)
- **40-60**: Neutral (routine news, mixed coverage)
- **60-100**: Positive (goals, assists, praise)

Only players with 50+ news articles receive sentiment scores.

## Advanced: LLM Integration

To integrate with an LLM (OpenAI, Anthropic, etc.):

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Add your API key to `.env`:
   ```
   OPENAI_API_KEY=your_key_here
   ```

3. Modify `src/llm_integration.py` to uncomment LLM code

4. Run:
   ```bash
   python src/llm_integration.py
   ```

## Tips

1. **First Run**: Use `--test` flag to process only 10 players and verify everything works
2. **Full Run**: Takes 1-2 hours to fetch news for all ~600 players
3. **Rate Limiting**: The scraper includes delays to be respectful to servers
4. **Storage**: Full dataset requires ~100-200 MB of disk space
5. **Updates**: Re-run the pipeline weekly to get fresh news and sentiment

## Troubleshooting

### "Please run data_fetcher.py first"
- You need to fetch data before running other components
- Run: `python run_pipeline.py --test`

### Dashboard shows no sentiment scores
- Not all players have 50+ articles
- Lower the minimum articles filter in the sidebar
- Or adjust `min_articles` in the pipeline

### RAG system not available
- Ensure you've run `src/rag_system.py` to create the vector database
- Or run the complete pipeline

### News fetching fails
- Check internet connection
- Some articles may fail to download (this is normal)
- The system will continue with available data

## Command Line Options

```bash
# Run pipeline with custom settings
python run_pipeline.py --limit 50 --min-articles 30

# Options:
#   --test              Process only 10 players
#   --limit N           Process only N players
#   --min-articles N    Require N articles for sentiment (default: 50)
```

## Next Steps

After setup:

1. ✅ Run pipeline: `python run_pipeline.py --test`
2. ✅ Launch dashboard: `streamlit run src/dashboard.py`
3. ✅ Explore player sentiment scores
4. ✅ Try RAG search queries
5. ✅ Integrate with your favorite LLM

Good luck with your Fantasy Premier League team! ⚽
