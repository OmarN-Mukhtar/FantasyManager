---
title: Fantasy Premier League Manager
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.39.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Fantasy Premier League Manager

A Fantasy Premier League assistant: daily-updated player data, XGBoost predictions for each player's next 5 fixtures (weighted by closeness and opponent difficulty), news sentiment weighted by recency, a player browser, and a chatbot that answers FPL questions using that data.

## How it works

The pipeline runs daily via GitHub Actions ([.github/workflows/daily_update.yml](.github/workflows/daily_update.yml)) and commits refreshed data:

1. **Player data** — `auxilliary/data_fetcher.py` fetches all current players from the FPL API → `data/players.json` / `data/players.csv`.
2. **Match history** — `prediction/update_current_season.py` fetches gameweek-by-gameweek stats and upserts them into `data/cleaned_merged_seasons.csv` (historical seasons + current season).
3. **News sentiment** — `sentiment/sentiment_analyzer.py` pulls each player's recent Google News RSS headlines (last 7 days) and scores them with DistilBERT (SST-2). Each headline's score is weighted by recency (3-day half-life), producing a -1..1 score per player → `data/news.json`, `data/sentiment_analysis.csv`.
4. **Predictions** — `prediction/predictor.py` trains an XGBoost regressor on rolling-window features (1/3/5/7 GW averages of points, minutes, xG, etc.) and predicts each of the player's **next 5 fixtures** using the real opponent and venue. The next-GW prediction is blended with FPL's own `ep_next` and the sentiment score. The headline number, `predicted_next_5_weighted`, discounts each fixture by distance (×0.8 per GW) and opponent difficulty (FPL FDR: easier → up-weighted, harder → down-weighted) → `data/predictions.json` / `data/predictions.csv`. The model choice is justified in `notebooks/model_exploration.ipynb`, which compares it against a naive baseline, Ridge, RandomForest, and LightGBM on a time-based split.
5. **Chatbot + player browser** — `RAG/langchain_rag.py` gives a LangChain agent (Llama 3.3 70B via Groq's free tier) three tools over the data files: fuzzy player lookup, filtered top-player rankings, and topical news search (the model supplies keyword synonyms, the tool greps all headlines — no embedding stack needed). FPL rules live in the system prompt. `auxilliary/app.py` is the Streamlit UI: a Chat tab, plus a Players tab with a sortable table (price, points, sentiment, next-5 predictions) filterable by name, position, team, and price.

Pushes to `main` auto-deploy the app to Hugging Face Spaces ([.github/workflows/deploy_spaces.yml](.github/workflows/deploy_spaces.yml)).

## Setup

```bash
pip install -r requirements.txt            # app only
pip install -r requirements-pipeline.txt   # + data pipeline (torch, transformers, xgboost)
echo "GROQ_API_KEY=your_key_here" > .env   # free key from https://console.groq.com/keys
```

## Usage

```bash
# Refresh data (each step is optional if its outputs already exist)
python auxilliary/data_fetcher.py
python prediction/update_current_season.py
python sentiment/sentiment_analyzer.py
python prediction/predictor.py

# Launch the chatbot
streamlit run auxilliary/app.py
```

## Data files

| File | Contents |
|------|----------|
| `data/players.json` / `.csv` | Current FPL players (price, form, points) |
| `data/cleaned_merged_seasons.csv` | Historical + current season gameweek stats |
| `data/news.json` | Recent headlines per player |
| `data/sentiment_analysis.csv` / `.json` | Recency-weighted sentiment score per player (-1..1) |
| `data/predictions.json` / `.csv` | Per-fixture predictions for the next 5 GWs + weighted total per player |

## Stack (all free)

- **Data**: FPL API, Google News RSS
- **ML**: XGBoost; DistilBERT sentiment (both run locally, pipeline-only)
- **LLM**: Llama 3.3 70B on Groq free tier
- **App**: Streamlit on Hugging Face Spaces
- **Automation**: GitHub Actions

## Disclaimer

For informational purposes only — do your own research before making transfers!
