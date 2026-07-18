import os
import json
from difflib import get_close_matches
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

#1) Chat Model
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_api_key)

#2) Data — loaded once at startup; the daily redeploy brings fresh files
_predictions = pd.read_csv(PROJECT_ROOT / "data" / "predictions.csv")
_sentiment = pd.read_csv(PROJECT_ROOT / "data" / "sentiment_analysis.csv")
_players = _predictions.merge(_sentiment, on="player_name", how="left")
_news = json.loads((PROJECT_ROOT / "data" / "news.json").read_text(encoding="utf-8"))

#3) Tools

@tool
def player_info(name: str) -> str:
    """Look up one player's price, predictions, sentiment, and recent headlines.
    Works with partial or misspelled names (e.g. 'Salah', 'Bruno Fernandes')."""
    names = _players['player_name'].tolist()
    match = [n for n in names if name.lower() in n.lower()]
    if not match:
        match = get_close_matches(name, names, n=1, cutoff=0.4)
    if not match:
        return f"No player found matching '{name}'."

    row = _players[_players['player_name'] == match[0]].iloc[0]
    lines = [f"{k}: {v}" for k, v in row.to_dict().items()]
    headlines = _news.get(match[0], {}).get('headlines', [])[:10]
    if headlines:
        lines.append("Recent headlines:")
        lines.extend(f"- {h}" for h in headlines)
    return "\n".join(lines)


@tool
def top_players(position: str = "", max_price: str = "", limit: str = "10") -> str:
    """Top players ranked by predicted points over the next 5 gameweeks.
    Optional filters: position (GK/DEF/MID/FWD) and max_price in £M."""
    # ponytail: string params — Llama emits "10" not 10 and Groq rejects schema mismatches
    df = _players
    if position:
        df = df[df['position'].str.upper() == position.upper()]
    try:
        if max_price and float(max_price) > 0:
            df = df[df['now_cost'] <= float(max_price)]
    except ValueError:
        pass
    try:
        n = max(1, min(int(float(limit)), 25))
    except (ValueError, TypeError):
        n = 10
    cols = ['player_name', 'position', 'team', 'now_cost',
            'predicted_next_5_weighted', 'predicted_next_gw_points',
            'sentiment_score', 'next_5_fixtures']
    return df.sort_values('predicted_next_5_weighted', ascending=False).head(n)[cols].to_string(index=False)


@tool
def search_news(keywords: list[str]) -> str:
    """Search every player's recent headlines for a topic. Matching is plain
    substring, so pass several variants and synonyms, e.g.
    ["injury", "injured", "knock", "doubt", "sidelined"]."""
    # ponytail: LLM supplies the synonyms, tool just greps — replaces the embedding stack
    kws = [k.lower() for k in keywords]
    hits = []
    for player, payload in _news.items():
        for h in payload.get('headlines', []):
            if any(k in h.lower() for k in kws):
                hits.append(f"{player}: {h}")
    return "\n".join(hits[:30]) if hits else "No headlines matched."


tools = [player_info, top_players, search_news]
prompt = """You are a helpful assistant for Fantasy Premier League (FPL) managers. Use the tools to look up players, rankings, and news before answering. Do not ask too many follow up questions.
Always give 3-5 concrete player suggestions first, even when the user is vague.
Ask at most one short follow-up question after giving suggestions.
These are the rules for FPL: # Fantasy Premier League Team Selection Rules

## Budget and Squad Size
- Total budget: £100.0 million
- Squad size: 15 players total
  * 2 Goalkeepers (GK)
  * 5 Defenders (DEF)
  * 5 Midfielders (MID)
  * 3 Forwards (FWD)

## Team Limits
- Maximum 3 players from any single Premier League team
- Cannot have more than 3 players from the same club

## Starting XI Formation
- Must select 11 players from your 15-player squad
- Must include:
  * Exactly 1 Goalkeeper
  * At least 3 Defenders
  * At least 2 Midfielders
  * At least 1 Forward
- Valid formations:
  * 3-4-3 (3 DEF, 4 MID, 3 FWD)
  * 3-5-2 (3 DEF, 5 MID, 2 FWD)
  * 4-3-3 (4 DEF, 3 MID, 3 FWD)
  * 4-4-2 (4 DEF, 4 MID, 2 FWD)
  * 4-5-1 (4 DEF, 5 MID, 1 FWD)
  * 5-3-2 (5 DEF, 3 MID, 2 FWD)
  * 5-4-1 (5 DEF, 4 MID, 1 FWD)

## Captain and Vice-Captain
- Select 1 Captain: receives double points
- Select 1 Vice-Captain: receives double points if Captain doesn't play

## Transfers
- 1 free transfer per gameweek
- Additional transfers cost 4 points each
- Unused free transfers roll over (max 2 free transfers)
- Wildcard: make unlimited free transfers (2 per season)

## Chips (one-time use each season)
- Bench Boost: Get points from all bench players for one gameweek
- Triple Captain: Captain gets triple points instead of double
- Free Hit: Make unlimited transfers for one gameweek, team reverts after
- Wildcard: Unlimited free transfers (available twice per season)

## Points Scoring System

### All Players
- Playing 0-60 minutes: 1 point
- Playing 60+ minutes: 2 points
- Yellow card: -1 point
- Red card: -3 points

### Goalkeepers & Defenders
- Clean sheet (no goals conceded while playing 60+ mins): 4 points (GK/DEF)
- Goal scored: 6 points (GK/DEF)
- Assist: 3 points
- Save (GK only): 1 point per 3 saves
- Penalty save (GK only): 5 points
- Penalty miss: -2 points
- Own goal: -2 points
- Conceding 2+ goals: -1 point per goal

### Midfielders
- Clean sheet: 1 point
- Goal scored: 5 points
- Assist: 3 points
- Penalty miss: -2 points
- Own goal: -2 points

### Forwards
- Goal scored: 4 points
- Assist: 3 points
- Penalty miss: -2 points
- Own goal: -2 points

### Bonus Points (BPS)
- Top 3 players in each match get bonus points
- 1st place: 3 bonus points
- 2nd place: 2 bonus points
- 3rd place: 1 bonus point
"""

agent = create_agent(model=model, tools=tools, system_prompt=prompt)
