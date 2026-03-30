import os
import json
from pathlib import Path
from dotenv import load_dotenv
# Chat Imports
from langchain_google_genai import ChatGoogleGenerativeAI
# Embeddings Imports
from langchain_huggingface import HuggingFaceEmbeddings
# Vector Store Imports
import faiss
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
# RAG Imports
from langchain.agents import create_agent
from langchain.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

#1) Chat Model
google_api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite', google_api_key=google_api_key)

#2) Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#3) Vector Store Model
embeddings_dim = 384
index = faiss.IndexFlatL2(embeddings_dim)

vectorstore = FAISS(embedding_function=embeddings, index=index, docstore=InMemoryDocstore(), index_to_docstore_id={})
vector_db_path = PROJECT_ROOT / "vector_db" / "langchain_faiss"

if __name__ == "__main__":
    #4) Indexing the data
    # Load, Split, Store

    news_data_path = PROJECT_ROOT / "data" / "news.json"
    predictions_data_path = PROJECT_ROOT / "data" / "predictions.json"

    news_data = json.loads(news_data_path.read_text(encoding="utf-8"))
    predictions_data = json.loads(predictions_data_path.read_text(encoding="utf-8"))

    docs = []

    for player_name, payload in news_data.items():
        if not isinstance(payload, dict):
            continue

        search_name = payload.get("search_name", player_name)
        headlines = payload.get("headlines", [])

        for i, headline in enumerate(headlines):

            if not isinstance(headline, str) or not headline.strip():
                continue

            docs.append(
                Document(
                    page_content=f"Player: {player_name}\nHeadline: {headline.strip()}",
                    metadata={
                        "type": "news",
                        "player_name": player_name,
                        "search_name": search_name,
                        "item_index": i,
                    },
                )
            )

    for player_name, payload in predictions_data.items():
        if not isinstance(payload, dict):
            continue

        current_points = payload.get("current_season_points", 0)
        position = payload.get("position", "Unknown")

        docs.append(
            Document(
                page_content=f"Player: {player_name}\nCurrent Season Points: {current_points}\nPosition: {position}",
                metadata={
                    "type": "prediction",
                    "player_name": player_name,
                    "current_season_points": current_points,
                    "position": position,
                },
            )
        )

    vectorstore.add_documents(docs)
    vector_db_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vector_db_path))
else:
    vectorstore = FAISS.load_local(
        str(vector_db_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

# 5) Retrieval and Generation
@tool(response_format='content_and_artifact')
def retrieve_context(query: str, limit: int = 10) -> str:
    "Search the vector database for relevant documents."
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    serialized = "\n\n".join([f"{doc.page_content}\nMetadata: {doc.metadata}" for doc in retrieved_docs])
    return serialized, retrieved_docs

tools = [retrieve_context]
prompt = """You are a helpful assistant for Fantasy Premier League (FPL) managers. Use the following retrieved information to answer the user's question. If the retrieved information is not relevant, use your own knowledge to answer. These are the rules for FPL: # Fantasy Premier League Team Selection Rules

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

## Strategy Considerations
- Look for players with good upcoming fixtures (easy opponents)
- Consider form over the last 5-10 gameweeks
- Balance premium players (expensive) with budget options (value picks)
- Defenders from teams with good defensive records get clean sheet points
- Attacking players (goals/assists) generally outscore others
- Midfielders who play as forwards offer best value (MID price, FWD returns)
- Rotation risk: avoid players who don't play regularly
- Injury prone players are risky despite talent
- Monitor pre-match press conferences for team news
"""

agent = create_agent(model=model, tools=tools, prompt=prompt)

