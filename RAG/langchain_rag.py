import os
import json
from pathlib import Path
from dotenv import load_dotenv
# Chat Imports
from langchain_google_genai import ChatGoogleGenerativeAI
# Embeddings Imports
from langchain_huggingface import HuggingFaceEmbeddings
# Vector Store Imports
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_pinecone import PineconeVectorStore
# RAG Imports
from langchain.agents import create_agent
from langchain.tools import tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

#1) Chat Model
google_api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model='gemma-3-1B', google_api_key=google_api_key)

#2) Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#3) Vector Store Model
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("fantasy-manager-vectors")

_vectorstore_cache = None

def build_vectorstore():
    """Build vectorstore from news.json and predictions.json and upload to Pinecone"""
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

    vs = PineconeVectorStore(embedding=embeddings, index=index)
    vs.add_documents(docs)
    return vs

def get_vectorstore():
    """Lazily initialize and cache the vectorstore to avoid duplicate uploads"""
    global _vectorstore_cache
    if _vectorstore_cache is None:
        _vectorstore_cache = PineconeVectorStore(embedding=embeddings, index=index)
    return _vectorstore_cache

if __name__ == "__main__":
    # Initialize vectorstore and upload documents to Pinecone
    print("Building vectorstore and uploading to Pinecone...")
    build_vectorstore()
    print("Vectorstore successfully uploaded to Pinecone!")

# 5) Retrieval and Generation
@tool(response_format='content_and_artifact')
def retrieve_context(query: str, limit: int = 10) -> str:
    "Search the vector database for relevant documents."
    vectorstore = get_vectorstore()
    retrieved_docs = vectorstore.similarity_search(query, k=limit)
    serialized = "\n\n".join([f"{doc.page_content}\nMetadata: {doc.metadata}" for doc in retrieved_docs])
    return serialized, retrieved_docs

tools = [retrieve_context]
prompt = """You are a helpful assistant for Fantasy Premier League (FPL) managers. Use the information you can get and always make suggestions of players. Do not ask too many follow up questions.
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

