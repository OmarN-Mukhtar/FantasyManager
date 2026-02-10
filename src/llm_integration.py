"""
Example script showing how to integrate the RAG system with OpenAI for Q&A.
"""

import os
from dotenv import load_dotenv
from rag_system import PlayerNewsRAG

# Uncomment to use with OpenAI
# from openai import OpenAI


def ask_with_llm(question: str, rag: PlayerNewsRAG, n_context: int = 5):
    """
    Answer a question using RAG + LLM.
    
    Args:
        question: User's question
        rag: RAG system instance
        n_context: Number of context articles to retrieve
    """
    # Retrieve relevant context
    results = rag.query(question, n_results=n_context)
    
    # Build context
    context_parts = []
    for i, result in enumerate(results, 1):
        context_parts.append(
            f"Article {i}:\n"
            f"Player: {result['player_name']} ({result['team']})\n"
            f"Title: {result['title']}\n"
            f"Sentiment Score: {result['sentiment_score']:.1f}/100\n"
            f"Content: {result['content'][:800]}\n"
        )
    
    context = "\n---\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are a Fantasy Premier League expert. Based on the following news articles, 
answer the user's question with specific recommendations and reasoning.

News Articles:
{context}

User Question: {question}

Please provide a detailed answer with specific player recommendations and analysis."""
    
    print("\n" + "="*60)
    print("CONTEXT RETRIEVED:")
    print("="*60)
    print(context)
    print("\n" + "="*60)
    print("PROMPT TO LLM:")
    print("="*60)
    print(prompt)
    
    # Uncomment to use with OpenAI
    """
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a Fantasy Premier League expert assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content
    
    print("\n" + "="*60)
    print("LLM ANSWER:")
    print("="*60)
    print(answer)
    
    return answer
    """
    
    print("\n" + "="*60)
    print("NOTE:")
    print("="*60)
    print("Uncomment the OpenAI code and add your API key to .env to get LLM-generated answers")
    return context


def main():
    """Example usage."""
    # Initialize RAG system
    rag = PlayerNewsRAG()
    
    if not rag.load_data():
        print("Please run data_fetcher.py first")
        return
    
    # Example questions
    questions = [
        "Who should I captain this week?",
        "Which midfielders are in the best form?",
        "Are there any injury concerns I should know about?",
        "Which budget strikers are good value?",
    ]
    
    print("Fantasy Premier League RAG + LLM System")
    print("="*60)
    
    # Ask first question as example
    ask_with_llm(questions[0], rag)
    
    print("\n" + "="*60)
    print("Other example questions you can ask:")
    for q in questions[1:]:
        print(f"  - {q}")


if __name__ == "__main__":
    main()
