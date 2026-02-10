"""
Example script showing how to integrate the RAG system with Google Gemini for Q&A.
"""

import os
from dotenv import load_dotenv
from rag_system import PlayerNewsRAG

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  google-generativeai not installed. Run: pip install google-generativeai")


def ask_with_llm(question: str, rag: PlayerNewsRAG, n_context: int = 5):
    """
    Answer a question using RAG + Google Gemini LLM (FREE API).
    
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
        if result['type'] == 'news':
            context_parts.append(
                f"Article {i}:\n"
                f"Player: {result['player_name']} ({result['team']})\n"
                f"Title: {result['title']}\n"
                f"Sentiment Score: {result.get('sentiment_score', 0):.1f}/100\n"
                f"Content: {result['content'][:800]}\n"
            )
        else:  # prediction
            context_parts.append(
                f"Prediction {i}:\n"
                f"Player: {result['player_name']} ({result['team']})\n"
                f"Predicted Points: {result.get('predicted_points', 0):.1f}\n"
                f"Details: {result['content'][:400]}\n"
            )
    
    context = "\n---\n".join(context_parts)
    
    # Create prompt for Gemini
    prompt = f"""You are a Fantasy Premier League (FPL) expert assistant. Based on the following player data, 
answer the user's question with specific recommendations and reasoning.

FPL Context:
- Budget: £100M total for 15 players
- Formation rules: 2 GK, 5 DEF, 5 MID, 3 FWD
- Max 3 players from same team
- Points awarded for goals, assists, clean sheets, bonus points

Player Data:
{context}

User Question: {question}

Provide a detailed answer with specific player recommendations, considering their form, sentiment, predicted points, and price."""
    
    print("\n" + "="*60)
    print("CONTEXT RETRIEVED:")
    print("="*60)
    print(context)
    print("\n" + "="*60)
    print("QUERYING GEMINI API...")
    print("="*60)
    
    # Use Google Gemini API
    if GEMINI_AVAILABLE:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            print("\n⚠️  GOOGLE_API_KEY not found in .env file")
            print("Get your FREE API key at: https://makersuite.google.com/app/apikey")
            print("Add it to .env file: GOOGLE_API_KEY=your_key_here\n")
            return context
        
        try:
            # Configure Gemini
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Generate response
            response = model.generate_content(prompt)
            answer = response.text
            
            print("\n" + "="*60)
            print("GEMINI ANSWER:")
            print("="*60)
            print(answer)
            print("\n" + "="*60)
            
            return answer
            
        except Exception as e:
            print(f"\n❌ Error calling Gemini API: {e}")
            print("Returning context only...\n")
            return context
    else:
        print("\n⚠️  Install google-generativeai to use Gemini LLM")
        print("Run: pip install google-generativeai")
        return context


def main():
    """Example usage with Google Gemini."""
    print("="*70)
    print("FANTASY PREMIER LEAGUE - RAG + GEMINI LLM SYSTEM")
    print("="*70)
    print("\n🔧 Using Google Gemini API (FREE)")
    print("Get your key at: https://makersuite.google.com/app/apikey\n")
    
    # Initialize RAG system
    rag = PlayerNewsRAG()
    
    if not rag.load_data():
        print("\n❌ No data found. Please run the pipeline first:")
        print("   python run_pipeline.py --test")
        return
    
    # Load or create vector database
    if not rag.load_index():
        print("\nCreating vector database...")
        rag.create_vector_db()
    
    # Example questions
    questions = [
        "Who should I captain this week based on form and fixtures?",
        "Which midfielders are in the best form and good value?",
        "Are there any injury concerns I should know about?",
        "Which budget strikers under £7M are good value?",
        "Who are the best differential picks right now?",
    ]
    
    print("\n" + "="*70)
    print("EXAMPLE QUESTIONS:")
    print("="*70)
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")
    
    print("\n" + "="*70)
    print("TESTING WITH FIRST QUESTION:")
    print("="*70)
    
    # Ask first question as example
    ask_with_llm(questions[0], rag, n_context=5)
    
    print("\n" + "="*70)
    print("✅ LLM Integration Ready!")
    print("="*70)
    print("\nTo use:")
    print("1. Get FREE Gemini API key: https://makersuite.google.com/app/apikey")
    print("2. Add to .env: GOOGLE_API_KEY=your_key_here")
    print("3. Run: python src/llm_integration.py")
    print("\nOr integrate into your dashboard using the ask_with_llm() function!")


if __name__ == "__main__":
    main()
