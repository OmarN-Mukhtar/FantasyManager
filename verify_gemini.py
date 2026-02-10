#!/usr/bin/env python3
"""
Quick verification script to test Google Gemini API setup.
"""

import os
from dotenv import load_dotenv

def verify_gemini():
    """Verify Gemini API is properly configured."""
    print("="*70)
    print("GOOGLE GEMINI API - VERIFICATION")
    print("="*70)
    
    # Check if package is installed
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package installed")
    except ImportError:
        print("❌ google-generativeai not installed")
        print("\nInstall with: pip install google-generativeai")
        return False
    
    # Check for API key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        print("\n📝 Setup instructions:")
        print("1. Get FREE API key: https://makersuite.google.com/app/apikey")
        print("2. Create .env file with: GOOGLE_API_KEY=your_key_here")
        return False
    
    print(f"✅ GOOGLE_API_KEY found (length: {len(api_key)})")
    
    # Test API call
    print("\n🔄 Testing API connection...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(
            "Say 'Fantasy Premier League ready!' if you can help with FPL advice."
        )
        
        print(f"✅ API call successful!")
        print(f"\n📝 Response from Gemini:")
        print(f"   {response.text}")
        
        print("\n" + "="*70)
        print("✅ GEMINI SETUP VERIFIED - READY TO USE!")
        print("="*70)
        print("\n🚀 Next steps:")
        print("1. Run: python src/llm_integration.py")
        print("2. Or use in dashboard RAG search tab")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        print("\n🔍 Troubleshooting:")
        print("1. Check API key is valid")
        print("2. Ensure you have internet connection")
        print("3. Verify API key has proper permissions")
        return False


if __name__ == "__main__":
    verify_gemini()
