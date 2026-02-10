"""
Hugging Face Spaces entry point for Fantasy Premier League Manager.
This file is required for Hugging Face Spaces deployment.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import and run the Streamlit dashboard
from src.dashboard import main

if __name__ == "__main__":
    main()
