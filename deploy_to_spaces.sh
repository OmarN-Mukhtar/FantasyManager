#!/bin/bash
# Quick deployment script for Hugging Face Spaces

set -e  # Exit on error

echo "🚀 Fantasy PL Manager - Hugging Face Spaces Deployment"
echo "======================================================"
echo ""

# Check if HF username is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide your Hugging Face username"
    echo "Usage: ./deploy_to_spaces.sh YOUR_HF_USERNAME"
    exit 1
fi

HF_USERNAME=$1
SPACE_NAME="fantasy-premier-league-manager"
SPACE_URL="https://huggingface.co/spaces/$HF_USERNAME/$SPACE_NAME"

echo "📦 Deploying to: $SPACE_URL"
echo ""

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "📥 Installing Hugging Face CLI..."
    pip3 install --user huggingface_hub
fi

# Login check
echo "🔐 Checking Hugging Face authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "Please login to Hugging Face:"
    huggingface-cli login
fi

# Create temporary directory for deployment
DEPLOY_DIR="hf_deploy_temp"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

echo "📋 Copying files..."

# Copy necessary files
cp auxilliary/app.py $DEPLOY_DIR/app.py
cp requirements.txt $DEPLOY_DIR/requirements.txt
cp -r auxilliary $DEPLOY_DIR/
cp -r sentiment $DEPLOY_DIR/
cp -r prediction $DEPLOY_DIR/
cp -r RAG $DEPLOY_DIR/

# Copy data if exists
if [ -d "data" ]; then
    echo "📊 Copying data directory..."
    cp -r data $DEPLOY_DIR/
else
    echo "⚠️  No data directory found - creating placeholder"
    mkdir -p $DEPLOY_DIR/data
fi

# Create README for Hugging Face
cat > $DEPLOY_DIR/README.md << 'EOF'
---
title: Fantasy Premier League Manager
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.29.0"
app_file: app.py
pinned: false
license: mit
python_version: "3.11"
---

# ⚽ Fantasy Premier League Manager

AI-powered Fantasy Premier League analysis with ML predictions, sentiment analysis, and FPL-aware recommendations.

## Features

🔮 **Performance Predictions** - ML models predict player points using historical data  
📊 **Sentiment Analysis** - News sentiment scoring (0-100 scale)  
🔍 **RAG Search** - Natural language queries with FPL rules knowledge  
📈 **Interactive Dashboard** - 5 comprehensive analysis tabs  
💰 **100% Free** - No API costs (FAISS, scikit-learn)

## How to Use

1. **Player Table** - Browse and filter all Premier League players
2. **Predictions** - View ML-powered performance forecasts
3. **Analytics** - Explore sentiment distributions and correlations
4. **RAG Search** - Ask questions about players in natural language
5. **About** - Learn about the system

## Technology

- **Data**: Fantasy Premier League API, Google News RSS
- **ML**: scikit-learn Random Forest with rolling window features
- **NLP**: TextBlob sentiment, Sentence-BERT embeddings
- **Vector DB**: FAISS for fast semantic search
- **Dashboard**: Streamlit with Plotly visualizations

## Example Queries

- "Which defenders under £5M are predicted to score well?"
- "Players with good sentiment and high predicted points"
- "Best captain choice this week"
- "Value midfielders in form"

---

Built with ❤️ for Fantasy PL managers
EOF

cd $DEPLOY_DIR

# Initialize git if needed
if [ ! -d ".git" ]; then
    git init
    git config user.email "deploy@fantasy-pl.com"
    git config user.name "Fantasy PL Deploy"
fi

# Add all files
git add .
git commit -m "Deploy Fantasy Premier League Manager" || true

# Push to Hugging Face
echo ""
echo "🚀 Pushing to Hugging Face Spaces..."
git remote remove space 2>/dev/null || true
git remote add space $SPACE_URL
git push --force space main:main

cd ..
rm -rf $DEPLOY_DIR

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your dashboard will be available at:"
echo "   $SPACE_URL"
echo ""
echo "⏱️  It may take 2-3 minutes to build and deploy."
echo "📊 Check the logs at $SPACE_URL/logs"
echo ""
