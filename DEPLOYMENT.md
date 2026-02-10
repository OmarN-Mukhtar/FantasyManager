# 🚀 Deployment Guide - All Options

Complete guide for deploying Fantasy Premier League Manager to various platforms.

## 📋 Quick Comparison

| Platform | Cost | Setup Time | Auto-Updates | Best For |
|----------|------|------------|--------------|----------|
| **Hugging Face Spaces** | Free | 5 min | ✅ Yes | Public dashboards |
| **Streamlit Cloud** | Free | 3 min | ✅ Yes | Streamlit apps |
| **Railway.app** | $5 credit | 10 min | ✅ Yes | Full control |
| **Render.com** | Free tier | 10 min | ✅ Yes | Always-on apps |

---

## 🎯 Option 1: Hugging Face Spaces (Recommended)

**Best for**: Public dashboards, ML projects, free hosting

### Quick Deploy:

```bash
# 1. Run deployment script
./deploy_to_spaces.sh YOUR_HF_USERNAME

# 2. Configure GitHub Actions (one-time setup)
# Add these secrets in GitHub repo settings:
# - HF_TOKEN: Your HF token from https://huggingface.co/settings/tokens
# - HF_SPACE_REPO: YOUR_USERNAME/fantasy-premier-league-manager
```

**Live URL**: `https://huggingface.co/spaces/YOUR_USERNAME/fantasy-premier-league-manager`

**Features**:
- ✅ Free forever
- ✅ Auto-updates via GitHub Actions
- ✅ Public URL to share
- ✅ CPU/GPU options
- ✅ Built-in analytics

📖 **Full Guide**: See [README_SPACES.md](README_SPACES.md)

---

## 🌊 Option 2: Streamlit Community Cloud

**Best for**: Streamlit-specific apps, easy deployment

### Setup:

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Connect your GitHub repository
5. Set:
   - Main file path: `src/dashboard.py`
   - Python version: `3.10`
6. Click **Deploy**

**Live URL**: `https://YOUR_APP_NAME.streamlit.app`

**Configuration**: Create `streamlit_config.toml`:

```toml
[server]
maxUploadSize = 200
enableCORS = false

[theme]
primaryColor = "#00ff85"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
font = "sans serif"
```

**Automatic Updates**: Enable in Streamlit Cloud settings

---

## 🚂 Option 3: Railway.app

**Best for**: Full control, database integration

### Setup:

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Deploy
railway up
```

**Configuration**: Create `railway.json`:

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "streamlit run src/dashboard.py --server.port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

**Automatic Updates**: Connect GitHub repository for auto-deploy

**Cost**: $5 free credit, then ~$5/month

---

## 🎨 Option 4: Render.com

**Best for**: Always-free tier available

### Setup:

1. Go to [render.com](https://render.com)
2. Click **New Web Service**
3. Connect GitHub repository
4. Configure:
   - **Name**: fantasy-premier-league
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt')" && python -m textblob.download_corpora`
   - **Start Command**: `streamlit run src/dashboard.py --server.port $PORT --server.address 0.0.0.0`
   - **Plan**: Free

**Configuration**: Create `render.yaml`:

```yaml
services:
  - type: web
    name: fantasy-pl-manager
    env: python
    buildCommand: pip install -r requirements.txt && python -c "import nltk; nltk.download('punkt')" && python -m textblob.download_corpora
    startCommand: streamlit run src/dashboard.py --server.port $PORT --server.address 0.0.0.0
    plan: free
    
  # Optional: Cron job for updates
  - type: cron
    name: fantasy-updater
    env: python
    schedule: "0 2 * * *"
    buildCommand: pip install -r requirements.txt
    startCommand: python run_pipeline.py --skip-news
```

**Note**: Free tier spins down after inactivity (15 min startup time)

---

## 🔄 Automatic Updates Configuration

### GitHub Actions (Works with All Platforms)

Your repository is already configured with:

1. **`.github/workflows/update_data.yml`** - Updates predictions daily
2. **`.github/workflows/deploy_spaces.yml`** - Auto-deploys to Hugging Face

**Required Secrets** (add in GitHub repo settings):
- `HF_TOKEN` - Hugging Face access token
- `HF_SPACE_REPO` - Your Space name (e.g., `username/fantasy-premier-league-manager`)

**Customize Schedule**: Edit `.github/workflows/update_data.yml`:

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
    # - cron: '0 */6 * * *'  # Every 6 hours
    # - cron: '0 2 * * 1'  # Weekly on Monday
```

---

## 📊 Data Management

### Initial Data Setup:

```bash
# Generate all data locally first
python run_pipeline.py --test  # Test with 10 players
python run_pipeline.py         # Full run (1-2 hours)

# Copy to deployment
cp -r data/ hf_deploy_temp/
```

### Incremental Updates:

```bash
# Update only predictions (fast)
python src/predictor.py

# Update sentiment (if you have news data)
python src/sentiment_analyzer.py

# Rebuild RAG index
python src/rag_system.py
```

---

## 🔒 Environment Variables

Some platforms require environment variables. Create `.env`:

```bash
# Optional: OpenAI API for LLM integration
OPENAI_API_KEY=your_key_here

# Optional: Custom data sources
FPL_API_URL=https://fantasy.premierleague.com/api/bootstrap-static/
```

**Security**: Never commit `.env` to git! It's already in `.gitignore`.

---

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution**: 
```bash
# Add to deployment start command:
export PYTHONPATH="${PYTHONPATH}:./src"
streamlit run src/dashboard.py
```

### Issue: "Out of memory"
**Solution**: 
- Reduce model size in `spaces_requirements.txt`
- Use pre-computed predictions
- Enable caching in Streamlit

### Issue: Deployment takes too long
**Solution**:
- Pre-build data locally
- Use Docker for consistent deploys
- Enable build caching

### Issue: Data not updating
**Solution**:
```bash
# Check GitHub Actions logs
# Manually trigger workflow
# Verify secrets are set correctly
```

---

## 📈 Monitoring & Analytics

### Hugging Face Spaces:
- Built-in analytics dashboard
- Real-time logs
- Resource usage metrics

### Streamlit Cloud:
- Access logs in dashboard
- View app metrics
- Monitor user activity

### Custom Analytics:
Add to `src/dashboard.py`:

```python
import streamlit as st

# Track page views
st.session_state.page_views = st.session_state.get('page_views', 0) + 1
st.sidebar.metric("Page Views", st.session_state.page_views)
```

---

## 💡 Best Practices

1. ✅ **Test locally** before deploying
2. ✅ **Use Git LFS** for large files (`.csv`, `.json`)
3. ✅ **Cache data** in Streamlit with `@st.cache_data`
4. ✅ **Monitor costs** on paid platforms
5. ✅ **Set up alerts** for deployment failures
6. ✅ **Keep secrets secure** (use platform-specific secret management)
7. ✅ **Version your data** (commit predictions with timestamps)
8. ✅ **Document your deployment** for future reference

---

## 🎉 Next Steps

After deployment:

1. ✅ **Share your URL** with friends
2. ✅ **Star the repo** on GitHub
3. ✅ **Set up monitoring** for uptime
4. ✅ **Join communities** (Hugging Face, Streamlit Discord)
5. ✅ **Iterate and improve** based on feedback

---

## 📚 Additional Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Streamlit Deployment Docs](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)
- [Railway Docs](https://docs.railway.app)
- [Render Docs](https://render.com/docs)
- [GitHub Actions Docs](https://docs.github.com/en/actions)

---

**Questions?** Check the troubleshooting section or open an issue on GitHub!

Good luck with your deployment! ⚽🚀
