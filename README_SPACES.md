# 🚀 Deploy Fantasy PL Manager to Hugging Face Spaces

Complete guide to deploying your Fantasy Premier League Manager with **automatic updates** and **free hosting**.

## 📋 Prerequisites

1. **GitHub Account** (free)
2. **Hugging Face Account** (free) - Sign up at [huggingface.co](https://huggingface.co/join)
3. Your Fantasy Manager repository on GitHub

## 🎯 Quick Deployment (5 Minutes)

### Step 1: Create Hugging Face Space

1. Go to [https://huggingface.co/new-space](https://huggingface.co/new-space)
2. Fill in details:
   - **Space name**: `fantasy-premier-league-manager`
   - **License**: `MIT`
   - **SDK**: `Streamlit`
   - **Hardware**: `CPU basic` (free tier)
   - **Visibility**: `Public`
3. Click **Create Space**

### Step 2: Get Hugging Face Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Name: `github-actions-deploy`
4. Type: `Write`
5. Click **Generate**
6. **Copy the token** (you'll need it in Step 3)

### Step 3: Configure GitHub Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add two secrets:

   **Secret 1:**
   - Name: `HF_TOKEN`
   - Value: (paste the token from Step 2)
   
   **Secret 2:**
   - Name: `HF_SPACE_REPO`
   - Value: `YOUR_USERNAME/fantasy-premier-league-manager`
   - Example: `john-doe/fantasy-premier-league-manager`

### Step 4: Enable GitHub Actions

1. In your repository, go to **Settings** → **Actions** → **General**
2. Under **Actions permissions**, select:
   - ✅ **Allow all actions and reusable workflows**
3. Under **Workflow permissions**, select:
   - ✅ **Read and write permissions**
4. Click **Save**

### Step 5: Initial Deploy

#### Option A: Manual Deploy (Recommended First Time)

```bash
# Clone your HF Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/fantasy-premier-league-manager
cd fantasy-premier-league-manager

# Copy files from your local project
cp /path/to/FantasyManager/app.py .
cp /path/to/FantasyManager/spaces_requirements.txt requirements.txt
cp -r /path/to/FantasyManager/src .
mkdir -p data

# If you have pre-built data, copy it
cp -r /path/to/FantasyManager/data/* data/

# Commit and push
git add .
git commit -m "Initial deployment"
git push
```

#### Option B: Trigger GitHub Action

```bash
# In your main repository
git add .
git commit -m "Deploy to Hugging Face Spaces"
git push origin main

# GitHub Actions will automatically deploy!
```

## 🔄 Automatic Updates

Your system is now configured for automatic updates!

### What Gets Updated Automatically:

✅ **Daily at 2 AM UTC**:
- Predictions refresh using latest data
- Data committed to repository
- Auto-deployed to Hugging Face Spaces

✅ **On Every Push to Main**:
- Code changes auto-deploy
- Dashboard updates live

### Manual Trigger:

You can manually trigger updates anytime:

1. Go to **Actions** tab in GitHub
2. Select **Update Fantasy PL Data**
3. Click **Run workflow**
4. Select **main** branch
5. Click **Run workflow**

## 📊 Monitoring

### Check Deployment Status:

1. **GitHub Actions**: 
   - Repository → Actions tab
   - See workflow runs and logs

2. **Hugging Face Spaces**: 
   - Your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/fantasy-premier-league-manager`
   - Check **App** tab for live dashboard
   - Check **Files** tab for deployed files
   - Check **Logs** tab for any errors

### Common Issues:

#### ❌ Workflow fails with "HF_TOKEN not found"
**Solution**: Make sure you added the `HF_TOKEN` secret in GitHub repository settings

#### ❌ Space shows "Building" forever
**Solution**: 
- Check Hugging Face Space logs
- Might need to reduce `spaces_requirements.txt` dependencies
- Try restarting the Space

#### ❌ "No data directory" warning
**Solution**: This is normal! The workflow will generate data automatically

## 🎛️ Customization

### Change Update Frequency:

Edit `.github/workflows/update_data.yml`:

```yaml
# Daily at 2 AM
- cron: '0 2 * * *'

# Every 6 hours
- cron: '0 */6 * * *'

# Weekly on Monday
- cron: '0 2 * * 1'
```

### Deploy to Different Space:

Update GitHub secret `HF_SPACE_REPO` to point to a different Space.

### Use Private Space:

1. Create a private Space on Hugging Face
2. Update the secret accordingly
3. Everything else stays the same!

## 💰 Cost Breakdown

| Service | Free Tier | Used For |
|---------|-----------|----------|
| **Hugging Face Spaces** | ✅ Unlimited (CPU) | Dashboard hosting |
| **GitHub Actions** | ✅ 2,000 min/month | Auto-updates |
| **GitHub Storage** | ✅ 1 GB | Code + data |

**Total: $0/month** 🎉

## 🔒 Security Best Practices

✅ **Never commit tokens** to repository  
✅ **Use GitHub Secrets** for all credentials  
✅ **Keep Space public** or upgrade for private  
✅ **Review workflow logs** regularly  

## 📱 Access Your Dashboard

After deployment:
- **Live URL**: `https://huggingface.co/spaces/YOUR_USERNAME/fantasy-premier-league-manager`
- **Shareable**: Send this URL to anyone!
- **Always up**: 24/7 availability
- **Auto-updates**: Fresh data daily

## 🚨 Troubleshooting

### Data Not Updating?

```bash
# Manually run the pipeline locally
python run_pipeline.py --skip-news

# Commit the data
git add data/
git commit -m "Update data"
git push

# Will auto-deploy to HF Spaces
```

### Space Running Slow?

**Solution**: 
- Reduce `spaces_requirements.txt` dependencies
- Pre-compute predictions locally
- Use cached data files

### Out of GitHub Actions Minutes?

**Solution**:
- Reduce update frequency (weekly instead of daily)
- Skip news fetching in workflow (already configured)
- Use workflow_dispatch for manual triggers only

## 📚 Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Streamlit Documentation](https://docs.streamlit.io)

## 🎉 You're Done!

Your Fantasy PL Manager is now:
- ✅ **Hosted for free** on Hugging Face Spaces
- ✅ **Auto-updating daily** via GitHub Actions
- ✅ **Accessible to anyone** with a URL
- ✅ **Zero maintenance** required

Share your dashboard URL and enjoy! ⚽📊
