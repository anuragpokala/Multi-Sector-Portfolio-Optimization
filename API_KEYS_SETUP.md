# API Keys Setup Guide

This guide will help you obtain the required API keys for sentiment-based portfolio optimization.

## Required API Keys

### 1. Polygon.io API Key (Required)

**Purpose:** Fetch financial news articles for sentiment analysis

**How to get:**
1. Go to https://polygon.io/
2. Click "Sign Up" (top right)
3. Choose the **FREE tier** (5 requests/minute)
4. Verify your email
5. Go to Dashboard → API Keys
6. Copy your API key

**Free tier limits:** 5 requests/minute (sufficient for 8 stocks)

### 2. Finnhub API Key (Required)

**Purpose:** Validate sentiment scores as a sanity check

**How to get:**
1. Go to https://finnhub.io/
2. Click "Get free API key"
3. Sign up with email or GitHub
4. Your API key will be displayed immediately
5. Copy and save it

**Free tier limits:** 60 requests/minute (more than enough)

### 3. Hugging Face Token (Optional)

**Purpose:** Access private/gated models (NOT needed for FinBERT)

**Note:** FinBERT (ProsusAI/finbert) is a public model and doesn't require authentication. You can skip this unless you plan to use private models.

**How to get (if needed):**
1. Go to https://huggingface.co/
2. Sign up for a free account
3. Go to Settings → Access Tokens
4. Create a new token (read access is sufficient)
5. Copy and save it

## Installation Steps

### 1. Copy `.env.example` to `.env`
```bash
cp .env.example .env
```

### 2. Edit `.env` and add your API keys
```bash
# Open in your text editor
nano .env
# or
code .env
```

Replace the placeholder values:
```bash
POLYGON_API_KEY=pk_your_actual_polygon_key_here
FINNHUB_API_KEY=your_actual_finnhub_key_here
HUGGINGFACE_TOKEN=hf_your_token_here  # Optional
```

### 3. Install python-dotenv
```bash
pip install python-dotenv
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Security Notes

- ✅ `.env` is in `.gitignore` - your keys will NOT be committed to Git
- ✅ Never share your `.env` file publicly
- ✅ Use `.env.example` as a template for team members (without real keys)
- ✅ If you accidentally commit API keys, regenerate them immediately

## Testing Your Setup

After adding your API keys, test the connection:

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Check if keys are loaded
polygon_key = os.getenv('POLYGON_API_KEY')
finnhub_key = os.getenv('FINNHUB_API_KEY')

print(f"Polygon key loaded: {polygon_key[:10]}..." if polygon_key else "❌ Polygon key missing")
print(f"Finnhub key loaded: {finnhub_key[:10]}..." if finnhub_key else "❌ Finnhub key missing")
```

## Cost Summary

| Service | Free Tier | Cost for This Project |
|---------|-----------|---------------------|
| Polygon | 5 req/min | **FREE** (8 stocks = 8 requests) |
| Finnhub | 60 req/min | **FREE** (8 stocks = 8 requests) |
| Hugging Face | Unlimited for public models | **FREE** (FinBERT is public) |

**Total cost: $0** 🎉

## Troubleshooting

### "ModuleNotFoundError: No module named 'dotenv'"
```bash
pip install python-dotenv
```

### "API key not found" error
- Make sure `.env` file is in the project root directory
- Check that you've replaced the placeholder values
- Verify there are no extra spaces or quotes around the keys
- Make sure `.env` doesn't have `.txt` extension (check with `ls -a`)

### Rate limit errors
- Polygon: Wait 12 seconds between requests (free tier = 5/min)
- Add delays in the code: `time.sleep(1)` between requests

## Questions?

If you need help:
1. Check the [Polygon docs](https://polygon.io/docs/stocks/getting-started)
2. Check the [Finnhub docs](https://finnhub.io/docs/api)
3. Verify your keys are active in their respective dashboards
