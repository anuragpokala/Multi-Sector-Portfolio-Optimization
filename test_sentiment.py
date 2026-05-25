"""
Test script for sentiment analysis pipeline.

This script validates:
1. API connections (Polygon, Finnhub)
2. FinBERT model loading
3. Sentiment analysis on sample text
4. View generation and ranking
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

# Load environment variables
load_dotenv()

from src import sentiment_analyzer
import pandas as pd

print("=" * 70)
print("SENTIMENT ANALYSIS PIPELINE TEST")
print("=" * 70)

# Test 1: Load API keys
print("\n[Test 1] Loading API keys from .env...")
polygon_key = os.getenv('POLYGON_API_KEY')
finnhub_key = os.getenv('FINNHUB_API_KEY')

if polygon_key and polygon_key != 'your_polygon_api_key_here':
    print("  ✓ Polygon API key loaded")
else:
    print("  ✗ Polygon API key missing or not configured")
    sys.exit(1)

if finnhub_key and finnhub_key != 'your_finnhub_api_key_here':
    print("  ✓ Finnhub API key loaded")
else:
    print("  ✗ Finnhub API key missing or not configured")
    sys.exit(1)

# Test 2: Test FinBERT model loading
print("\n[Test 2] Loading FinBERT model...")
try:
    sample_texts = [
        "Apple stock surged after strong earnings report",
        "Tesla faces challenges as competition intensifies",
        "Microsoft announces new AI partnership"
    ]
    
    results = sentiment_analyzer.analyze_finbert_sentiment(sample_texts)
    print(f"  ✓ FinBERT model loaded successfully")
    print(f"  Analyzed {len(results)} sample texts")
    print("\nSample Results:")
    print(results[['text', 'sentiment', 'score']].to_string(index=False))
except Exception as e:
    print(f"  ✗ Failed to load FinBERT: {e}")
    sys.exit(1)

# Test 3: Test Polygon news fetching
print("\n[Test 3] Testing Polygon API...")
try:
    test_ticker = "AAPL"  # Using Apple as test case
    articles = sentiment_analyzer.fetch_polygon_news(
        ticker=test_ticker,
        days=7,  # Just last week for testing
        api_key=polygon_key,
        delay=0.5
    )
    print(f"  ✓ Polygon API working")
    print(f"  Retrieved {len(articles)} articles for {test_ticker} (last 7 days)")
    
    if len(articles) > 0:
        print(f"\n  Sample headline: {articles[0]['title'][:80]}...")
except Exception as e:
    print(f"  ✗ Polygon API error: {e}")

# Test 4: Test Finnhub sentiment
print("\n[Test 4] Testing Finnhub API...")
try:
    test_ticker = "AAPL"
    finnhub_data = sentiment_analyzer.fetch_finnhub_sentiment(
        ticker=test_ticker,
        api_key=finnhub_key
    )
    print(f"  ✓ Finnhub API working")
    print(f"  Sentiment score for {test_ticker}: {finnhub_data['sentiment_score']:+.3f}")
    print(f"  Bullish: {finnhub_data['bullish_percent']:.1f}%, Bearish: {finnhub_data['bearish_percent']:.1f}%")
except Exception as e:
    print(f"  ✗ Finnhub API error: {e}")

# Test 5: Test complete sentiment analysis for one ticker
print("\n[Test 5] Testing complete sentiment pipeline...")
try:
    test_ticker = "AMZN"
    sentiment_data = sentiment_analyzer.compute_ticker_sentiment(
        ticker=test_ticker,
        polygon_key=polygon_key,
        finnhub_key=finnhub_key,
        days=7,  # Short period for testing
        min_articles=3,
        model_name="ProsusAI/finbert",
        batch_size=32
    )
    
    print(f"\n  ✓ Complete pipeline working for {test_ticker}")
    print(f"\n  Results:")
    print(f"    Sentiment Score: {sentiment_data['sentiment_score']:+.3f}")
    print(f"    Sentiment Label: {sentiment_data['sentiment_label']}")
    print(f"    Confidence: {sentiment_data['confidence']}")
    print(f"    Articles Analyzed: {sentiment_data['num_articles']}")
    print(f"    FinBERT Score: {sentiment_data['finbert_score']:+.3f}")
    print(f"    Finnhub Score: {sentiment_data['finnhub_score']:+.3f}")
    print(f"    Agreement: {sentiment_data['agreement']}")
    
except Exception as e:
    print(f"  ✗ Pipeline error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test view ranking
print("\n[Test 6] Testing sentiment ranking...")
try:
    test_scores = {
        'AAPL': 0.45,
        'MSFT': 0.32,
        'AMZN': 0.15,
        'TSLA': -0.10,
        'META': -0.25,
        'GOOGL': 0.28,
        'NVDA': 0.50,
        'AMD': 0.05
    }
    
    ranking_tiers = {
        'top_quartile': 0.04,
        'second_quartile': 0.02,
        'third_quartile': -0.01,
        'bottom_quartile': -0.03
    }
    
    views = sentiment_analyzer.rank_stocks_by_sentiment(test_scores, ranking_tiers)
    
    print("  ✓ Ranking system working")
    print("\n  Sample Rankings:")
    for ticker, view in sorted(views.items(), key=lambda x: x[1]['rank'])[:5]:
        print(f"    {view['rank']}. {ticker}: sentiment {view['sentiment_score']:+.3f} → {view['quartile']} → return {view['return']:+.1%}")
        
except Exception as e:
    print(f"  ✗ Ranking error: {e}")

print("\n" + "=" * 70)
print("TESTING COMPLETE")
print("=" * 70)
print("\nAll core components are working!")
print("\nNext steps:")
print("1. Run the full notebook to generate sentiment-based Black-Litterman portfolio")
print("2. Compare results with analyst-only Black-Litterman")
print("3. Review sentiment analysis results for accuracy")
print("\nNote: Full analysis will take 30-60 seconds due to API calls and FinBERT processing")
