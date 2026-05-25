"""
Sentiment Analysis Module for Black-Litterman Optimization

Uses financial news sentiment to generate portfolio views:
- Polygon API for news retrieval
- FinBERT (Hugging Face) for sentiment analysis
- Finnhub for validation
- Relative ranking to generate expected returns
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from polygon import RESTClient
import finnhub
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Suppress transformer warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# Global variables for model caching (load once, reuse)
_finbert_tokenizer = None
_finbert_model = None


def _load_finbert_model(model_name: str = "ProsusAI/finbert"):
    """
    Load FinBERT model and tokenizer (cached for reuse).
    
    Parameters
    ----------
    model_name : str
        Hugging Face model name
        
    Returns
    -------
    Tuple[AutoTokenizer, AutoModelForSequenceClassification]
        Tokenizer and model
        
    Notes
    -----
    - Model is downloaded on first use (~440MB)
    - Subsequent calls use cached version
    - CPU inference is sufficient (no GPU needed)
    """
    global _finbert_tokenizer, _finbert_model
    
    if _finbert_tokenizer is None or _finbert_model is None:
        print(f"Loading FinBERT model: {model_name}...")
        print("(This may take a minute on first run - model will be cached)")
        
        _finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use use_safetensors=True to bypass torch.load security restrictions
        _finbert_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            use_safetensors=True
        )
        _finbert_model.eval()  # Set to evaluation mode
        
        print("✓ FinBERT model loaded successfully")
    
    return _finbert_tokenizer, _finbert_model


def fetch_polygon_news(
    ticker: str,
    days: int,
    api_key: str,
    delay: float = 1.0
) -> List[Dict]:
    """
    Fetch financial news for a ticker from Polygon API.
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    days : int
        Number of days to look back
    api_key : str
        Polygon API key
    delay : float
        Delay after request (respect rate limits)
        
    Returns
    -------
    List[Dict]
        News articles with keys: title, description, published_utc, url
        
    Notes
    -----
    - Free tier: 5 requests/minute
    - Adds delay to respect rate limits
    """
    client = RESTClient(api_key)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Fetch news
        news_response = client.list_ticker_news(
            ticker=ticker,
            published_utc_gte=start_date.strftime('%Y-%m-%d'),
            published_utc_lte=end_date.strftime('%Y-%m-%d'),
            limit=100  # Max 100 articles
        )
        
        articles = []
        for article in news_response:
            articles.append({
                'title': article.title or "",
                'description': article.description or "",
                'published_utc': article.published_utc,
                'url': article.article_url
            })
        
        # Respect rate limits
        time.sleep(delay)
        
        return articles
        
    except Exception as e:
        print(f"Warning: Failed to fetch Polygon news for {ticker}: {e}")
        return []


def analyze_finbert_sentiment(
    texts: List[str],
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Analyze sentiment of texts using FinBERT (batch processing).
    
    Parameters
    ----------
    texts : List[str]
        List of text to analyze (titles + descriptions)
    model_name : str
        FinBERT model name
    batch_size : int
        Number of texts to process simultaneously
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: text, sentiment, score, positive_prob, negative_prob, neutral_prob
        
    Notes
    -----
    - Batch processing is 30-50x faster than sequential
    - Sentiment: 'positive', 'negative', or 'neutral'
    - Score: confidence score (0-1)
    """
    if not texts:
        return pd.DataFrame(columns=['text', 'sentiment', 'score', 'positive_prob', 'negative_prob', 'neutral_prob'])
    
    tokenizer, model = _load_finbert_model(model_name)
    
    results = []
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Process results
        label_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}
        
        for j, text in enumerate(batch):
            probs = predictions[j].tolist()
            max_prob_idx = probs.index(max(probs))
            
            results.append({
                'text': text[:100] + '...' if len(text) > 100 else text,  # Truncate for display
                'sentiment': label_mapping[max_prob_idx],
                'score': max(probs),
                'positive_prob': probs[0],
                'negative_prob': probs[1],
                'neutral_prob': probs[2]
            })
    
    return pd.DataFrame(results)


def fetch_finnhub_sentiment(
    ticker: str,
    api_key: str
) -> Dict:
    """
    Get aggregate news sentiment from Finnhub (validation/sanity check).
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    api_key : str
        Finnhub API key
        
    Returns
    -------
    Dict
        Contains: sentiment_score, bearish_percent, bullish_percent, avg_score
        
    Notes
    -----
    - Provides aggregated sentiment (not article-level)
    - Use as validation for FinBERT results
    - Free tier: 60 requests/minute
    """
    client = finnhub.Client(api_key=api_key)
    
    try:
        # Get news sentiment
        sentiment_data = client.news_sentiment(ticker)
        
        if sentiment_data and 'sentiment' in sentiment_data:
            buzz = sentiment_data['buzz']
            sentiment = sentiment_data['sentiment']
            
            return {
                'sentiment_score': sentiment.get('sentimentScore', 0),
                'bearish_percent': sentiment.get('bearishPercent', 0),
                'bullish_percent': sentiment.get('bullishPercent', 0),
                'articles_in_last_week': buzz.get('articlesInLastWeek', 0)
            }
        else:
            print(f"Warning: No sentiment data from Finnhub for {ticker}")
            return {'sentiment_score': 0, 'bearish_percent': 0, 'bullish_percent': 0, 'articles_in_last_week': 0}
            
    except Exception as e:
        print(f"Warning: Failed to fetch Finnhub sentiment for {ticker}: {e}")
        return {'sentiment_score': 0, 'bearish_percent': 0, 'bullish_percent': 0, 'articles_in_last_week': 0}


def compute_ticker_sentiment(
    ticker: str,
    polygon_key: str,
    finnhub_key: str,
    days: int = 30,
    min_articles: int = 5,
    model_name: str = "ProsusAI/finbert",
    batch_size: int = 32
) -> Dict:
    """
    Compute comprehensive sentiment for a ticker (orchestrates all sources).
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol
    polygon_key : str
        Polygon API key
    finnhub_key : str
        Finnhub API key
    days : int
        Days to look back for news
    min_articles : int
        Minimum articles needed for valid score
    model_name : str
        FinBERT model name
    batch_size : int
        Batch size for FinBERT
        
    Returns
    -------
    Dict
        Complete sentiment analysis with scores, confidence, agreement
        
    Notes
    -----
    Sentiment score ranges from -1 (very negative) to +1 (very positive)
    Confidence based on:
    - Agreement between FinBERT and Finnhub
    - Number of articles analyzed
    - Consistency of sentiment across articles
    """
    print(f"\n{'='*60}")
    print(f"Analyzing sentiment for {ticker}")
    print(f"{'='*60}")
    
    # 1. Fetch Polygon news
    print(f"Fetching news from Polygon (last {days} days)...")
    articles = fetch_polygon_news(ticker, days, polygon_key)
    num_articles = len(articles)
    print(f"  Found {num_articles} articles")
    
    # Only skip FinBERT if there are literally 0 articles
    if num_articles == 0:
        print(f"  ⚠️  Warning: No articles found")
        print(f"  Falling back to Finnhub only")
        
        # Use Finnhub only
        finnhub_data = fetch_finnhub_sentiment(ticker, finnhub_key)
        return {
            'ticker': ticker,
            'sentiment_score': finnhub_data['sentiment_score'],
            'confidence': 'low',
            'num_articles': num_articles,
            'finbert_score': None,
            'finnhub_score': finnhub_data['sentiment_score'],
            'agreement': 'n/a',
            'sentiment_label': 'neutral' if abs(finnhub_data['sentiment_score']) < 0.1 else ('positive' if finnhub_data['sentiment_score'] > 0 else 'negative')
        }
    
    # Proceed with FinBERT even if only 1-2 articles
    if num_articles < min_articles:
        print(f"  ⚠️  Note: Only {num_articles} articles (less than typical minimum of {min_articles})")
        print(f"  Proceeding with FinBERT analysis anyway...")
    
    # 2. Analyze with FinBERT
    print(f"Analyzing sentiment with FinBERT...")
    texts = [f"{article['title']} {article['description']}" for article in articles]
    finbert_results = analyze_finbert_sentiment(texts, model_name, batch_size)
    
    # Calculate aggregate FinBERT score (-1 to +1)
    positive_pct = (finbert_results['sentiment'] == 'positive').sum() / len(finbert_results)
    negative_pct = (finbert_results['sentiment'] == 'negative').sum() / len(finbert_results)
    finbert_score = positive_pct - negative_pct  # Range: -1 to +1
    
    print(f"  FinBERT: {positive_pct:.1%} positive, {negative_pct:.1%} negative")
    print(f"  FinBERT score: {finbert_score:+.3f}")
    
    # 3. Get Finnhub validation
    print(f"Validating with Finnhub...")
    finnhub_data = fetch_finnhub_sentiment(ticker, finnhub_key)
    finnhub_score = finnhub_data['sentiment_score']
    print(f"  Finnhub score: {finnhub_score:+.3f}")
    
    # 4. Combine scores and assess agreement
    # Weight FinBERT more heavily (0.7) since it's analyzing actual article text
    combined_score = 0.7 * finbert_score + 0.3 * finnhub_score
    
    # Assess agreement
    agreement_diff = abs(finbert_score - finnhub_score)
    if agreement_diff < 0.2:
        agreement = 'strong'
        confidence = 'high'
    elif agreement_diff < 0.4:
        agreement = 'moderate'
        confidence = 'medium'
    else:
        agreement = 'weak'
        confidence = 'low'
    
    # Determine sentiment label
    if combined_score > 0.1:
        sentiment_label = 'positive'
    elif combined_score < -0.1:
        sentiment_label = 'negative'
    else:
        sentiment_label = 'neutral'
    
    print(f"  Combined score: {combined_score:+.3f} ({sentiment_label})")
    print(f"  Agreement: {agreement} (confidence: {confidence})")
    
    return {
        'ticker': ticker,
        'sentiment_score': combined_score,
        'confidence': confidence,
        'num_articles': num_articles,
        'finbert_score': finbert_score,
        'finnhub_score': finnhub_score,
        'agreement': agreement,
        'sentiment_label': sentiment_label,
        'positive_pct': positive_pct,
        'negative_pct': negative_pct
    }


def rank_stocks_by_sentiment(
    sentiment_scores: Dict[str, float],
    ranking_tiers: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Rank stocks by sentiment and assign expected returns (relative ranking).
    
    Parameters
    ----------
    sentiment_scores : Dict[str, float]
        Mapping of ticker to sentiment score (-1 to +1)
    ranking_tiers : Dict[str, float]
        Expected returns for each quartile
        
    Returns
    -------
    Dict[str, Dict]
        Views for each ticker: {ticker: {'return': float, 'rank': int, 'quartile': str}}
        
    Notes
    -----
    Relative ranking is more robust than absolute scaling:
    - Avoids overfitting to absolute sentiment levels
    - Only requires relative ordering to be correct
    - Less sensitive to sentiment score calibration
    """
    # Sort tickers by sentiment (descending)
    sorted_tickers = sorted(sentiment_scores.items(), key=lambda x: x[1], reverse=True)
    n_stocks = len(sorted_tickers)
    
    views = {}
    for rank, (ticker, score) in enumerate(sorted_tickers, 1):
        # Determine quartile
        quartile_index = int((rank - 1) / n_stocks * 4)  # 0, 1, 2, or 3
        
        if quartile_index == 0:
            quartile = 'top_quartile'
            expected_return = ranking_tiers['top_quartile']
        elif quartile_index == 1:
            quartile = 'second_quartile'
            expected_return = ranking_tiers['second_quartile']
        elif quartile_index == 2:
            quartile = 'third_quartile'
            expected_return = ranking_tiers['third_quartile']
        else:
            quartile = 'bottom_quartile'
            expected_return = ranking_tiers['bottom_quartile']
        
        views[ticker] = {
            'return': expected_return,
            'rank': rank,
            'quartile': quartile,
            'sentiment_score': score
        }
    
    return views


def combine_views(
    analyst_views: Dict,
    sentiment_views: Dict[str, Dict],
    sentiment_weight: float,
    confidence_levels: Dict[str, float],
    sentiment_data: Dict[str, Dict]
) -> Dict:
    """
    Combine analyst views with sentiment views using weighted average.
    
    Parameters
    ----------
    analyst_views : Dict
        Manual analyst views from config.yaml
    sentiment_views : Dict[str, Dict]
        Sentiment-based views from ranking
    sentiment_weight : float
        Weight for sentiment (0-1), remaining weight goes to analyst
    confidence_levels : Dict[str, float]
        Confidence values (variance) for high/medium/low
    sentiment_data : Dict[str, Dict]
        Full sentiment analysis data (for confidence assessment)
        
    Returns
    -------
    Dict
        Combined views in same format as config.yaml
        
    Notes
    -----
    Combination strategy:
    - Returns: weighted average
    - Confidence: higher when views agree, lower when they conflict
    - If only one source has a view, use that source entirely
    """
    combined = {}
    
    for ticker in sentiment_views.keys():
        sentiment_view = sentiment_views[ticker]
        analyst_view = analyst_views.get(ticker, {})
        
        # Get sentiment confidence level
        ticker_sentiment = sentiment_data.get(ticker, {})
        sentiment_confidence_level = ticker_sentiment.get('confidence', 'medium')
        
        if analyst_view:
            # Both sources have views - combine them
            analyst_return = analyst_view['return']
            sentiment_return = sentiment_view['return']
            
            # Weighted average
            combined_return = (sentiment_weight * sentiment_return + 
                             (1 - sentiment_weight) * analyst_return)
            
            # Determine confidence based on agreement
            views_agree = (analyst_return * sentiment_return > 0)  # Same sign
            
            if views_agree and sentiment_confidence_level == 'high':
                confidence = confidence_levels['high']
            elif views_agree or sentiment_confidence_level == 'high':
                confidence = confidence_levels['medium']
            else:
                confidence = confidence_levels['low']
            
            combined[ticker] = {
                'type': 'absolute',
                'return': combined_return,
                'confidence': confidence,
                'description': f"Combined: analyst {analyst_return:+.1%} + sentiment {sentiment_return:+.1%}"
            }
        else:
            # Only sentiment view available
            combined[ticker] = {
                'type': 'absolute',
                'return': sentiment_view['return'],
                'confidence': confidence_levels[sentiment_confidence_level],
                'description': f"Sentiment only: {sentiment_view['return']:+.1%} (rank {sentiment_view['rank']})"
            }
    
    return combined


def analyze_portfolio_sentiment(
    tickers: List[str],
    config: Dict,
    polygon_key: str,
    finnhub_key: str
) -> Tuple[Dict[str, Dict], Dict[str, Dict], pd.DataFrame]:
    """
    Complete sentiment analysis pipeline for portfolio.
    
    Parameters
    ----------
    tickers : List[str]
        List of stock tickers
    config : Dict
        Configuration dictionary with sentiment settings
    polygon_key : str
        Polygon API key
    finnhub_key : str
        Finnhub API key
        
    Returns
    -------
    Tuple[Dict, Dict, pd.DataFrame]
        (combined_views, sentiment_data, summary_df)
        - combined_views: Ready for Black-Litterman optimization
        - sentiment_data: Full sentiment analysis results
        - summary_df: Summary table for display
        
    Notes
    -----
    This is the main entry point for sentiment analysis.
    """
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("="*70)
    
    sentiment_config = config['black_litterman_sentiment']
    
    # 1. Compute sentiment for each ticker
    sentiment_data = {}
    for ticker in tickers:
        result = compute_ticker_sentiment(
            ticker=ticker,
            polygon_key=polygon_key,
            finnhub_key=finnhub_key,
            days=sentiment_config['news_lookback_days'],
            min_articles=sentiment_config['min_news_articles'],
            model_name=sentiment_config['finbert_model'],
            batch_size=sentiment_config['batch_size']
        )
        sentiment_data[ticker] = result
    
    # 2. Rank stocks and generate sentiment views
    print("\n" + "="*70)
    print("GENERATING SENTIMENT-BASED VIEWS")
    print("="*70)
    
    sentiment_scores = {t: data['sentiment_score'] for t, data in sentiment_data.items()}
    sentiment_views = rank_stocks_by_sentiment(
        sentiment_scores,
        sentiment_config['ranking_tiers']
    )
    
    print("\nSentiment Rankings:")
    for ticker, view in sorted(sentiment_views.items(), key=lambda x: x[1]['rank']):
        print(f"  {view['rank']}. {ticker}: {view['sentiment_score']:+.3f} → {view['quartile']} → {view['return']:+.1%}")
    
    # 3. Combine with analyst views
    print("\n" + "="*70)
    print("COMBINING WITH ANALYST VIEWS")
    print("="*70)
    
    analyst_views = config['black_litterman']['views']
    combined_views = combine_views(
        analyst_views=analyst_views,
        sentiment_views=sentiment_views,
        sentiment_weight=sentiment_config['sentiment_weight'],
        confidence_levels=sentiment_config['confidence_levels'],
        sentiment_data=sentiment_data
    )
    
    print(f"\nBlending: {sentiment_config['sentiment_weight']:.0%} sentiment + {1-sentiment_config['sentiment_weight']:.0%} analyst")
    print("\nCombined Views:")
    for ticker, view in combined_views.items():
        print(f"  {ticker}: {view['return']:+.1%} (confidence: {view['confidence']:.4f})")
    
    # 4. Create summary DataFrame
    summary_data = []
    for ticker in tickers:
        sent_data = sentiment_data[ticker]
        sent_view = sentiment_views.get(ticker, {})
        comb_view = combined_views.get(ticker, {})
        analyst_view = analyst_views.get(ticker, {})
        
        summary_data.append({
            'Ticker': ticker,
            'Sentiment Score': sent_data['sentiment_score'],
            'Sentiment Label': sent_data['sentiment_label'],
            'Confidence': sent_data['confidence'],
            'Articles': sent_data['num_articles'],
            'Rank': sent_view.get('rank', '-'),
            'Sentiment View': sent_view.get('return', 0),
            'Analyst View': analyst_view.get('return', 0) if analyst_view else 0,
            'Combined View': comb_view.get('return', 0)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\n" + "="*70)
    print("SENTIMENT ANALYSIS COMPLETE")
    print("="*70)
    
    return combined_views, sentiment_data, summary_df


if __name__ == "__main__":
    print("Sentiment Analyzer Module")
    print("="*60)
    print("This module provides sentiment analysis for portfolio optimization.")
    print("Use analyze_portfolio_sentiment() as the main entry point.")
