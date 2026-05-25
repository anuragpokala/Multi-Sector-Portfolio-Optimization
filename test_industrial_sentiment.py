"""
Test sentiment analysis for Industrial sector tickers

This script tests if sentiment analysis works properly for Industrial stocks
before running the full optimization notebook.
"""

import yaml
from src import sentiment_analyzer

def test_industrial_sentiment():
    """Test sentiment analysis with Industrial sector tickers."""
    print("="*70)
    print("TESTING INDUSTRIAL SECTOR SENTIMENT ANALYSIS")
    print("="*70)
    
    # Load config
    with open('config_industrial.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    tickers = config['tickers']
    sentiment_config = config['black_litterman_sentiment']
    
    print(f"\nTesting sentiment for {len(tickers)} Industrial stocks:")
    print(f"  {', '.join(tickers)}\n")
    
    # Test sentiment analysis
    try:
        result = sentiment_analyzer.analyze_portfolio_sentiment(
            tickers=tickers,
            config=sentiment_config
        )
        
        print("\n" + "="*70)
        print("SENTIMENT ANALYSIS SUCCESSFUL")
        print("="*70)
        
        # Display results
        sentiment_scores = result['sentiment_scores']
        sentiment_views = result['sentiment_views']
        
        print("\nSentiment Scores:")
        for ticker in tickers:
            if ticker in sentiment_scores:
                score = sentiment_scores[ticker]
                print(f"  {ticker}: {score:+.3f}")
        
        print("\nSentiment-Based Views:")
        for ticker, view in sentiment_views.items():
            print(f"  {ticker}: Return = {view['return']:+.2%}, "
                  f"Confidence = {view['confidence']:.6f}")
        
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("✓ Ready to run industrial_portfolio_optimization.ipynb")
        print("="*70)
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR in sentiment analysis")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nPossible issues:")
        print("  - API keys not set in .env")
        print("  - Network connectivity problems")
        print("  - Ticker symbols not recognized by news APIs")
        print("  - Rate limiting (try increasing request_delay in config)")
        
        return False


if __name__ == '__main__':
    success = test_industrial_sentiment()
    
    if not success:
        print("\n⚠️  Sentiment analysis test failed")
        print("   The notebook will still work for Mean-Variance optimization")
        print("   Fix API issues to enable Black-Litterman sentiment analysis")
