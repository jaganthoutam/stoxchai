"""
Pytest configuration and fixtures for StoxChai tests
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_stock_data():
    """Create sample stock data for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    return pd.DataFrame({
        'Open': [100 + i for i in range(len(dates))],
        'High': [105 + i for i in range(len(dates))],
        'Low': [95 + i for i in range(len(dates))],
        'Close': [102 + i for i in range(len(dates))],
        'Volume': [1000000 + i*10000 for i in range(len(dates))]
    }, index=dates)

@pytest.fixture
def sample_company_info():
    """Create sample company info for testing"""
    return {
        'longName': 'Test Company Limited',
        'sector': 'Technology',
        'industry': 'Software',
        'marketCap': 1000000000,
        'trailingPE': 15.5,
        'trailingEps': 85.2,
        'dividendYield': 0.02,
        'bookValue': 150.0,
        'priceToBook': 2.5,
        'beta': 1.2,
        'fiftyTwoWeekHigh': 200.0,
        'fiftyTwoWeekLow': 80.0,
        'averageVolume': 1500000,
        'longBusinessSummary': 'A technology company focused on software development.',
        'website': 'https://www.testcompany.com'
    }

@pytest.fixture
def sample_news_articles():
    """Create sample news articles for testing"""
    return [
        {
            'title': 'Company Reports Strong Q4 Results',
            'url': 'https://example.com/news1',
            'source': 'Financial Times',
            'published_date': '2024-01-15',
            'content': 'The company reported strong quarterly results...',
            'sentiment_score': 0.8,
            'sentiment_label': 'POSITIVE',
            'relevance_score': 0.9
        },
        {
            'title': 'Market Concerns Over Regulatory Changes',
            'url': 'https://example.com/news2',
            'source': 'Economic Times',
            'published_date': '2024-01-14',
            'content': 'Investors are concerned about new regulations...',
            'sentiment_score': -0.6,
            'sentiment_label': 'NEGATIVE',
            'relevance_score': 0.7
        }
    ]

@pytest.fixture
def mock_yfinance_ticker():
    """Mock yfinance Ticker for testing"""
    mock_ticker = Mock()
    mock_ticker.history.return_value = pd.DataFrame({
        'Open': [100, 102, 104],
        'High': [105, 107, 109],
        'Low': [95, 97, 99],
        'Close': [103, 105, 107],
        'Volume': [1000000, 1100000, 1200000]
    })
    mock_ticker.info = {
        'longName': 'Mock Company Limited',
        'sector': 'Technology',
        'industry': 'Software'
    }
    return mock_ticker

@pytest.fixture
def test_environment():
    """Set up test environment variables"""
    os.environ['ENVIRONMENT'] = 'testing'
    os.environ['DEBUG'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    
    yield
    
    # Cleanup
    test_vars = ['ENVIRONMENT', 'DEBUG', 'LOG_LEVEL']
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary directory for test data"""
    return tmp_path_factory.mktemp("test_data")

@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing"""
    mock_cache = Mock()
    mock_cache.get.return_value = None
    mock_cache.set.return_value = True
    mock_cache.delete.return_value = True
    mock_cache.clear_all.return_value = True
    return mock_cache

@pytest.fixture
def indian_stock_symbols():
    """List of common Indian stock symbols for testing"""
    return [
        'RELIANCE.NS',
        'TCS.NS',
        'HDFCBANK.NS',
        'INFY.NS',
        'ICICIBANK.NS',
        'HINDUNILVR.NS',
        'SBIN.NS',
        'BHARTIARTL.NS',
        'ITC.NS',
        'KOTAKBANK.NS'
    ]

@pytest.fixture
def market_indices():
    """List of Indian market indices for testing"""
    return {
        'NIFTY_50': '^NSEI',
        'NIFTY_BANK': '^NSEBANK',
        'SENSEX': '^BSESN',
        'NIFTY_IT': '^CNXIT',
        'NIFTY_PHARMA': '^CNXPHARMA'
    }

# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "api: mark test as requiring external API"
    )

# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark API tests
        if any(keyword in item.nodeid.lower() for keyword in ["api", "yfinance", "external"]):
            item.add_marker(pytest.mark.api)
        
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["slow", "performance", "load"]):
            item.add_marker(pytest.mark.slow)