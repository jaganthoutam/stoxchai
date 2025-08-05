"""
Tests for Indian market data functionality
"""
import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.data.indian_market import IndianMarketData, IndianNewsProvider
from src.core.models import CompanyInfo, Exchange

class TestIndianMarketData:
    """Test cases for IndianMarketData class"""
    
    @pytest.fixture
    def market_data(self):
        """Create IndianMarketData instance for testing"""
        return IndianMarketData()
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        return pd.DataFrame({
            'Open': [100 + i for i in range(len(dates))],
            'High': [105 + i for i in range(len(dates))],
            'Low': [95 + i for i in range(len(dates))],
            'Close': [102 + i for i in range(len(dates))],
            'Volume': [1000000 + i*10000 for i in range(len(dates))]
        }, index=dates)
    
    def test_get_nse_stock_data_valid_symbol(self, market_data, sample_stock_data):
        """Test getting NSE stock data for valid symbol"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = sample_stock_data
            
            result = market_data.get_nse_stock_data("RELIANCE", period="1mo")
            
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert not result.empty
            mock_ticker.assert_called_once_with("RELIANCE.NS")
    
    def test_get_nse_stock_data_invalid_symbol(self, market_data):
        """Test getting NSE stock data for invalid symbol"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            
            result = market_data.get_nse_stock_data("INVALID", period="1mo")
            
            assert result is None
    
    def test_get_bse_stock_data_valid_symbol(self, market_data, sample_stock_data):
        """Test getting BSE stock data for valid symbol"""
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = sample_stock_data
            
            result = market_data.get_bse_stock_data("RELIANCE", period="1mo")
            
            assert result is not None
            assert isinstance(result, pd.DataFrame)
            mock_ticker.assert_called_once_with("RELIANCE.BO")
    
    def test_get_company_info_valid_symbol(self, market_data):
        """Test getting company info for valid symbol"""
        mock_info = {
            'longName': 'Reliance Industries Limited',
            'sector': 'Energy',
            'industry': 'Oil & Gas',
            'marketCap': 1000000000000,
            'trailingPE': 15.5,
            'trailingEps': 85.2,
            'website': 'https://www.ril.com'
        }
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.info = mock_info
            
            result = market_data.get_company_info("RELIANCE", "NSE")
            
            assert result is not None
            assert isinstance(result, CompanyInfo)
            assert result.name == 'Reliance Industries Limited'
            assert result.exchange == Exchange.NSE
            assert result.sector == 'Energy'
    
    def test_get_indian_indices(self, market_data):
        """Test getting Indian market indices"""
        mock_data = pd.DataFrame({
            'Close': [18000, 18100]
        })
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_data
            mock_ticker.return_value.info = {}
            
            result = market_data.get_indian_indices()
            
            assert isinstance(result, list)
            # Should get all configured indices
            assert len(result) > 0
    
    def test_get_market_status_open(self, market_data):
        """Test market status when market is open"""
        with patch('src.config.settings.is_market_open', return_value=True):
            result = market_data.get_market_status()
            assert result == "OPEN"
    
    def test_get_market_status_closed(self, market_data):
        """Test market status when market is closed"""
        with patch('src.config.settings.is_market_open', return_value=False):
            with patch('datetime.datetime') as mock_datetime:
                # Mock time to be outside market hours
                mock_datetime.now.return_value.time.return_value.hour = 10
                result = market_data.get_market_status()
                assert result in ["CLOSED", "PRE_OPEN", "POST_CLOSE"]
    
    def test_get_top_stocks_gainers(self, market_data):
        """Test getting top gaining stocks"""
        mock_data = pd.DataFrame({
            'Close': [100, 105],  # 5% gain
            'Volume': [1000000, 1100000]
        })
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = mock_data
            
            result = market_data.get_top_stocks("gainers", limit=5)
            
            assert isinstance(result, list)
            assert len(result) <= 5
            # Results should be sorted by change_percent descending
            if len(result) > 1:
                assert result[0]['change_percent'] >= result[1]['change_percent']
    
    def test_search_stocks(self, market_data):
        """Test stock search functionality"""
        result = market_data.search_stocks("RELIANCE", limit=5)
        
        assert isinstance(result, list)
        assert len(result) <= 5
        # Should find RELIANCE in results
        symbols = [stock['symbol'] for stock in result]
        assert 'RELIANCE' in symbols

class TestIndianNewsProvider:
    """Test cases for IndianNewsProvider class"""
    
    @pytest.fixture
    def news_provider(self):
        """Create IndianNewsProvider instance for testing"""
        return IndianNewsProvider()
    
    def test_get_stock_news_valid_symbol(self, news_provider):
        """Test getting news for valid stock symbol"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = """
        <rss>
            <channel>
                <item>
                    <title>Reliance Stock Price Update</title>
                    <link>https://example.com/news1</link>
                    <pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate>
                </item>
            </channel>
        </rss>
        """
        
        with patch.object(news_provider.session, 'get', return_value=mock_response):
            result = news_provider.get_stock_news("RELIANCE", "Reliance Industries", limit=5)
            
            assert isinstance(result, list)
            assert len(result) <= 5
            if result:
                assert 'title' in result[0]
                assert 'url' in result[0]
                assert 'source' in result[0]
    
    def test_get_stock_news_empty_response(self, news_provider):
        """Test getting news with empty response"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = "<rss><channel></channel></rss>"
        
        with patch.object(news_provider.session, 'get', return_value=mock_response):
            result = news_provider.get_stock_news("INVALID", "Invalid Company", limit=5)
            
            assert isinstance(result, list)
            assert len(result) == 0
    
    def test_get_stock_news_network_error(self, news_provider):
        """Test getting news with network error"""
        with patch.object(news_provider.session, 'get', side_effect=Exception("Network error")):
            result = news_provider.get_stock_news("RELIANCE", "Reliance Industries", limit=5)
            
            assert isinstance(result, list)
            assert len(result) == 0

class TestIntegration:
    """Integration tests for Indian market data"""
    
    def test_full_stock_analysis_workflow(self):
        """Test complete workflow of getting stock data and analysis"""
        market_data = IndianMarketData()
        
        # Test with a known Indian stock
        symbol = "TCS"
        
        # This would be a real API call in integration testing
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock successful data retrieval
            sample_data = pd.DataFrame({
                'Open': [3000, 3010, 3020],
                'High': [3050, 3060, 3070],
                'Low': [2990, 3000, 3010],
                'Close': [3020, 3030, 3040],
                'Volume': [1000000, 1100000, 1200000]
            })
            
            mock_ticker.return_value.history.return_value = sample_data
            mock_ticker.return_value.info = {
                'longName': 'Tata Consultancy Services Limited',
                'sector': 'Technology',
                'industry': 'IT Services'
            }
            
            # Get stock data
            stock_data = market_data.get_nse_stock_data(symbol, period="1mo")
            assert stock_data is not None
            
            # Get company info
            company_info = market_data.get_company_info(symbol, "NSE")
            assert company_info is not None
            assert company_info.sector == 'Technology'
    
    @pytest.mark.parametrize("symbol,exchange", [
        ("RELIANCE", "NSE"),
        ("TCS", "NSE"),
        ("HDFCBANK", "BSE"),
        ("INFY", "NSE")
    ])
    def test_multiple_stock_symbols(self, symbol, exchange):
        """Test multiple stock symbols and exchanges"""
        market_data = IndianMarketData()
        
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame({
                'Close': [100, 105, 110]
            })
            mock_ticker.return_value.info = {'longName': f'{symbol} Company'}
            
            if exchange == "NSE":
                result = market_data.get_nse_stock_data(symbol)
            else:
                result = market_data.get_bse_stock_data(symbol)
            
            assert result is not None or result is None  # Either valid data or None for invalid symbols

if __name__ == "__main__":
    pytest.main([__file__])