"""
Core data models for StoxChai
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum
import pandas as pd

class Exchange(Enum):
    """Indian stock exchanges"""
    NSE = "NSE"
    BSE = "BSE"

class MarketStatus(Enum):
    """Market status enumeration"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PRE_OPEN = "PRE_OPEN"
    POST_CLOSE = "POST_CLOSE"

class SentimentLabel(Enum):
    """News sentiment labels"""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

@dataclass
class StockPrice:
    """Stock price data model"""
    open: float
    high: float
    low: float
    close: float
    volume: int
    date: datetime
    adjusted_close: Optional[float] = None
    
    @property
    def change(self) -> float:
        """Calculate absolute change"""
        return self.close - self.open
    
    @property
    def change_percent(self) -> float:
        """Calculate percentage change"""
        if self.open == 0:
            return 0
        return ((self.close - self.open) / self.open) * 100

@dataclass
class CompanyInfo:
    """Company information model"""
    symbol: str
    name: str
    exchange: Exchange
    sector: Optional[str] = None
    industry: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = None
    book_value: Optional[float] = None
    price_to_book: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    avg_volume: Optional[int] = None
    business_summary: Optional[str] = None
    employees: Optional[int] = None
    founded: Optional[str] = None
    headquarters: Optional[str] = None
    website: Optional[str] = None
    
@dataclass
class NewsArticle:
    """News article model"""
    title: str
    url: str
    source: str
    published_date: datetime
    content: str
    sentiment_score: float
    sentiment_label: SentimentLabel
    relevance_score: Optional[float] = None
    summary: Optional[str] = None
    
    @property
    def is_positive(self) -> bool:
        return self.sentiment_label == SentimentLabel.POSITIVE
    
    @property
    def is_negative(self) -> bool:
        return self.sentiment_label == SentimentLabel.NEGATIVE

@dataclass
class MarketIndex:
    """Market index data model"""
    name: str
    symbol: str
    current_value: float
    change: float
    change_percent: float
    last_updated: datetime
    
@dataclass
class TechnicalIndicator:
    """Technical indicator data model"""
    name: str
    value: float
    signal: str  # BUY, SELL, HOLD
    description: str
    
@dataclass
class StockAnalysis:
    """Complete stock analysis model"""
    symbol: str
    company_info: CompanyInfo
    current_price: StockPrice
    historical_data: pd.DataFrame
    news_articles: List[NewsArticle]
    technical_indicators: List[TechnicalIndicator]
    ai_analysis: Optional[str] = None
    recommendation: Optional[str] = None
    target_price: Optional[float] = None
    risk_level: Optional[str] = None
    
    @property
    def avg_sentiment(self) -> float:
        """Calculate average sentiment from news articles"""
        if not self.news_articles:
            return 0.0
        return sum(article.sentiment_score for article in self.news_articles) / len(self.news_articles)
    
    @property
    def sentiment_distribution(self) -> Dict[str, int]:
        """Get sentiment distribution"""
        distribution = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        for article in self.news_articles:
            distribution[article.sentiment_label.value] += 1
        return distribution

@dataclass
class Portfolio:
    """User portfolio model"""
    user_id: str
    stocks: List[Dict[str, Union[str, float, int]]]
    total_value: float
    total_investment: float
    profit_loss: float
    profit_loss_percent: float
    last_updated: datetime
    
    @property
    def is_profitable(self) -> bool:
        return self.profit_loss > 0

@dataclass
class Alert:
    """Price alert model"""
    id: str
    user_id: str
    symbol: str
    alert_type: str  # PRICE_ABOVE, PRICE_BELOW, VOLUME_SPIKE, etc.
    threshold: float
    is_active: bool
    created_at: datetime
    triggered_at: Optional[datetime] = None

@dataclass
class MarketSummary:
    """Daily market summary"""
    date: datetime
    indices: List[MarketIndex]
    top_gainers: List[Dict[str, Union[str, float]]]
    top_losers: List[Dict[str, Union[str, float]]]
    most_active: List[Dict[str, Union[str, float]]]
    market_mood: str  # BULLISH, BEARISH, NEUTRAL
    total_volume: int
    advances: int
    declines: int
    unchanged: int

@dataclass
class APIResponse:
    """Standard API response model"""
    success: bool
    data: Optional[Dict] = None
    message: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

class StockValidator:
    """Validator for stock symbols and data"""
    
    @staticmethod
    def is_valid_indian_symbol(symbol: str) -> bool:
        """Validate Indian stock symbol format"""
        if not symbol:
            return False
        
        symbol = symbol.upper().strip()
        
        # Check for NSE format (SYMBOL.NS)
        if symbol.endswith('.NS'):
            base_symbol = symbol[:-3]
            return base_symbol.isalnum() and 1 <= len(base_symbol) <= 10
        
        # Check for BSE format (SYMBOL.BO)
        if symbol.endswith('.BO'):
            base_symbol = symbol[:-3]
            return base_symbol.isalnum() and 1 <= len(base_symbol) <= 10
        
        # Plain symbol (assume NSE)
        return symbol.isalnum() and 1 <= len(symbol) <= 10
    
    @staticmethod
    def normalize_symbol(symbol: str, exchange: str = "NSE") -> str:
        """Normalize stock symbol for Indian markets"""
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        
        symbol = symbol.upper().strip()
        
        # If already has exchange suffix, return as is
        if '.' in symbol:
            return symbol
        
        # Add appropriate suffix
        if exchange.upper() in ['NSE', 'NS']:
            return f"{symbol}.NS"
        elif exchange.upper() in ['BSE', 'BO']:
            return f"{symbol}.BO"
        else:
            return f"{symbol}.NS"  # Default to NSE