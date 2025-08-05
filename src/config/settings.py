"""
Configuration settings for StoxChai - Indian Stock Market Analysis Tool
"""
import os
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "stoxchai")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")
    
@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    password: str = os.getenv("REDIS_PASSWORD", "")
    db: int = int(os.getenv("REDIS_DB", "0"))

@dataclass
class APIConfig:
    """External API configuration"""
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    nse_api_base: str = "https://www.nseindia.com/api"
    bse_api_base: str = "https://api.bseindia.com"
    yahoo_finance_base: str = "https://query1.finance.yahoo.com"
    
@dataclass
class OllamaConfig:
    """Ollama AI configuration"""
    host: str = os.getenv("OLLAMA_HOST", "localhost")
    port: int = int(os.getenv("OLLAMA_PORT", "11434"))
    default_model: str = os.getenv("OLLAMA_DEFAULT_MODEL", "qwen2.5:latest")
    available_models: List[str] = None
    
    def __post_init__(self):
        if self.available_models is None:
            self.available_models = [
                "qwen2.5:latest",
                "deepseek-r1:7b",
                "llama3.1:8b",
                "mistral:7b",
                "gemma2:9b"
            ]

class Settings:
    """Main application settings"""
    
    # Application Info
    APP_NAME = "StoxChai"
    APP_VERSION = "2.0.0"
    DESCRIPTION = "Advanced Indian Stock Market Analysis Tool"
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "true").lower() == "true"
    
    # Security
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "*").split(",")
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    CACHE_DIR = BASE_DIR / "cache"
    LOGS_DIR = BASE_DIR / "logs"
    STATIC_DIR = BASE_DIR / "static"
    
    # Indian Market Specific
    INDIAN_EXCHANGES = ["NSE", "BSE"]
    INDIAN_INDICES = {
        "NIFTY_50": "^NSEI",
        "NIFTY_BANK": "^NSEBANK",
        "SENSEX": "^BSESN",
        "NIFTY_IT": "^CNXIT",
        "NIFTY_PHARMA": "^CNXPHARMA",
        "NIFTY_AUTO": "^CNXAUTO",
        "NIFTY_FMCG": "^CNXFMCG",
        "NIFTY_METAL": "^CNXMETAL",
        "NIFTY_REALTY": "^CNXREALTY",
        "NIFTY_ENERGY": "^CNXENERGY"
    }
    
    MARKET_HOURS = {
        "PRE_OPEN": "09:00",
        "OPEN": "09:15",
        "CLOSE": "15:30",
        "POST_CLOSE": "16:00"
    }
    
    # Indian holidays (major trading holidays)
    TRADING_HOLIDAYS_2025 = [
        "2025-01-26",  # Republic Day
        "2025-03-14",  # Holi
        "2025-04-14",  # Ram Navami
        "2025-04-18",  # Good Friday
        "2025-05-01",  # Maharashtra Day
        "2025-08-15",  # Independence Day
        "2025-10-02",  # Gandhi Jayanti
        "2025-11-01",  # Diwali
        "2025-11-15",  # Guru Nanak Jayanti
        "2025-12-25"   # Christmas
    ]
    
    # Currency
    BASE_CURRENCY = "INR"
    CURRENCY_SYMBOL = "₹"
    
    # Data Sources Priority
    DATA_SOURCES_PRIORITY = [
        "yahoo_finance",
        "nse_api",
        "bse_api",
        "alpha_vantage"
    ]
    
    # Cache Settings
    CACHE_TTL = {
        "stock_data": 300,      # 5 minutes
        "news_data": 1800,      # 30 minutes  
        "company_info": 3600,   # 1 hour
        "market_status": 60,    # 1 minute
        "indices": 60           # 1 minute
    }
    
    # Streamlit Settings
    STREAMLIT_CONFIG = {
        "page_title": "StoxChai - Indian Stock Market Analysis",
        "page_icon": "📈",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Rate Limiting
    RATE_LIMITS = {
        "api_calls_per_minute": 60,
        "news_requests_per_hour": 100,
        "ai_queries_per_minute": 10
    }
    
    # Performance
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "30"))
    
    # Database configs
    database = DatabaseConfig()
    redis = RedisConfig()
    api = APIConfig()
    ollama = OllamaConfig()
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [cls.DATA_DIR, cls.CACHE_DIR, cls.LOGS_DIR, cls.STATIC_DIR]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def is_market_open(cls) -> bool:
        """Check if Indian market is currently open"""
        from datetime import datetime, time
        import pytz
        
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist)
        current_time = now.time()
        current_date = now.date().isoformat()
        
        # Check if it's a weekend
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
            
        # Check if it's a trading holiday
        if current_date in cls.TRADING_HOLIDAYS_2025:
            return False
            
        # Check market hours
        market_open = time.fromisoformat(cls.MARKET_HOURS["OPEN"])
        market_close = time.fromisoformat(cls.MARKET_HOURS["CLOSE"])
        
        return market_open <= current_time <= market_close
    
    @classmethod
    def get_indian_stock_symbol(cls, symbol: str, exchange: str = "NS") -> str:
        """Format stock symbol for Indian exchanges"""
        symbol = symbol.upper().strip()
        
        # If already has exchange suffix, return as is
        if "." in symbol:
            return symbol
            
        # Add appropriate exchange suffix
        if exchange.upper() == "NSE" or exchange.upper() == "NS":
            return f"{symbol}.NS"
        elif exchange.upper() == "BSE" or exchange.upper() == "BO":
            return f"{symbol}.BO"
        else:
            return f"{symbol}.NS"  # Default to NSE

# Create global settings instance
settings = Settings()

# Create directories on import
settings.create_directories()