"""
Centralized logging system for StoxChai
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from src.config import settings

class StoxChaiLogger:
    """Custom logger for StoxChai application"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger instance"""
        if name not in cls._loggers:
            cls._loggers[name] = cls._create_logger(name)
        return cls._loggers[name]
    
    @classmethod
    def _create_logger(cls, name: str) -> logging.Logger:
        """Create a new logger instance"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatter
        formatter = logging.Formatter(settings.LOG_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler for application logs
        log_file = settings.LOGS_DIR / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Error file handler
        error_log_file = settings.LOGS_DIR / f"{name}_error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)
        
        return logger

def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = StoxChaiLogger.get_logger("function_calls")
            logger.info(f"Calling {func_name} with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {str(e)}")
                raise
        return wrapper
    return decorator

def log_api_call(api_name: str, endpoint: str):
    """Decorator to log API calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = StoxChaiLogger.get_logger("api_calls")
            logger.info(f"API Call: {api_name} - {endpoint}")
            try:
                result = func(*args, **kwargs)
                logger.info(f"API Call successful: {api_name} - {endpoint}")
                return result
            except Exception as e:
                logger.error(f"API Call failed: {api_name} - {endpoint} - Error: {str(e)}")
                raise
        return wrapper
    return decorator

# Create default loggers
app_logger = StoxChaiLogger.get_logger("stoxchai")
data_logger = StoxChaiLogger.get_logger("data")
ui_logger = StoxChaiLogger.get_logger("ui")
api_logger = StoxChaiLogger.get_logger("api")
cache_logger = StoxChaiLogger.get_logger("cache")
error_logger = StoxChaiLogger.get_logger("error")