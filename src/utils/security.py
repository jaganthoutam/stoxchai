"""
Security utilities and input validation for StoxChai
"""
import re
import hashlib
import secrets
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from functools import wraps
import streamlit as st

from src.config import settings
from src.utils.logger import app_logger

class InputValidator:
    """Input validation utilities"""
    
    # Regex patterns
    STOCK_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9]{1,10}(\.(NS|BO))?$')
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    
    @staticmethod
    def validate_stock_symbol(symbol: str) -> bool:
        """Validate stock symbol format"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        symbol = symbol.strip().upper()
        return bool(InputValidator.STOCK_SYMBOL_PATTERN.match(symbol))
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 100) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\';\\]', '', input_str)
        
        # Limit length
        sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> bool:
        """Validate date range"""
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            # Check if start is before end
            if start >= end:
                return False
            
            # Check if dates are not too far in the future
            max_future_date = datetime.now() + timedelta(days=365)
            if start > max_future_date or end > max_future_date:
                return False
            
            # Check if dates are not too far in the past (10 years)
            min_past_date = datetime.now() - timedelta(days=3650)
            if start < min_past_date:
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_numeric_input(value: Any, min_val: float = None, max_val: float = None) -> bool:
        """Validate numeric input"""
        try:
            num_value = float(value)
            
            if min_val is not None and num_value < min_val:
                return False
            
            if max_val is not None and num_value > max_val:
                return False
            
            return True
            
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_period(period: str) -> bool:
        """Validate time period input"""
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max']
        return period in valid_periods

class RateLimiter:
    """Rate limiting for API calls and user actions"""
    
    def __init__(self):
        self.request_counts = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def _cleanup_old_entries(self):
        """Remove old entries from rate limit tracking"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            cutoff_time = current_time - 3600  # 1 hour ago
            
            keys_to_remove = []
            for key, timestamps in self.request_counts.items():
                # Remove timestamps older than 1 hour
                self.request_counts[key] = [ts for ts in timestamps if ts > cutoff_time]
                
                # Remove empty entries
                if not self.request_counts[key]:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.request_counts[key]
            
            self.last_cleanup = current_time
    
    def is_rate_limited(self, identifier: str, limit: int, window: int = 3600) -> bool:
        """Check if identifier is rate limited"""
        self._cleanup_old_entries()
        
        current_time = time.time()
        cutoff_time = current_time - window
        
        # Get or create request history for this identifier
        if identifier not in self.request_counts:
            self.request_counts[identifier] = []
        
        # Remove old requests
        self.request_counts[identifier] = [
            ts for ts in self.request_counts[identifier] if ts > cutoff_time
        ]
        
        # Check if limit exceeded
        return len(self.request_counts[identifier]) >= limit
    
    def record_request(self, identifier: str):
        """Record a new request"""
        if identifier not in self.request_counts:
            self.request_counts[identifier] = []
        
        self.request_counts[identifier].append(time.time())

class SecurityManager:
    """Main security manager"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.failed_attempts = {}
    
    def generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        pwdhash = hashlib.pbkdf2_hmac('sha256', 
                                     password.encode('utf-8'), 
                                     salt.encode('utf-8'), 
                                     100000)
        return salt + pwdhash.hex()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        salt = hashed[:32]
        stored_hash = hashed[32:]
        
        pwdhash = hashlib.pbkdf2_hmac('sha256',
                                     password.encode('utf-8'),
                                     salt.encode('utf-8'),
                                     100000)
        
        return pwdhash.hex() == stored_hash
    
    def check_rate_limit(self, identifier: str, action: str) -> bool:
        """Check if action is rate limited"""
        limits = settings.RATE_LIMITS
        
        if action == "api_calls":
            return self.rate_limiter.is_rate_limited(
                f"api:{identifier}", 
                limits["api_calls_per_minute"], 
                60
            )
        elif action == "news_requests":
            return self.rate_limiter.is_rate_limited(
                f"news:{identifier}", 
                limits["news_requests_per_hour"], 
                3600
            )
        elif action == "ai_queries":
            return self.rate_limiter.is_rate_limited(
                f"ai:{identifier}", 
                limits["ai_queries_per_minute"], 
                60
            )
        
        return False
    
    def record_action(self, identifier: str, action: str):
        """Record an action for rate limiting"""
        if action == "api_calls":
            self.rate_limiter.record_request(f"api:{identifier}")
        elif action == "news_requests":
            self.rate_limiter.record_request(f"news:{identifier}")
        elif action == "ai_queries":
            self.rate_limiter.record_request(f"ai:{identifier}")
    
    def sanitize_user_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize all user inputs"""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = InputValidator.sanitize_string(value)
            elif isinstance(value, (int, float)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = [
                    InputValidator.sanitize_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized

def rate_limit(action: str, identifier_func=None):
    """Decorator for rate limiting functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier (default to session state or IP)
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                # Use Streamlit session state as identifier
                identifier = getattr(st.session_state, 'session_id', 'anonymous')
            
            if security_manager.check_rate_limit(identifier, action):
                st.error(f"Rate limit exceeded for {action}. Please try again later.")
                app_logger.warning(f"Rate limit exceeded for {identifier} on {action}")
                return None
            
            # Record the action
            security_manager.record_action(identifier, action)
            
            # Execute the function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def validate_input(validation_rules: Dict[str, Any]):
    """Decorator for input validation"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate arguments based on rules
            for i, (param_name, rule) in enumerate(validation_rules.items()):
                if i < len(args):
                    value = args[i]
                elif param_name in kwargs:
                    value = kwargs[param_name]
                else:
                    continue
                
                # Apply validation rule
                if rule == 'stock_symbol':
                    if not InputValidator.validate_stock_symbol(value):
                        st.error(f"Invalid stock symbol: {value}")
                        app_logger.warning(f"Invalid stock symbol validation: {value}")
                        return None
                elif rule == 'period':
                    if not InputValidator.validate_period(value):
                        st.error(f"Invalid period: {value}")
                        app_logger.warning(f"Invalid period validation: {value}")
                        return None
                elif isinstance(rule, dict) and 'type' in rule:
                    if rule['type'] == 'numeric':
                        min_val = rule.get('min')
                        max_val = rule.get('max')
                        if not InputValidator.validate_numeric_input(value, min_val, max_val):
                            st.error(f"Invalid numeric value: {value}")
                            app_logger.warning(f"Invalid numeric validation: {value}")
                            return None
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def secure_api_call(api_name: str):
    """Decorator for securing API calls"""
    def decorator(func):
        @wraps(func)
        @rate_limit("api_calls")
        def wrapper(*args, **kwargs):
            try:
                app_logger.info(f"Secure API call: {api_name}")
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                app_logger.error(f"API call failed: {api_name} - {str(e)}")
                raise
        
        return wrapper
    return decorator

class SecurityHeadersMiddleware:
    """Security headers for web application"""
    
    @staticmethod
    def apply_security_headers():
        """Apply security headers to Streamlit app"""
        # This would be applied at the server level in production
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
        
        # In Streamlit, we can only add some headers via HTML
        csp_meta = """
        <meta http-equiv="Content-Security-Policy" 
              content="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';">
        """
        st.markdown(csp_meta, unsafe_allow_html=True)

def get_client_ip() -> str:
    """Get client IP address for rate limiting"""
    # In production, this would get the real client IP
    # For Streamlit, we'll use a session-based identifier
    if 'client_id' not in st.session_state:
        st.session_state.client_id = secrets.token_hex(8)
    
    return st.session_state.client_id

def initialize_session_security():
    """Initialize security for the session"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = SecurityManager().generate_session_token()
    
    if 'session_start' not in st.session_state:
        st.session_state.session_start = datetime.now()
    
    # Session timeout (4 hours)
    session_duration = datetime.now() - st.session_state.session_start
    if session_duration > timedelta(hours=4):
        st.session_state.clear()
        st.rerun()

# Global security manager instance
security_manager = SecurityManager()
input_validator = InputValidator()

# Apply security headers
SecurityHeadersMiddleware.apply_security_headers()