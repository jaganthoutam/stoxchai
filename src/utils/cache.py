"""
Caching system for StoxChai
"""
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Union
from pathlib import Path

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from src.config import settings
from src.utils.logger import cache_logger

class CacheManager:
    """Centralized cache management"""
    
    def __init__(self):
        self.redis_client = None
        self.file_cache_dir = settings.CACHE_DIR / "file_cache"
        self.file_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis if available and configured
        if REDIS_AVAILABLE and settings.redis.host:
            try:
                self.redis_client = redis.Redis(
                    host=settings.redis.host,
                    port=settings.redis.port,
                    password=settings.redis.password,
                    db=settings.redis.db,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                cache_logger.info("Redis cache initialized successfully")
            except Exception as e:
                cache_logger.warning(f"Redis connection failed: {str(e)}. Falling back to file cache.")
                self.redis_client = None
    
    def _generate_key(self, key: str, prefix: str = "stoxchai") -> str:
        """Generate cache key with prefix"""
        return f"{prefix}:{key}"
    
    def _generate_file_path(self, key: str) -> Path:
        """Generate file path for cache key"""
        # Create hash to avoid filesystem issues with special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.file_cache_dir / f"{key_hash}.cache"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, prefix: str = "stoxchai") -> bool:
        """Set cache value"""
        try:
            cache_key = self._generate_key(key, prefix)
            
            # Try Redis first
            if self.redis_client:
                serialized_value = json.dumps(value, default=str)
                if ttl:
                    result = self.redis_client.setex(cache_key, ttl, serialized_value)
                else:
                    result = self.redis_client.set(cache_key, serialized_value)
                
                if result:
                    cache_logger.debug(f"Cached to Redis: {cache_key}")
                    return True
            
            # Fallback to file cache
            cache_data = {
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'ttl': ttl
            }
            
            file_path = self._generate_file_path(cache_key)
            with open(file_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            cache_logger.debug(f"Cached to file: {cache_key}")
            return True
            
        except Exception as e:
            cache_logger.error(f"Error setting cache for {key}: {str(e)}")
            return False
    
    def get(self, key: str, prefix: str = "stoxchai") -> Optional[Any]:
        """Get cache value"""
        try:
            cache_key = self._generate_key(key, prefix)
            
            # Try Redis first
            if self.redis_client:
                value = self.redis_client.get(cache_key)
                if value:
                    try:
                        deserialized_value = json.loads(value)
                        cache_logger.debug(f"Cache hit (Redis): {cache_key}")
                        return deserialized_value
                    except json.JSONDecodeError:
                        cache_logger.warning(f"Failed to deserialize Redis cache: {cache_key}")
            
            # Fallback to file cache
            file_path = self._generate_file_path(cache_key)
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # Check TTL
                    if cache_data.get('ttl'):
                        timestamp = datetime.fromisoformat(cache_data['timestamp'])
                        if datetime.now() - timestamp > timedelta(seconds=cache_data['ttl']):
                            # Cache expired
                            file_path.unlink()
                            cache_logger.debug(f"Cache expired: {cache_key}")
                            return None
                    
                    cache_logger.debug(f"Cache hit (File): {cache_key}")
                    return cache_data['value']
                    
                except Exception as e:
                    cache_logger.warning(f"Failed to load file cache {cache_key}: {str(e)}")
                    # Remove corrupted cache file
                    file_path.unlink(missing_ok=True)
            
            cache_logger.debug(f"Cache miss: {cache_key}")
            return None
            
        except Exception as e:
            cache_logger.error(f"Error getting cache for {key}: {str(e)}")
            return None
    
    def delete(self, key: str, prefix: str = "stoxchai") -> bool:
        """Delete cache value"""
        try:
            cache_key = self._generate_key(key, prefix)
            
            # Delete from Redis
            if self.redis_client:
                self.redis_client.delete(cache_key)
            
            # Delete file cache
            file_path = self._generate_file_path(cache_key)
            file_path.unlink(missing_ok=True)
            
            cache_logger.debug(f"Cache deleted: {cache_key}")
            return True
            
        except Exception as e:
            cache_logger.error(f"Error deleting cache for {key}: {str(e)}")
            return False
    
    def clear_all(self, prefix: str = "stoxchai") -> bool:
        """Clear all cache with prefix"""
        try:
            # Clear Redis cache
            if self.redis_client:
                keys = self.redis_client.keys(f"{prefix}:*")
                if keys:
                    self.redis_client.delete(*keys)
            
            # Clear file cache
            for cache_file in self.file_cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            cache_logger.info(f"Cleared all cache with prefix: {prefix}")
            return True
            
        except Exception as e:
            cache_logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            'redis_available': self.redis_client is not None,
            'file_cache_files': len(list(self.file_cache_dir.glob("*.cache"))),
            'cache_dir_size': sum(f.stat().st_size for f in self.file_cache_dir.glob("*.cache"))
        }
        
        if self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats['redis_memory_used'] = redis_info.get('used_memory_human', 'N/A')
                stats['redis_keys'] = self.redis_client.dbsize()
            except Exception as e:
                cache_logger.warning(f"Error getting Redis stats: {str(e)}")
        
        return stats

class CacheDecorator:
    """Decorator for caching function results"""
    
    def __init__(self, ttl: int = 300, prefix: str = "func", key_func=None):
        self.ttl = ttl
        self.prefix = prefix
        self.key_func = key_func
        self.cache = cache_manager
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if self.key_func:
                cache_key = self.key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = self.cache.get(cache_key, self.prefix)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.cache.set(cache_key, result, self.ttl, self.prefix)
            
            return result
        
        return wrapper

# Specific cache decorators for common use cases
def cache_stock_data(ttl: int = None):
    """Cache decorator for stock data"""
    if ttl is None:
        ttl = settings.CACHE_TTL['stock_data']
    return CacheDecorator(ttl=ttl, prefix="stock_data")

def cache_news_data(ttl: int = None):
    """Cache decorator for news data"""
    if ttl is None:
        ttl = settings.CACHE_TTL['news_data']
    return CacheDecorator(ttl=ttl, prefix="news_data")

def cache_company_info(ttl: int = None):
    """Cache decorator for company info"""
    if ttl is None:
        ttl = settings.CACHE_TTL['company_info']
    return CacheDecorator(ttl=ttl, prefix="company_info")

# Global cache manager instance
cache_manager = CacheManager()

# Cache helper functions
def cache_stock_price(symbol: str, data: Any, ttl: int = None) -> bool:
    """Cache stock price data"""
    if ttl is None:
        ttl = settings.CACHE_TTL['stock_data']
    return cache_manager.set(f"price:{symbol}", data, ttl, "stock")

def get_cached_stock_price(symbol: str) -> Optional[Any]:
    """Get cached stock price data"""
    return cache_manager.get(f"price:{symbol}", "stock")

def cache_market_status(status: str, ttl: int = None) -> bool:
    """Cache market status"""
    if ttl is None:
        ttl = settings.CACHE_TTL['market_status']
    return cache_manager.set("market_status", status, ttl, "market")

def get_cached_market_status() -> Optional[str]:
    """Get cached market status"""
    return cache_manager.get("market_status", "market")

def cache_indices_data(data: Any, ttl: int = None) -> bool:
    """Cache indices data"""
    if ttl is None:
        ttl = settings.CACHE_TTL['indices']
    return cache_manager.set("indices", data, ttl, "market")

def get_cached_indices_data() -> Optional[Any]:
    """Get cached indices data"""
    return cache_manager.get("indices", "market")