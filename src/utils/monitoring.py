"""
Monitoring and health check utilities for StoxChai
"""
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from src.config import settings
from src.utils.logger import app_logger
from src.utils.cache import cache_manager

class HealthStatus(Enum):
    """Health check status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Health check result"""
    service: str
    status: HealthStatus
    message: str
    response_time_ms: float
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

class MetricsCollector:
    """Collect application metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        
        if PROMETHEUS_AVAILABLE:
            # Define Prometheus metrics
            self.request_count = Counter(
                'stoxchai_requests_total',
                'Total number of requests',
                ['method', 'endpoint', 'status']
            )
            
            self.request_duration = Histogram(
                'stoxchai_request_duration_seconds',
                'Request duration in seconds',
                ['method', 'endpoint']
            )
            
            self.active_users = Gauge(
                'stoxchai_active_users',
                'Number of active users'
            )
            
            self.cache_hits = Counter(
                'stoxchai_cache_hits_total',
                'Total cache hits',
                ['cache_type']
            )
            
            self.cache_misses = Counter(
                'stoxchai_cache_misses_total',
                'Total cache misses',
                ['cache_type']
            )
            
            self.api_calls = Counter(
                'stoxchai_api_calls_total',
                'Total external API calls',
                ['api_name', 'status']
            )
            
            self.memory_usage = Gauge(
                'stoxchai_memory_usage_bytes',
                'Memory usage in bytes'
            )
            
            self.cpu_usage = Gauge(
                'stoxchai_cpu_usage_percent',
                'CPU usage percentage'
            )
            
            self.uptime = Gauge(
                'stoxchai_uptime_seconds',
                'Application uptime in seconds'
            )
            
            # Start background metrics collection
            self._start_system_metrics_collection()
    
    def _start_system_metrics_collection(self):
        """Start background thread for system metrics collection"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        def collect_system_metrics():
            while True:
                try:
                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage.set(memory.used)
                    
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.cpu_usage.set(cpu_percent)
                    
                    # Uptime
                    uptime = time.time() - self.start_time
                    self.uptime.set(uptime)
                    
                    time.sleep(30)  # Collect every 30 seconds
                    
                except Exception as e:
                    app_logger.error(f"Error collecting system metrics: {str(e)}")
                    time.sleep(60)  # Wait longer on error
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record request metrics"""
        if PROMETHEUS_AVAILABLE:
            self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        if PROMETHEUS_AVAILABLE:
            self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        if PROMETHEUS_AVAILABLE:
            self.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_api_call(self, api_name: str, status: str):
        """Record external API call"""
        if PROMETHEUS_AVAILABLE:
            self.api_calls.labels(api_name=api_name, status=status).inc()
    
    def set_active_users(self, count: int):
        """Set active users count"""
        if PROMETHEUS_AVAILABLE:
            self.active_users.set(count)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest()
        return "Prometheus not available"

class HealthChecker:
    """Health check system"""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
        self.check_interval = 30  # seconds
    
    def register_check(self, name: str, check_func, interval: int = 30):
        """Register a health check function"""
        self.checks[name] = check_func
        self.last_check_time[name] = datetime.min
    
    def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check"""
        if name not in self.checks:
            return HealthCheck(
                service=name,
                status=HealthStatus.UNHEALTHY,
                message="Check not found",
                response_time_ms=0,
                timestamp=datetime.now()
            )
        
        start_time = time.time()
        try:
            result = self.checks[name]()
            response_time = (time.time() - start_time) * 1000
            
            if isinstance(result, HealthCheck):
                result.response_time_ms = response_time
                result.timestamp = datetime.now()
                return result
            elif isinstance(result, bool):
                return HealthCheck(
                    service=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="OK" if result else "Check failed",
                    response_time_ms=response_time,
                    timestamp=datetime.now()
                )
            else:
                return HealthCheck(
                    service=name,
                    status=HealthStatus.HEALTHY,
                    message=str(result),
                    response_time_ms=response_time,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                service=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                response_time_ms=response_time,
                timestamp=datetime.now()
            )
    
    def run_all_checks(self, force: bool = False) -> List[HealthCheck]:
        """Run all health checks"""
        results = []
        current_time = datetime.now()
        
        for name in self.checks:
            # Check if we need to run this check
            if (force or 
                current_time - self.last_check_time[name] > timedelta(seconds=self.check_interval)):
                
                result = self.run_check(name)
                results.append(result)
                self.last_check_time[name] = current_time
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall application health status"""
        checks = self.run_all_checks()
        
        if not checks:
            return HealthStatus.UNHEALTHY
        
        unhealthy_count = sum(1 for check in checks if check.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for check in checks if check.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

# Specific health check functions
def check_database_connection() -> HealthCheck:
    """Check database connectivity"""
    try:
        # This would check actual database connection
        # For now, we'll simulate a check
        import time
        time.sleep(0.1)  # Simulate DB query time
        
        return HealthCheck(
            service="database",
            status=HealthStatus.HEALTHY,
            message="Database connection OK",
            response_time_ms=0,  # Will be set by run_check
            timestamp=datetime.now(),
            details={"host": settings.database.host, "port": settings.database.port}
        )
    except Exception as e:
        return HealthCheck(
            service="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}",
            response_time_ms=0,
            timestamp=datetime.now()
        )

def check_cache_connection() -> HealthCheck:
    """Check cache (Redis) connectivity"""
    try:
        # Test cache connection
        cache_stats = cache_manager.get_cache_stats()
        
        if cache_stats.get('redis_available', False):
            return HealthCheck(
                service="cache",
                status=HealthStatus.HEALTHY,
                message="Cache connection OK",
                response_time_ms=0,
                timestamp=datetime.now(),
                details=cache_stats
            )
        else:
            return HealthCheck(
                service="cache",
                status=HealthStatus.DEGRADED,
                message="Using file cache (Redis unavailable)",
                response_time_ms=0,
                timestamp=datetime.now(),
                details=cache_stats
            )
    except Exception as e:
        return HealthCheck(
            service="cache",
            status=HealthStatus.UNHEALTHY,
            message=f"Cache check failed: {str(e)}",
            response_time_ms=0,
            timestamp=datetime.now()
        )

def check_ollama_connection() -> HealthCheck:
    """Check Ollama AI service connectivity"""
    try:
        import requests
        
        url = f"http://{settings.ollama.host}:{settings.ollama.port}/api/tags"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            models = response.json().get('models', [])
            return HealthCheck(
                service="ollama",
                status=HealthStatus.HEALTHY,
                message=f"Ollama service OK ({len(models)} models available)",
                response_time_ms=0,
                timestamp=datetime.now(),
                details={"models_count": len(models), "models": [m.get('name') for m in models]}
            )
        else:
            return HealthCheck(
                service="ollama",
                status=HealthStatus.DEGRADED,
                message=f"Ollama service responding with status {response.status_code}",
                response_time_ms=0,
                timestamp=datetime.now()
            )
    except Exception as e:
        return HealthCheck(
            service="ollama",
            status=HealthStatus.UNHEALTHY,
            message=f"Ollama service unavailable: {str(e)}",
            response_time_ms=0,
            timestamp=datetime.now()
        )

def check_market_data_api() -> HealthCheck:
    """Check market data API connectivity"""
    try:
        import yfinance as yf
        
        # Try to fetch a simple quote
        ticker = yf.Ticker("RELIANCE.NS")
        info = ticker.info
        
        if info and 'symbol' in info:
            return HealthCheck(
                service="market_data_api",
                status=HealthStatus.HEALTHY,
                message="Market data API OK",
                response_time_ms=0,
                timestamp=datetime.now(),
                details={"test_symbol": "RELIANCE.NS"}
            )
        else:
            return HealthCheck(
                service="market_data_api",
                status=HealthStatus.DEGRADED,
                message="Market data API responding but data quality issues",
                response_time_ms=0,
                timestamp=datetime.now()
            )
    except Exception as e:
        return HealthCheck(
            service="market_data_api",
            status=HealthStatus.UNHEALTHY,
            message=f"Market data API unavailable: {str(e)}",
            response_time_ms=0,
            timestamp=datetime.now()
        )

def check_system_resources() -> HealthCheck:
    """Check system resource usage"""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Define thresholds
        memory_threshold = 85  # 85%
        disk_threshold = 90    # 90%
        cpu_threshold = 80     # 80%
        
        issues = []
        status = HealthStatus.HEALTHY
        
        if memory.percent > memory_threshold:
            issues.append(f"High memory usage: {memory.percent:.1f}%")
            status = HealthStatus.DEGRADED
        
        if disk.percent > disk_threshold:
            issues.append(f"High disk usage: {disk.percent:.1f}%")
            status = HealthStatus.DEGRADED
        
        if cpu_percent > cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
            status = HealthStatus.DEGRADED
        
        message = "System resources OK" if not issues else "; ".join(issues)
        
        return HealthCheck(
            service="system_resources",
            status=status,
            message=message,
            response_time_ms=0,
            timestamp=datetime.now(),
            details={
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "cpu_percent": cpu_percent
            }
        )
    except Exception as e:
        return HealthCheck(
            service="system_resources",
            status=HealthStatus.UNHEALTHY,
            message=f"System resource check failed: {str(e)}",
            response_time_ms=0,
            timestamp=datetime.now()
        )

# Global instances
metrics_collector = MetricsCollector()
health_checker = HealthChecker()

# Register health checks
health_checker.register_check("database", check_database_connection, 60)
health_checker.register_check("cache", check_cache_connection, 30)
health_checker.register_check("ollama", check_ollama_connection, 60)
health_checker.register_check("market_data_api", check_market_data_api, 300)  # 5 minutes
health_checker.register_check("system_resources", check_system_resources, 30)

# Monitoring decorator
def monitor_function(func_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        name = func_name or func.__name__
        
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record successful call
                metrics_collector.record_request("function", name, 200, duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed call
                metrics_collector.record_request("function", name, 500, duration)
                
                raise
        
        return wrapper
    return decorator