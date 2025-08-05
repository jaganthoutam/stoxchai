"""
Dynamic User Agent management to avoid blocking from data sources
"""
import random
import time
from typing import List, Dict
from datetime import datetime, timedelta

class UserAgentRotator:
    """Rotate user agents to avoid detection and blocking"""
    
    def __init__(self):
        self.user_agents = [
            # Chrome on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
            
            # Chrome on macOS
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            
            # Chrome on Linux
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            
            # Firefox on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            
            # Firefox on macOS
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0',
            
            # Safari on macOS
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            
            # Edge on Windows
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
            
            # Mobile User Agents (Android)
            'Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
            'Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36',
            
            # Mobile User Agents (iOS)
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (iPad; CPU OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
            
            # Financial data specific user agents
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 FinanceBot/1.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 MarketData/2.0',
        ]
        
        self.last_used = {}
        self.usage_count = {}
        self.cooldown = 300  # 5 minutes cooldown per user agent
        
    def get_random_user_agent(self) -> str:
        """Get a random user agent"""
        return random.choice(self.user_agents)
    
    def get_rotated_user_agent(self) -> str:
        """Get a user agent with rotation logic to avoid overuse"""
        current_time = datetime.now()
        available_agents = []
        
        for agent in self.user_agents:
            last_used = self.last_used.get(agent, datetime.min)
            if current_time - last_used > timedelta(seconds=self.cooldown):
                available_agents.append(agent)
        
        # If no agents are available, use the least recently used one
        if not available_agents:
            agent = min(self.user_agents, key=lambda x: self.last_used.get(x, datetime.min))
        else:
            agent = random.choice(available_agents)
        
        # Update usage tracking
        self.last_used[agent] = current_time
        self.usage_count[agent] = self.usage_count.get(agent, 0) + 1
        
        return agent
    
    def get_headers(self, referer: str = None, accept_language: str = "en-US,en;q=0.9") -> Dict[str, str]:
        """Get complete headers with rotated user agent"""
        headers = {
            'User-Agent': self.get_rotated_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': accept_language,
            'Accept-Encoding': 'gzip, deflate, br, zstd',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        }
        
        if referer:
            headers['Referer'] = referer
        
        return headers
    
    def get_indian_market_headers(self) -> Dict[str, str]:
        """Get headers optimized for Indian market data sources"""
        return self.get_headers(
            referer="https://finance.yahoo.com/",
            accept_language="en-IN,en;q=0.9,hi;q=0.8"
        )
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for monitoring"""
        return self.usage_count.copy()

class RequestHandler:
    """Enhanced request handler with retry logic and rate limiting"""
    
    def __init__(self):
        self.user_agent_rotator = UserAgentRotator()
        self.request_history = []
        self.min_request_interval = 1.0  # Minimum 1 second between requests
        self.max_retries = 3
        self.retry_delays = [1, 3, 5]  # Exponential backoff
        
    def should_delay_request(self) -> float:
        """Check if we should delay the request and return delay time"""
        if not self.request_history:
            return 0
        
        last_request_time = self.request_history[-1]
        time_since_last = time.time() - last_request_time
        
        if time_since_last < self.min_request_interval:
            return self.min_request_interval - time_since_last
        
        return 0
    
    def make_request(self, url: str, session=None, **kwargs) -> 'requests.Response':
        """Make a request with dynamic user agent and retry logic"""
        import requests
        
        if session is None:
            session = requests.Session()
        
        # Apply delay if needed
        delay = self.should_delay_request()
        if delay > 0:
            time.sleep(delay)
        
        # Get headers with rotated user agent
        headers = self.user_agent_rotator.get_indian_market_headers()
        
        # Merge with any provided headers
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        kwargs['headers'] = headers
        
        # Set default timeout
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 10
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                # Record request time
                self.request_history.append(time.time())
                
                # Keep only last 100 requests in history
                if len(self.request_history) > 100:
                    self.request_history = self.request_history[-100:]
                
                response = session.get(url, **kwargs)
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', self.retry_delays[attempt]))
                    time.sleep(retry_after)
                    continue
                
                # Check for other client errors
                if response.status_code >= 400:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delays[attempt])
                        # Rotate user agent for next attempt
                        headers = self.user_agent_rotator.get_indian_market_headers()
                        kwargs['headers'] = headers
                        continue
                
                return response
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt])
                    # Rotate user agent for next attempt
                    headers = self.user_agent_rotator.get_indian_market_headers()
                    kwargs['headers'] = headers
                    continue
        
        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise requests.exceptions.RequestException(f"Failed to fetch {url} after {self.max_retries} attempts")

# Global instances
user_agent_rotator = UserAgentRotator()
request_handler = RequestHandler()

# Convenience functions
def get_dynamic_headers() -> Dict[str, str]:
    """Get headers with dynamic user agent for Indian market"""
    return user_agent_rotator.get_indian_market_headers()

def make_safe_request(url: str, session=None, **kwargs) -> 'requests.Response':
    """Make a safe request with automatic retry and user agent rotation"""
    return request_handler.make_request(url, session, **kwargs)