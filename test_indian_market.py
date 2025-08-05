#!/usr/bin/env python3
"""
Test script to verify enhanced Indian market data implementation
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import time
from datetime import datetime
import pandas as pd

# Import our enhanced modules
from src.data.indian_market import IndianMarketData, IndianNewsProvider
from src.utils.user_agents import user_agent_rotator
from src.config import settings

def test_user_agent_rotation():
    """Test dynamic user agent rotation"""
    print("🔄 Testing User Agent Rotation...")
    
    # Test multiple user agents
    agents = []
    for i in range(10):
        agent = user_agent_rotator.get_rotated_user_agent()
        agents.append(agent)
        print(f"  Agent {i+1}: {agent[:50]}...")
        time.sleep(0.1)
    
    # Check if we got different agents
    unique_agents = len(set(agents))
    print(f"✅ Generated {unique_agents} unique user agents out of {len(agents)} requests")
    
    # Test headers
    headers = user_agent_rotator.get_indian_market_headers()
    print(f"📋 Sample headers: {len(headers)} headers generated")
    print(f"   Accept-Language: {headers.get('Accept-Language', 'Not set')}")
    
    return True

def test_indian_market_status():
    """Test Indian market status detection"""
    print("\n📊 Testing Indian Market Status...")
    
    market_data = IndianMarketData()
    
    # Test market status
    status = market_data.get_market_status()
    print(f"✅ Current market status: {status}")
    
    # Test market hours logic
    is_open = settings.is_market_open()
    print(f"✅ Is market open (settings): {is_open}")
    
    return True

def test_stock_data_fetching():
    """Test enhanced stock data fetching"""
    print("\n📈 Testing Enhanced Stock Data Fetching...")
    
    market_data = IndianMarketData()
    
    # Test popular Indian stocks
    test_stocks = [
        ("RELIANCE", "NSE"),
        ("TCS", "NSE"),  
        ("HDFCBANK", "BSE")
    ]
    
    results = {}
    
    for symbol, exchange in test_stocks:
        print(f"\n  Testing {symbol} on {exchange}...")
        
        try:
            if exchange == "NSE":
                data = market_data.get_nse_stock_data(symbol, period="5d")
            else:
                data = market_data.get_bse_stock_data(symbol, period="5d")
            
            if data is not None and not data.empty:
                results[symbol] = {
                    'success': True,
                    'rows': len(data),
                    'columns': list(data.columns),
                    'latest_close': data['Close'].iloc[-1] if 'Close' in data.columns else 'N/A',
                    'date_range': f"{data.index[0].date()} to {data.index[-1].date()}"
                }
                print(f"    ✅ Success: {len(data)} rows, latest close: ₹{data['Close'].iloc[-1]:.2f}")
                print(f"    📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
            else:
                results[symbol] = {'success': False, 'error': 'No data returned'}
                print(f"    ❌ Failed: No data returned")
                
        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"    ❌ Error: {str(e)}")
        
        # Add delay between requests
        time.sleep(2)
    
    # Summary
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\n📊 Stock Data Test Summary: {successful}/{len(test_stocks)} successful")
    
    return results

def test_company_info():
    """Test company information retrieval"""
    print("\n🏢 Testing Company Information...")
    
    market_data = IndianMarketData()
    
    test_symbols = ["RELIANCE", "TCS"]
    
    for symbol in test_symbols:
        print(f"\n  Getting info for {symbol}...")
        
        try:
            info = market_data.get_company_info(symbol, "NSE")
            
            if info:
                print(f"    ✅ Company: {info.name}")
                print(f"    🏭 Sector: {info.sector}")
                print(f"    💰 Market Cap: ₹{info.market_cap/10000000:.0f} Cr" if info.market_cap else "N/A")
                print(f"    📊 P/E Ratio: {info.pe_ratio:.2f}" if info.pe_ratio else "N/A")
            else:
                print(f"    ❌ No info retrieved for {symbol}")
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
        
        time.sleep(1)
    
    return True

def test_market_indices():
    """Test Indian market indices"""
    print("\n📊 Testing Indian Market Indices...")
    
    market_data = IndianMarketData()
    
    try:
        indices = market_data.get_indian_indices()
        
        if indices:
            print(f"✅ Retrieved {len(indices)} indices:")
            
            for index in indices[:5]:  # Show first 5
                change_symbol = "📈" if index.change >= 0 else "📉"
                print(f"    {change_symbol} {index.name}: {index.current_value:,.2f} ({index.change_percent:+.2f}%)")
        else:
            print("❌ No indices data retrieved")
            
    except Exception as e:
        print(f"❌ Error retrieving indices: {str(e)}")
    
    return True

def test_news_fetching():
    """Test enhanced news fetching"""
    print("\n📰 Testing Enhanced News Fetching...")
    
    news_provider = IndianNewsProvider()
    
    test_stocks = [
        ("RELIANCE", "Reliance Industries"),
        ("TCS", "Tata Consultancy Services")
    ]
    
    for symbol, company_name in test_stocks:
        print(f"\n  Fetching news for {symbol}...")
        
        try:
            articles = news_provider.get_stock_news(symbol, company_name, limit=3)
            
            if articles:
                print(f"    ✅ Retrieved {len(articles)} articles:")
                for i, article in enumerate(articles[:2], 1):
                    print(f"      {i}. {article['title'][:60]}...")
                    print(f"         Source: {article['source']} | {article['published_date']}")
            else:
                print(f"    ❌ No news articles found for {symbol}")
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)}")
        
        time.sleep(3)  # Longer delay for news requests
    
    return True

def test_caching_functionality():
    """Test caching system"""
    print("\n💾 Testing Cache Functionality...")
    
    try:
        from src.utils.cache import cache_manager
        
        # Test cache stats
        stats = cache_manager.get_cache_stats()
        print(f"✅ Cache system initialized:")
        print(f"    Redis available: {stats.get('redis_available', False)}")
        print(f"    File cache files: {stats.get('file_cache_files', 0)}")
        
        # Test basic cache operations
        test_key = "test_indian_market"
        test_value = {"symbol": "RELIANCE.NS", "price": 2500.50}
        
        # Set cache
        cache_set = cache_manager.set(test_key, test_value, ttl=60)
        print(f"    Cache set operation: {'✅ Success' if cache_set else '❌ Failed'}")
        
        # Get cache
        cached_value = cache_manager.get(test_key)
        cache_get = cached_value == test_value
        print(f"    Cache get operation: {'✅ Success' if cache_get else '❌ Failed'}")
        
        # Delete cache
        cache_delete = cache_manager.delete(test_key)
        print(f"    Cache delete operation: {'✅ Success' if cache_delete else '❌ Failed'}")
        
    except Exception as e:
        print(f"❌ Cache test error: {str(e)}")
    
    return True

def main():
    """Run all tests"""
    print("🇮🇳 StoxChai Enhanced Indian Market Data Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run tests
    tests = [
        ("User Agent Rotation", test_user_agent_rotation),
        ("Market Status", test_indian_market_status),
        ("Stock Data Fetching", test_stock_data_fetching),
        ("Company Information", test_company_info),
        ("Market Indices", test_market_indices),
        ("News Fetching", test_news_fetching),
        ("Cache Functionality", test_caching_functionality)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        try:
            result = test_func()
            results[test_name] = "✅ PASSED" if result else "❌ FAILED"
        except Exception as e:
            results[test_name] = f"❌ ERROR: {str(e)}"
            print(f"❌ Test failed with error: {str(e)}")
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n{'='*60}")
    print("🏁 TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        print(f"{test_name:.<30} {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed")
    print(f"⏱️  Total duration: {duration:.2f} seconds")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced Indian market implementation is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)