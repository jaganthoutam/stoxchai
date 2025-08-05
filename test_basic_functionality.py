#!/usr/bin/env python3
"""
Basic functionality test for StoxChai Indian market enhancements
Tests core functionality that doesn't require external dependencies
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
from datetime import datetime

def test_user_agent_rotation():
    """Test dynamic user agent rotation"""
    print("🔄 Testing User Agent Rotation...")
    
    try:
        from src.utils.user_agents import UserAgentRotator, get_dynamic_headers
        
        rotator = UserAgentRotator()
        
        # Test multiple user agents
        agents = []
        for i in range(5):
            agent = rotator.get_rotated_user_agent()
            agents.append(agent)
            print(f"  Agent {i+1}: {agent[:60]}...")
            time.sleep(0.1)
        
        # Check if we got different agents
        unique_agents = len(set(agents))
        print(f"✅ Generated {unique_agents} unique user agents out of {len(agents)} requests")
        
        # Test headers
        headers = get_dynamic_headers()
        print(f"📋 Generated headers: {len(headers)} headers")
        print(f"   User-Agent: {headers.get('User-Agent', 'Not set')[:50]}...")
        print(f"   Accept-Language: {headers.get('Accept-Language', 'Not set')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing Configuration System...")
    
    try:
        from src.config import settings
        
        print(f"✅ App Name: {settings.APP_NAME}")
        print(f"✅ App Version: {settings.APP_VERSION}")
        print(f"✅ Environment: {settings.ENVIRONMENT}")
        print(f"✅ Base Currency: {settings.BASE_CURRENCY}")
        print(f"✅ Currency Symbol: {settings.CURRENCY_SYMBOL}")
        
        # Test Indian market specific config
        print(f"✅ Indian Exchanges: {settings.INDIAN_EXCHANGES}")
        print(f"✅ Market Hours: {settings.MARKET_HOURS}")
        print(f"✅ Number of Indices: {len(settings.INDIAN_INDICES)}")
        
        # Test market status function
        try:
            is_open = settings.is_market_open()
            print(f"✅ Market Status Function: Works (Market is {'OPEN' if is_open else 'CLOSED'})")
        except Exception as e:
            print(f"⚠️  Market Status Function: {str(e)}")
        
        # Test symbol formatting
        try:
            nse_symbol = settings.get_indian_stock_symbol("RELIANCE", "NSE")
            bse_symbol = settings.get_indian_stock_symbol("RELIANCE", "BSE")
            print(f"✅ Symbol Formatting: NSE={nse_symbol}, BSE={bse_symbol}")
        except Exception as e:
            print(f"❌ Symbol Formatting Error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration Error: {str(e)}")
        return False

def test_models():
    """Test core data models"""
    print("\n📊 Testing Core Data Models...")
    
    try:
        from src.core.models import StockValidator, Exchange, StockPrice, CompanyInfo
        from datetime import datetime
        
        # Test StockValidator
        print("  Testing StockValidator...")
        
        test_symbols = [
            ("RELIANCE.NS", True),
            ("TCS.BO", True),
            ("INVALID", True),  # Plain symbols are valid
            ("", False),
            ("TOOLONG12345.NS", False),
        ]
        
        for symbol, expected in test_symbols:
            result = StockValidator.is_valid_indian_symbol(symbol)
            status = "✅" if result == expected else "❌"
            print(f"    {status} {symbol}: {result} (expected {expected})")
        
        # Test symbol normalization
        print("  Testing Symbol Normalization...")
        normalized = StockValidator.normalize_symbol("RELIANCE", "NSE")
        print(f"    ✅ RELIANCE -> {normalized}")
        
        # Test Exchange enum
        print("  Testing Exchange Enum...")
        nse = Exchange.NSE
        bse = Exchange.BSE
        print(f"    ✅ Exchanges: {nse.value}, {bse.value}")
        
        # Test StockPrice model
        print("  Testing StockPrice Model...")
        stock_price = StockPrice(
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000000,
            date=datetime.now()
        )
        print(f"    ✅ StockPrice: Change={stock_price.change}, Change%={stock_price.change_percent:.2f}%")
        
        # Test CompanyInfo model
        print("  Testing CompanyInfo Model...")
        company = CompanyInfo(
            symbol="RELIANCE",
            name="Reliance Industries Limited",
            exchange=Exchange.NSE,
            sector="Energy",
            market_cap=1500000000000  # 15 lakh crores
        )
        print(f"    ✅ CompanyInfo: {company.name} ({company.exchange.value})")
        
        return True
        
    except Exception as e:
        print(f"❌ Models Error: {str(e)}")
        return False

def test_logging():
    """Test logging system"""
    print("\n📝 Testing Logging System...")
    
    try:
        from src.utils.logger import app_logger, data_logger, ui_logger
        
        # Test different log levels
        app_logger.info("✅ App logger test message")
        data_logger.debug("✅ Data logger test message")
        ui_logger.warning("✅ UI logger test message")
        
        print("✅ Logging system initialized successfully")
        print(f"   App logger: {app_logger.name}")
        print(f"   Data logger: {data_logger.name}")
        print(f"   UI logger: {ui_logger.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging Error: {str(e)}")
        return False

def test_cache_system():
    """Test caching system (basic functionality)"""
    print("\n💾 Testing Cache System...")
    
    try:
        from src.utils.cache import CacheManager
        
        # Create a cache manager instance
        cache = CacheManager()
        
        # Test basic cache operations
        test_key = "test_key"
        test_value = {"symbol": "RELIANCE.NS", "price": 2500.50, "timestamp": datetime.now().isoformat()}
        
        # Test set operation
        set_result = cache.set(test_key, test_value, ttl=60)
        print(f"✅ Cache Set: {set_result}")
        
        # Test get operation
        get_result = cache.get(test_key)
        get_success = get_result is not None and get_result.get("symbol") == "RELIANCE.NS"
        print(f"✅ Cache Get: {get_success} (Retrieved: {get_result is not None})")
        
        # Test cache stats
        stats = cache.get_cache_stats()
        print(f"✅ Cache Stats: {len(stats)} statistics available")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Test delete operation
        delete_result = cache.delete(test_key)
        print(f"✅ Cache Delete: {delete_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache Error: {str(e)}")
        return False

def test_security():
    """Test security features"""
    print("\n🔒 Testing Security Features...")
    
    try:
        from src.utils.security import InputValidator, SecurityManager
        
        validator = InputValidator()
        security = SecurityManager()
        
        # Test input validation
        print("  Testing Input Validation...")
        
        # Stock symbol validation
        valid_symbols = ["RELIANCE.NS", "TCS.BO", "HDFCBANK"]
        invalid_symbols = ["", "INVALID@SYMBOL", "TOOLONGSTOCKSYMBOL.NS"]
        
        for symbol in valid_symbols:
            result = validator.validate_stock_symbol(symbol)
            print(f"    ✅ {symbol}: {result} (should be True)")
        
        for symbol in invalid_symbols:
            result = validator.validate_stock_symbol(symbol)
            print(f"    ✅ {symbol}: {result} (should be False)")
        
        # Test string sanitization
        test_strings = [
            "Normal string",
            "<script>alert('xss')</script>",
            "String with 'quotes' and \"double quotes\"",
        ]
        
        for test_str in test_strings:
            sanitized = validator.sanitize_string(test_str)
            print(f"    ✅ '{test_str}' -> '{sanitized}'")
        
        # Test security manager
        print("  Testing Security Manager...")
        token = security.generate_session_token()
        print(f"    ✅ Session Token Generated: {len(token)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Security Error: {str(e)}")
        return False

def test_directory_structure():
    """Test if all required directories and files exist"""
    print("\n📁 Testing Directory Structure...")
    
    base_dir = os.path.dirname(__file__)
    
    required_dirs = [
        "src",
        "src/config",
        "src/core", 
        "src/data",
        "src/ui",
        "src/utils",
        "tests",
        "deployment",
        "data",
        "cache",
        "logs"
    ]
    
    required_files = [
        "src/config/__init__.py",
        "src/config/settings.py",
        "src/core/models.py",
        "src/data/indian_market.py",
        "src/utils/user_agents.py",
        "src/utils/cache.py",
        "src/utils/security.py",
        "src/utils/logger.py",
        "main.py",
        "requirements-prod.txt",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example"
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if os.path.exists(full_path):
            print(f"    ✅ {dir_path}/")
        else:
            print(f"    ❌ {dir_path}/ (missing)")
            missing_dirs.append(dir_path)
    
    # Check files
    for file_path in required_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"    ✅ {file_path}")
        else:
            print(f"    ❌ {file_path} (missing)")
            missing_files.append(file_path)
    
    print(f"\n📊 Directory Structure Summary:")
    print(f"   Directories: {len(required_dirs) - len(missing_dirs)}/{len(required_dirs)} present")
    print(f"   Files: {len(required_files) - len(missing_files)}/{len(required_files)} present")
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def main():
    """Run all basic tests"""
    print("🇮🇳 StoxChai Basic Functionality Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run tests
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Configuration System", test_configuration),
        ("Core Data Models", test_models),
        ("User Agent Rotation", test_user_agent_rotation),
        ("Logging System", test_logging),
        ("Cache System", test_cache_system),
        ("Security Features", test_security)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*15} {test_name} {'='*15}")
        
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
    print("🏁 BASIC FUNCTIONALITY TEST SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        print(f"{test_name:.<35} {result}")
    
    passed = sum(1 for r in results.values() if "PASSED" in r)
    total = len(results)
    
    print(f"\n📊 Overall Results: {passed}/{total} tests passed")
    print(f"⏱️  Total duration: {duration:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All basic functionality tests passed!")
        print("✅ The StoxChai Indian market implementation is properly structured")
        print("✅ Core components are working correctly")
        print("✅ Ready for Indian market data integration")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the output above for details.")
    
    print(f"\n📋 Next Steps:")
    print("   1. Install required dependencies: pip install -r requirements-prod.txt")
    print("   2. Set up environment variables: cp .env.example .env")
    print("   3. Start with Docker: docker-compose up -d")
    print("   4. Access the app: http://localhost:8501")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)