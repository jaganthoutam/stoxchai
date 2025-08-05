#!/usr/bin/env python3
"""
Quick test to verify the main application can start
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_main_imports():
    """Test the main imports that are causing issues"""
    print("🧪 Testing main imports...")
    
    try:
        # Test the specific import that was failing
        from src.ui.components import create_sidebar, create_header, create_footer, IndianMarketDashboard
        print("✅ UI components import successful")
        
        # Test other critical imports
        from src.config import settings
        print("✅ Settings import successful")
        
        from src.utils.logger import ui_logger
        print("✅ Logger import successful")
        
        from src.data.indian_market import indian_market
        print("✅ Indian market import successful")
        
        from src.core.models import StockValidator
        print("✅ Models import successful")
        
        # Test the main app import
        from src.ui.streamlit_app import StoxChaiApp
        print("✅ Streamlit app import successful")
        
        print("🎉 All critical imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Quick import test...")
    print(f"📁 Project root: {project_root}")
    print()
    
    if test_main_imports():
        print("\n✅ All imports working! The application should start correctly.")
        print("💡 You can now run:")
        print("   podman-compose -f podman-compose-simple.yml up -d")
    else:
        print("\n❌ Some imports failed. Check the error messages above.")
        sys.exit(1) 