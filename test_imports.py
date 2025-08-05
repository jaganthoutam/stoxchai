#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path (same as run_app.py)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        print("✅ Testing basic imports...")
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import yfinance as yf
        print("✅ Basic imports successful")
        
        # Test src imports
        print("✅ Testing src imports...")
        from src.config import settings
        from src.utils.logger import ui_logger
        from src.utils.cache import cache_manager
        from src.data.indian_market import indian_market
        from src.core.models import StockValidator
        print("✅ Src imports successful")
        
        # Test UI components
        print("✅ Testing UI components...")
        from src.ui.components import create_sidebar, create_header, create_footer, IndianMarketDashboard
        print("✅ UI components successful")
        
        # Test AI/ML imports
        print("✅ Testing AI/ML imports...")
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        import ollama
        print("✅ AI/ML imports successful")
        
        # Test LangChain imports
        print("✅ Testing LangChain imports...")
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document
        print("✅ LangChain imports successful")
        
        print("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_streamlit_app():
    """Test that the Streamlit app can be imported"""
    print("🧪 Testing Streamlit app import...")
    
    try:
        # This should work now with the Python path set up
        from src.ui.streamlit_app import StoxChaiApp
        print("✅ Streamlit app import successful")
        return True
    except Exception as e:
        print(f"❌ Streamlit app import failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting import tests...")
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[0]}")
    print()
    
    success = True
    success &= test_imports()
    success &= test_streamlit_app()
    
    if success:
        print("\n🎉 All tests passed! The application should work correctly.")
        print("💡 You can now run the application with:")
        print("   python run_app.py")
        print("   or")
        print("   ./run_with_podman.sh")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 