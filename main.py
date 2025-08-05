"""
Main entry point for StoxChai application
Production-ready Indian Stock Market Analysis Tool
"""
import os
import sys
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import settings
from src.utils.logger import app_logger
from src.utils.security import initialize_session_security
from src.ui.streamlit_app import main as run_streamlit_app

def setup_production_environment():
    """Setup production environment"""
    # Create necessary directories
    settings.create_directories()
    
    # Configure logging for production
    if settings.ENVIRONMENT == "production":
        logging.getLogger("streamlit").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Initialize security
    initialize_session_security()
    
    app_logger.info(f"StoxChai v{settings.APP_VERSION} starting in {settings.ENVIRONMENT} mode")

def main():
    """Main application entry point"""
    try:
        # Setup environment
        setup_production_environment()
        
        # Run Streamlit app
        run_streamlit_app()
        
    except Exception as e:
        app_logger.error(f"Application startup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()