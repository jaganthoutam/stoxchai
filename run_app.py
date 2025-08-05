#!/usr/bin/env python3
"""
StoxChai Application Entry Point
This script sets up the Python path and runs the Streamlit application.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for production
os.environ.setdefault('ENVIRONMENT', 'production')
os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
os.environ.setdefault('STREAMLIT_SERVER_ENABLE_CORS', 'false')
os.environ.setdefault('STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION', 'true')

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    import sys
    
    # Set the streamlit app path
    app_path = str(project_root / "src" / "ui" / "streamlit_app.py")
    
    # Run streamlit with the app
    sys.argv = [
        "streamlit", "run", app_path,
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=true"
    ]
    
    sys.exit(stcli.main()) 