"""
UI Components for StoxChai - Indian Market Focus
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta

def create_sidebar():
    """Create the main sidebar for the application"""
    with st.sidebar:
        st.image("logo/StoxChaiLogo.png", width=100)
        st.title("StoxChai Tool")
        
        # Stock symbol input
        ticker_input = st.text_input(
            "Enter Stock Symbol (e.g., HDFCBANK.NS, RELIANCE.NS):", 
            value="NAVA.NS"
        )
        
        # Handle multiple tickers
        tickers = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip()]
        
        # Time period selection
        period = st.selectbox(
            "Select Time Period:",
            options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
            index=3  # Default to 1y
        )
        
        # Data source selection
        data_source = st.selectbox(
            "Select Data Source:",
            options=["Yahoo Finance", "Alternative Source (if Yahoo fails)"],
            index=0
        )
        
        # Fetch data button
        fetch_data = st.button("Fetch Data", use_container_width=True)
        
        # RAG query section
        st.subheader("RAG Query")
        rag_query = st.text_input(
            "Ask something about the stock:", 
            placeholder="e.g., How has the stock performed in the last month?"
        )
        run_rag_query = st.button("Run Query", use_container_width=True)
        
        # AI analysis section
        st.subheader("AI Analysis")
        ai_model = st.selectbox(
            "Select AI Model:",
            options=["qwen2.5:latest", "deepseek-r1:7b"],
            index=0
        )
        generate_ai_analysis = st.button("Generate AI Analysis", use_container_width=True)
        
        # Notification for API issues
        st.markdown("---")
        st.info("**Note:** Yahoo Finance API had changes in February 2025. If you encounter issues, try updating to the latest yfinance version (0.2.54+).")
        
        # About section
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool fetches stock data from Yahoo Finance, analyzes news sentiment, and provides AI-powered analysis using Ollama and RAG techniques.
        """)
        
        # App version
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #666;'>v1.2.0 - May 2025</div>", unsafe_allow_html=True)
        
        return {
            'tickers': tickers,
            'period': period,
            'data_source': data_source,
            'fetch_data': fetch_data,
            'rag_query': rag_query,
            'run_rag_query': run_rag_query,
            'ai_model': ai_model,
            'generate_ai_analysis': generate_ai_analysis
        }

def create_header():
    """Create the main header for the application"""
    st.markdown("<h1 class='main-header'>Advanced StoxChai Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='indian-flag-header'></div>", unsafe_allow_html=True)

def create_footer():
    """Create the footer for the application"""
    st.markdown("<footer>", unsafe_allow_html=True)
    st.markdown("Data source: Yahoo Finance | Built with Streamlit, Plotly, and Ollama")
    st.markdown("</footer>", unsafe_allow_html=True)

class IndianMarketDashboard:
    """Dashboard class for Indian market analysis"""
    
    def __init__(self):
        self.setup_page_config()
        
    def setup_page_config(self):
        """Configure Streamlit page for Indian market theme"""
        st.set_page_config(
            page_title="StoxChai - Indian Market Analysis",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def display_market_status(self, market_status: str):
        """Display current market status"""
        if market_status == "OPEN":
            st.markdown("<div class='market-status-open'>🟢 Market is OPEN</div>", unsafe_allow_html=True)
        elif market_status == "CLOSED":
            st.markdown("<div class='market-status-closed'>🔴 Market is CLOSED</div>", unsafe_allow_html=True)
        elif market_status == "PRE_OPEN":
            st.markdown("<div class='market-status-pre-open'>🟡 Market is PRE-OPEN</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='market-status-closed'>❓ Market status unknown</div>", unsafe_allow_html=True)
    
    def display_stock_metrics(self, data: pd.DataFrame, symbol: str):
        """Display key stock metrics"""
        if data is None or data.empty:
            st.warning(f"No data available for {symbol}")
            return
        
        # Calculate metrics
        current_price = data['Close'].iloc[-1]
        previous_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"₹{current_price:.2f}",
                delta=f"{price_change_pct:.2f}%"
            )
        
        with col2:
            st.metric(
                label="Volume",
                value=f"{data['Volume'].iloc[-1]:,}"
            )
        
        with col3:
            if len(data) >= 20:
                sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
                st.metric(
                    label="20-Day MA",
                    value=f"₹{sma_20:.2f}"
                )
            else:
                st.metric(label="20-Day MA", value="N/A")
        
        with col4:
            if len(data) >= 50:
                sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
                st.metric(
                    label="50-Day MA",
                    value=f"₹{sma_50:.2f}"
                )
            else:
                st.metric(label="50-Day MA", value="N/A")
    
    def display_news_summary(self, news_articles: List[Dict]):
        """Display news sentiment summary"""
        if not news_articles:
            st.info("No news articles available")
            return
        
        # Calculate sentiment statistics
        sentiment_scores = [article.get('sentiment_score', 0) for article in news_articles]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        positive_news = sum(1 for s in sentiment_scores if s > 0.2)
        negative_news = sum(1 for s in sentiment_scores if s < -0.2)
        neutral_news = len(sentiment_scores) - positive_news - negative_news
        
        # Display sentiment summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Positive News", positive_news, delta=f"{positive_news}")
        
        with col2:
            st.metric("Neutral News", neutral_news, delta=f"{neutral_news}")
        
        with col3:
            st.metric("Negative News", negative_news, delta=f"{negative_news}")
        
        # Display average sentiment
        st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
    
    def display_company_info(self, company_info: Dict):
        """Display company information"""
        if not company_info:
            st.info("Company information not available")
            return
        
        st.subheader("Company Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Company Details**")
            st.write(f"**Name**: {company_info.get('longName', 'N/A')}")
            st.write(f"**Sector**: {company_info.get('sector', 'N/A')}")
            st.write(f"**Industry**: {company_info.get('industry', 'N/A')}")
            st.write(f"**Exchange**: {company_info.get('exchange', 'N/A')}")
        
        with col2:
            st.markdown("**Financial Metrics**")
            st.write(f"**Market Cap**: ₹{company_info.get('marketCap', 'N/A'):,}" if isinstance(company_info.get('marketCap'), (int, float)) else f"**Market Cap**: {company_info.get('marketCap', 'N/A')}")
            st.write(f"**P/E Ratio**: {company_info.get('trailingPE', 'N/A')}")
            st.write(f"**Dividend Yield**: {company_info.get('dividendYield', 'N/A')}")
            st.write(f"**Beta**: {company_info.get('beta', 'N/A')}")
        
        # Business summary if available
        if 'longBusinessSummary' in company_info:
            st.subheader("Business Summary")
            st.write(company_info['longBusinessSummary']) 