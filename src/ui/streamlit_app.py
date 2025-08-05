"""
Enhanced Streamlit UI for StoxChai - Indian Market Focus
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio

from src.config import settings
from src.utils.logger import ui_logger
from src.utils.cache import cache_manager, get_cached_market_status, cache_market_status
from src.data.indian_market import indian_market, indian_news
from src.core.models import StockValidator
from src.ui.components import create_sidebar, create_header, create_footer, IndianMarketDashboard

class StoxChaiApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.setup_page_config()
        self.load_custom_css()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(**settings.STREAMLIT_CONFIG)
        
    def load_custom_css(self):
        """Load custom CSS for Indian market theme"""
        css = """
        <style>
        /* Indian Market Theme */
        :root {
            --primary-color: #FF6B35;
            --secondary-color: #138808;
            --accent-color: #FF9500;
            --background-color: #FFFFFF;
            --text-color: #000000;
            --border-color: #E5E5E5;
            --success-color: #00C851;
            --danger-color: #FF4444;
            --warning-color: #FFBB33;
        }
        
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .indian-flag-header {
            background: linear-gradient(90deg, #FF9933, #FFFFFF, #138808);
            height: 4px;
            margin-bottom: 1rem;
            border-radius: 2px;
        }
        
        .market-status-open {
            background-color: var(--success-color);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        
        .market-status-closed {
            background-color: var(--danger-color);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        
        .market-status-pre-open {
            background-color: var(--warning-color);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        
        .stock-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid var(--primary-color);
            margin-bottom: 1rem;
        }
        
        .price-positive {
            color: var(--success-color);
            font-weight: bold;
        }
        
        .price-negative {
            color: var(--danger-color);
            font-weight: bold;
        }
        
        .indian-metric-card {
            background: linear-gradient(135deg, #FFF5F5, #F0FFF4);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border-color);
        }
        
        .nse-badge {
            background-color: #1f77b4;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .bse-badge {
            background-color: #ff7f0e;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .rupee-symbol {
            color: var(--primary-color);
            font-weight: bold;
        }
        
        .sidebar-indian-theme {
            background: linear-gradient(180deg, #FFF5F5, #F0FFF4);
        }
        
        .indices-container {
            background: linear-gradient(135deg, #E3F2FD, #FFF3E0);
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .news-sentiment-positive {
            border-left: 4px solid var(--success-color);
            background: #F0FFF4;
        }
        
        .news-sentiment-negative {
            border-left: 4px solid var(--danger-color);
            background: #FFF5F5;
        }
        
        .news-sentiment-neutral {
            border-left: 4px solid #9E9E9E;
            background: #FAFAFA;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main-header {
                font-size: 1.8rem;
            }
            
            .stock-card {
                padding: 1rem;
            }
        }
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'selected_stocks' not in st.session_state:
            st.session_state.selected_stocks = []
        
        if 'market_data' not in st.session_state:
            st.session_state.market_data = {}
        
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        if 'user_portfolio' not in st.session_state:
            st.session_state.user_portfolio = []
        
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []
    
    def display_header(self):
        """Display application header with Indian theme"""
        st.markdown('<div class="indian-flag-header"></div>', unsafe_allow_html=True)
        st.markdown('<h1 class="main-header">🇮🇳 StoxChai - Indian Stock Market Analysis</h1>', unsafe_allow_html=True)
        
        # Market status
        market_status = get_cached_market_status()
        if not market_status:
            market_status = indian_market.get_market_status()
            cache_market_status(market_status, 60)  # Cache for 1 minute
        
        status_class = f"market-status-{market_status.lower().replace('_', '-')}"
        status_text = market_status.replace('_', ' ').title()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f'<div class="{status_class}">Market Status: {status_text}</div>', 
                       unsafe_allow_html=True)
    
    def display_indices_ticker(self):
        """Display Indian market indices as a ticker"""
        indices = indian_market.get_indian_indices()
        
        if indices:
            st.markdown('<div class="indices-container">', unsafe_allow_html=True)
            st.markdown("### 📊 Market Indices")
            
            cols = st.columns(len(indices[:5]))  # Show top 5 indices
            
            for i, index in enumerate(indices[:5]):
                with cols[i]:
                    change_class = "price-positive" if index.change >= 0 else "price-negative"
                    change_symbol = "▲" if index.change >= 0 else "▼"
                    
                    st.markdown(f"""
                    <div class="indian-metric-card">
                        <div style="font-weight: bold; font-size: 0.9rem;">{index.name}</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{index.current_value:,.2f}</div>
                        <div class="{change_class}">
                            {change_symbol} {index.change:+.2f} ({index.change_percent:+.2f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def create_sidebar(self):
        """Create enhanced sidebar for Indian market"""
        with st.sidebar:
            # Logo and branding
            st.image("logo/StoxChaiLogo.png", width=120)
            st.markdown("### 🇮🇳 Indian Stock Market Tool")
            
            # Stock symbol input with validation
            st.markdown("#### Stock Selection")
            symbol_input = st.text_input(
                "Enter Stock Symbol:",
                value="NAVA",
                help="Enter NSE/BSE symbol (e.g., RELIANCE, TCS, HDFCBANK)"
            )
            
            # Exchange selection
            exchange = st.selectbox(
                "Exchange:",
                options=["NSE", "BSE"],
                index=0,
                help="Select the stock exchange"
            )
            
            # Validate and format symbol
            if symbol_input:
                try:
                    formatted_symbol = StockValidator.normalize_symbol(symbol_input, exchange)
                    if StockValidator.is_valid_indian_symbol(formatted_symbol):
                        st.success(f"✓ Valid symbol: {formatted_symbol}")
                        st.session_state.current_symbol = formatted_symbol
                    else:
                        st.error("❌ Invalid stock symbol format")
                        st.session_state.current_symbol = None
                except Exception as e:
                    st.error(f"Error validating symbol: {str(e)}")
                    st.session_state.current_symbol = None
            
            # Time period selection
            period = st.selectbox(
                "Time Period:",
                options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
                index=5,  # Default to 1y
                help="Select data time period"
            )
            
            # Quick stock search
            st.markdown("#### Quick Search")
            search_query = st.text_input("Search Indian stocks:", placeholder="e.g., Reliance, TCS")
            
            if search_query:
                search_results = indian_market.search_stocks(search_query, limit=5)
                if search_results:
                    st.markdown("**Search Results:**")
                    for result in search_results:
                        if st.button(f"{result['symbol']} - {result['name'][:30]}...", 
                                   key=f"search_{result['symbol']}"):
                            st.session_state.current_symbol = f"{result['symbol']}.NS"
                            st.rerun()
            
            # Watchlist
            st.markdown("#### 📝 Watchlist")
            if st.session_state.watchlist:
                for symbol in st.session_state.watchlist:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button(symbol, key=f"watch_{symbol}"):
                            st.session_state.current_symbol = symbol
                            st.rerun()
                    with col2:
                        if st.button("❌", key=f"remove_{symbol}"):
                            st.session_state.watchlist.remove(symbol)
                            st.rerun()
            else:
                st.info("No stocks in watchlist")
            
            # Add to watchlist
            if hasattr(st.session_state, 'current_symbol') and st.session_state.current_symbol:
                if st.button("➕ Add to Watchlist"):
                    if st.session_state.current_symbol not in st.session_state.watchlist:
                        st.session_state.watchlist.append(st.session_state.current_symbol)
                        st.success("Added to watchlist!")
                        st.rerun()
            
            # Market insights
            st.markdown("#### 💡 Market Insights")
            top_gainers = indian_market.get_top_stocks("gainers", limit=3)
            if top_gainers:
                st.markdown("**Top Gainers:**")
                for stock in top_gainers:
                    st.markdown(f"• {stock['symbol']}: +{stock['change_percent']:.2f}%")
            
            top_losers = indian_market.get_top_stocks("losers", limit=3)
            if top_losers:
                st.markdown("**Top Losers:**")
                for stock in top_losers:
                    st.markdown(f"• {stock['symbol']}: {stock['change_percent']:.2f}%")
            
            # App info
            st.markdown("---")
            st.markdown("#### ℹ️ About")
            st.info(f"""
            **{settings.APP_NAME} v{settings.APP_VERSION}**
            
            Advanced Indian stock market analysis with AI-powered insights.
            
            • Real-time NSE/BSE data
            • AI-driven analysis
            • News sentiment tracking
            • Technical indicators
            """)
            
            # Cache stats (for debugging)
            if settings.DEBUG:
                st.markdown("#### 🔧 Debug Info")
                cache_stats = cache_manager.get_cache_stats()
                st.json(cache_stats)
    
    def display_stock_analysis(self, symbol: str, period: str):
        """Display comprehensive stock analysis"""
        if not symbol:
            st.warning("Please select a valid stock symbol")
            return
        
        # Load stock data
        with st.spinner("Loading stock data..."):
            stock_data = indian_market.get_nse_stock_data(symbol, period)
            company_info = indian_market.get_company_info(symbol.replace('.NS', '').replace('.BO', ''))
        
        if stock_data is None or stock_data.empty:
            st.error(f"Could not load data for {symbol}")
            return
        
        # Display company header
        if company_info:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"## {company_info.name}")
                exchange_badge = "nse-badge" if symbol.endswith('.NS') else "bse-badge"
                exchange_name = "NSE" if symbol.endswith('.NS') else "BSE"
                st.markdown(f'<span class="{exchange_badge}">{exchange_name}</span>', 
                           unsafe_allow_html=True)
            
            with col2:
                if company_info.sector:
                    st.metric("Sector", company_info.sector)
            
            with col3:
                if company_info.market_cap:
                    market_cap_cr = company_info.market_cap / 10000000  # Convert to crores
                    st.metric("Market Cap", f"₹{market_cap_cr:,.0f} Cr")
        
        # Key metrics
        current_price = stock_data['Close'].iloc[-1]
        prev_close = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
        change = current_price - prev_close
        change_percent = (change / prev_close) * 100 if prev_close != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="indian-metric-card">
                <div style="font-size: 0.9rem; color: #666;">Current Price</div>
                <div style="font-size: 1.8rem; font-weight: bold;">
                    <span class="rupee-symbol">₹</span>{:,.2f}
                </div>
            </div>
            """.format(current_price), unsafe_allow_html=True)
        
        with col2:
            change_class = "price-positive" if change >= 0 else "price-negative"
            change_symbol = "▲" if change >= 0 else "▼"
            st.markdown(f"""
            <div class="indian-metric-card">
                <div style="font-size: 0.9rem; color: #666;">Change</div>
                <div class="{change_class}" style="font-size: 1.4rem; font-weight: bold;">
                    {change_symbol} ₹{abs(change):.2f} ({change_percent:+.2f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            volume = stock_data['Volume'].iloc[-1]
            volume_lakhs = volume / 100000  # Convert to lakhs
            st.markdown(f"""
            <div class="indian-metric-card">
                <div style="font-size: 0.9rem; color: #666;">Volume</div>
                <div style="font-size: 1.4rem; font-weight: bold;">
                    {volume_lakhs:.1f}L
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            day_high = stock_data['High'].iloc[-1]
            day_low = stock_data['Low'].iloc[-1]
            st.markdown(f"""
            <div class="indian-metric-card">
                <div style="font-size: 0.9rem; color: #666;">Day Range</div>
                <div style="font-size: 1.2rem; font-weight: bold;">
                    ₹{day_low:.2f} - ₹{day_high:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts and detailed analysis
        self.display_stock_charts(stock_data, symbol)
        
        if company_info:
            self.display_company_details(company_info)
    
    def display_stock_charts(self, data: pd.DataFrame, symbol: str):
        """Display stock charts with Indian market styling"""
        tab1, tab2, tab3 = st.tabs(["📈 Price Chart", "📊 Volume Analysis", "📰 News & Sentiment"])
        
        with tab1:
            # Candlestick chart
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#00C851',
                decreasing_line_color='#FF4444'
            ))
            
            # Add moving averages
            if len(data) >= 20:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'].rolling(window=20).mean(),
                    mode='lines',
                    name='20-Day MA',
                    line=dict(color='orange', width=1)
                ))
            
            if len(data) >= 50:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'].rolling(window=50).mean(),
                    mode='lines',
                    name='50-Day MA',
                    line=dict(color='blue', width=1)
                ))
            
            fig.update_layout(
                title=f"{symbol} Stock Price Chart",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                template="plotly_white",
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Volume analysis
            fig = go.Figure()
            
            colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                     for i in range(len(data))]
            
            fig.add_trace(go.Bar(
                x=data.index,
                y=data['Volume'],
                marker_color=colors,
                name='Volume',
                opacity=0.7
            ))
            
            # Volume moving average
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Volume'].rolling(window=20).mean(),
                mode='lines',
                name='20-Day Volume MA',
                line=dict(color='purple', width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} Trading Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # News and sentiment
            self.display_news_sentiment(symbol)
    
    def display_news_sentiment(self, symbol: str):
        """Display news sentiment analysis"""
        base_symbol = symbol.replace('.NS', '').replace('.BO', '')
        
        with st.spinner("Fetching latest news..."):
            news_articles = indian_news.get_stock_news(base_symbol, base_symbol, limit=10)
        
        if news_articles:
            st.markdown("### 📰 Latest News")
            
            for article in news_articles:
                # Create sentiment styling
                sentiment_class = "news-sentiment-neutral"  # Default
                
                st.markdown(f"""
                <div class="stock-card {sentiment_class}">
                    <h4 style="margin-bottom: 0.5rem;">{article['title']}</h4>
                    <p style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">
                        {article['source']} • {article['published_date']}
                    </p>
                    <a href="{article['url']}" target="_blank">Read more →</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No recent news found for this stock.")
    
    def display_company_details(self, company_info):
        """Display detailed company information"""
        st.markdown("### 🏢 Company Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Financial Metrics**")
            metrics = [
                ("P/E Ratio", company_info.pe_ratio, lambda x: f"{x:.2f}" if x else "N/A"),
                ("EPS", company_info.eps, lambda x: f"₹{x:.2f}" if x else "N/A"),
                ("Book Value", company_info.book_value, lambda x: f"₹{x:.2f}" if x else "N/A"),
                ("Dividend Yield", company_info.dividend_yield, lambda x: f"{x*100:.2f}%" if x else "N/A")
            ]
            
            for label, value, formatter in metrics:
                st.metric(label, formatter(value))
        
        with col2:
            st.markdown("**Market Data**")
            if company_info.fifty_two_week_high and company_info.fifty_two_week_low:
                st.metric("52W High", f"₹{company_info.fifty_two_week_high:.2f}")
                st.metric("52W Low", f"₹{company_info.fifty_two_week_low:.2f}")
            
            if company_info.beta:
                st.metric("Beta", f"{company_info.beta:.2f}")
        
        if company_info.business_summary:
            st.markdown("**Business Summary**")
            st.write(company_info.business_summary)
    
    def run(self):
        """Run the main application"""
        try:
            ui_logger.info("Starting StoxChai application")
            
            # Display header
            self.display_header()
            
            # Display indices ticker
            self.display_indices_ticker()
            
            # Create sidebar
            self.create_sidebar()
            
            # Main content
            if hasattr(st.session_state, 'current_symbol') and st.session_state.current_symbol:
                period = st.selectbox("Select Period:", 
                                     ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], 
                                     index=5)
                self.display_stock_analysis(st.session_state.current_symbol, period)
            else:
                # Welcome screen
                st.markdown("""
                ## Welcome to StoxChai! 🇮🇳
                
                Your comprehensive Indian stock market analysis platform.
                
                ### Features:
                - 📊 Real-time NSE & BSE stock data
                - 📈 Interactive charts and technical analysis
                - 📰 News sentiment analysis
                - 🤖 AI-powered insights
                - 📱 Mobile-responsive design
                
                **Get started by selecting a stock from the sidebar!**
                """)
                
                # Display top movers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🚀 Top Gainers")
                    gainers = indian_market.get_top_stocks("gainers", limit=5)
                    for stock in gainers:
                        st.markdown(f"**{stock['symbol']}**: +{stock['change_percent']:.2f}%")
                
                with col2:
                    st.markdown("### 📉 Top Losers")
                    losers = indian_market.get_top_stocks("losers", limit=5)
                    for stock in losers:
                        st.markdown(f"**{stock['symbol']}**: {stock['change_percent']:.2f}%")
            
            # Footer
            create_footer()
            
        except Exception as e:
            ui_logger.error(f"Application error: {str(e)}")
            st.error(f"An error occurred: {str(e)}")

def create_footer():
    """Create application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>StoxChai v2.0.0 - Indian Stock Market Analysis Platform</p>
        <p>Data sources: NSE, BSE, Yahoo Finance | Built with ❤️ for Indian investors</p>
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational purposes only. 
        Please consult with financial advisors before making investment decisions.</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize and run the app
def main():
    """Main function to run the Streamlit app"""
    app = StoxChaiApp()
    app.run()

if __name__ == "__main__":
    main()