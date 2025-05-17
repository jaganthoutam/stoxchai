#!/usr/bin/env python
# StoxChai Tool - Part 1: Imports and Setup
# Filename: stock_analyzer.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import os
import json
import re
from newspaper import Article
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Fix deprecated LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
import time
from collections import deque
import pkg_resources
import warnings

# Try to import sentence_transformers directly (for fallback)
try:
    from sentence_transformers import SentenceTransformer
    from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
except ImportError:
    pass

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Setup directories for data storage
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VECTOR_DB_DIR = os.path.join(DATA_DIR, "vector_db")
NEWS_CACHE_DIR = os.path.join(DATA_DIR, "news_cache")

# Create directories if they don't exist
for directory in [DATA_DIR, VECTOR_DB_DIR, NEWS_CACHE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Configure page
st.set_page_config(
    page_title="StoxChai Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fix for asyncio/tornado issues in Streamlit
# This helps prevent the "no running event loop" errors
import nest_asyncio
try:
    nest_asyncio.apply()
except Exception:
    pass

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4B5563;
    }
    .up-value {
        color: #059669;
    }
    .down-value {
        color: #DC2626;
    }
    .news-title {
        font-weight: 600;
        color: #1E3A8A;
    }
    .news-source {
        font-size: 0.8rem;
        color: #4B5563;
    }
    .news-sentiment {
        font-size: 0.9rem;
        font-weight: 500;
    }
    .positive-sentiment {
        color: #059669;
    }
    .negative-sentiment {
        color: #DC2626;
    }
    .neutral-sentiment {
        color: #9CA3AF;
    }
    .tab-content {
        padding: 1rem 0;
    }
    footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6B7280;
    }
    .chat-user-message {
        background-color: #F0F2F6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .chat-assistant-message {
        background-color: #E1F5FE;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .suggestion-button {
        background-color: #f8f9fa;
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .suggestion-button:hover {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# StoxChai Tool - Part 2: Data Functions
# Functions for loading, processing, and storing stock data

# Function to load stock data from Yahoo Finance
def load_stock_data(ticker, period='1y'):
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period)
        
        # Handle the February 2025 Yahoo Finance API change that introduced MultiIndex
        if isinstance(hist_data.columns, pd.MultiIndex):
            hist_data = hist_data.xs(key=ticker, axis=1, level=1, drop_level=False)
        
        # Get stock info with error handling
        try:
            info = stock.info
        except Exception as info_error:
            st.warning(f"Could not retrieve complete info for {ticker}: {info_error}")
            # Create minimal info dictionary with default values
            info = {"longName": ticker, "currentPrice": hist_data['Close'].iloc[-1] if not hist_data.empty else None}
        
        if hist_data.empty:
            st.warning(f"No historical data found for {ticker}. Trying alternative method...")
            # Try alternative method - direct download
            alt_data = yf.download(ticker, period=period)
            if not alt_data.empty:
                hist_data = alt_data
                if isinstance(hist_data.columns, pd.MultiIndex):
                    hist_data = hist_data.xs(key=ticker, axis=1, level=1, drop_level=False)
            else:
                st.error(f"Could not retrieve data for {ticker} using alternative method either.")
        
        return hist_data, info
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        # Try alternative data source
        try:
            st.info("Attempting to use alternative data source...")
            # Placeholder for alternative data source (could be Alpha Vantage, IEX, etc.)
            # For this example, we'll just use a different method from yfinance
            alt_data = yf.download(ticker, period=period)
            if not alt_data.empty:
                if isinstance(alt_data.columns, pd.MultiIndex):
                    alt_data = alt_data.xs(key=ticker, axis=1, level=1, drop_level=False)
                return alt_data, {"longName": ticker, "currentPrice": alt_data['Close'].iloc[-1]}
            else:
                return None, None
        except Exception as alt_e:
            st.error(f"Alternative data source also failed: {alt_e}")
            return None, None

# Function to save stock data locally
def save_stock_data(ticker, data, info):
    ticker_file = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    info_file = os.path.join(DATA_DIR, f"{ticker}_info.json")
    
    data.to_csv(ticker_file)
    with open(info_file, 'w') as f:
        json.dump(info, f)

# Function to load stock data from local storage
def load_local_stock_data(ticker):
    ticker_file = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    info_file = os.path.join(DATA_DIR, f"{ticker}_info.json")
    
    if os.path.exists(ticker_file) and os.path.exists(info_file):
        data = pd.read_csv(ticker_file, index_col=0, parse_dates=True)
        with open(info_file, 'r') as f:
            info = json.load(f)
        return data, info
    return None, None

# Function to check if local data is outdated
def is_data_outdated(ticker):
    ticker_file = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    
    if not os.path.exists(ticker_file):
        return True
    
    file_modified_time = datetime.fromtimestamp(os.path.getmtime(ticker_file))
    current_time = datetime.now()
    
    # Return True if data is older than 24 hours
    return (current_time - file_modified_time) > timedelta(hours=24)

# StoxChai Tool - Part 3: News Sentiment Functions
# Functions for retrieving and analyzing news articles

# Function to fetch and process news articles
def fetch_news(ticker, company_name, max_articles=5):
    cache_file = os.path.join(NEWS_CACHE_DIR, f"{ticker}_news.json")
    
    # Check if we have cached news that's less than 6 hours old
    if os.path.exists(cache_file):
        file_modified_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if (datetime.now() - file_modified_time) < timedelta(hours=6):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # If cache file is corrupted, continue to fetch new data
                pass
    
    # Search queries to try
    search_queries = [
        f"{ticker} stock news",
        f"{company_name} financial news",
        f"{company_name} stock performance"
    ]
    
    all_articles = []
    
    for query in search_queries:
        if len(all_articles) >= max_articles:
            break
            
        search_query = query.replace(' ', '+')
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        try:
            # Try different news sources
            sources = [
                f"https://www.google.com/search?q={search_query}&tbm=nws",
                f"https://news.search.yahoo.com/search?p={search_query}",
                f"https://www.bing.com/news/search?q={search_query}"
            ]
            
            for source_url in sources:
                if len(all_articles) >= max_articles:
                    break
                    
                try:
                    response = requests.get(source_url, headers=headers, timeout=10)
                    if response.status_code != 200:
                        continue
                        
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Google News parser
                    if "google.com" in source_url:
                        news_elements = soup.select('div.SoaBEf')
                        for element in news_elements:
                            if len(all_articles) >= max_articles:
                                break
                                
                            try:
                                title_element = element.select_one('div.mCBkyc')
                                link_element = element.select_one('a')
                                source_element = element.select_one('.CEMjEf span')
                                date_element = element.select_one('.OSrXXb span')
                                
                                if not all([title_element, link_element, source_element]):
                                    continue
                                    
                                title = title_element.text
                                url = link_element['href']
                                if url.startswith('/url?q='):
                                    url = url.split('/url?q=')[1].split('&sa=')[0]
                                
                                source = source_element.text if source_element else "Unknown"
                                date = date_element.text if date_element else "Recent"
                                
                                # Add article to list if not already present
                                if not any(article['url'] == url for article in all_articles):
                                    process_article(title, url, source, date, all_articles)
                            except Exception as e:
                                continue
                    
                    # Yahoo News parser
                    elif "yahoo.com" in source_url:
                        news_elements = soup.select('div.NewsArticle')
                        for element in news_elements:
                            if len(all_articles) >= max_articles:
                                break
                                
                            try:
                                title_element = element.select_one('h4')
                                link_element = element.select_one('a')
                                source_element = element.select_one('.s-source')
                                date_element = element.select_one('.s-time')
                                
                                if not all([title_element, link_element]):
                                    continue
                                    
                                title = title_element.text
                                url = link_element['href']
                                if not url.startswith('http'):
                                    url = 'https://news.yahoo.com' + url
                                
                                source = source_element.text if source_element else "Yahoo News"
                                date = date_element.text if date_element else "Recent"
                                
                                # Add article to list if not already present
                                if not any(article['url'] == url for article in all_articles):
                                    process_article(title, url, source, date, all_articles)
                            except Exception as e:
                                continue
                    
                    # Bing News parser
                    elif "bing.com" in source_url:
                        news_elements = soup.select('.news-card')
                        for element in news_elements:
                            if len(all_articles) >= max_articles:
                                break
                                
                            try:
                                title_element = element.select_one('a.title')
                                source_element = element.select_one('.source')
                                date_element = element.select_one('.source span')
                                
                                if not title_element:
                                    continue
                                    
                                title = title_element.text
                                url = title_element['href']
                                
                                source = source_element.text.split(' on ')[0] if source_element else "Unknown"
                                date = date_element.text if date_element else "Recent"
                                
                                # Add article to list if not already present
                                if not any(article['url'] == url for article in all_articles):
                                    process_article(title, url, source, date, all_articles)
                            except Exception as e:
                                continue
                except Exception as e:
                    continue
        except Exception as e:
            st.warning(f"Could not fetch news from search query: {e}")
    
    # Cache the results
    if all_articles:
        with open(cache_file, 'w') as f:
            json.dump(all_articles, f)
    
    return all_articles

# Helper function to process an article
def process_article(title, url, source, date, all_articles):
    # Try to get the full article text
    try:
        article = Article(url)
        article.download()
        article.parse()
        content = article.text
    except:
        content = "Content unavailable"
    
    # Get sentiment
    if content != "Content unavailable":
        sentiment = sia.polarity_scores(content)
        sentiment_score = sentiment['compound']
    else:
        sentiment_score = sia.polarity_scores(title)['compound']
    
    if sentiment_score > 0.2:
        sentiment_label = "Positive"
    elif sentiment_score < -0.2:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"
    
    all_articles.append({
        'title': title,
        'url': url,
        'source': source,
        'date': date,
        'content': content[:1000] + '...' if len(content) > 1000 else content,
        'sentiment_score': sentiment_score,
        'sentiment': sentiment_label
    })

# StoxChai Tool - Part 4: RAG and AI Functions
# Functions for RAG system and AI analysis

# Initialize RAG system with HuggingFace embeddings
def initialize_rag_system():
    try:
        # Fix for torch issues
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Try different approaches to initialize embeddings
        try:
            # First, try the simplest approach with just device specification
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            st.success("Successfully initialized embeddings with standard method.")
            return embeddings
        except Exception as e1:
            st.warning(f"Primary embedding initialization failed: {e1}. Trying direct method...")
            
            # Try direct initialization without LangChain wrapper
            try:
                # Use direct sentence-transformers if LangChain wrapper fails
                from sentence_transformers import SentenceTransformer
                
                class CustomEmbeddings:
                    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                        self.model = SentenceTransformer(model_name, device="cpu")
                    
                    def embed_documents(self, texts):
                        return self.model.encode(texts)
                    
                    def embed_query(self, text):
                        return self.model.encode(text)
                
                # Create custom embeddings
                embeddings = CustomEmbeddings()
                st.success("Using custom embeddings implementation.")
                return embeddings
            except Exception as e2:
                # Last resort - try older version of HuggingFaceEmbeddings
                try:
                    st.warning(f"Direct embedding method failed: {e2}. Trying final fallback...")
                    
                    # Try with minimal parameters for compatibility
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    st.success("Using minimal parameter embeddings initialization.")
                    return embeddings
                except Exception as e3:
                    st.error(f"All embedding methods failed. Last error: {e3}")
                    return None
                
    except Exception as e:
        st.error(f"Could not initialize RAG system: {e}")
        return None

# Function to create or update vector store
def update_vector_store(ticker, data, news_articles, embeddings):
    if embeddings is None:
        return None
        
    try:
        # Prepare documents from stock data
        documents = []
        
        # Convert historical data to documents
        for date, row in data.iterrows():
            content = f"Date: {date.strftime('%Y-%m-%d')}\n"
            content += f"Open: {row['Open']}, High: {row['High']}, Low: {row['Low']}, Close: {row['Close']}\n"
            content += f"Volume: {row['Volume']}\n"
            
            doc = Document(
                page_content=content,
                metadata={
                    "ticker": ticker,
                    "date": date.strftime("%Y-%m-%d"),
                    "type": "historical_data"
                }
            )
            documents.append(doc)
        
        # Add news articles to documents
        for article in news_articles:
            content = f"Title: {article['title']}\n"
            content += f"Source: {article['source']}, Date: {article['date']}\n"
            content += f"Content: {article['content']}\n"
            content += f"Sentiment: {article['sentiment']} ({article['sentiment_score']})\n"
            
            doc = Document(
                page_content=content,
                metadata={
                    "ticker": ticker,
                    "type": "news",
                    "url": article['url'],
                    "sentiment": article['sentiment'],
                    "sentiment_score": article['sentiment_score']
                }
            )
            documents.append(doc)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)
        
        # Check if we're using custom embeddings
        is_custom_embeddings = hasattr(embeddings, 'model') and not hasattr(embeddings, 'client')
        
        if is_custom_embeddings:
            # For custom embeddings, create FAISS index directly
            import faiss
            import numpy as np
            import pickle
            import os
            
            # Create vector store directory if it doesn't exist
            vector_store_path = os.path.join(VECTOR_DB_DIR, ticker)
            if not os.path.exists(vector_store_path):
                os.makedirs(vector_store_path)
            
            # Embed documents
            embeddings_list = []
            docstore = {}
            
            for i, doc in enumerate(split_docs):
                # Embed the document
                embedding = embeddings.embed_documents([doc.page_content])[0]
                embeddings_list.append(embedding)
                
                # Store the document with its index as key
                docstore[str(i)] = doc
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings_list).astype('float32')
            
            # Create the FAISS index
            dimension = len(embeddings_array[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            # Save index and docstore
            faiss.write_index(index, os.path.join(vector_store_path, "index.faiss"))
            with open(os.path.join(vector_store_path, "docstore.pkl"), "wb") as f:
                pickle.dump(docstore, f)
            
            st.success(f"Created vector store for {ticker} with custom embeddings.")
            return True
        else:
            # For LangChain embeddings, use the standard method
            try:
                # Create or load vector store
                vector_store_path = os.path.join(VECTOR_DB_DIR, ticker)
                
                if os.path.exists(vector_store_path):
                    try:
                        # Load and update existing vector store with allow_dangerous_deserialization flag
                        vectorstore = FAISS.load_local(
                            vector_store_path, 
                            embeddings, 
                            allow_dangerous_deserialization=True
                        )
                        # Add new documents
                        vectorstore.add_documents(split_docs)
                    except Exception as load_error:
                        st.warning(f"Could not load existing vector store: {load_error}. Creating new one.")
                        # Create new vector store if loading fails
                        vectorstore = FAISS.from_documents(split_docs, embeddings)
                else:
                    # Create new vector store
                    vectorstore = FAISS.from_documents(split_docs, embeddings)
                
                # Save updated vector store
                vectorstore.save_local(vector_store_path)
                
                return vectorstore
            except Exception as e:
                st.error(f"Error with LangChain vector store: {e}")
                return None
    
    except Exception as e:
        st.error(f"Error updating vector store: {e}")
        return None

# Function to query vector store
def query_vector_store(ticker, query, embeddings, k=5):
    vector_store_path = os.path.join(VECTOR_DB_DIR, ticker)
    
    if not os.path.exists(vector_store_path) or embeddings is None:
        return []
        
    try:
        # Check if we're using custom embeddings
        if hasattr(embeddings, 'model') and not hasattr(embeddings, 'client'):
            # For custom embeddings, we need to load FAISS differently
            import faiss
            import pickle
            import numpy as np
            
            # Load the index
            index = faiss.read_index(os.path.join(vector_store_path, "index.faiss"))
            
            # Load the docstore
            with open(os.path.join(vector_store_path, "docstore.pkl"), "rb") as f:
                docstore = pickle.load(f)
            
            # Embed the query
            query_embedding = embeddings.embed_query(query)
            query_embedding = np.array([query_embedding]).astype("float32")
            
            # Search the index
            scores, indices = index.search(query_embedding, k)
            
            # Get the documents
            results = []
            for j, i in enumerate(indices[0]):
                if i == -1:  # This happens when there are not enough docs
                    continue
                doc = docstore.get(str(i))
                if doc is not None:
                    results.append(doc)
            
            return results
        else:
            # For LangChain embeddings, use the standard method
            vectorstore = FAISS.load_local(
                vector_store_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            results = vectorstore.similarity_search(query, k=k)
            return results
    except Exception as e:
        st.error(f"Error querying vector store: {e}")
        st.info("Try fetching data again to rebuild the vector store.")
        return []

# Function to generate analysis with Ollama
def generate_analysis(ticker, data, info, news_articles, query_results=None, model="llama3"):
    try:
        # Create a prompt for Ollama
        prompt = f"""Analyze the stock {ticker} based on the following information:

Company: {info.get('longName', ticker)}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}

Current Price: {info.get('currentPrice', 'N/A')}
Previous Close: {info.get('previousClose', 'N/A')}
Market Cap: {info.get('marketCap', 'N/A')}
52-Week Range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}
Average Volume: {info.get('averageVolume', 'N/A')}

Recent Performance Summary:
"""
        # Add recent performance data
        if not data.empty:
            recent_data = data.tail(10)
            # Use try-except to handle potential NaN values in calculations
            try:
                price_change = recent_data['Close'].pct_change().mean() * 100
                volatility = recent_data['Close'].pct_change().std() * 100
                
                prompt += f"- Price Change (Last 10 Days): {price_change:.2f}%\n"
                prompt += f"- Volatility (Last 10 Days): {volatility:.2f}%\n"
            except Exception as calc_error:
                prompt += f"- Price Change (Last 10 Days): Not available\n"
                prompt += f"- Volatility (Last 10 Days): Not available\n"
            
            if len(data) >= 50:
                try:
                    sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
                    prompt += f"- 50-Day Moving Average: {sma_50:.2f}\n"
                    
                    if data['Close'].iloc[-1] > sma_50:
                        prompt += f"- Price is ABOVE the 50-day moving average\n"
                    else:
                        prompt += f"- Price is BELOW the 50-day moving average\n"
                except Exception:
                    prompt += f"- 50-Day Moving Average: Not available\n"
                    
        # Add news sentiment
        if news_articles:
            sentiment_scores = [article['sentiment_score'] for article in news_articles]
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            
            prompt += f"\nNews Sentiment Analysis:\n"
            prompt += f"- Average Sentiment Score: {avg_sentiment:.2f}\n"
            
            positive_news = sum(1 for s in sentiment_scores if s > 0.2)
            negative_news = sum(1 for s in sentiment_scores if s < -0.2)
            neutral_news = len(sentiment_scores) - positive_news - negative_news
            
            prompt += f"- Positive News: {positive_news}\n"
            prompt += f"- Neutral News: {neutral_news}\n"
            prompt += f"- Negative News: {negative_news}\n"
        
        # Add RAG query results if available
        if query_results:
            prompt += "\nAdditional Context from Historical Data:\n"
            for doc in query_results:
                prompt += f"- {doc.page_content}\n"
        
        prompt += "\nBased on the above information, provide a comprehensive analysis of the stock including:\n"
        prompt += "1. Current market position\n"
        prompt += "2. Recent performance trends\n"
        prompt += "3. News sentiment implications\n"
        prompt += "4. Key factors to watch\n"
        prompt += "5. Overall outlook (bullish, bearish, or neutral)\n\n"
        prompt += "Keep your analysis professional, data-driven, and concise."
        
        # Send the request to Ollama
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            )
            
            return response['response']
        except Exception as ollama_error:
            st.error(f"Error connecting to Ollama: {ollama_error}")
            return f"Could not generate analysis. Please ensure Ollama is running with the {model} model installed."
        
    except Exception as e:
        st.error(f"Error generating analysis with Ollama: {e}")
        st.info(f"Please make sure Ollama is running with the {model} model installed.")
        return f"Could not generate analysis. Please ensure Ollama is properly configured with the {model} model."

# StoxChai Tool - Part 5: Chat Interface Functions
# Functions for the chat interface

# Function to create chat interface
def create_chat_interface(selected_ticker=None, stocks_data=None, embeddings=None):
    # Initialize chat history in session state if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = deque(maxlen=50)  # Limit history to 50 messages
    
    # Initialize callback for suggestion buttons if it doesn't exist
    if 'set_suggestion' not in st.session_state:
        st.session_state.set_suggestion = ""
    
    # Function to add messages to chat history
    def add_message(role, content):
        timestamp = time.strftime("%H:%M:%S")
        st.session_state.chat_history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
    
    # Function to handle user input
    def handle_user_input():
        user_input = st.session_state.user_question
        if user_input:
            add_message("user", user_input)
            
            # Process user input
            with st.spinner("Analyzing..."):
                # Determine if this is a stock-related query
                if any(keyword in user_input.lower() for keyword in ["stock", "price", "market", "trend", "analysis", "performance"]):
                    # Process StoxChai query
                    if embeddings is not None and selected_ticker:
                        # Use RAG to get relevant information
                        query_results = query_vector_store(selected_ticker, user_input, embeddings, k=3)
                        
                        # Generate AI response
                        response = generate_chat_response(user_input, selected_ticker, stocks_data.get(selected_ticker, {}), query_results)
                    else:
                        response = "Please select a stock and fetch data first before asking stock-specific questions."
                else:
                    # General financial query
                    response = generate_chat_response(user_input)
                
                add_message("assistant", response)
    
    # Function to set suggestion
    def set_suggestion(suggestion_text):
        st.session_state.set_suggestion = suggestion_text
    
    # If a suggestion was clicked, copy it to the input field
    if st.session_state.set_suggestion:
        initial_value = st.session_state.set_suggestion
        st.session_state.set_suggestion = ""  # Clear for next use
    else:
        initial_value = ""
    
    # Display chat interface
    st.markdown("<h3 class='sub-header'>AI Assistant Chat</h3>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f"<div class='chat-user-message'>"
                    f"<strong>You ({message['timestamp']}):</strong><br>{message['content']}</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-assistant-message'>"
                    f"<strong>AI Assistant ({message['timestamp']}):</strong><br>{message['content']}</div>", 
                    unsafe_allow_html=True
                )
    
    # Input field and button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.text_input("Ask about StoxChai, market trends, or investment strategies:", 
                      key="user_question", 
                      value=initial_value,
                      on_change=handle_user_input)
    with col2:
        st.button("Send", on_click=handle_user_input, use_container_width=True)
    
    # Suggested questions
    st.markdown("<div style='margin-top: 10px;'>", unsafe_allow_html=True)
    st.markdown("<strong>Suggested questions:</strong>", unsafe_allow_html=True)
    suggestion_cols = st.columns(3)
    
    suggestions = [
        "What's the outlook for this stock?",
        "Explain the recent price movements",
        "How does this compare to sector performance?",
        "What do news articles say about this stock?",
        "Should I consider investing now?",
        "What are the key risks for this stock?"
    ]
    
    for i, suggestion in enumerate(suggestions):
        with suggestion_cols[i % 3]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True, 
                         help=f"Click to ask: {suggestion}",
                         on_click=set_suggestion, 
                         args=(suggestion,)):
                pass  # The on_click handler takes care of it
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Function to generate chat responses
def generate_chat_response(query, ticker=None, stock_data=None, query_results=None):
    try:
        # Build a context prompt for Ollama
        if ticker and stock_data:
            data = stock_data.get('data')
            info = stock_data.get('info', {})
            
            # Create a context-rich prompt with error handling
            prompt = f"""As a financial analyst AI assistant, please respond to the following question about {ticker}:

User question: {query}

Stock context:"""
            
            # Add current price with error handling
            prompt += f"\n- Current price: {info.get('currentPrice', 'N/A')}"
            
            # Add recent change with error handling
            try:
                if data is not None and not data.empty and len(data) >= 5:
                    recent_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100
                    prompt += f"\n- Recent change: {recent_change:.2f}% (5-day)"
                else:
                    prompt += "\n- Recent change: N/A"
            except Exception:
                prompt += "\n- Recent change: N/A"
            
            # Add other stock information
            prompt += f"\n- 52-week range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}"
            prompt += f"\n- Sector: {info.get('sector', 'N/A')}"
            prompt += f"\n- Industry: {info.get('industry', 'N/A')}"
            
            # Add RAG query results for additional context
            if query_results:
                prompt += "\n\nAdditional context from analysis:\n"
                for i, doc in enumerate(query_results):
                    if i < 3:  # Limit to top 3 results
                        prompt += f"- {doc.page_content}\n"
            
            prompt += "\nProvide a concise, informative response to the user's question based on this context. Be helpful but avoid making specific investment recommendations."
            
            try:
                # Send to Ollama
                response = ollama.generate(
                    model="llama3",
                    prompt=prompt,
                    options={
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                )
                return response['response']
            except Exception as e:
                # Fallback response if Ollama is unavailable
                recent_change = 0
                try:
                    if data is not None and not data.empty and len(data) >= 5:
                        recent_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-5]) - 1) * 100
                except Exception:
                    pass
                
                sentiment = "positive" if recent_change > 0 else "negative"
                
                return (f"Based on the data for {ticker}, the stock is currently trading at "
                        f"{info.get('currentPrice', 'N/A')} with recent performance showing a "
                        f"{recent_change:.2f}% change over the last 5 days. "
                        f"The overall market sentiment appears to be {sentiment} in the short term.")
        else:
            # General financial questions
            general_prompt = f"""As a financial analyst AI assistant, please respond to the following general finance question:

User question: {query}

Provide a concise, informative response that is helpful but avoids making specific investment recommendations.
"""
            try:
                # Send to Ollama
                response = ollama.generate(
                    model="llama3",
                    prompt=general_prompt,
                    options={
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                )
                return response['response']
            except Exception as e:
                # Fallback response if Ollama is unavailable
                return ("I can help answer your finance questions, but my ability to provide detailed responses "
                       "is currently limited. Could you try asking about a specific stock after selecting it in the sidebar?")
    
    except Exception as e:
        return f"I encountered an error while processing your request: {str(e)}. Please try again or rephrase your question."


# StoxChai Tool - Part 6: Main Application Function (Beginning)
# Main function to run the Streamlit application

# Main app function
def main():
    # Handle missing modules and print helpful error message
    missing_modules = []
    
    # Check for required packages
    required_packages = {
        "sentence_transformers": "sentence-transformers",
        "langchain_community": "langchain-community",
        "ollama": "ollama",
        "yfinance": "yfinance>=0.2.54",
        "newspaper": "newspaper3k",
        "nltk": "nltk",
        "bs4": "beautifulsoup4",
        "plotly": "plotly"
    }
    
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(package)
    
    if missing_modules:
        st.error("Missing required packages. Please install the following packages:")
        install_cmd = f"pip install {' '.join(missing_modules)}"
        st.code(install_cmd)
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("logo/StoxChaiLogo.png", width=80)
        st.title("StoxChai Tool")
        
        # Input for stock symbol
        ticker_input = st.text_input("Enter Stock Symbol (e.g., HDFCBANK.NS, RELIANCE.NS):", value="HDFCBANK.NS")
        
        # Handle multiple tickers
        tickers = [ticker.strip() for ticker in ticker_input.split(',') if ticker.strip()]
        
        # Select time period
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
        
        # RAG query
        st.subheader("RAG Query")
        rag_query = st.text_input("Ask something about the stock:", placeholder="e.g., How has the stock performed in the last month?")
        run_rag_query = st.button("Run Query", use_container_width=True)
        
        # Ollama analysis
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

    # Main area
    st.markdown("<h1 class='main-header'>Advanced StoxChai Dashboard</h1>", unsafe_allow_html=True)
    
    # Display app status
    status_container = st.empty()
    
    # Check yfinance version
    import pkg_resources
    yf_version = pkg_resources.get_distribution("yfinance").version
    if pkg_resources.parse_version(yf_version) < pkg_resources.parse_version("0.2.54"):
        status_container.warning(f"You are using yfinance version {yf_version}. For best results, upgrade to version 0.2.54 or higher to handle the February 2025 API changes.")
    else:
        status_container.success(f"Using yfinance version {yf_version} ✓")
    
    # Initialize stocks data dictionary
    stocks_data = {}
    embeddings = initialize_rag_system()
    selected_ticker = None if not tickers else tickers[0]


    # StoxChai Tool - Part 6b: Main App Function (Middle)
# Continuing the main function with tabs and visualizations

    # Process each ticker
    for ticker in tickers:
        if not ticker:
            continue
            
        selected_ticker = ticker  # Set the current ticker as selected
        st.markdown(f"<h2 class='sub-header'>{ticker}</h2>", unsafe_allow_html=True)
        
        # Check if we need to fetch new data
        should_fetch = fetch_data or is_data_outdated(ticker)
        
        if should_fetch:
            with st.spinner(f"Fetching data for {ticker}..."):
                data, info = load_stock_data(ticker, period=period)
                if data is not None and info is not None:
                    save_stock_data(ticker, data, info)
                    stocks_data[ticker] = {'data': data, 'info': info}
        else:
            # Load from local storage
            data, info = load_local_stock_data(ticker)
            if data is not None and info is not None:
                stocks_data[ticker] = {'data': data, 'info': info}
            else:
                with st.spinner(f"Local data not found. Fetching data for {ticker}..."):
                    data, info = load_stock_data(ticker, period=period)
                    if data is not None and info is not None:
                        save_stock_data(ticker, data, info)
                        stocks_data[ticker] = {'data': data, 'info': info}
        
        # Display stock data if available
        if ticker in stocks_data:
            data = stocks_data[ticker]['data']
            info = stocks_data[ticker]['info']
            
            # Create columns for key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                current_price = info.get('currentPrice', data['Close'].iloc[-1] if not data.empty else 'N/A')
                if isinstance(current_price, (int, float)):
                    st.markdown(f"<div class='metric-value'>{current_price:.2f}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='metric-value'>{current_price}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Current Price</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                if not data.empty and len(data) > 1:
                    try:
                        daily_change = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
                        change_class = "up-value" if daily_change >= 0 else "down-value"
                        st.markdown(f"<div class='metric-value {change_class}'>{daily_change:.2f}%</div>", unsafe_allow_html=True)
                    except Exception:
                        st.markdown("<div class='metric-value'>N/A</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='metric-value'>N/A</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Daily Change</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col3:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                volume = info.get('averageVolume', data['Volume'].iloc[-1] if not data.empty else 'N/A')
                if isinstance(volume, (int, float)):
                    st.markdown(f"<div class='metric-value'>{volume:,}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='metric-value'>{volume}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Volume</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col4:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                market_cap = info.get('marketCap', 'N/A')
                if isinstance(market_cap, (int, float)):
                    if market_cap >= 1e9:
                        market_cap_str = f"{market_cap/1e9:.2f}B"
                    elif market_cap >= 1e6:
                        market_cap_str = f"{market_cap/1e6:.2f}M"
                    else:
                        market_cap_str = f"{market_cap:,}"
                    st.markdown(f"<div class='metric-value'>{market_cap_str}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='metric-value'>{market_cap}</div>", unsafe_allow_html=True)
                st.markdown("<div class='metric-label'>Market Cap</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Volume Analysis", "News Sentiment", "Company Info"])
            
            with tab1:
                if not data.empty:
                    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                    
                    # Price chart with candlestick
                    fig = go.Figure()
                    
                    # Add candlestick chart
                    fig.add_trace(go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Price'
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
                            line=dict(color='green', width=1)
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{ticker} Stock Price",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        template="plotly_white",
                        height=500,
                        xaxis_rangeslider_visible=False,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
            with tab2:
                if not data.empty:
                    st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                    
                    # Volume chart
                    fig = go.Figure()
                    
                    # Add volume bars with error handling
                    try:
                        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
                                for i in range(len(data))]
                        
                        fig.add_trace(go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            marker_color=colors,
                            name='Volume'
                        ))
                    except Exception:
                        # Fallback if there's an error with the colors
                        fig.add_trace(go.Bar(
                            x=data.index,
                            y=data['Volume'],
                            name='Volume'
                        ))
                    
                    # Add 20-day average volume
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Volume'].rolling(window=20).mean(),
                        mode='lines',
                        name='20-Day Avg Volume',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{ticker} Trading Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        template="plotly_white",
                        height=400,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional volume analysis
                    vol_col1, vol_col2 = st.columns(2)
                    
                    with vol_col1:
                        # Recent volume trend with error handling
                        if len(data) >= 10:
                            try:
                                recent_volume = data['Volume'].tail(10)
                                avg_recent_volume = recent_volume.mean()
                                avg_total_volume = data['Volume'].mean()
                                vol_change = ((avg_recent_volume / avg_total_volume) - 1) * 100
                                
                                st.markdown("<div class='card'>", unsafe_allow_html=True)
                                st.markdown("<div class='sub-header'>Recent Volume Trend</div>", unsafe_allow_html=True)
                                
                                if vol_change > 15:
                                    st.markdown("📈 **Volume is significantly higher** than historical average")
                                elif vol_change > 5:
                                    st.markdown("📈 **Volume is higher** than historical average")
                                elif vol_change < -15:
                                    st.markdown("📉 **Volume is significantly lower** than historical average")
                                elif vol_change < -5:
                                    st.markdown("📉 **Volume is lower** than historical average")
                                else:
                                    st.markdown("📊 **Volume is in line** with historical average")
                                    
                                st.markdown(f"Recent 10-day average volume: {avg_recent_volume:,.0f}")
                                st.markdown(f"Historical average volume: {avg_total_volume:,.0f}")
                                st.markdown(f"Change: {vol_change:.2f}%")
                                st.markdown("</div>", unsafe_allow_html=True)
                            except Exception as vol_error:
                                st.markdown("<div class='card'>", unsafe_allow_html=True)
                                st.markdown("<div class='sub-header'>Recent Volume Trend</div>", unsafe_allow_html=True)
                                st.markdown("Could not calculate volume trends due to data issues.")
                                st.markdown("</div>", unsafe_allow_html=True)
                    
                    with vol_col2:
                        # Volume vs price correlation with error handling
                        if len(data) >= 20:
                            try:
                                # Calculate rolling correlation
                                data['Price_Change'] = data['Close'].pct_change()
                                data['Volume_Change'] = data['Volume'].pct_change()
                                
                                # Remove NaN values
                                correlation_data = data.dropna()
                                
                                if len(correlation_data) > 0:
                                    # Calculate correlation
                                    correlation = correlation_data['Price_Change'].corr(correlation_data['Volume_Change'])
                                    
                                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                                    st.markdown("<div class='sub-header'>Volume-Price Correlation</div>", unsafe_allow_html=True)



# StoxChai Tool - Part 6c: Main App Function (End)
# Final part of the main function with RAG, AI analysis, and chat interface

                                    if correlation > 0.5:
                                        st.markdown("📈 **Strong positive correlation** between volume and price changes")
                                    elif correlation > 0.2:
                                        st.markdown("📈 **Moderate positive correlation** between volume and price changes")
                                    elif correlation < -0.5:
                                        st.markdown("📉 **Strong negative correlation** between volume and price changes")
                                    elif correlation < -0.2:
                                        st.markdown("📉 **Moderate negative correlation** between volume and price changes")
                                    else:
                                        st.markdown("📊 **Weak or no correlation** between volume and price changes")
                                        
                                    st.markdown(f"Correlation coefficient: {correlation:.2f}")
                                    st.markdown("</div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                                    st.markdown("<div class='sub-header'>Volume-Price Correlation</div>", unsafe_allow_html=True)
                                    st.markdown("Insufficient data to calculate correlation.")
                                    st.markdown("</div>", unsafe_allow_html=True)
                            except Exception as corr_error:
                                st.markdown("<div class='card'>", unsafe_allow_html=True)
                                st.markdown("<div class='sub-header'>Volume-Price Correlation</div>", unsafe_allow_html=True)
                                st.markdown("Could not calculate correlation due to data issues.")
                                st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
            with tab3:
                st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                
                # Fetch news and perform sentiment analysis
                with st.spinner("Fetching news articles..."):
                    company_name = info.get('longName', ticker)
                    news_articles = fetch_news(ticker, company_name)
                
                if news_articles:
                    # Update RAG system with news data
                    if embeddings is not None:
                        update_vector_store(ticker, data, news_articles, embeddings)
                    
                    # Display sentiment summary
                    sentiment_scores = [article['sentiment_score'] for article in news_articles]
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    
                    positive_news = sum(1 for s in sentiment_scores if s > 0.2)
                    negative_news = sum(1 for s in sentiment_scores if s < -0.2)
                    neutral_news = len(sentiment_scores) - positive_news - negative_news
                    
                    # Create sentiment chart
                    sentiment_data = [
                        {"category": "Positive", "count": positive_news, "color": "#059669"},
                        {"category": "Neutral", "count": neutral_news, "color": "#9CA3AF"},
                        {"category": "Negative", "count": negative_news, "color": "#DC2626"}
                    ]
                    
                    fig = px.bar(
                        sentiment_data,
                        x="category",
                        y="count",
                        color="category",
                        color_discrete_map={
                            "Positive": "#059669",
                            "Neutral": "#9CA3AF",
                            "Negative": "#DC2626"
                        },
                        title=f"News Sentiment Distribution for {ticker}"
                    )
                    
                    fig.update_layout(
                        xaxis_title="Sentiment Category",
                        yaxis_title="Number of Articles",
                        showlegend=False,
                        template="plotly_white",
                        height=300
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display average sentiment score gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=avg_sentiment,
                        title={"text": "Average Sentiment Score"},
                        gauge={
                            "axis": {"range": [-1, 1]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [-1, -0.2], "color": "firebrick"},
                                {"range": [-0.2, 0.2], "color": "lightgray"},
                                {"range": [0.2, 1], "color": "forestgreen"}
                            ],
                            "threshold": {
                                "line": {"color": "black", "width": 4},
                                "thickness": 0.75,
                                "value": avg_sentiment
                            }
                        }
                    ))
                    
                    fig.update_layout(
                        height=250,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display news articles
                    st.markdown("<div class='sub-header'>Recent News Articles</div>", unsafe_allow_html=True)
                    
                    for article in news_articles:
                        # Determine sentiment class for styling
                        if article['sentiment'] == "Positive":
                            sentiment_class = "positive-sentiment"
                        elif article['sentiment'] == "Negative":
                            sentiment_class = "negative-sentiment"
                        else:
                            sentiment_class = "neutral-sentiment"
                        
                        # Create card for each news article
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown(f"<div class='news-title'>{article['title']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='news-source'>{article['source']} • {article['date']}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='news-sentiment {sentiment_class}'>Sentiment: {article['sentiment']} ({article['sentiment_score']:.2f})</div>", unsafe_allow_html=True)
                        
                        # Show snippet of content
                        if article['content'] != "Content unavailable":
                            snippet = article['content'][:200] + "..." if len(article['content']) > 200 else article['content']
                            st.write(snippet)
                        
                        st.markdown(f"[Read full article]({article['url']})")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                else:
                    st.info("No news articles found for this stock.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            with tab4:
                st.markdown("<div class='tab-content'>", unsafe_allow_html=True)
                
                # Company information
                if info:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("<div class='sub-header'>Company Overview</div>", unsafe_allow_html=True)
                        
                        # Company details
                        st.markdown(f"**Name**: {info.get('longName', ticker)}")
                        st.markdown(f"**Symbol**: {ticker}")
                        st.markdown(f"**Sector**: {info.get('sector', 'N/A')}")
                        st.markdown(f"**Industry**: {info.get('industry', 'N/A')}")
                        st.markdown(f"**Exchange**: {info.get('exchange', 'N/A')}")
                        st.markdown(f"**Currency**: {info.get('currency', 'N/A')}")
                        
                        # Add business summary if available
                        if 'longBusinessSummary' in info:
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown("**Business Summary**:")
                            st.write(info['longBusinessSummary'])
                            
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown("<div class='card'>", unsafe_allow_html=True)
                        st.markdown("<div class='sub-header'>Key Statistics</div>", unsafe_allow_html=True)
                        
                        # Financial metrics
                        metrics = [
                            ("Market Cap", info.get('marketCap', 'N/A'), lambda x: f"₹{x:,.0f}" if isinstance(x, (int, float)) else x),
                            ("P/E Ratio", info.get('trailingPE', 'N/A'), lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x),
                            ("EPS (TTM)", info.get('trailingEps', 'N/A'), lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x),
                            ("Dividend Yield", info.get('dividendYield', 'N/A'), lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x),
                            ("52-Week High", info.get('fiftyTwoWeekHigh', 'N/A'), lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x),
                            ("52-Week Low", info.get('fiftyTwoWeekLow', 'N/A'), lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x),
                            ("50-Day Average", info.get('fiftyDayAverage', 'N/A'), lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x),
                            ("200-Day Average", info.get('twoHundredDayAverage', 'N/A'), lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x),
                            ("Beta", info.get('beta', 'N/A'), lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x),
                            ("Book Value", info.get('bookValue', 'N/A'), lambda x: f"₹{x:.2f}" if isinstance(x, (int, float)) else x),
                            ("Price to Book", info.get('priceToBook', 'N/A'), lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x),
                            ("Return on Equity", info.get('returnOnEquity', 'N/A'), lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x),
                            ("Profit Margins", info.get('profitMargins', 'N/A'), lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else x)
                        ]
                        
                        # Display metrics in a simple table
                        for label, value, formatter in metrics:
                            formatted_value = formatter(value)
                            st.markdown(f"**{label}**: {formatted_value}")
                            
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                else:
                    st.info("Company information not available.")
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Handle RAG query
            if run_rag_query and rag_query and embeddings is not None:
                st.markdown("<div class='sub-header'>RAG Query Results</div>", unsafe_allow_html=True)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                with st.spinner("Processing query..."):
                    query_results = query_vector_store(ticker, rag_query, embeddings)
                    
                    if query_results:
                        st.markdown(f"**Query**: {rag_query}")
                        st.markdown("**Results**:")
                        
                        for i, doc in enumerate(query_results):
                            st.markdown(f"**Result {i+1}**:")
                            st.write(doc.page_content)
                            st.markdown("---")
                    else:
                        st.info("No relevant information found in the database. Try a different query or fetch more data.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            # Handle Ollama analysis
            if generate_ai_analysis and embeddings is not None:
                st.markdown("<div class='sub-header'>AI Analysis</div>", unsafe_allow_html=True)
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                
                with st.spinner("Generating AI analysis..."):
                    # Get RAG query results for additional context
                    query_results = query_vector_store(ticker, "recent performance trends sentiment", embeddings, k=3)
                    
                    # Generate analysis
                    analysis = generate_analysis(ticker, data, info, news_articles, query_results, model=ai_model)
                    
                    if analysis:
                        st.markdown(analysis)
                    else:
                        st.error(f"Failed to generate AI analysis. Make sure Ollama is running with the {ai_model} model installed.")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            st.error(f"Failed to load data for {ticker}. Please check if the ticker symbol is correct.")
    
    # Add chat interface
    st.markdown("---")
    create_chat_interface(selected_ticker, stocks_data, embeddings)
    
    # Footer
    st.markdown("<footer>", unsafe_allow_html=True)
    st.markdown("Data source: Yahoo Finance | Built with Streamlit, Plotly, and Ollama")
    st.markdown("</footer>", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()