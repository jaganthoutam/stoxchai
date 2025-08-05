-- Database initialization script for StoxChai
-- This script sets up the initial database schema for the Indian stock market analysis tool

-- Create database if not exists (handled by Docker)
-- CREATE DATABASE IF NOT EXISTS stoxchai;

-- Use the database
\c stoxchai;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS stocks;
CREATE SCHEMA IF NOT EXISTS users;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Set search path
SET search_path TO stocks, users, analytics, public;

-- Create tables for stock data
CREATE TABLE IF NOT EXISTS stocks.companies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(200) NOT NULL,
    exchange VARCHAR(10) NOT NULL CHECK (exchange IN ('NSE', 'BSE')),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stocks.stock_prices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(15,4),
    high_price DECIMAL(15,4),
    low_price DECIMAL(15,4),
    close_price DECIMAL(15,4),
    volume BIGINT,
    adjusted_close DECIMAL(15,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, date)
);

CREATE TABLE IF NOT EXISTS stocks.market_indices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    current_value DECIMAL(15,4),
    change_value DECIMAL(15,4),
    change_percent DECIMAL(8,4),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, DATE(last_updated))
);

CREATE TABLE IF NOT EXISTS stocks.news_articles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20),
    title TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    source VARCHAR(100),
    published_date TIMESTAMP WITH TIME ZONE,
    content TEXT,
    sentiment_score DECIMAL(5,4),
    sentiment_label VARCHAR(20) CHECK (sentiment_label IN ('POSITIVE', 'NEGATIVE', 'NEUTRAL')),
    relevance_score DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create user-related tables
CREATE TABLE IF NOT EXISTS users.user_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users.user_portfolios (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.user_profiles(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    purchase_price DECIMAL(15,4) NOT NULL,
    purchase_date DATE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS users.watchlists (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.user_profiles(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, symbol)
);

CREATE TABLE IF NOT EXISTS users.price_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.user_profiles(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    alert_type VARCHAR(20) NOT NULL CHECK (alert_type IN ('PRICE_ABOVE', 'PRICE_BELOW', 'VOLUME_SPIKE', 'PERCENTAGE_CHANGE')),
    threshold_value DECIMAL(15,4) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    triggered_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create analytics tables
CREATE TABLE IF NOT EXISTS analytics.user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users.user_profiles(id),
    ip_address INET,
    user_agent TEXT,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    page_views INTEGER DEFAULT 0,
    actions JSONB
);

CREATE TABLE IF NOT EXISTS analytics.api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id UUID REFERENCES users.user_profiles(id),
    ip_address INET,
    status_code INTEGER,
    response_time_ms INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stocks.stock_prices(symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stocks.stock_prices(date DESC);
CREATE INDEX IF NOT EXISTS idx_companies_symbol ON stocks.companies(symbol);
CREATE INDEX IF NOT EXISTS idx_companies_exchange ON stocks.companies(exchange);
CREATE INDEX IF NOT EXISTS idx_news_articles_symbol ON stocks.news_articles(symbol);
CREATE INDEX IF NOT EXISTS idx_news_articles_published_date ON stocks.news_articles(published_date DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_sentiment ON stocks.news_articles(sentiment_label);
CREATE INDEX IF NOT EXISTS idx_user_portfolios_user_id ON users.user_portfolios(user_id);
CREATE INDEX IF NOT EXISTS idx_watchlists_user_id ON users.watchlists(user_id);
CREATE INDEX IF NOT EXISTS idx_price_alerts_user_id ON users.price_alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_price_alerts_symbol ON users.price_alerts(symbol);
CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON analytics.user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON analytics.api_usage(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON analytics.api_usage(endpoint);

-- Create GIN indexes for text search
CREATE INDEX IF NOT EXISTS idx_news_articles_content_gin ON stocks.news_articles USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_companies_name_gin ON stocks.companies USING GIN (to_tsvector('english', name));

-- Create triggers for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_companies_updated_at BEFORE UPDATE ON stocks.companies FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_profiles_updated_at BEFORE UPDATE ON users.user_profiles FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_portfolios_updated_at BEFORE UPDATE ON users.user_portfolios FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some initial data
INSERT INTO stocks.companies (symbol, name, exchange, sector, industry) VALUES
('RELIANCE', 'Reliance Industries Limited', 'NSE', 'Energy', 'Oil & Gas Refining & Marketing'),
('TCS', 'Tata Consultancy Services Limited', 'NSE', 'Information Technology', 'IT - Services'),
('HDFCBANK', 'HDFC Bank Limited', 'NSE', 'Financial Services', 'Banks - Private Sector'),
('INFY', 'Infosys Limited', 'NSE', 'Information Technology', 'IT - Services'),
('ICICIBANK', 'ICICI Bank Limited', 'NSE', 'Financial Services', 'Banks - Private Sector'),
('HINDUNILVR', 'Hindustan Unilever Limited', 'NSE', 'Fast Moving Consumer Goods', 'Personal Products'),
('SBIN', 'State Bank of India', 'NSE', 'Financial Services', 'Banks - Public Sector'),
('BHARTIARTL', 'Bharti Airtel Limited', 'NSE', 'Telecommunication', 'Telecom - Services'),
('ITC', 'ITC Limited', 'NSE', 'Fast Moving Consumer Goods', 'Tobacco Products'),
('KOTAKBANK', 'Kotak Mahindra Bank Limited', 'NSE', 'Financial Services', 'Banks - Private Sector')
ON CONFLICT (symbol) DO NOTHING;

-- Create views for common queries
CREATE OR REPLACE VIEW stocks.latest_prices AS
SELECT DISTINCT ON (symbol) 
    symbol, date, open_price, high_price, low_price, close_price, volume
FROM stocks.stock_prices
ORDER BY symbol, date DESC;

CREATE OR REPLACE VIEW analytics.daily_active_users AS
SELECT 
    DATE(start_time) as date,
    COUNT(DISTINCT user_id) as active_users
FROM analytics.user_sessions
WHERE user_id IS NOT NULL
GROUP BY DATE(start_time)
ORDER BY date DESC;

-- Grant permissions
GRANT USAGE ON SCHEMA stocks TO stoxchai;
GRANT USAGE ON SCHEMA users TO stoxchai;
GRANT USAGE ON SCHEMA analytics TO stoxchai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA stocks TO stoxchai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA users TO stoxchai;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO stoxchai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA stocks TO stoxchai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA users TO stoxchai;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO stoxchai;

-- Create function to get stock performance
CREATE OR REPLACE FUNCTION stocks.get_stock_performance(
    p_symbol VARCHAR(20),
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    symbol VARCHAR(20),
    current_price DECIMAL(15,4),
    change_amount DECIMAL(15,4),
    change_percent DECIMAL(8,4),
    high_52w DECIMAL(15,4),
    low_52w DECIMAL(15,4)
) AS $$
BEGIN
    RETURN QUERY
    WITH price_data AS (
        SELECT 
            sp.symbol,
            sp.close_price,
            sp.date,
            ROW_NUMBER() OVER (PARTITION BY sp.symbol ORDER BY sp.date DESC) as rn
        FROM stocks.stock_prices sp
        WHERE sp.symbol = p_symbol
        AND sp.date >= CURRENT_DATE - INTERVAL '1 year'
    ),
    latest_price AS (
        SELECT close_price as current_price
        FROM price_data
        WHERE rn = 1
    ),
    comparison_price AS (
        SELECT close_price as old_price
        FROM price_data
        WHERE rn = p_days + 1
    ),
    year_high_low AS (
        SELECT 
            MAX(close_price) as high_52w,
            MIN(close_price) as low_52w
        FROM price_data
    )
    SELECT 
        p_symbol,
        lp.current_price,
        (lp.current_price - cp.old_price) as change_amount,
        CASE 
            WHEN cp.old_price > 0 THEN 
                ((lp.current_price - cp.old_price) / cp.old_price * 100)
            ELSE 0
        END as change_percent,
        yhl.high_52w,
        yhl.low_52w
    FROM latest_price lp
    CROSS JOIN comparison_price cp
    CROSS JOIN year_high_low yhl;
END;
$$ LANGUAGE plpgsql;

-- Create function to get market sentiment
CREATE OR REPLACE FUNCTION analytics.get_market_sentiment(
    p_symbol VARCHAR(20) DEFAULT NULL,
    p_days INTEGER DEFAULT 7
)
RETURNS TABLE (
    symbol VARCHAR(20),
    positive_count BIGINT,
    negative_count BIGINT,
    neutral_count BIGINT,
    avg_sentiment DECIMAL(5,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        na.symbol,
        COUNT(*) FILTER (WHERE na.sentiment_label = 'POSITIVE') as positive_count,
        COUNT(*) FILTER (WHERE na.sentiment_label = 'NEGATIVE') as negative_count,
        COUNT(*) FILTER (WHERE na.sentiment_label = 'NEUTRAL') as neutral_count,
        AVG(na.sentiment_score) as avg_sentiment
    FROM stocks.news_articles na
    WHERE (p_symbol IS NULL OR na.symbol = p_symbol)
    AND na.published_date >= CURRENT_DATE - INTERVAL '1 day' * p_days
    GROUP BY na.symbol
    ORDER BY avg_sentiment DESC;
END;
$$ LANGUAGE plpgsql;