# 🇮🇳 StoxChai - Production-Ready Indian Stock Market Analysis Platform

[![CI/CD Pipeline](https://github.com/your-username/stoxchai/workflows/StoxChai%20CI/CD%20Pipeline/badge.svg)](https://github.com/your-username/stoxchai/actions)
[![Code Coverage](https://codecov.io/gh/your-username/stoxchai/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/stoxchai)
[![Docker Image](https://img.shields.io/docker/image-size/stoxchai/app)](https://hub.docker.com/r/stoxchai/app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

StoxChai is a comprehensive, production-ready Indian stock market analysis platform built with modern technologies. It provides real-time data analysis, AI-powered insights, and advanced visualization capabilities specifically optimized for NSE and BSE markets.

### Key Highlights

- **🇮🇳 Indian Market Focus**: Optimized for NSE and BSE with support for Indian stock symbols, currency (₹), and market timings
- **🚀 Production Ready**: Docker containerization, monitoring, caching, and CI/CD pipeline
- **🤖 AI-Powered**: Integration with Ollama for advanced stock analysis and insights
- **📊 Real-time Data**: Live market data with intelligent caching and rate limiting
- **🔒 Enterprise Security**: Comprehensive security features, input validation, and rate limiting
- **📈 Advanced Analytics**: Technical indicators, sentiment analysis, and performance metrics

## ✨ Features

### Core Features
- **Real-time Stock Data**: Live NSE/BSE stock prices, volumes, and market indices
- **Interactive Charts**: Candlestick charts, volume analysis, and technical indicators
- **News Sentiment Analysis**: AI-powered sentiment analysis of Indian financial news
- **Market Insights**: Top gainers/losers, market mood, and sector performance
- **Watchlist Management**: Personal stock watchlists with real-time updates
- **AI Analysis**: Advanced stock analysis using local LLM models

### Technical Features
- **Modular Architecture**: Clean separation of concerns with proper MVC structure
- **Caching System**: Redis/File-based caching for optimal performance
- **Database Integration**: PostgreSQL with optimized schemas and indexes
- **Monitoring & Logging**: Comprehensive monitoring with Prometheus and Grafana
- **Security**: Rate limiting, input validation, and secure authentication
- **Testing**: Comprehensive test suite with unit and integration tests

### Indian Market Specific
- **Currency Support**: All prices displayed in Indian Rupees (₹)
- **Market Hours**: Automatic detection of Indian market trading hours
- **Holiday Calendar**: Support for Indian trading holidays
- **Exchange Support**: Both NSE (.NS) and BSE (.BO) symbols
- **Indian Indices**: NIFTY 50, SENSEX, NIFTY Bank, and sector indices

## 🏗️ Architecture

```
StoxChai/
├── src/
│   ├── config/          # Configuration management
│   ├── core/            # Core business models
│   ├── data/            # Data sources and providers
│   ├── ui/              # Streamlit UI components
│   └── utils/           # Utilities (cache, logging, security)
├── tests/               # Test suite
├── deployment/          # Deployment configurations
├── docker/              # Docker configurations
└── docs/                # Documentation
```

### Technology Stack
- **Backend**: Python 3.9+, Streamlit, FastAPI
- **Database**: PostgreSQL 15+ with Redis caching
- **AI/ML**: Ollama, LangChain, Sentence Transformers
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker, Docker Compose, Kubernetes
- **CI/CD**: GitHub Actions

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/stoxchai.git
   cd stoxchai
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the application**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Main App: http://localhost:8501
   - Monitoring: http://localhost:3000 (Grafana)
   - Metrics: http://localhost:9090 (Prometheus)

### Local Development

1. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-prod.txt
   ```

2. **Set up services**
   ```bash
   # Start PostgreSQL and Redis
   docker-compose up -d postgres redis ollama
   ```

3. **Run the application**
   ```bash
   streamlit run main.py
   ```

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis 7+
- Ollama (for AI features)

### Detailed Installation

1. **System Dependencies**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3-pip python3-venv postgresql-client redis-tools
   
   # macOS
   brew install python@3.9 postgresql redis
   ```

2. **Install Ollama**
   ```bash
   # Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # macOS
   brew install ollama
   
   # Pull required models
   ollama pull qwen2.5:latest
   ollama pull deepseek-r1:7b
   ```

3. **Database Setup**
   ```bash
   # Create database
   createdb stoxchai
   
   # Run migrations
   psql -d stoxchai -f deployment/init.sql
   ```

## 🔧 Configuration

### Environment Variables

```bash
# Application
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stoxchai
DB_USER=stoxchai
DB_PASSWORD=secure-password

# Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis-password

# External APIs
ALPHA_VANTAGE_API_KEY=your-api-key

# AI Service
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
OLLAMA_DEFAULT_MODEL=qwen2.5:latest

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PASSWORD=admin-password
```

### Indian Market Configuration

The application comes pre-configured for Indian markets:

- **Exchanges**: NSE (primary), BSE (secondary)
- **Indices**: NIFTY 50, SENSEX, NIFTY Bank, sector indices
- **Market Hours**: 9:15 AM - 3:30 PM IST
- **Currency**: Indian Rupees (₹)
- **Trading Holidays**: Pre-configured for 2025

## 🚢 Deployment

### Production Deployment

1. **Using Docker Compose**
   ```bash
   # Production environment
   docker-compose -f docker-compose.yml up -d
   ```

2. **Using Kubernetes**
   ```bash
   kubectl apply -f k8s/
   ```

3. **Environment Setup**
   - Set up SSL certificates
   - Configure domain and DNS
   - Set up monitoring and alerting
   - Configure backup strategies

### Scaling

- **Horizontal Scaling**: Multiple Streamlit instances behind load balancer
- **Database Scaling**: Read replicas and connection pooling
- **Cache Scaling**: Redis cluster setup
- **AI Scaling**: Multiple Ollama instances

## 📊 Monitoring

### Health Checks

The application includes comprehensive health checks:

- **Database connectivity**
- **Cache availability**
- **AI service status**
- **Market data API**
- **System resources**

Access health status at: `http://localhost:8501/health`

### Metrics

Prometheus metrics available at: `http://localhost:9090`

Key metrics:
- Request count and latency
- Cache hit/miss ratios
- API call success rates
- System resource usage
- Active user count

### Dashboards

Grafana dashboards available at: `http://localhost:3000`

- Application performance
- System metrics
- Business metrics
- Error tracking

## 🧪 Testing

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest tests/ -v --cov=src --cov-report=html

# Performance tests
pytest tests/performance/ -v
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability scanning

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Standards

- **Code Formatting**: Black, isort
- **Linting**: Flake8, mypy
- **Testing**: pytest with >90% coverage
- **Documentation**: Comprehensive docstrings

## 📖 API Documentation

### REST API Endpoints

- `GET /api/v1/stocks/{symbol}` - Get stock data
- `GET /api/v1/indices` - Get market indices
- `GET /api/v1/news/{symbol}` - Get stock news
- `GET /api/v1/health` - Health check

### WebSocket Events

- `stock_price_update` - Real-time price updates
- `market_status_change` - Market open/close events
- `news_alert` - Breaking news alerts

Full API documentation available at: `http://localhost:8501/docs`

## 🔐 Security

### Security Features

- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: API and user action rate limits
- **Authentication**: JWT-based authentication (optional)
- **HTTPS**: SSL/TLS encryption in production
- **CORS**: Proper cross-origin resource sharing
- **CSP**: Content Security Policy headers

### Security Best Practices

- Regular security updates
- Dependency vulnerability scanning
- Secure configuration management
- Access logging and monitoring

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Sources**: Yahoo Finance, NSE, BSE
- **AI Models**: Ollama community models
- **Indian Financial Community**: For feedback and requirements
- **Open Source Libraries**: All the amazing Python packages used

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/stoxchai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/stoxchai/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/stoxchai/wiki)

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. Please consult with qualified financial advisors before making investment decisions. The developers are not responsible for any financial losses incurred through the use of this application.

**Made with ❤️ for the Indian investment community**