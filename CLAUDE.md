# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project titled "1 LLM vs. 100 persona given sLLMs" that compares the decision-making capabilities of a single high-performance LLM against 100 smaller LLMs for stock investment decisions based on news events.

## Core Architecture

### Main Components

1. **baseline.py** - Core implementation of the 1-vs-100 experiment framework
   - `OpenAIAgent` class for wrapping OpenRouter API calls to different models
   - `StockPredictor` class that orchestrates expert vs crowd predictions
   - Uses OpenRouter API for accessing both GPT-5 (expert) and GPT-5-nano (crowd) models

2. **eventResponse.py** - Simplified experimental template for Colab environments
   - `LLMExperiment` class for managing investment decision experiments
   - Aggregates weighted responses from multiple small models
   - Compares against single high-performance model decisions

3. **Data Pipeline Scripts** (in `yujin/` directory):
   - `02_crawling_articles.py` - News crawling for specific stock events
   - `03_crawling_articles_OOP.py` - Object-oriented version of news crawler
   - `04_attach_marketcap.py` - Adds market capitalization data to articles
   - `01_data_related.ipynb` - Data preprocessing and S&P 100 stock analysis

### Data Structure

- **Event Data**: S&P 100 stocks with significant price movements
- **News Data**: Crawled articles from the day before each event using GNews
- **Stock Data**: S&P 100 company information (symbols, names, sectors)
- **Market Data**: Market capitalization information attached to news articles

## Development Environment

### Dependencies
- Python >=3.13,<4.0
- pandas - Data manipulation
- langchain - LLM framework integration
- openai - OpenAI API client (used with OpenRouter)
- dotenv - Environment variable management
- Additional dependencies: yfinance, gnews, beautifulsoup4, requests

### Environment Setup
```bash
# Install dependencies
poetry install

# Set up environment variables
# Create .env file with:
OPENROUTER_API_KEY=your_api_key_here
```

### Key Configuration
- Uses OpenRouter API as proxy for accessing multiple LLM models
- Expert model: `openai/gpt-5`
- Crowd models: `openai/gpt-5-nano` (100 instances)
- Default response format: JSON with `decision` and `confidence` fields

## Data Processing Pipeline

1. **Stock Event Identification**: Identify S&P 100 stocks with significant movements
2. **News Collection**: Crawl relevant news articles from the day before each event
3. **Data Enrichment**: Add market capitalization and sector information
4. **Experiment Execution**: Run 1-vs-100 comparison experiments

## File Locations

- **Data**: `data/` directory contains CSV files and processed articles
- **Raw Articles**: `data/articles/` with JSON and CSV subdirectories
- **Processed Data**: Event data, stock data, and filtered news datasets
- **Notebooks**: `yujin/01_data_related.ipynb` for data exploration and analysis

## Testing and Validation

The project uses real financial data and news articles to validate the experimental framework. Test runs can be performed with sample events to ensure the pipeline works correctly before running full experiments.