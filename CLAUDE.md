# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project titled "1 LLM vs. 100 persona given sLLMs" that compares the decision-making capabilities of a single high-performance LLM against 100 smaller LLMs for stock investment decisions based on news events.

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry
poetry install

# Run the main experiment framework
python baseline.py

# Run the simplified Colab-friendly experiment
python eventResponse.py

# Execute specific data processing scripts
python yujin/B_crawling_articles.py
python yujin/C_crawling_articles_OOP.py
python yujin/D_attach_marketcap.py
python yujin/E_data_form_making.py
python yujin/F_call_openrouter.py

# View Jupyter notebook for data analysis
jupyter notebook yujin/A_data_related.ipynb
```

### Environment Variables
Create a `.env` file with:
```
OPENROUTER_API_KEY=your_api_key_here
```

## Core Architecture

### Experiment Framework Architecture

The project follows a modular agent-based architecture for comparing expert vs crowd intelligence:

1. **Agent Abstraction Layer** (`baseline.py`):
   - `BaseModelAgent` abstract class defines the prediction interface
   - `OpenAIAgent` implements OpenRouter API calls with error handling
   - `AgentPrediction` TypedDict enforces response format: `{"decision": str, "confidence": int}`

2. **Prediction Orchestration** (`StockPredictor` class):
   - Manages expert agent (single high-performance model)
   - Coordinates crowd agents (100 smaller models)
   - Aggregates crowd decisions using confidence-weighted voting

3. **Experimental Templates**:
   - **baseline.py**: Full-featured framework with modular agent design
   - **eventResponse.py**: Simplified template optimized for Colab environments with hardcoded configurations

### Data Processing Pipeline

Sequential data processing architecture in `yujin/` directory:

1. **A_data_related.ipynb**: Initial data exploration and S&P 100 analysis
2. **B_crawling_articles.py**: Basic news crawler implementation (uses GNews API)
3. **C_crawling_articles_OOP.py**: Object-oriented news crawler with enhanced features and immediate saving
4. **D_attach_marketcap.py**: Market capitalization data enrichment using yfinance
5. **E_data_form_making.py**: Data formatting for LLM agent consumption (`convert_df_to_agent_format()`)
6. **F_call_openrouter.py**: Direct OpenRouter API integration example (deprecated - use baseline.py instead)
7. **G_scoring_answer.py**: Scoring and evaluation of prediction results

### OpenRouter API Integration

All LLM interactions use OpenRouter as a proxy for accessing multiple models:
- **Expert Model**: `openai/gpt-5` for high-performance reasoning
- **Crowd Models**: `openai/gpt-5-nano` for cost-effective bulk predictions
- **Response Format**: JSON objects with `decision`, `confidence`, and optional `reason` fields
- **Error Handling**: Fallback to default predictions (`{"decision": "hold", "confidence": 50}`)

## Data Structure and Flow

### Input Data Format
```python
agent_data = {
    'symbol': 'AAPL',
    'search_date': '2024-12-11',
    'titles': 'News title 1 / News title 2 / ...',
    'descriptions': 'Description 1 / Description 2 / ...',
    'sector': 'Technology'
}
```

### File Organization
- **Raw Data**: `data/articles/json/` and `data/articles/csv/`
- **Processed Data**: `data/event_data.csv`, `data/stock_data.csv`
- **Market Data**: Market cap enriched datasets with timestamps (e.g., `news_with_market_cap_YYYYMMDD_HHMMSS.csv`)
- **Prediction Results**: `data/answer/prediction_results.csv`
- **Configuration**: `.env` for API keys, `pyproject.toml` for dependencies

### Experiment Flow
1. Load and filter event data by symbol and date
2. Format data for LLM consumption using `E_data_form_making.convert_df_to_agent_format()`
3. Execute expert prediction (single model)
4. Execute crowd predictions (100 models in parallel)
5. Aggregate crowd results using confidence-weighted voting
6. Compare expert vs crowd final decisions
7. Evaluate results using `G_scoring_answer.py`

## Key Implementation Details

### Model Configuration
- All models use JSON response format enforcement
- Temperature settings vary: expert (0.0-0.2), crowd (0.6-1.0 with variance)
- Max tokens limited to 50-256 for cost efficiency
- Retry logic with exponential backoff for API failures

### Decision Aggregation
Crowd decisions use confidence-weighted voting:
```python
buy_sum = sum(res["confidence"] for res in crowd_results if res["decision"] == "buy")
hold_sum = sum(res["confidence"] for res in crowd_results if res["decision"] == "hold")
final_decision = "buy" if buy_sum > hold_sum else "hold"
```

## External Dependencies

### Required Python Packages
- **pandas**: Data manipulation and CSV processing
- **langchain**: LLM framework integration
- **openai**: OpenAI/OpenRouter API client
- **dotenv**: Environment variable management
- **gnews**: Google News API for article crawling (`B_crawling_articles.py`, `C_crawling_articles_OOP.py`)
- **yfinance**: Yahoo Finance API for market cap data (`D_attach_marketcap.py`)
- **requests**: HTTP client for OpenRouter API (`eventResponse.py`)

### API Services
- **OpenRouter**: Proxy service for accessing multiple LLM providers (requires `OPENROUTER_API_KEY`)
- **GNews**: News article collection (used in crawling scripts)
- **Yahoo Finance**: Historical market data and market capitalization