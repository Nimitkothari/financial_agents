# AI Agents for Financial Analysis and Risk Management

This project implements a class-based agentic system for financial analysis and risk management using Langchain, Groq LLMs, and Streamlit. The system is containerized with Docker for easy deployment.

## Features

- **Multi-Agent System**: Specialized agents for risk management, market analysis, and financial planning
- **Financial Analysis Tools**: Stock data retrieval, risk metrics calculation, portfolio analysis
- **Interactive UI**: Streamlit-based interface for interacting with financial agents
- **Containerized Deployment**: Docker configuration for easy setup and deployment

## Components

1. **Agent Architecture**:
   - Base `FinancialAgent` class
   - Specialized agent classes for different financial domains
   - Orchestrator for coordinating between agents

2. **Financial Tools**:
   - Stock data retrieval
   - Risk metrics calculation
   - Portfolio analysis
   - Market indicators
   - Stock comparison
   - Chart generation

3. **User Interface**:
   - Chat interface for querying agents
   - Portfolio management
   - Market dashboard
   - Risk assessment dashboard

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Groq API key/ NVAPI key

### Setup and Running

1. **Clone the repository**

2. **Set environment variables**

   Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```
   OR
   ```
   NGC_API_KEY=your_nv_key
   ```

3. **Build and run with Docker Compose**

   ```bash
   sudo docker compose up --build
   ```

4. **Access the application**

   Open your browser and go to `http://localhost:8501`

### Running Without Docker

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

## Using the Application

### Chat with Financial AI
Ask any financial question to get insights and analysis from the specialized agents.

### Portfolio Analysis
- Add stocks to your portfolio with custom weights
- Analyze portfolio performance and risk metrics
- View comparative stock performance

### Market Dashboard
- Get current market indicators and economic data
- Receive insights on market trends

### Risk Assessment
- Analyze risk profiles for individual stocks
- Set custom risk thresholds
- Get risk mitigation recommendations

## Extending the System

### Adding New Tools
To add new financial tools, create new functions in `tools.py` and add them to the `FINANCIAL_TOOLS` list.

### Creating New Specialized Agents
Extend the `FinancialAgent` base class to create new specialized agents for specific financial domains.

## Future Enhancements

- Integration with real-time market data APIs
- Backtesting capabilities for investment strategies
- Enhanced data visualization
- User authentication and saved portfolios
- Export capabilities for reports and analysis
