# tools.py

# Ensure typing compatibility first
import sys
if sys.version_info >= (3, 9):
    from typing import Annotated, Dict, List, Any, Optional
else:
    from typing import Dict, List, Any, Optional
    from typing_extensions import Annotated

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain.tools import BaseTool, tool
from langchain.pydantic_v1 import BaseModel, Field
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import io
import base64

# Models for structured inputs
class StockTickerInput(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol")
    period: str = Field("1y", description="Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)")

class PortfolioInput(BaseModel):
    tickers: str = Field(..., description="Comma-separated list of stock tickers")
    weights: str = Field(..., description="Comma-separated list of weights (should sum to 1)")
    period: str = Field("1y", description="Time period for historical data")

class TickerComparisonInput(BaseModel):
    tickers: str = Field(..., description="Comma-separated list of stock tickers to compare")
    metrics: str = Field("return,volatility,sharpe", description="Comma-separated list of metrics to compare")
    period: str = Field("1y", description="Time period for historical data")

# Basic Stock Data Tool
@tool("get_stock_data", description="Get historical data and key metrics for a stock ticker")
def get_stock_data(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Fetch historical stock data for a given ticker symbol.
    
    Args:
        ticker: The stock ticker symbol (e.g., AAPL, MSFT)
        period: Time period for data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
    Returns:
        Dictionary with stock data summary
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        print("hist...: ",hist)
        if hist.empty:
            return {"error": f"No data found for ticker {ticker}"}
        
        # Calculate key metrics
        returns = hist['Close'].pct_change().dropna()
        
        result = {
            "ticker": ticker,
            "current_price": hist['Close'].iloc[-1],
            "price_change": hist['Close'].iloc[-1] - hist['Open'].iloc[0],
            "percent_change": (hist['Close'].iloc[-1] / hist['Open'].iloc[0] - 1) * 100,
            "average_volume": hist['Volume'].mean(),
            "volatility": returns.std() * np.sqrt(252) * 100,  # Annualized volatility as percentage
            "data_period": f"{hist.index.min().date()} to {hist.index.max().date()}",
            "price_data": hist['Close'].tolist()[-10:],  # Last 10 data points
            "dates": [d.strftime('%Y-%m-%d') for d in hist.index.tolist()[-10:]]
        }
        print("yahoo data: ",result)
        return result
    except Exception as e:
        return {"error": f"Error retrieving data for {ticker}: {str(e)}"}

# Risk Metrics Tool
@tool("calculate_risk_metrics", description="Calculate risk metrics for a list of stocks including volatility, Sharpe ratio, and VaR")
def calculate_risk_metrics(tickers_str: str, period: str = "1y") -> Dict[str, Any]:
    """
    Calculate key risk metrics for a list of stocks.
    
    Args:
        tickers_str: Comma-separated list of ticker symbols (e.g., "AAPL,MSFT,GOOGL")
        period: Time period for historical data
        
    Returns:
        Dictionary with risk metrics for each ticker and portfolio
    """
    tickers = [t.strip() for t in tickers_str.split(',')]
    
    try:
        # Get data for all tickers
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                data[ticker] = hist['Close']
        
        if not data:
            return {"error": "No data found for the provided tickers"}
        
        # Create a dataframe with all close prices
        df = pd.DataFrame(data)
        
        # Calculate daily returns
        returns = df.pct_change().dropna()
        
        # Calculate metrics for each ticker
        results = {}
        for ticker in tickers:
            if ticker in returns.columns:
                ticker_returns = returns[ticker]
                
                results[ticker] = {
                    "volatility": ticker_returns.std() * np.sqrt(252) * 100,  # Annualized volatility
                    "sharpe_ratio": (ticker_returns.mean() / ticker_returns.std()) * np.sqrt(252),  # Sharpe ratio assuming 0% risk-free rate
                    "var_95": ticker_returns.quantile(0.05) * 100,  # Value at Risk (95% confidence)
                    "max_drawdown": (df[ticker] / df[ticker].cummax() - 1).min() * 100,  # Maximum drawdown
                    "average_return": ticker_returns.mean() * 100  # Average daily return
                }
        
        return results
    except Exception as e:
        return {"error": f"Error calculating risk metrics: {str(e)}"}

# Portfolio Analysis Tool
@tool("analyze_portfolio", description="Analyze a portfolio of stocks with given weights to get risk-return metrics")
def analyze_portfolio(tickers_str: str, weights_str: str, period: str = "1y") -> Dict[str, Any]:
    """
    Analyze a portfolio of stocks with given weights.
    
    Args:
        tickers_str: Comma-separated list of ticker symbols (e.g., "AAPL,MSFT,GOOGL")
        weights_str: Comma-separated list of portfolio weights (e.g., "0.4,0.3,0.3")
        period: Time period for historical data
        
    Returns:
        Dictionary with portfolio analysis results
    """
    tickers = [t.strip() for t in tickers_str.split(',')]
    weights = [float(w.strip()) for w in weights_str.split(',')]
    
    if len(tickers) != len(weights):
        return {"error": "Number of tickers must match number of weights"}
    
    if abs(sum(weights) - 1.0) > 0.0001:
        return {"error": "Weights must sum to 1"}
    
    try:
        # Get data for all tickers
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                data[ticker] = hist['Close']
        
        if not data:
            return {"error": "No data found for the provided tickers"}
        
        # Create a dataframe with all close prices
        df = pd.DataFrame(data)
        
        # Calculate daily returns
        returns = df.pct_change().dropna()
        
        # Calculate portfolio returns
        portfolio_returns = returns.dot(weights)
        
        # Calculate covariance matrix and portfolio volatility
        cov_matrix = returns.cov()
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * np.sqrt(252) * 100
        
        # Calculate other portfolio metrics
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252)
        var_95 = portfolio_returns.quantile(0.05) * 100
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1
        
        result = {
            "portfolio_composition": {ticker: weight for ticker, weight in zip(tickers, weights)},
            "expected_annual_return": portfolio_returns.mean() * 252 * 100,  # Annualized return as percentage
            "portfolio_volatility": portfolio_volatility,  # Annualized volatility as percentage
            "sharpe_ratio": sharpe_ratio,
            "value_at_risk_95": var_95,
            "max_drawdown": (cumulative_returns + 1).div((cumulative_returns + 1).cummax()).sub(1).min() * 100,
            "data_period": f"{returns.index.min().date()} to {returns.index.max().date()}",
        }
        
        return result
    except Exception as e:
        return {"error": f"Error analyzing portfolio: {str(e)}"}

# Market Indicators Tool
@tool("get_market_indicators", description="Get current market indicators including major indices and treasury yields")
def get_market_indicators() -> Dict[str, Any]:
    """
    Fetch current market indicators and economic data.
    
    Returns:
        Dictionary with market indicators
    """
    try:
        # Get major index data
        indices = {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ": "^IXIC",
            "Russell 2000": "^RUT",
            "VIX": "^VIX"
        }
        
        results = {}
        for name, ticker in indices.items():
            idx = yf.Ticker(ticker)
            hist = idx.history(period="5d")
            if not hist.empty:
                results[name] = {
                    "current": hist['Close'].iloc[-1],
                    "daily_change": (hist['Close'].iloc[-1] / hist['Close'].iloc[-2] - 1) * 100,
                    "weekly_change": (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                }
        
        # Get treasury yield data
        treasury_tickers = {
            "10Y Treasury Yield": "^TNX",
            "30Y Treasury Yield": "^TYX",
            "5Y Treasury Yield": "^FVX"
        }
        
        results["Treasury Yields"] = {}
        for name, ticker in treasury_tickers.items():
            idx = yf.Ticker(ticker)
            hist = idx.history(period="5d")
            if not hist.empty:
                results["Treasury Yields"][name] = hist['Close'].iloc[-1]
        
        return results
    except Exception as e:
        return {"error": f"Error retrieving market indicators: {str(e)}"}

# Stock Comparison Tool
@tool("compare_stocks", description="Compare multiple stocks based on selected metrics like returns, volatility, and Sharpe ratio")
def compare_stocks(tickers_str: str, metrics_str: str = "return,volatility,sharpe", period: str = "1y") -> Dict[str, Any]:
    """
    Compare multiple stocks based on selected metrics.
    
    Args:
        tickers_str: Comma-separated list of ticker symbols to compare
        metrics_str: Comma-separated list of metrics to compare (return,volatility,sharpe,var,drawdown)
        period: Time period for historical data
        
    Returns:
        Dictionary with comparison results
    """
    tickers = [t.strip() for t in tickers_str.split(',')]
    metrics = [m.strip().lower() for m in metrics_str.split(',')]
    
    valid_metrics = ["return", "volatility", "sharpe", "var", "drawdown"]
    metrics = [m for m in metrics if m in valid_metrics]
    
    if not metrics:
        metrics = ["return", "volatility", "sharpe"]
    
    try:
        # Get data for all tickers
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                data[ticker] = hist['Close']
        
        if not data:
            return {"error": "No data found for the provided tickers"}
        
        # Create a dataframe with all close prices
        df = pd.DataFrame(data)
        
        # Calculate daily returns
        returns = df.pct_change().dropna()
        
        # Calculate metrics for each ticker
        results = {"comparison": {}}
        
        for ticker in tickers:
            if ticker in returns.columns:
                ticker_returns = returns[ticker]
                ticker_data = {}
                
                if "return" in metrics:
                    ticker_data["annual_return"] = ticker_returns.mean() * 252 * 100
                
                if "volatility" in metrics:
                    ticker_data["volatility"] = ticker_returns.std() * np.sqrt(252) * 100
                
                if "sharpe" in metrics:
                    ticker_data["sharpe_ratio"] = (ticker_returns.mean() / ticker_returns.std()) * np.sqrt(252)
                
                if "var" in metrics:
                    ticker_data["value_at_risk_95"] = ticker_returns.quantile(0.05) * 100
                
                if "drawdown" in metrics:
                    ticker_data["max_drawdown"] = (df[ticker] / df[ticker].cummax() - 1).min() * 100
                
                results["comparison"][ticker] = ticker_data
        
        results["period"] = f"{returns.index.min().date()} to {returns.index.max().date()}"
        results["metrics_included"] = metrics
        
        return results
    except Exception as e:
        return {"error": f"Error comparing stocks: {str(e)}"}

# Function to generate chart for visualization
def generate_stock_chart(tickers_str: str, period: str = "1y") -> str:
    """
    Generate a chart comparing stock performance and return as base64 image.
    
    Args:
        tickers_str: Comma-separated list of ticker symbols
        period: Time period for data
        
    Returns:
        Base64 encoded PNG image
    """
    tickers = [t.strip() for t in tickers_str.split(',')]
    
    try:
        # Get data
        data = {}
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                data[ticker] = hist['Close']
        
        if not data:
            return "Error: No data found for the provided tickers"
        
        # Create normalized dataframe (start at 100)
        df = pd.DataFrame(data)
        normalized = df / df.iloc[0] * 100
        
        # Create plot
        plt.figure(figsize=(10, 6))
        for ticker in normalized.columns:
            plt.plot(normalized.index, normalized[ticker], label=ticker)
        
        plt.title(f"Comparative Performance (Normalized to 100)")
        plt.xlabel("Date")
        plt.ylabel("Normalized Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_str
    except Exception as e:
        return f"Error generating chart: {str(e)}"

# Create tools from the functions above
# Using simple tool decorators instead of StructuredTool to avoid typing issues
stock_data_tool = get_stock_data
risk_metrics_tool = calculate_risk_metrics
portfolio_analysis_tool = analyze_portfolio
market_indicators_tool = get_market_indicators
stock_comparison_tool = compare_stocks

# List of all financial analysis tools
FINANCIAL_TOOLS = [
    stock_data_tool,
    risk_metrics_tool,
    portfolio_analysis_tool,
    market_indicators_tool,
    stock_comparison_tool
]