# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import time
import os
from dotenv import load_dotenv

# Import our financial agent components
from main import FinancialOrchestrator
from tools import FINANCIAL_TOOLS, generate_stock_chart

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add the page title
def add_page_title(title):
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)

# Initialize session state variables
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = FinancialOrchestrator()
    # Add tools to each agent
    for agent_name, agent in st.session_state.orchestrator.agents.items():
        for tool in FINANCIAL_TOOLS:
            agent.add_tool(tool)
    
    # Initialize all agents
    st.session_state.orchestrator.initialize_all_agents()

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'tickers': [],
        'weights': [],
        'data': None
    }

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-bottom: 1rem;
    }
    .info-text {
        font-size: 1rem;
        color: #424242;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 0.5rem;
        border-radius: 0.3rem;
    }
    .agent-message {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #FAFAFA;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.markdown('<div class="main-header">AI Financial Analyst</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Chat with Financial AI", "Portfolio Analysis", "Market Dashboard", "Risk Assessment"])

# API Key Input
groq_api_key = st.sidebar.text_input("GROQ API Key", type="password")
if groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key

# Helper function to display chat messages
def display_conversation():
    for i, message in enumerate(st.session_state.conversation_history):
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="agent-message">AI: {message["content"]}</div>', unsafe_allow_html=True)
            
            # Display any included charts
            if "chart" in message and message["chart"]:
                st.image(f"data:image/png;base64,{message['chart']}")

# Main application
if not groq_api_key:
    st.warning("Please enter your GROQ API key in the sidebar to use the application.")
else:
    # Chat with Financial AI page
    if page == "Chat with Financial AI":
        add_page_title("Chat with Financial AI")
        
        st.markdown("""
        <div class="info-text">
        Ask me anything about financial markets, stock analysis, risk management, 
        or financial planning. I can help you analyze stocks, assess portfolio risk, 
        and provide market insights.
        </div>
        """, unsafe_allow_html=True)
        
        # Display conversation history
        display_conversation()
        
        # User input
        user_query = st.text_input("Ask a financial question:", key="user_input")
        
        if st.button("Submit") and user_query:
            # Add user message to conversation
            st.session_state.conversation_history.append({"role": "user", "content": user_query})
            
            # Get response from the appropriate agent
            with st.spinner("Analyzing..."):
                response = st.session_state.orchestrator.route_query(user_query)
                
                # Check if we need to include a chart
                chart_base64 = None
                if "stock" in user_query.lower() and "chart" in user_query.lower():
                    # Try to extract ticker symbols from the query
                    import re
                    tickers = re.findall(r'\b[A-Z]{1,5}\b', user_query)
                    if tickers:
                        tickers_str = ",".join(tickers)
                        chart_base64 = generate_stock_chart(tickers_str)
                
                # Add response to conversation
                st.session_state.conversation_history.append({
                    "role": "assistant", 
                    "content": response.get("output", "I had trouble processing that request."),
                    "chart": chart_base64
                })
            
            # Rerun to display the updated conversation
            st.rerun()
    
    # Portfolio Analysis page
    elif page == "Portfolio Analysis":
        add_page_title("Portfolio Analysis")
        manage_portfolio()
    
    # Market Dashboard page
    elif page == "Market Dashboard":
        add_page_title("Market Dashboard")
        
        st.markdown("""
        <div class="info-text">
        This dashboard provides an overview of current market conditions, major indices, 
        and economic indicators.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Refresh Market Data"):
            with st.spinner("Fetching market data..."):
                # Get market indicators using the agent
                result = st.session_state.orchestrator.market_agent.run(
                    "Provide a comprehensive market overview with current indicators"
                )
                
                st.markdown('<div class="sub-header">Market Overview</div>', unsafe_allow_html=True)
                st.json(result)
                
                # Query the market agent for insights
                insights = st.session_state.orchestrator.market_agent.run(
                    "Based on current market data, what are the key insights and trends investors should be aware of?"
                )
                
                st.markdown('<div class="sub-header">Market Insights</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="agent-message">{insights.get("output", "No insights available.")}</div>', unsafe_allow_html=True)
    
    # Risk Assessment page
    elif page == "Risk Assessment":
        add_page_title("Risk Assessment")
        
        st.markdown("""
        <div class="info-text">
        Evaluate and analyze financial risks for specific stocks or your portfolio.
        </div>
        """, unsafe_allow_html=True)
        
        # Stock risk analysis
        st.markdown('<div class="sub-header">Individual Stock Risk Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            ticker = st.text_input("Stock Ticker", "AAPL")
            period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        with col2:
            risk_threshold = st.slider("Risk Threshold (%)", 0.0, 50.0, 20.0, 0.5)
            st.session_state.orchestrator.risk_agent.set_risk_threshold(risk_threshold / 100.0)
        
        if st.button("Analyze Risk"):
            with st.spinner(f"Analyzing risk for {ticker}..."):
                # Query the risk agent
                query = f"Analyze the risk profile for {ticker} over a {period} period. Consider my risk threshold of {risk_threshold}%."
                result = st.session_state.orchestrator.risk_agent.run(query)
                
                # Display result
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.write(f"Risk Analysis for {ticker}:")
                st.write(result.get("output", "Could not analyze risk at this time."))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Generate and display chart
                chart_base64 = generate_stock_chart(ticker, period)
                if chart_base64 and not chart_base64.startswith("Error"):
                    st.image(f"data:image/png;base64,{chart_base64}")

# Helper function for portfolio management
def manage_portfolio():
    st.markdown('<div class="sub-header">Portfolio Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_ticker = st.text_input("Add Stock (Ticker Symbol)", "")
        new_weight = st.number_input("Weight (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0)
        
        if st.button("Add to Portfolio"):
            if new_ticker and new_weight > 0:
                # Check if ticker already exists
                if new_ticker in st.session_state.portfolio['tickers']:
                    st.warning(f"{new_ticker} is already in your portfolio.")
                else:
                    st.session_state.portfolio['tickers'].append(new_ticker)
                    st.session_state.portfolio['weights'].append(new_weight / 100.0)
                    st.success(f"Added {new_ticker} to portfolio with weight {new_weight}%")
    
    with col2:
        if st.session_state.portfolio['tickers']:
            # Display current portfolio
            portfolio_df = pd.DataFrame({
                'Ticker': st.session_state.portfolio['tickers'],
                'Weight (%)': [w * 100 for w in st.session_state.portfolio['weights']]
            })
            
            st.markdown('<div class="info-text">Current Portfolio</div>', unsafe_allow_html=True)
            st.dataframe(portfolio_df)
            
            # Check if weights sum to 100%
            weight_sum = sum(st.session_state.portfolio['weights']) * 100
            if abs(weight_sum - 100.0) > 0.01:
                st.warning(f"Portfolio weights sum to {weight_sum:.2f}%. They should sum to 100%.")
            
            if st.button("Reset Portfolio"):
                st.session_state.portfolio = {
                    'tickers': [],
                    'weights': [],
                    'data': None
                }
                st.success("Portfolio has been reset.")
        else:
            st.info("Your portfolio is empty. Add some stocks to get started.")
    
    # Portfolio analysis
    if st.session_state.portfolio['tickers'] and abs(sum(st.session_state.portfolio['weights']) - 1.0) <= 0.01:
        st.markdown('<div class="sub-header">Portfolio Analysis</div>', unsafe_allow_html=True)
        
        period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        
        if st.button("Analyze Portfolio"):
            with st.spinner("Analyzing portfolio..."):
                # Prepare inputs for the portfolio analysis tool
                tickers_str = ",".join(st.session_state.portfolio['tickers'])
                weights_str = ",".join([str(w) for w in st.session_state.portfolio['weights']])
                
                # Generate portfolio analysis using the agent
                query = f"Analyze this portfolio: tickers={tickers_str}, weights={weights_str}, period={period}"
                result = st.session_state.orchestrator.risk_agent.run(query)
                
                # Store the result in session state
                st.session_state.portfolio['data'] = result
                
                # Generate chart
                chart_base64 = generate_stock_chart(tickers_str, period)
                
                # Display the results
                st.markdown('<div class="highlight">', unsafe_allow_html=True)
                st.write("Portfolio Analysis Results:")
                st.json(result)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display chart
                if chart_base64 and not chart_base64.startswith("Error"):
                    st.image(f"data:image/png;base64,{chart_base64}")
                else:
                    st.error("Could not generate chart.")