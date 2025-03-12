# Structure of the Financial Analysis Agent System
# main.py

import os
from typing import List, Dict, Any, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

# Base Agent class
class FinancialAgent:
    """Base class for financial analysis agents"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        # self.llm = ChatGroq(
        #     api_key=os.getenv("GROQ_API_KEY"),
        #     model_name="llama3-70b-8192"  # Can be changed to other Groq models
        # )
        self.llm = ChatNVIDIA(base_url="http://10.141.0.5:31430/v1", NVIDIA_API_KEY=os.getenv("NGC_API_KEY"),
        model="nvidia/llama-3.1-nemotron-70b-instruct", temperature=0.1, max_tokens=1000, top_p=1.0)
        self.tools: List[BaseTool] = []
        self.agent_executor = None
    
    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent's toolkit"""
        self.tools.append(tool)
    
    def initialize_agent(self) -> None:
        """Initialize the agent with tools and prompt"""
        prompt = PromptTemplate.from_template(
            """You are a financial analysis expert named {name}.
            {description}
            
            To solve a user's request, think step-by-step to determine the best approach.
            
            You have access to the following tools:
            {tools}
            
            Use the following format:
            
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
            
            User's request: {input}
            
            {agent_scratchpad}
            """
        )
        
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run the agent with a user query"""
        if not self.agent_executor:
            self.initialize_agent()
        
        return self.agent_executor.invoke({"input": query, "name": self.name, "description": self.description})


# Risk Management Agent - specialized for risk assessment
class RiskManagementAgent(FinancialAgent):
    """Agent specialized in financial risk management"""
    
    def __init__(self):
        super().__init__(
            name="Risk Analyzer",
            description="I specialize in analyzing financial risk factors, calculating risk metrics, and providing risk mitigation strategies."
        )
        self.risk_threshold = 0.7  # Default risk threshold
    
    def set_risk_threshold(self, threshold: float) -> None:
        """Set the risk threshold for alerts"""
        self.risk_threshold = threshold
    
    def analyze_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze risk metrics for a portfolio"""
        # This would contain more complex risk calculations in a real implementation
        return self.agent_executor.invoke({
            "input": f"Analyze the risk profile of this portfolio: {portfolio_data}",
            "name": self.name,
            "description": self.description
        })


# Market Analysis Agent - specialized for market trends and analysis
class MarketAnalysisAgent(FinancialAgent):
    """Agent specialized in market analysis and forecasting"""
    
    def __init__(self):
        super().__init__(
            name="Market Analyst",
            description="I specialize in analyzing market trends, economic indicators, and providing market forecasts."
        )
    
    def analyze_market_trends(self, market_data: Dict[str, Any], timeframe: str = "short-term") -> Dict[str, Any]:
        """Analyze market trends based on provided data"""
        return self.agent_executor.invoke({
            "input": f"Analyze {timeframe} market trends based on this data: {market_data}",
            "name": self.name,
            "description": self.description
        })


# Financial Planning Agent - specialized for financial planning and strategy
class FinancialPlanningAgent(FinancialAgent):
    """Agent specialized in financial planning and strategy"""
    
    def __init__(self):
        super().__init__(
            name="Financial Planner",
            description="I specialize in creating financial plans, investment strategies, and optimizing financial decisions."
        )
    
    def create_investment_strategy(self, financial_goals: Dict[str, Any], risk_profile: str) -> Dict[str, Any]:
        """Create an investment strategy based on financial goals and risk profile"""
        return self.agent_executor.invoke({
            "input": f"Create an investment strategy for these financial goals: {financial_goals}, with a risk profile of: {risk_profile}",
            "name": self.name,
            "description": self.description
        })


# Orchestrator Agent - coordinates between specialized agents
class FinancialOrchestrator:
    """Orchestrator to coordinate between different financial agents"""
    
    def __init__(self):
        self.risk_agent = RiskManagementAgent()
        self.market_agent = MarketAnalysisAgent()
        self.planning_agent = FinancialPlanningAgent()
        self.agents = {
            "risk": self.risk_agent,
            "market": self.market_agent,
            "planning": self.planning_agent
        }
    
    def initialize_all_agents(self) -> None:
        """Initialize all agents with their tools"""
        for agent in self.agents.values():
            agent.initialize_agent()
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """Route a query to the appropriate agent based on content"""
        # In a more complex system, we would use an LLM to determine routing
        if "risk" in query.lower():
            return self.risk_agent.run(query)
        elif "market" in query.lower() or "trend" in query.lower():
            return self.market_agent.run(query)
        elif "plan" in query.lower() or "strategy" in query.lower() or "investment" in query.lower():
            return self.planning_agent.run(query)
        else:
            # Default to planning agent if unclear
            return self.planning_agent.run(query)
    
    def comprehensive_analysis(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a comprehensive analysis using all agents"""
        results = {}
        
        # Get risk analysis
        results["risk_analysis"] = self.risk_agent.run(
            f"Provide a comprehensive risk analysis of this financial data: {financial_data}"
        )
        
        # Get market analysis
        results["market_analysis"] = self.market_agent.run(
            f"Analyze market trends and opportunities based on this financial data: {financial_data}"
        )
        
        # Get financial planning recommendations
        results["financial_planning"] = self.planning_agent.run(
            f"Provide financial planning recommendations based on this data: {financial_data}"
        )
        
        return results