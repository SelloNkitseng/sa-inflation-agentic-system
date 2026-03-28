from langgraph.graph import StateGraph, END
from typing import Dict, Any
from agents.data_collector import data_collector
from agents.data_cleaner import data_cleaner
from agents.eda_agent import eda_agent
from agents.feature_engineer import feature_engineer
from agents.modeler import modeler
from agents.evaluator import evaluator
from agents.insights_agent import insights_agent
import time
import logging

logger = logging.getLogger(__name__)

def timed_agent(agent_func, agent_name):
    """Wrapper to time agent execution."""
    def wrapper(state):
        start_time = time.time()
        logger.info(f"Starting {agent_name}...")
        result = agent_func(state)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{agent_name} completed in {duration:.2f} seconds")
        return result
    return wrapper

def create_workflow():
    """Create LangGraph workflow."""
    workflow = StateGraph(state_schema=Dict[str, Any])
    
    # Add nodes with timing
    workflow.add_node("data_collect", timed_agent(data_collector, "Data Collector"))
    workflow.add_node("data_clean", timed_agent(data_cleaner, "Data Cleaner"))
    workflow.add_node("eda", timed_agent(eda_agent, "EDA Agent"))
    workflow.add_node("features", timed_agent(feature_engineer, "Feature Engineer"))
    workflow.add_node("model", timed_agent(modeler, "Modeler"))
    workflow.add_node("eval", timed_agent(evaluator, "Evaluator"))
    workflow.add_node("insights", timed_agent(insights_agent, "Insights Agent"))
    
    # Edges
    workflow.set_entry_point("data_collect")
    workflow.add_edge("data_collect", "data_clean")
    workflow.add_edge("data_clean", "eda")
    workflow.add_edge("eda", "features")
    workflow.add_edge("features", "model")
    workflow.add_edge("model", "eval")
    workflow.add_edge("eval", "insights")
    workflow.add_edge("insights", END)
    
    app = workflow.compile()
    return app
