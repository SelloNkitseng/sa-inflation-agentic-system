from utils.llm import get_llm, generate_insight
import logging

logger = logging.getLogger(__name__)

def insights_agent(state):
    """Generate LLM insights."""
    logger.info("=== Insights Agent ===")
    
    state['agent_status'] = state.get('agent_status', {})
    
    try:
        metrics = state['metrics']
        best = state['best_model']
        eda_narrative = state.get('eda_narrative', 'EDA analysis not available')
        eda_results = state.get('eda_results', {})
        
        context = f"""
COMPREHENSIVE CPI NOWCAST ANALYSIS REPORT

DATA OVERVIEW:
- Dataset shape: {eda_results.get('shape', 'N/A')}
- CPI % change stats: {eda_results.get('pct_change_stats', 'N/A')}

EDA ANALYSIS SUMMARY:
{eda_narrative}

MODEL PERFORMANCE:
{metrics.to_string()}

BEST MODEL: {best}

Provide a comprehensive analysis that tells the complete story:
1. Data characteristics and key trends from EDA
2. Model performance comparison and why the best model excels
3. Economic implications for South African inflation nowcasting
4. Integration of EDA insights with modeling results
5. Recommendations for policymakers and future improvements

Make this a cohesive narrative (300-400 words) that flows from data exploration to final predictions.
"""
        
        llm = get_llm()
        insights = generate_insight(llm, {"context": context})
        
        llm = get_llm()
        insights = generate_insight(llm, {"context": context})
        
        state['insights'] = insights
        
        logger.info("Insights generated")
        state['agent_status']['insights_agent'] = 'success'
        logger.info("Insights complete")
        print()  # Blank line for readability
    except Exception as e:
        state['agent_status']['insights_agent'] = f"failure: {e}"
        logger.error(f"Insights failed: {e}")
        raise
    
    return state
