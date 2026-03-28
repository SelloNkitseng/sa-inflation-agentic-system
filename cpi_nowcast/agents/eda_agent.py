from utils.data import plot_cpi_changes, plot_cpi_trend, plot_feature_correlation
from utils.llm import get_llm, generate_insight
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def eda_agent(state):
    """Perform EDA, plot, LLM narrative."""
    logger.info("=== EDA Agent ===")
    
    state['agent_status'] = state.get('agent_status', {})
    
    try:
        clean_data = state['clean_data']
        
        # Stats
        stats = clean_data.describe()
        pct_change = clean_data['CPI_pct_change'].describe()
        
        eda_results = {
            "stats": stats.to_dict(),
            "pct_change_stats": pct_change.to_dict(),
            "shape": clean_data.shape,
            "head": clean_data.head().to_dict()
        }
        
        # Create 3 different types of plots
        plot1_path = plot_cpi_changes(state['raw_data']['cpi'], "plot1_cpi_pct_change.png")
        plot2_path = plot_cpi_trend(state['raw_data']['cpi'], "plot2_cpi_trend.png")
        plot3_path = plot_feature_correlation(clean_data, "plot3_correlation_heatmap.png")
        
        # LLM narrative with descriptions of each plot
        llm = get_llm()
        prompt_data = {
            "context": f"""
Analyze this CPI dataset and three plots:
- Dataset: {eda_results['shape']} rows, CPI % change stats: {eda_results['pct_change_stats']}
- Plot 1: CPI % change bar chart
- Plot 2: CPI trend line chart  
- Plot 3: Feature correlation heatmap

Provide brief economic insights (100-150 words).
"""
        }
        narrative = generate_insight(llm, prompt_data)
        
        eda_results['narrative'] = narrative
        eda_results['plots'] = {
            'pct_change': plot1_path,
            'trend': plot2_path,
            'correlation': plot3_path
        }
        
        state['eda_results'] = eda_results
        state['eda_narrative'] = narrative
        
        # Print narrative to console for visibility
        print("\n" + "="*80)
        print("📊 EDA Analysis - Chart Descriptions & Insights:")
        print("="*80)
        print(narrative)
        print("="*80 + "\n")
        
        state['agent_status']['eda_agent'] = 'success'
        logger.info("EDA complete")
        print()  # Blank line for readability
    except Exception as e:
        state['agent_status']['eda_agent'] = f"failure: {e}"
        logger.error(f"EDA failed: {e}")
        raise
    
    return state
