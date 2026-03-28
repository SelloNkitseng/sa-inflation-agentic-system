#!/usr/bin/env python3
"""
CPI Nowcast Agentic Workflow - Main Entry Point
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph import create_workflow
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting CPI Nowcast Agentic Workflow...")
    print("🚀 Starting CPI Nowcast Agentic Workflow...")
    
    # Create and run graph
    app = create_workflow()
    initial_state = {}
    
    try:
        logger.info("Invoking workflow...")
        result = app.invoke(initial_state)
        logger.info("Workflow completed successfully")
        
        print("\n✅ Workflow complete!")
        print("\n📊 Key Results:")
        print(f"Best Model: {result.get('best_model', 'N/A')}")
        print(f"Metrics saved to predictions.csv")
        
        # Display all 3 plots
        plots = result.get('eda_results', {}).get('plots', {})
        if plots:
            print(f"Plots generated:")
            print(f"  - {plots.get('pct_change', 'N/A')}")
            print(f"  - {plots.get('trend', 'N/A')}")
            print(f"  - {plots.get('correlation', 'N/A')}")
        
        print(f"\nInsights: {result.get('insights', 'N/A')}")
        
        # Print agent statuses
        statuses = result.get('agent_status', {})
        print("\n🔍 Agent Status Summary:")
        for agent, status in statuses.items():
            print(f"  {agent}: {status}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        print(f"\n❌ Workflow failed: {e}")
        # If partial result, show statuses
        if 'result' in locals() and result.get('agent_status'):
            statuses = result['agent_status']
            print("\n🔍 Partial Agent Status:")
            for agent, status in statuses.items():
                print(f"  {agent}: {status}")
        logging.shutdown()
        sys.exit(1)
    
    logging.shutdown()
