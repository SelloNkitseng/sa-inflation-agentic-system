from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def evaluator(state):
    """Evaluate models, select best."""
    logger.info("=== Evaluation Agent ===")
    
    state['agent_status'] = state.get('agent_status', {})
    
    try:
        X_test, y_test, predictions = state['test_data']
        
        metrics = {}
        for name, pred in predictions.items():
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            metrics[name] = {'RMSE': rmse, 'MAE': mae}
            
            if rmse > 5.0:
                logger.warning(f"{name} RMSE >5: {rmse:.2f}")
        
        metrics_df = pd.DataFrame(metrics).T
        best_model = metrics_df['RMSE'].idxmin()
        
        state['metrics'] = metrics_df
        state['best_model'] = best_model
        
        # Save predictions
        results = pd.DataFrame({'true': y_test, **predictions})
        results.to_csv('predictions.csv', index=False)

        # Print aligned metrics for readability with full model names
        model_name_map = state.get('model_name_map', {})
        print("\nEvaluation Metrics:")
        print("{:<30} {:>15} {:>15}".format('Model', 'RMSE', 'MAE'))
        print("{:<30} {:>15} {:>15}".format('-'*30, '-'*15, '-'*15))
        for key, row in metrics_df.iterrows():
            full_name = model_name_map.get(key, key)
            print("{:<30} {:>15.4f} {:>15.4f}".format(full_name, row['RMSE'], row['MAE']))
        
        print(f"\nBest model: {model_name_map.get(best_model, best_model)} (RMSE {metrics_df.loc[best_model, 'RMSE']:.4f})")

        logger.info(f"Metrics: {metrics_df}\nBest: {best_model}")
        state['agent_status']['evaluator'] = 'success'
        logger.info("Evaluation complete")
        print()  # Blank line for readability
    except Exception as e:
        state['agent_status']['evaluator'] = f"failure: {e}"
        logger.error(f"Evaluation failed: {e}")
        raise
    
    return state
