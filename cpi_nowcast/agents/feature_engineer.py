import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def feature_engineer(state):
    """Engineer time-series features."""
    logger.info("=== Feature Engineering Agent ===")
    
    state['agent_status'] = state.get('agent_status', {})
    
    try:
        data = state['clean_data'].set_index('date').sort_index()
        
        # Validate required columns
        if 'CPI' not in data.columns:
            raise ValueError("CPI column missing in clean_data")
        if data.index.name != 'date':
            raise ValueError("date not set as index")
        
        # Lag features (CPI, vehicle sales proxy)
        data['CPI_lag1'] = data['CPI'].shift(1)
        data['CPI_lag2'] = data['CPI'].shift(2)
        
        # Rolling stats
        data['CPI_roll_mean_3'] = data['CPI'].rolling(3).mean()
        data['vehicle_total'] = data.select_dtypes(include=np.number).iloc[:, :-1].sum(axis=1)  # Exclude CPI
        data['vehicle_roll_mean_3'] = data['vehicle_total'].rolling(3).mean()
        
        # % changes
        data['vehicle_pct_change'] = data['vehicle_total'].pct_change()
        
        # Air quality proxy (if available)
        if 'NitrogenDioxide_NO2_column_number_density' in data.columns:
            data['air_no2_roll_mean'] = data['NitrogenDioxide_NO2_column_number_density'].rolling(3).mean()
        
        # Drop NaN rows
        features = data.fillna(0)
        
        state['features'] = features
        logger.info(f"Features shape: {features.shape}")
        state['agent_status']['feature_engineer'] = 'success'
        logger.info("Feature Engineering complete")
        print()  # Blank line for readability
    except Exception as e:
        state['agent_status']['feature_engineer'] = f"failure: {e}"
        logger.error(f"Feature Engineering failed: {e}")
        raise
    
    return state
