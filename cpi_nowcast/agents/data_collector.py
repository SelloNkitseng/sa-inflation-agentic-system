from utils.data import load_data
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def data_collector(state):
    """Load and standardize datasets."""
    logger.info("=== Data Collector Agent ===")
    
    state['agent_status'] = state.get('agent_status', {})
    
    try:
        raw_data = load_data(parent_dir=os.getenv('DATA_DIR'))
        
        # Standardize dates to month-end (handle Excel serial dates)
        for key, df in raw_data.items():
            if 'Month' in df.columns:
                month_col = df['Month']
                if not pd.api.types.is_datetime64_any_dtype(month_col):
                    if month_col.dtype in ['int64', 'float64']:  # Excel serial
                        month_series = pd.to_datetime(month_col, origin='1899-12-30', unit='D')
                    else:
                        month_series = pd.to_datetime(month_col)
                else:
                    month_series = month_col
                df['date'] = month_series.dt.normalize().dt.to_period('M').dt.to_timestamp('M') if pd.api.types.is_datetime64_any_dtype(month_series) else month_series.dt.floor('D')
            else:
                df['date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='MS')
            logger.info(f"{key}: {len(df)} rows")
        
        state['raw_data'] = raw_data
        state['agent_status']['data_collector'] = 'success'
        logger.info("Data Collector complete")
        print()  # Blank line for readability
    except Exception as e:
        state['agent_status']['data_collector'] = f"failure: {e}"
        logger.error(f"Data Collector failed: {e}")
        raise
    
    return state
