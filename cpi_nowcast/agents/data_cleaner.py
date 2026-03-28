import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def data_cleaner(state):
    """Clean, align, merge datasets."""
    logger.info("=== Data Cleaning Agent ===")
    
    state['agent_status'] = state.get('agent_status', {})
    
    try:
        raw_data = state['raw_data']
        
        # Validate required data
        if 'cpi' not in raw_data or raw_data['cpi'].empty:
            raise ValueError("CPI data missing or empty")
        if 'naamsa' not in raw_data or raw_data['naamsa'].empty:
            raise ValueError("Naamsa data missing or empty")
        
        # Filter CPI to Headline only to avoid duplicates
        raw_data['cpi'] = raw_data['cpi'][raw_data['cpi']['Category'] == 'Headline_CPI'].copy()
        logger.info(f"CPI after Headline filter: {raw_data['cpi'].shape}")
        
        # Align to common monthly dates (month-end 2022-2023)
        common_dates = pd.date_range('2022-01-31', '2023-04-30', freq='ME')
        
        clean_dfs = {}
        for name, df in raw_data.items():
            # Remove any duplicate dates
            df = df.drop_duplicates(subset=['date'])
            df_clean = df.set_index('date').reindex(common_dates)
            if name == 'cpi':
                df_clean = df_clean.ffill()  # Don't fill CPI with 0, keep NaN for missing
            else:
                df_clean = df_clean.ffill().fillna(0)  # Fill others with 0
            clean_dfs[name] = df_clean
        
        # Detect CPI outliers (z-score >3)
        cpi = clean_dfs['cpi'].reset_index()
        cpi['z_score'] = np.abs(stats.zscore(cpi['Value']))
        outliers = cpi[cpi['z_score'] > 3]
        if not outliers.empty:
            logger.warning(f"CPI outliers detected: {len(outliers)} rows")
        
        # Merge all (multi-index if needed)
        merged = clean_dfs['naamsa'].join(clean_dfs['cpi']['Value'], how='outer')
        if 'air' in clean_dfs:
            merged = merged.join(clean_dfs['air'][['NitrogenDioxide_NO2_column_number_density']], how='outer')
        
        merged.index.name = 'date'
        merged = merged.reset_index().rename(columns={'Value': 'CPI'})
        merged['CPI_pct_change'] = merged['CPI'].pct_change()
        
        # Filter to months with valid CPI data (not NaN)
        valid_months = merged['CPI'].notna()
        merged = merged[valid_months].copy()
        
        if merged.empty:
            raise ValueError("No valid CPI data found after filtering missing values")
        
        logger.info(f"Filtered to {len(merged)} months with valid CPI data")
        
        state['clean_data'] = merged
        logger.info(f"Merged dataset: {merged.shape}")
        state['agent_status']['data_cleaner'] = 'success'
        logger.info("Data Cleaning complete")
        print()  # Blank line for readability
    except Exception as e:
        state['agent_status']['data_cleaner'] = f"failure: {e}"
        logger.error(f"Data Cleaning failed: {e}")
        raise
    
    return state
