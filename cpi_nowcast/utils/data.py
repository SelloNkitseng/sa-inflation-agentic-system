import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def load_data(parent_dir: str = None, fallback_dir: str = "Data"):
    """
    Load all datasets with environment path support and robust error handling.
    
    Args:
        parent_dir: Optional explicit data directory path. If None, uses DATA_DIR env var
                   or falls back to fallback_dir.
        fallback_dir: Default fallback directory if env var and parent_dir not provided.
    
    Returns:
        dict: Dictionary containing 'cpi', 'naamsa', and optionally 'air' DataFrames.
    
    Raises:
        FileNotFoundError: If required data files are not found.
        ValueError: If data parsing or validation fails.
    """
    data = {}
    
    # Determine data directory with environment support
    if parent_dir is None:
        parent_dir = os.getenv('DATA_DIR', fallback_dir)
    
    data_path = Path(parent_dir).resolve()
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    
    # Load CPI data
    try:
        cpi_file = data_path / "CPI_Historic_Values_Zindi_Apr_23.csv"
        if not cpi_file.exists():
            raise FileNotFoundError(f"CPI data file not found: {cpi_file}")
        
        data['cpi'] = pd.read_csv(cpi_file)
        
        # Validate required columns
        required_cpi_cols = ['Month', 'Category', 'Value']
        missing_cols = [col for col in required_cpi_cols if col not in data['cpi'].columns]
        if missing_cols:
            raise ValueError(f"CPI data missing required columns: {missing_cols}")
        
        data['cpi']['Month'] = pd.to_datetime(data['cpi']['Month'], dayfirst=True)
        data['cpi'] = data['cpi'].sort_values('Month').reset_index(drop=True)
        logger.info(f"CPI data loaded: {len(data['cpi'])} rows")
        
    except Exception as e:
        logger.error(f"Failed to load CPI data: {e}")
        raise
    
    # Load Naamsa data
    try:
        naamsa_file = data_path / "Naamsa_Vehicle_Sales.csv"
        if not naamsa_file.exists():
            logger.warning(f"Naamsa data file not found: {naamsa_file}")
            data['naamsa'] = pd.DataFrame()
        else:
            data['naamsa'] = pd.read_csv(naamsa_file)
            
            # Convert numeric columns (skip first column assumed to be date/index)
            for col in data['naamsa'].columns[1:]:
                data['naamsa'][col] = pd.to_numeric(data['naamsa'][col], errors='coerce')
            
            logger.info(f"Naamsa data loaded: {len(data['naamsa'])} rows")
    
    except Exception as e:
        logger.error(f"Failed to load Naamsa data: {e}")
        data['naamsa'] = pd.DataFrame()
    
    # Load Air Quality data (optional)
    try:
        air_file = data_path / "AirQualityData" / "zaf_grouped_sentinel5p.csv"
        if air_file.exists():
            air = pd.read_csv(air_file)
            
            # Aggregate by year if 'year' column exists
            if 'year' in air.columns:
                data['air'] = air.groupby('year').mean(numeric_only=True).reset_index()
            else:
                data['air'] = air
            
            logger.info(f"Air Quality data loaded: {len(data['air'])} rows")
        else:
            logger.info(f"Air Quality data file not found (optional): {air_file}")
            data['air'] = pd.DataFrame()
    
    except Exception as e:
        logger.warning(f"Failed to load Air Quality data (optional): {e}")
        data['air'] = pd.DataFrame()
    
    logger.info("Data loading completed successfully")
    return data


def plot_cpi_changes(df: pd.DataFrame, output_path: str = None):
    """
    Plot monthly CPI % changes with robust error handling.
    
    Args:
        df: DataFrame containing 'Category', 'Value', and 'Month' columns.
        output_path: Optional output file path. If None, uses OUTPUT_DIR env var
                    or defaults to 'eda_plot.png' in current directory.
    
    Returns:
        str: Path to the saved plot file.
    
    Raises:
        ValueError: If required columns are missing or data is invalid.
    """
    # Validate input DataFrame
    required_cols = ['Category', 'Value', 'Month']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Filter for Headline CPI
    headline = df[df['Category'] == 'Headline_CPI'].copy()
    
    if headline.empty:
        raise ValueError("No 'Headline_CPI' category found in input data")
    
    # Calculate percent change
    headline['pct_change'] = headline['Value'].pct_change() * 100
    
    # Determine output path with environment support
    if output_path is None:
        output_dir = os.getenv('OUTPUT_DIR', '.')
        output_path = os.path.join(output_dir, 'eda_plot.png')
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create robust plot
        plt.figure(figsize=(12, 6))
        
        # Filter out NaN values for plotting
        valid_data = headline[headline['pct_change'].notna()].copy()
        
        if valid_data.empty:
            logger.warning("All pct_change values are NaN; creating empty plot")
        else:
            plt.bar(valid_data['Month'], valid_data['pct_change'], alpha=0.7, color='steelblue')
        
        plt.title('Monthly Headline CPI % Change')
        plt.xlabel('Date')
        plt.ylabel('% Change')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save with error handling
        plt.savefig(str(output_path_obj), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plot saved successfully: {output_path_obj}")
        return str(output_path_obj)
    
    except Exception as e:
        plt.close()
        logger.error(f"Failed to save plot: {e}")
        raise


def plot_cpi_trend(df: pd.DataFrame, output_path: str = None):
    """
    Plot CPI trend over time as a line chart.
    
    Args:
        df: DataFrame containing 'Category', 'Value', and 'Month' columns.
        output_path: Optional output file path.
    
    Returns:
        str: Path to the saved plot file.
    """
    required_cols = ['Category', 'Value', 'Month']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Filter for Headline CPI
    headline = df[df['Category'] == 'Headline_CPI'].copy()
    
    if headline.empty:
        raise ValueError("No 'Headline_CPI' category found in input data")
    
    # Determine output path
    if output_path is None:
        output_dir = os.getenv('OUTPUT_DIR', '.')
        output_path = os.path.join(output_dir, 'cpi_trend.png')
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        plt.figure(figsize=(12, 6))
        
        valid_data = headline[headline['Value'].notna()].copy()
        
        if valid_data.empty:
            logger.warning("All CPI values are NaN; creating empty plot")
        else:
            plt.plot(valid_data['Month'], valid_data['Value'], marker='o', linewidth=2, markersize=6, color='darkgreen')
        
        plt.title('CPI Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('CPI Value')
        plt.xticks(rotation=45, ha='right')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig(str(output_path_obj), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trend plot saved successfully: {output_path_obj}")
        return str(output_path_obj)
    
    except Exception as e:
        plt.close()
        logger.error(f"Failed to save trend plot: {e}")
        raise


def plot_feature_correlation(df: pd.DataFrame, output_path: str = None):
    """
    Plot correlation heatmap of numeric features.
    
    Args:
        df: DataFrame with numeric features for correlation analysis.
        output_path: Optional output file path.
    
    Returns:
        str: Path to the saved plot file.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    # Determine output path
    if output_path is None:
        output_dir = os.getenv('OUTPUT_DIR', '.')
        output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import seaborn as sns
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            logger.warning("Not enough numeric columns for correlation plot")
            # Create a minimal plot
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'Insufficient numeric data for correlation analysis', 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
        else:
            # Limit to top 8 features to avoid clutter
            if len(numeric_cols) > 8:
                numeric_cols = numeric_cols[:8]
            
            corr_matrix = df[numeric_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
        
        plt.savefig(str(output_path_obj), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Correlation heatmap saved successfully: {output_path_obj}")
        return str(output_path_obj)
    
    except Exception as e:
        plt.close()
        logger.error(f"Failed to save correlation heatmap: {e}")
        raise
