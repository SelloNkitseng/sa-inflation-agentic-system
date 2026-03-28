# SA Inflation Agentic System

A LangGraph-based multi-agent system for South African CPI nowcasting using modern ML and LLM integration.

## Overview

This project implements an **agentic workflow** that orchestrates 7 specialized agents to:
1. **Collect** historical CPI, vehicle sales, and air quality data
2. **Clean & normalize** multi-source datasets
3. **Explore** data with statistical analysis and visualizations
4. **Engineer** time-series features
5. **Model** predictions using 6 ML algorithms
6. **Evaluate** performance and select the best model
7. **Generate insights** using LLM (Llama3 via Ollama)

## Architecture

```
Data Collection → Data Cleaning → EDA → Feature Engineering → Modeling → Evaluation → LLM Insights
```

Each agent is a modular node in a LangGraph state machine, with automatic state passing and error handling.

## Setup

### Prerequisites
- Python 3.10+
- Ollama with Llama3 model
- CSV data files in `../Data/` directory

### Installation

```bash
# Navigate to project
cd cpi_nowcast

# Create virtual environment
python -m venv cpi_env

# Activate (Windows)
cpi_env\Scripts\activate.bat

# Install dependencies
pip install -r requirements_fixed.txt
```

### Ollama Setup

```bash
# Install Ollama (Windows)
winget install Ollama.Ollama

# Pull Llama3 model
ollama pull llama3

# Start Ollama server (in separate terminal)
ollama serve
```

## Running the Workflow

```bash
python main.py
```

### Expected Output

- **workflow.log**: Detailed execution log with timestamps
- **predictions.csv**: Model predictions for test set
- **Visualizations**:
  - `plot1_cpi_pct_change.png`: Monthly CPI % change
  - `plot2_cpi_trend.png`: CPI trend over time
  - `plot3_correlation_heatmap.png`: Feature correlations
- **Console output**:
  - Agent status summary
  - Model training progress
  - Evaluation metrics (RMSE, MAE)
  - LLM-generated economic insights

## Data Requirements

Place CSV files in `../Data/`:

- **CPI_Historic_Values_Zindi_Apr_23.csv** (required)
  - Columns: Month, Category, Value
  - Must contain 'Headline_CPI' category

- **Naamsa_Vehicle_Sales.csv** (required)
  - Vehicle sales figures (used as economic proxy)

- **AirQualityData/zaf_grouped_sentinel5p.csv** (optional)
  - Environmental indicators

## Project Structure

```
cpi_nowcast/
├── main.py                 # Entry point
├── graph.py               # LangGraph workflow definition
├── agents/
│   ├── data_collector.py  # Load and standardize data
│   ├── data_cleaner.py    # Clean, align, merge datasets
│   ├── eda_agent.py       # Exploratory analysis + visualization
│   ├── feature_engineer.py# Time-series features
│   ├── modeler.py         # Train 6 ML models
│   ├── evaluator.py       # Evaluate and select best model
│   └── insights_agent.py  # LLM-based economic analysis
├── utils/
│   ├── data.py           # Data loading and visualization functions
│   └── llm.py            # Ollama LLM interface
└── requirements_fixed.txt
```

## Models Tested

1. **Linear Regression** - Baseline
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization & feature selection
4. **Support Vector Regression** - RBF kernel
5. **Random Forest** - Ensemble learning
6. **Gradient Boosting** - Sequential boosting

The best model (lowest RMSE) is selected for final predictions.

## Troubleshooting

### Ollama connection errors
- Ensure `ollama serve` is running in a separate terminal
- Verify connection: `curl http://localhost:11434`

### Data not found
- Check `../Data/` directory exists relative to `cpi_nowcast/`
- Verify CSV filenames match exactly

### Import errors
- Reinstall dependencies: `pip install -r requirements_fixed.txt`
- Check Python version: `python --version`

## Key Features

✅ **Modular agents** - Each step is independent and reusable  
✅ **State machine workflow** - LangGraph manages agent coordination  
✅ **Error handling** - Agents capture and report failures gracefully  
✅ **Multi-source data** - Integrates CPI, economic, and environmental data  
✅ **Ensemble modeling** - Compares 6 different algorithms  
✅ **LLM insights** - Uses local Llama3 for economic interpretation  
✅ **Rich visualization** - Three complementary chart types  
✅ **Comprehensive logging** - Full execution trace for debugging  

## Author

Sello Nkitseng

## License

MIT
