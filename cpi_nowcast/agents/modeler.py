from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)

def modeler(state):
    """Train baseline models."""
    logger.info("=== Modeling Agent ===")
    
    state['agent_status'] = state.get('agent_status', {})
    
    try:
        features = state['features']
        X = features.drop('CPI', axis=1)
        y = features['CPI']
        
        # Time-series split (80% train)
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Scale features for better model performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models and friendly names
        model_constructors = [
            ('lr', 'Linear Regression', LinearRegression()),
            ('ridge', 'Ridge Regression', Ridge(alpha=1.0)),
            ('lasso', 'Lasso Regression', Lasso(alpha=0.1)),
            ('svr', 'Support Vector Regression', SVR(kernel='rbf', C=1.0)),
            ('rf', 'Random Forest Regressor', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('gb', 'Gradient Boosting Regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]

        # Display model training order
        print("Training models:")
        for idx, (_, full_name, _) in enumerate(model_constructors, start=1):
            print(f" {idx}. {full_name}")
        print()

        models = {}
        model_name_map = {}

        for short_code, full_name, estimator in model_constructors:
            print(f"Training model: {full_name}")
            if short_code in ['lr', 'ridge', 'lasso', 'svr']:
                estimator.fit(X_train_scaled, y_train)
            else:
                estimator.fit(X_train, y_train)
            models[short_code] = estimator
            model_name_map[short_code] = full_name
            print(f"  - {full_name} completed")
        print()
        
        # Predictions for eval
        predictions = {}
        for name, model in models.items():
            if name in ['lr', 'ridge', 'lasso', 'svr']:
                predictions[name] = model.predict(X_test_scaled)
            else:
                predictions[name] = model.predict(X_test)
        
        state['models'] = models
        state['model_name_map'] = model_name_map
        state['test_data'] = (X_test, y_test, predictions)
        
        logger.info("Models trained")
        state['agent_status']['modeler'] = 'success'
        logger.info("Modeling complete")
        print()  # Blank line for readability
    except Exception as e:
        state['agent_status']['modeler'] = f"failure: {e}"
        logger.error(f"Modeling failed: {e}")
        raise
    
    return state
