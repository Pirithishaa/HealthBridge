import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess_data(file_path):
    """Loads and cleans the dataset for training."""
    print(f"--- Loading and preprocessing {file_path} ---")
    data = pd.read_csv(file_path)
    if "Patient_ID" in data.columns:
        data = data.drop(columns=["Patient_ID"])
    

    data['Overall_Risk_After'] = pd.to_numeric(data['Overall_Risk_After'], errors='coerce')


    data.dropna(subset=['Overall_Risk_After'], inplace=True)
    

    if data.empty:
        print("\n❌ Error: The dataset is empty after cleaning. This can happen if 'Overall_Risk_After' contains no valid numbers.")
        print("Please check the 'intervention_impact_dataset.csv' file and ensure the target column is correctly formatted.")
        return None, None 


    for col in data.columns:
        if data[col].isnull().any():
            if np.issubdtype(data[col].dtype, np.number):
                data[col].fillna(data[col].median(), inplace=True)
            else:
                data[col].fillna(data[col].mode()[0], inplace=True)

    X = data.drop(columns=["Overall_Risk_After"])
    y = data["Overall_Risk_After"]
    return X, y

def train_and_save_model(X, y):
    """Trains the model with hyperparameter tuning and saves it."""
    print("\n--- Starting Model Training ---")
    
  
    bins = pd.qcut(y, q=5, labels=False, duplicates='drop')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=bins
    )
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ])
    

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 15],
        "model__min_samples_split": [5, 10],
        "model__min_samples_leaf": [2, 4]
    }
    
    print("Starting hyperparameter tuning...")
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, n_jobs=-1, scoring="r2", verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"\nBest parameters found: {grid_search.best_params_}")
    
  
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("\n--- Model Performance on Test Set ---")
    print(f"R² Score: {r2:.3f}")
    print(f"RMSE: {rmse:.2f}%")
    
   
    joblib.dump(best_model, "intervention_model.joblib")
    print("\n✅ Model has been trained and saved to 'intervention_model.joblib'")
    
    return best_model

if __name__ == "__main__":
   
    try:
       
        X_train_data, y_train_data = load_and_preprocess_data("intervention_impact_dataset_updated.csv")
        
        
        if X_train_data is not None and y_train_data is not None:
            train_and_save_model(X_train_data, y_train_data)

    except FileNotFoundError:
        print("\n❌ Error: 'intervention_impact_dataset.csv' not found.")
        print("Please provide a dataset with features like 'Overall_Risk_Before', 'Intervention_Nutrition', etc., and a target 'Overall_Risk_After'.")
