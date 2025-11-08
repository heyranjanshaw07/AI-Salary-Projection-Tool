import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from flask import Flask, request, jsonify
from flask_cors import CORS 
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# --- Global Configuration  ---
CATEGORICAL_COLS = []
FEATURE_COLS = []
RANDOM_STATE = 42

EDUCATION_MAP = {"Bachelor's": "Bachelor's Degree", "Master's": "Master's Degree", "PhD": "PhD", "phD": "PhD", "Bachelor's Degree": "Bachelor's Degree", "Master's Degree": "Master's Degree", "High School": "High School"}
ORDINAL_ORDER = {'High School': 0, "Bachelor's Degree": 1, "Master's Degree": 2, "PhD": 3}

# --- Utility Functions  ---

def load_data(file_name='Salary_Data.csv'):
    try:
        df = pd.read_csv(file_name)
        if df.shape[1] <= 2 and 'Salary' not in df.columns:
             df = pd.read_csv(file_name, sep=';')
        return df
    except FileNotFoundError:
        logging.error(f"'{file_name}' not found. Ensure file is in the correct directory.")
        return None

def prepare_data(df):
    global CATEGORICAL_COLS
    global FEATURE_COLS
    df_cleaned = df.dropna().copy()
    df_cleaned['Education Level'] = df_cleaned['Education Level'].replace(EDUCATION_MAP)
    df_cleaned['Education_Encoded'] = df_cleaned['Education Level'].map(ORDINAL_ORDER)
    CATEGORICAL_COLS = ['Gender', 'Job Title']
    df_encoded = pd.get_dummies(df_cleaned, columns=CATEGORICAL_COLS, drop_first=True)
    df_final = df_encoded.drop(columns=['Education Level'])
    FEATURE_COLS = df_final.drop('Salary', axis=1).columns
    return df_final

# New Ensemble Training Function ---

def train_best_model(df_encoded):
    logging.info("Starting ENSEMBLE model training...")
    X = df_encoded.drop('Salary', axis=1)
    y = df_encoded['Salary']

    # 1. Base Models
    rf_model = RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1, max_depth=10)
    gbr_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)
    
    # 2. Ensemble Model (Voting Regressor)
    ensemble_model = VotingRegressor([
        ('rf', rf_model),
        ('gbr', gbr_model)
    ], n_jobs=-1)

    # Train the Ensemble
    ensemble_model.fit(X, y)
   
    logging.info("Ensemble Model Training Complete.")
    return ensemble_model

# --- predict_new_salary_api  ---

def predict_new_salary_api(model, new_data):
    new_df = pd.DataFrame([new_data])
    new_df['Education Level'] = new_df['Education Level'].replace(EDUCATION_MAP)
    new_df['Education_Encoded'] = new_df['Education Level'].map(ORDINAL_ORDER).fillna(0)
    new_df = new_df.drop(columns=['Education Level'])
    new_df_encoded = pd.get_dummies(new_df, columns=CATEGORICAL_COLS, drop_first=True)
    X_new = pd.DataFrame(0, index=new_df_encoded.index, columns=FEATURE_COLS)
    for col in new_df_encoded.columns:
        if col in X_new.columns:
            X_new[col] = new_df_encoded[col]
    predicted_salary = model.predict(X_new)[0]
    return predicted_salary

# --- Flask App Initialization  ---

app = Flask(__name__)
CORS(app)
best_model = None
data_loaded = False

def setup_model():
    global best_model
    global data_loaded
    if not data_loaded:
        logging.info("--- Backend Setup: Loading and Training Ensemble Model ---")
        salary_df = load_data()
        if salary_df is not None:
            processed_df = prepare_data(salary_df)
            best_model = train_best_model(processed_df)
            data_loaded = True
            logging.info("--- Setup Complete. Ready to serve predictions. ---")
        else:
            logging.error("[FATAL ERROR] Server started without a trained model.")

with app.app_context():
    setup_model()

@app.route('/predict', methods=['POST'])
def predict_salary():
    if best_model is None:
        return jsonify({"error": "Model is not trained. Check server logs."}), 500
    try:
        data = request.get_json(silent=False)
        required_keys = ['Age', 'Years of Experience', 'Gender', 'Education Level', 'Job Title']
        if not all(key in data for key in required_keys):
             return jsonify({"error": "Missing one or more required fields in the input data."}), 400
        data['Age'] = float(data['Age'])
        data['Years of Experience'] = float(data['Years of Experience'])
        predicted_salary = predict_new_salary_api(best_model, data)
        return jsonify({"status": "success", "predicted_salary": round(predicted_salary, 2), "input_data": data})
    except ValueError:
        return jsonify({"error": "Age and Years of Experience must be valid numbers."}), 400
    except Exception as e:
        logging.error(f"Prediction Error: {e}")
        return jsonify({"error": f"An unexpected processing error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)