import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# 1. Load the Model and Scaler
# Ensure these files are in the same folder as app.py
model = joblib.load('bikeshare_xgboost_model.pkl')
scaler = joblib.load('bikeshare_scaler.pkl')

# Get the EXACT list of features the model was trained on
model_features = model.get_booster().feature_names

st.set_page_config(page_title="EcoCycle Predictor", layout="centered")
st.title("🚲 EcoCycle: Smart-Rebalancing Dashboard")
st.markdown("Adjust conditions below to see predicted bike demand for 2025.")

# 2. Sidebar - User Inputs
st.sidebar.header("Input Parameters")
temp = st.sidebar.slider("Temperature (0.0 to 1.0)", 0.0, 1.0, 0.5, step=0.01)
hum = st.sidebar.slider("Humidity (0.0 to 1.0)", 0.0, 1.0, 0.5, step=0.01)
wind = st.sidebar.slider("Windspeed (0.0 to 1.0)", 0.0, 1.0, 0.1, step=0.01)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
is_holiday = st.sidebar.selectbox("Is it a Holiday?", [0, 1])
is_workingday = st.sidebar.selectbox("Is it a Working Day?", [1, 0])

# 3. Build the Input Dictionary
# --- 1. USER INPUTS (Keep your current sidebar code) ---
# ... [Your sliders for temp, hum, wind, etc.] ...

# --- 2. BUILD THE RAW INPUT DICTIONARY ---
input_dict = {
    'year': 2012, 
    'holiday': is_holiday, 
    'workingday': is_workingday,
    'temperature': temp, 
    'humidity': hum, 
    'windspeed': wind,
    'days_since_start': 730,
    'hour_sin': np.sin(2 * np.pi * hour / 24.0),
    'hour_cos': np.cos(2 * np.pi * hour / 24.0),
    'rentals_lag_1h': 100.0,
    'rentals_lag_2h': 80.0,
    'temp_hum_interaction': temp * hum,
    'is_commute_peak': 1 if (is_workingday == 1 and (7 <= hour <= 9 or 16 <= hour <= 19)) else 0
}

# --- 3. THE FIX: AUTO-ALIGN WITH MODEL SCHEMA ---
# Create a DataFrame with the initial data
df_input = pd.DataFrame([input_dict])

# Get the EXACT list and order of columns the model expects
expected_features = model.get_booster().feature_names

# REINDEX: This adds the missing 29 columns (like season_2, month_12, etc.) 
# and fills them with 0, matching the model's training order exactly.
df_final = df_input.reindex(columns=expected_features, fill_value=0)

# --- 4. SCALE THE NUMERICAL COLUMNS ---
cols_to_scale = ['temperature', 'humidity', 'windspeed', 'days_since_start', 
                 'rentals_lag_1h', 'rentals_lag_2h', 'temp_hum_interaction']
df_final[cols_to_scale] = scaler.transform(df_final[cols_to_scale])

# --- 5. PREDICT ---
# We use .values or .astype(float) to ensure XGBoost is happy
prediction = model.predict(df_final)
result = int(np.maximum(0, prediction[0]))


# 7. UI Output
st.subheader("Forecast Results")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Predicted Demand", value=f"{result} Bikes")

with col2:
    if result > 250:
        st.error("Status: High Demand")
    elif result > 100:
        st.warning("Status: Moderate Demand")
    else:
        st.success("Status: Low Demand")

st.info("The model uses XGBoost with 95.7% accuracy to help EcoCycle optimize rebalancing trucks.")
