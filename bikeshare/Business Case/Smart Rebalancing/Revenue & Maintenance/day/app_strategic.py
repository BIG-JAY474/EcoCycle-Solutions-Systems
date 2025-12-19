import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load Strategic Assets
model = joblib.load('bikeshare_strategic_rf_model.pkl')
scaler = joblib.load('bikeshare_strategic_scaler.pkl')
model_features = model.feature_names_in_

st.set_page_config(page_title="EcoCycle Strategic Planner", layout="wide")

st.title("🛠️ EcoCycle: Strategic Maintenance Planner")
st.markdown("Use this tool to forecast daily demand and schedule fleet repairs.")

# 2. Sidebar for Strategy Parameters
st.sidebar.header("Forecast Settings")
target_temp = st.sidebar.slider("Forecasted Temp (0.0 - 1.0)", 0.0, 1.0, 0.45)
target_hum = st.sidebar.slider("Forecasted Humidity", 0.0, 1.0, 0.60)
yesterday_cnt = st.sidebar.number_input("Yesterday's Total Rentals", value=4500)
last_week_cnt = st.sidebar.number_input("Same Day Last Week", value=4300)

# 3. Categorical Selections
col1, col2, col3 = st.columns(3)
with col1:
    working_day = st.selectbox("Working Day?", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    season = st.selectbox("Season", options=[1, 2, 3, 4], format_func=lambda x: ["Spring", "Summer", "Fall", "Winter"][x-1])
with col2:
    holiday = st.selectbox("Holiday?", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
    month = st.selectbox("Month", options=list(range(1, 13)))
with col3:
    weekday = st.selectbox("Day of Week", options=list(range(0, 7)), format_func=lambda x: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][x])
    weather = st.selectbox("Weather Condition", options=[1, 2, 3], format_func=lambda x: ["Clear", "Cloudy", "Rain/Snow"][x-1])

# 4. Process Input
input_data = {
    'year': 2012,
    'holiday': holiday,
    'workingday': working_day,
    'temperature': target_temp,
    'humidity': target_hum,
    'windspeed': 0.15,
    'temp_rolling_7d': target_temp, # Approximation
    'rentals_yesterday': yesterday_cnt,
    'rentals_last_week': last_week_cnt,
    'days_since_start': 720,
    'is_severe_weather': 1 if weather >= 3 else 0,
    'season': season,
    'weekday': weekday,
    'month': month,
    'quarter': (month-1)//3 + 1
}

# 5. Align and Scale
df_input = pd.DataFrame([input_data])
df_encoded = pd.get_dummies(df_input, columns=['season', 'weekday', 'month', 'quarter'])
df_final = df_encoded.reindex(columns=model_features, fill_value=0)

cols_to_scale = ['temperature', 'humidity', 'windspeed', 'temp_rolling_7d', 
                 'rentals_yesterday', 'rentals_last_week', 'days_since_start']
df_final[cols_to_scale] = scaler.transform(df_final[cols_to_scale])

# 6. Predict and Recommend
prediction = int(model.predict(df_final)[0])

st.divider()
kpi1, kpi2 = st.columns(2)

with kpi1:
    st.metric("Predicted Daily Demand", f"{prediction} Bikes")

with kpi2:
    # Logic: If demand is 30% below recent momentum, it's a good repair day
    if prediction < (yesterday_cnt * 0.7):
        st.success("STRIKE ZONE: Ideal for Mass Maintenance.")
    else:
        st.warning("PEAK LOAD: Minimize repairs; keep all bikes on street.")

# 7. Strategic Insight
st.info(f"Strategic Note: Based on a yesterday count of {yesterday_cnt}, the model suggests a change of {prediction - yesterday_cnt} bikes.")
