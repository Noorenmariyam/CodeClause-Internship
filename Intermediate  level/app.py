import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("clv_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Customer Lifetime Value Predictor", layout="wide")
st.title("🔮 Customer Lifetime Value Prediction App")

st.markdown("""
This app predicts the **Customer Lifetime Value (CLV)** based on user inputs.  
Please enter the customer details below:
""")

# Input fields (customize based on your dataset)
income = st.number_input("Income", min_value=0)
monthly_premium = st.number_input("Monthly Premium Auto", min_value=0)
total_claim = st.number_input("Total Claim Amount", min_value=0.0)
months_since_last_claim = st.slider("Months Since Last Claim", 0, 60, 12)
months_since_policy_inception = st.slider("Months Since Policy Inception", 0, 150, 60)
number_of_policies = st.slider("Number of Open Complaints", 0, 5, 0)

# Categorical inputs (example: encoded manually)
coverage = st.selectbox("Coverage", ["Basic", "Extended", "Premium"])
vehicle_class = st.selectbox("Vehicle Class", ["Two-Door Car", "Four-Door Car", "SUV", "Luxury Car", "Sports Car"])
gender = st.selectbox("Gender", ["Male", "Female"])
response = st.selectbox("Response", ["Yes", "No"])

# Manual encoding (simplified example)
coverage_map = {"Basic": 0, "Extended": 1, "Premium": 2}
vehicle_map = {"Two-Door Car": 0, "Four-Door Car": 1, "SUV": 2, "Luxury Car": 3, "Sports Car": 4}
gender_map = {"Male": 1, "Female": 0}
response_map = {"Yes": 1, "No": 0}

# Combine inputs
input_data = pd.DataFrame([[
    income,
    monthly_premium,
    total_claim,
    months_since_last_claim,
    months_since_policy_inception,
    number_of_policies,
    coverage_map[coverage],
    vehicle_map[vehicle_class],
    gender_map[gender],
    response_map[response]
]], columns=[
    "Income",
    "Monthly Premium Auto",
    "Total Claim Amount",
    "Months Since Last Claim",
    "Months Since Policy Inception",
    "Number of Open Complaints",
    "Coverage",
    "Vehicle Class",
    "Gender",
    "Response"
])

# Scale and predict
scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)[0]

st.subheader("📈 Predicted Customer Lifetime Value:")
st.success(f"${prediction:,.2f}")

# Feature importance plot
st.subheader("🔍 Feature Importance")
importances = model.feature_importances_
features = input_data.columns
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(importances)), importances[indices])
ax.set_xticks(range(len(importances)))
ax.set_xticklabels([features[i] for i in indices], rotation=45, ha='right')
ax.set_title("Feature Importance")
st.pyplot(fig)
