import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details to check if it's fraud or legit.")

# Features from dataset
feature_names = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']

# Collect user input
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(feature, value=0.0)

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Fraudulent Transaction (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legit Transaction (Probability: {prob:.2f})")
