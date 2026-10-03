import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Vehicle Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Load Serialized Model & Feature Names
@st.cache_resource
def load_assets():
    model = joblib.load('car_price_model.pkl')
    feature_names = joblib.load('model_features.pkl')
    return model, feature_names

try:
    model, feature_names = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Title & Description
st.title("🚗 Vehicle Resale Price Valuation Engine")
st.write("Enter vehicle specifications below to estimate fair market resale value.")
st.divider()

# Input Form Controls
st.subheader("Vehicle Specifications")

col1, col2 = st.columns(2)

with col1:
    kms_driven = st.number_input("Distance Driven (in Kms)", min_value=1000, max_value=500000, value=30000, step=1000)
    vehicle_age = st.slider("Vehicle Age (in Years)", min_value=0, max_value=25, value=4)

with col2:
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

st.divider()

# Prediction Execution
if st.button("Estimate Resale Price", type="primary", use_container_width=True):
    # Initialize feature vector dictionary with zero values matching trained model columns
    input_data = {feat: 0 for feat in feature_names}
    
    # Map numerical features dynamically (case-insensitive key lookup)
    for feat in feature_names:
        if feat.lower() in ['kms_driven', 'kms_driven']:
            input_data[feat] = kms_driven
        elif feat.lower() in ['vehicle_age']:
            input_data[feat] = vehicle_age

    # Map One-Hot Encoded Categorical Columns matching pandas get_dummies(drop_first=True)
    if fuel_type == "Diesel":
        for feat in feature_names:
            if 'fuel_type_diesel' in feat.lower() or 'fuel_diesel' in feat.lower():
                input_data[feat] = 1
    elif fuel_type == "Petrol":
        for feat in feature_names:
            if 'fuel_type_petrol' in feat.lower() or 'fuel_petrol' in feat.lower():
                input_data[feat] = 1

    if seller_type == "Individual":
        for feat in feature_names:
            if 'seller_type_individual' in feat.lower() or 'seller_individual' in feat.lower():
                input_data[feat] = 1

    if transmission == "Manual":
        for feat in feature_names:
            if 'transmission_manual' in feat.lower():
                input_data[feat] = 1

    # Convert to single-row DataFrame matching exact training schema
    input_df = pd.DataFrame([input_data])
    
    # Predict Price
    predicted_price = model.predict(input_df)[0]
    
    # Display Result
    st.success(f"### Estimated Resale Price: ₹ {max(0.1, round(predicted_price, 2))} Lakhs")
    
    st.info("""
    **Valuation Insights:**
    - **Depreciation Impact:** Vehicle age and total distance driven contribute most significantly to residual value loss.
    - **Fuel & Transmission Variant:** Diesel and automatic variants generally sustain higher baseline secondary market values.
    """)