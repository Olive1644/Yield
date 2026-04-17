#deploy model in streamlit
import pandas as pd
import joblib
import streamlit as st
# Load the saved model and preprocessing pipeline
model = joblib.load("best_ridge_model.pkl")
preprocessor = joblib.load("preprocessing_pipeline.pkl")
# Define a function to make predictions
def main():
    st.title("Crop Yield Prediction")
    # Create input fields for the features
    Area = st.number_input("Area (in hectares)", min_value=0.0)
    Item = st.text_input("Crop Type")
    Year = st.number_input("Year", min_value=1900, max_value=2026)
    average_rainfall_mm_per_year = st.number_input("Average Rainfall (mm/year)", min_value=0.0)
    pesticides_tonnes = st.number_input("Pesticides Used (tonnes)", min_value=0.0)
    avg_temp = st.number_input("Average Temperature (°C)", min_value=-0.0, max_value=50.0)
    input_data = {
            "Area": Area,
            "Item": Item,
            "Year": Year,
            "average_rainfall_mm_per_year": average_rainfall_mm_per_year,
            "pesticides_tonnes": pesticides_tonnes,
            "avg_temp": avg_temp
        }
    #create a button to make prediction
    if st.button("Predict Yield"):
        input_df = pd.DataFrame([input_data])
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)
        st.success(f"Predicted Yield: {prediction[0]:.2f} tonnes/hectare")