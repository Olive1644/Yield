#deploy model in streamlit
import pandas as pd
import joblib
import streamlit as st

st.markdown("""
    <style>
    
    /* Main background */
    .stApp {
        background-color: #f5f3e7;  /* light soil tone */
    }

    /* Title */
    h1 {
        color: #2e7d32;  /* deep green */
        text-align: center;
        font-weight: 700;
    }

    /* Labels */
    label {
        color: #4e342e;  /* brown */
        font-weight: 600;
    }

    /* Input boxes */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff;
        border: 2px solid #8d6e63;
        border-radius: 8px;
        color: #2e7d32;
    }

    /* Button */
    .stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: 600;
    }

    .stButton > button:hover {
        background-color: #1b5e20;
        color: #ffffff;
    }

    /* Success message */
    .stSuccess {
        background-color: #dcedc8;
        color: #33691e;
        border-radius: 10px;
        padding: 10px;
    }

    </style>
""", unsafe_allow_html=True)

# Load the saved model and preprocessing pipeline
model = joblib.load("best_ridge_model.pkl")
preprocessor = joblib.load("preprocessing_pipeline.pkl")

# Define a function to make predictions
def main():
    st.markdown("<h1>🌾 Crop Yield Prediction</h1>", unsafe_allow_html=True)
    st.write("Predict agricultural yield based on environmental and management factors.")
    Area = st.text_input("Country")
    Item = st.selectbox("Crop Type", options=['Potatoes', 'Maize', 'Wheat', 'Rice, paddy', 'Soybeans', 'Sorghum', 'Sweet potatoes', 'Cassava', 'Yams', 'Plantains and others']) 
    Year = st.number_input("Year", min_value=1960, max_value=2026, step=1)
    average_rain_fall_mm_per_year = st.number_input("Average_Rainfall_(mm/year)", min_value=0.0, max_value=5000.0)
    pesticides_tonnes = st.number_input("Pesticides (tonnes)", min_value=0.0, max_value=100000.0)
    avg_temp = st.number_input("Average Temperature (°C)", min_value=0.0, max_value=50.0)

    input_data = {
        "Area": Area,
        "Item": Item,
        "Year": Year,
        "average_rain_fall_mm_per_year": average_rain_fall_mm_per_year,
        "pesticides_tonnes": pesticides_tonnes,
        "avg_temp": avg_temp
    }

    if st.button("Predict Yield"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.success(f"Predicted Yield: {prediction[0]:.2f} tonnes/hectare")
if __name__ == "__main__":
    main()