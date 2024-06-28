import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set up Home page with an icon
st.set_page_config(
    page_title="Bulk Predict Page",
    layout="wide",
    page_icon="📊"  # Replace with your desired icon
)

st.title("Bulk Predict Page")

# Load Models
@st.cache_resource(show_spinner='Models Loading')
def load_model(model_path):
    try:
        model, threshold = joblib.load(model_path)
        return model, threshold
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_bulk(model, threshold, data):
    st.subheader("Predicting Customer Churn for Bulk Data")
    predictions = model.predict(data)  # Predicted classes
    probabilities = model.predict_proba(data)  # Predicted probabilities
    return predictions, probabilities

# Function to clean data
def clean_data(data):
    # Replace specific values
    data = data.replace({True: 'Yes', False: 'No', 1: 'Yes', 0: 'No', None: 'No'})
    # Fill NaN values with 'No'
    data = data.fillna('No')
    # Convert TotalCharges to numeric
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    return data

# Function to clean only the SeniorCitizen column
def clean_senior_citizen(data):
    data['SeniorCitizen'] = data['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})
    return data

# Load data
try:
    data = st.file_uploader("Upload CSV file", type=['csv'])
    if data is not None:
        df = pd.read_csv(data)
        
        # Clean data
        df = clean_data(df)
        df = clean_senior_citizen(df)
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        st.dataframe(df)
        st.info("Data loaded and cleaned successfully.")
        
        # Model selection (example with one model for simplicity)
        model_path = './Model/best_gb_model_and_threshold.joblib'
        model, threshold = load_model(model_path)
        
        if model is not None:
            st.subheader("Predicting Customer Churn")
            predictions, probabilities = predict_bulk(model, threshold, df)
            
            # Add predictions and probabilities to DataFrame
            df['Will Customer Churn'] = np.where(predictions == 1, 'Yes', 'No')
            df['Probability'] = np.where(predictions == 0,
                                         np.round(probabilities[:, 0] * 100, 2),
                                         np.round(probabilities[:, 1] * 100, 2))
            
            # Display updated DataFrame with predictions and probabilities
            st.dataframe(df)
            
except Exception as e:
    st.error(f"Error: {e}")
