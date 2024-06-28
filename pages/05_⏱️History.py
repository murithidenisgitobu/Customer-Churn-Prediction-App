import streamlit as st
import os
import pandas as pd
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# Set up Home page
st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTION APPLICATION",
    page_icon='📈',
    layout="wide"
)

# Load configuration from config.yaml
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize Streamlit Authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)

def read_history_data():
    file_path = 'Data/history.csv'  # Use forward slash for cross-platform compatibility

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.dataframe(df)
        return df
    else:
        st.warning(f"File '{file_path}' not found.")
        return None

if __name__ == '__main__':
    if st.session_state.get("authentication_status"):
        authenticator.logout(location='sidebar')
        st.title('History Page')
        read_history_data()
    else:
        st.info('Login on the Home page to access the History Page')
