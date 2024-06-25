import streamlit as st
import os
import pandas as pd
import yaml
from yaml.loader import  SafeLoader
import streamlit_authenticator as stauth

# Set up Home page
st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTION APPLICATION",
    page_icon='📈',
    layout="wide"
)

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)


authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['pre-authorized']
)


def read_history_date():

    file_path = 'Data\history.csv'

    file_exists = os.path.exists(file_path)

    if file_exists:
        df = pd.read_csv(file_path)
        st.dataframe(df)
        return df
    else:
        return None



if __name__ == '__main__':
    if st.session_state["authentication_status"]:
        authenticator.logout(location='sidebar')

        st.title('History Page')
        read_history_date()
    else:
                    st.info('Login in the Home page to access History Page')

