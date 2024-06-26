import streamlit as st
import pyodbc
import os
import pandas as pd
import streamlit_authenticator as stauth
import yaml
from yaml.loader import  SafeLoader
import pymssql



# Set up Home page
# Configure the home page
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


if st.session_state["authentication_status"]:

    authenticator.logout(location='sidebar')


    @st.cache_resource(show_spinner="Connecting to Database...")
    def get_db_connection():
        server = os.environ.get('SERVER')
        database = os.environ.get('DATABASE')
        username = os.environ.get('USERNAME')
        password = os.environ.get('PASSWORD')
        
        try:
            connection = pymssql.connect(server=server, user=username, password=password, database=database)
            return connection
        except pymssql.Error as e:
            st.error(f"Error connecting to database: {e}")
        return None
    @st.cache_data(show_spinner="Loading data...")
    def get_sql_data(_connection):
        churn_data_query = "SELECT * FROM dbo.LP2_Telco_churn_first_3000"
        return pd.read_sql_query(churn_data_query, _connection)

    def get_github_data():
        churn_data_csv = "Data/LP2_Telco-churn-second-2000.csv"
        churn_data = pd.read_csv(churn_data_csv)
        return churn_data

    # Function to replace boolean and None values
    def clean_data(data):
        # Replace specific values
        data = data.replace({True: 'Yes', False: 'No', 1: 'Yes', 0: 'No', None: 'No'})
        # Fill NaN values with 'No'
        # Convert TotalCharges to numeric
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        return data

    # Function to clean only the SeniorCitizen column
    def clean_senior_citizen(data):
        data['SeniorCitizen'] = data['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})
        return data

    # Function to get numerical columns
    def get_numerical_columns(data):
        return data.select_dtypes(include=['number']).columns.tolist()

    # Create tabs with "All Data" as the default selected tab
    tab_names = ["All Data", "Data from SQL Server", "Data from GitHub Repository"]
    tabs = st.tabs(tab_names)

    # Fetch data
    connection = get_db_connection()
    churn_data_sql = get_sql_data(connection)
    churn_data_github = get_github_data()

    # Clean data
    churn_data_sql = clean_data(churn_data_sql)
    churn_data_sql = clean_senior_citizen(churn_data_sql)
    churn_data_github = clean_data(churn_data_github)
    churn_data_github = clean_senior_citizen(churn_data_github)


    # Concatenate data
    combined_data = pd.concat([churn_data_sql, churn_data_github], ignore_index=True)

    # Clean concatenated data
    combined_data = clean_data(combined_data)

    def display_columns(data, num_columns, key_prefix):
        column_options = ["All Columns", "Categorical", "Numerical"]
        col_type = st.selectbox(
            "Select Column Type",
            column_options,
            key=f"{key_prefix}_selectbox"
        )
        
        if col_type == "All Columns":
            selected_columns = data.columns.tolist()
        elif col_type == "Categorical":
            selected_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        else:
            selected_columns = st.multiselect(
                "Select Numerical Columns", 
                num_columns, 
                default=num_columns, 
                key=f"{key_prefix}_numerical"
            )
        st.write(data[selected_columns])

    # Tab 1: All Data
    with tabs[0]:
        st.header("All Data")
        st.subheader("Data from SQL Server and GitHub Repository")

        display_columns(combined_data, get_numerical_columns(combined_data), "all_data")

    # Tab 2: Data from SQL Server
    with tabs[1]:
        st.header("Data from SQL Server")
        display_columns(churn_data_sql, get_numerical_columns(churn_data_sql), "sql_data")

    # Tab 3: Data from GitHub Repository
    with tabs[2]:
        st.header("Data from GitHub Repository")
        display_columns(churn_data_github, get_numerical_columns(churn_data_github), "github_data")
else:	
    st.info("Login in the Home page to access Data")

