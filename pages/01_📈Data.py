import streamlit as st
import pyodbc
import pandas as pd

# Set up Home page
# Configure the home page
st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTION APPLICATION",
    page_icon='🏠',
    layout="wide"
)

# Create tabs with "All Data" as the default selected tab
tab_names = ["All Data", "Data from SQL Server", "Data from GitHub Repository"]
tab_idx = 0  # Index for the default tab (0 for "All Data")

# Tab creation and selection handling
tabs = st.tabs(tab_names)

# Tab 1: All Data
with tabs[0]:
    st.header("All Data")
    st.subheader("Data from SQL Server and GitHub Repository")

    with st.spinner("Connecting to Database and Loading Data...."):
        # SQL Server data
        connection_string = (
            "DRIVER={SQL Server};"
            f"SERVER={st.secrets['SERVER']};"
            f"DATABASE={st.secrets['DATABASE']};"
            f"UID={st.secrets['USERNAME']};"
            f"PWD={st.secrets['PASSWORD']}"
        )
        connection = pyodbc.connect(connection_string)
        churn_data_query = "SELECT * FROM dbo.LP2_Telco_churn_first_3000"
        churn_data_sql = pd.read_sql_query(churn_data_query, connection)

        # GitHub Repository data
        churn_data_csv = "Data/LP2_Telco-churn-second-2000.csv"
        churn_data_github = pd.read_csv(churn_data_csv)

        # Concatenate the data
        combined_data = pd.concat([churn_data_sql, churn_data_github], ignore_index=True)
    
    # Display the concatenated data
    st.write(combined_data)

# Tab 2: Data from SQL Server
with tabs[1]:
    st.header("Data from SQL Server")
    with st.spinner("Connecting to Database...."):
        churn_data_sql = pd.read_sql_query(churn_data_query, connection)
    
    # Display the data
    st.write(churn_data_sql)

# Tab 3: Data from GitHub Repository
with tabs[2]:
    st.header("Data from GitHub Repository")
    with st.spinner("Loading data from GitHub Repository...."):
        churn_data_github = pd.read_csv(churn_data_csv)
    
    # Display the data
    st.write(churn_data_github)
