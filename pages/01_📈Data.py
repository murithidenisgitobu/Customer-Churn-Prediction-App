import streamlit as st
import pyodbc
import pandas as pd

# Set up Home page
# Configure the home page
st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTION APPLICATION",
    page_icon='📈',
    layout="wide"
)

@st.cache_resource
def get_db_connection():
    connection_string = (
        "DRIVER={SQL Server};"
        f"SERVER={st.secrets['SERVER']};"
        f"DATABASE={st.secrets['DATABASE']};"
        f"UID={st.secrets['USERNAME']};"
        f"PWD={st.secrets['PASSWORD']}"
    )
    return pyodbc.connect(connection_string)
@st.cache_data
def get_sql_data(_connection):
    churn_data_query = "SELECT * FROM dbo.LP2_Telco_churn_first_3000"
    return pd.read_sql_query(churn_data_query, _connection)


@st.cache_data
def get_github_data():
    churn_data_csv = "Data/LP2_Telco-churn-second-2000.csv"
    return pd.read_csv(churn_data_csv)

# Create tabs with "All Data" as the default selected tab
tab_names = ["All Data", "Data from SQL Server", "Data from GitHub Repository"]
tabs = st.tabs(tab_names)

# Fetch data
connection = get_db_connection()
churn_data_sql = get_sql_data(connection)
churn_data_github = get_github_data()

# Tab 1: All Data
with tabs[0]:
    st.header("All Data")
    st.subheader("Data from SQL Server and GitHub Repository")

    with st.spinner("Concatenating all data...."):
        combined_data = pd.concat([churn_data_sql, churn_data_github], ignore_index=True)
    
    # Display the concatenated data
    st.write(combined_data)

# Tab 2: Data from SQL Server
with tabs[1]:
    st.header("Data from SQL Server")
    # Display the data
    st.write(churn_data_sql)

# Tab 3: Data from GitHub Repository
with tabs[2]:
    st.header("Data from GitHub Repository")
    # Display the data
    st.write(churn_data_github)
