import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
from yaml.loader import  SafeLoader
import streamlit_authenticator as stauth

# Set up Home page
st.set_page_config(
    page_title="Dashboard",
    page_icon='🗃️',
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


# Load data
@st.cache_data()
def load_data():
    data = pd.read_csv('Data/customer_churn_clean_data.csv')
    return data

data = load_data()

def select_dashboard():
    if 'selected_dashboard' not in st.session_state:
        st.session_state['selected_dashboard'] = 'KPI Dashboard'  # Default selection
    
    col1, col2 = st.columns(2)
    with col1:
        pass

    with col2:
        st.markdown('### Select Dashboard')
        dashboard = st.selectbox('Select Dashboard', ['KPI Dashboard', 'EDA Dashboard'], key='selected_dashboard')
        if st.session_state['selected_dashboard'] != dashboard:
            st.session_state['selected_dashboard'] = dashboard

    if st.session_state['selected_dashboard'] == 'KPI Dashboard':
        kpi_dashboard()

    if st.session_state['selected_dashboard'] == 'EDA Dashboard':
        eda_dashboard()

def kpi_dashboard():
    col2, col3 = st.columns(2)
    with col2:
        st.header('Key Performance Indicator Dashboard')
    with col3:
        pass


    col1, col2 = st.columns(2)

    # Churn Rate vs Tenure
    churn_rate = data.groupby(['churn', 'tenure'])['churn'].count()
    churn_rate = churn_rate.reset_index(name='Number of Customers')
    churn_rate = churn_rate.pivot(index='tenure', columns='churn', values='Number of Customers')
    churn_rate['Churn Rate'] = churn_rate['Yes'] / (churn_rate['No'] + churn_rate['Yes'])
    churn_rate = churn_rate.reset_index()
    churn_rate = px.line(churn_rate, x='tenure', y='Churn Rate', title='Churn Rate vs Tenure')
    st.plotly_chart(churn_rate)
        
    
    with col1:
        
        cols_set_1 = ['gender', 'contract', 'seniorcitizen', 'partner', 'dependents', 'internetservice', 'onlinesecurity', 'paymentmethod']
        for col in cols_set_1:
            col_data = data.groupby(col)['churn'].count().reset_index(name='Number of Customers')
            if col == 'seniorcitizen':
                col_data[col] = col_data[col].replace({0: 'Non-Senior Citizen', 1: 'Senior Citizen'})
            fig = px.bar(col_data, x=col, y='Number of Customers', color=col, text='Number of Customers', 
                         title=f'{col.upper()} vs Churn')
            st.plotly_chart(fig)


    with col2:
        cols_set_2 = ['phoneservice', 'multiplelines', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'paperlessbilling']
        for col in cols_set_2:
            col_data = data.groupby(col)['churn'].count().reset_index(name='Number of Customers')
            fig = px.bar(col_data, x=col, y='Number of Customers', color=col, text='Number of Customers', title=f'{col.upper()} vs Churn')
            st.plotly_chart(fig)

def eda_dashboard():
    col1, col2 = st.columns(2)
    with col1:
        st.header('Exploratory Data Analysis Dashboard')
        col1, col2 = st.columns(2)
    
    with col2:
        pass
    
    col1, col2 = st.columns(2)
    with col1:
        hist = px.histogram(data, 'totalcharges', title='Total charges distribution', color='churn')
        st.plotly_chart(hist)

        hist = px.histogram(data, 'tenure', title='Tenure distribution', color='churn')
        st.plotly_chart(hist)

        corr = data[['tenure', 'monthlycharges', 'totalcharges']].corr()
        corr = px.imshow(corr, text_auto=True, title= 'Numerical columns correlations')
        st.plotly_chart(corr)

    with col2:
        hist = px.histogram(data, 'monthlycharges', title='Monthly charges distribution', color='churn')
        st.plotly_chart(hist)

        boxplot = px.box(data, ['tenure','monthlycharges'], title='Tenure and Monthly Charges Boxplots', color='churn')
        st.plotly_chart(boxplot)

        boxplot = px.box(data, ['totalcharges'], title='Total Charges Boxplot', color='churn')
        st.plotly_chart(boxplot)

        


if __name__ == '__main__':
    if st.session_state["authentication_status"]:
        authenticator.logout(location='sidebar')

        select_dashboard()
    else:	
        st.info('Login in the Home page to access Dashboard')
