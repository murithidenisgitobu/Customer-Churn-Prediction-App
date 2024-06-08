import streamlit as st

# Set up precict page
st.set_page_config(
    page_title="Predictions",
    page_icon='⚖️',
    layout="wide"
)

# Define title
st.title("Predict!")

# Create an option for users to select their preferred model
col1, col2 = st.columns(2)
with col1:
    st.cache_resource()
    st.selectbox(options=['XGB Regressor', 'Logistics Regressor'], label='Choose your preffered model')
with col2:
    pass
st.header("Enter Customer's Information")
# Create the form to get customer's information
with st.form(border=True, key=('Customer Information')):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.selectbox(options=['Male', 'Female'], key='customer gender', label='Gender')
        st.selectbox(options=['Senior Citizen', 'Non Senior Citizen'], key='senior citizen', label='Type of Citizen')
        st.selectbox(options=['Yes', 'No'], key='marital status', label='Has a partner')
        st.selectbox(options=['Yes', 'No'], key='dependents', label='Has dependents')
        st.selectbox(options=['Yes', 'No'], key='phoneservice', label='Has phone service')

        
    with col2:
        st.selectbox(options=['DSL', 'Fiber optic', 'No'], key='internetservice', label='Has internet service')
        st.selectbox(options=['Yes', 'No', 'No Internet Service'], key='onlinesecurity', label='Has online security')
        st.selectbox(options=['Yes', 'No', 'No Internet Service'], key='onlinebackup', label='Has online backup')
        st.selectbox(options=['Yes', 'No', 'No Internet Service'], key='deviceprotection', label='Has device protection')
        st.selectbox(options=['Yes', 'No', 'No Internet Service'], key='techsupport', label='Has tech support')
        
    with col3:
        st.selectbox(options=['Yes', 'No', 'No Internet Service'], key='streamingtv', label='Has streaming tv')
        st.selectbox(options=['Yes', 'No', 'No Internet Service'], key='streamingmovies', label='Has streaming movies')
        st.selectbox(options=['Month-Month', 'One Year', 'Two Years'], key='contract', label='Type of contract')
        st.selectbox(options=['Yes', 'No'], key='paperlessbilling', label='Has paperless billing')
        st.selectbox(options=['Yes', 'No', 'No Internet Service'], key='multiplelines', label='Has Multiple Lines')






        st.form_submit_button(label='Submit')
