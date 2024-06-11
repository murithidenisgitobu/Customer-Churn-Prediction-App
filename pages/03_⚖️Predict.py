import streamlit as st
import joblib
import pandas as pd
import datetime
import os

# Set up predict page
st.set_page_config(page_title="Predictions", page_icon='⚖️', layout="wide")
st.title("Predict Customer Churn!")

# Functions to load models and encoder
@st.cache_resource(show_spinner='Models Loading')
def load_xgb():
    model, threshold = joblib.load('./Model/best_gb_model_and_threshold.joblib')
    return model, threshold

@st.cache_resource(show_spinner='Models Loading')
def load_lr():
    model, threshold = joblib.load('./Model/best_LR_model_and_threshold.joblib')
    return model, threshold

def select_model():
    st.write("### Choose your preferred model")
    model_choice = st.selectbox('Select model', ['XGB Classifier', 'Logistic Regressor'], key='selected_model')
    
    if model_choice == 'XGB Classifier':
        pipeline, threshold = load_xgb()
    else:
        pipeline, threshold = load_lr()

    encoder = joblib.load("./Model/encoder.joblib")

    return pipeline, encoder, threshold

if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if 'probability' not in st.session_state:
    st.session_state['probability'] = None

# Function to make predictions
def make_predictions(pipeline, encoder, threshold):
    # Define columns
    columns = ['gender', 'seniorcitizen', 'partner', 'dependents', 'tenure',
               'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
               'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
               'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod',
               'monthlycharges', 'totalcharges']

    # Define data
    data = [[st.session_state['gender'], st.session_state['senior_citizen'], 
             st.session_state['partner'], st.session_state['dependents'], 
             st.session_state['tenure'], st.session_state['phoneservice'], 
             st.session_state['multiplelines'], st.session_state['internetservice'], 
             st.session_state['onlinesecurity'], st.session_state['onlinebackup'], 
             st.session_state['deviceprotection'], st.session_state['techsupport'], 
             st.session_state['streamingtv'], st.session_state['streamingmovies'], 
             st.session_state['contract'], st.session_state['paperlessbilling'], 
             st.session_state['paymentmethod'], st.session_state['monthlycharges'], 
             st.session_state['totalcharges']]]

    # Create dataframe for prediction
    df = pd.DataFrame(data, columns=columns)

    # Make predictions
    probability = pipeline.predict_proba(df)
    pred = (probability[:, 1] >= threshold).astype(int)
    
    pred = int(pred[0])
    prediction = encoder.inverse_transform([pred])[0]

    # Save the prediction history
    history_df = df.copy()
    history_df['Prediction Time'] = datetime.datetime.today()
    history_df['Model Used'] = st.session_state['selected_model']
    history_df['Prediction'] = prediction
    history_df.to_csv('./data/history.csv', mode='a', header=not os.path.exists('./data/history.csv'), index=False)

    # Updating state
    st.session_state['prediction'] = prediction
    st.session_state['probability'] = probability

    return prediction, probability

# Input customer information
def form(pipeline, encoder, threshold):
    st.write("### Enter Customer's Information")
    with st.form(key='Customer Information'):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tenure = st.number_input('Months of tenure', min_value=1, max_value=72, key='tenure')
            gender = st.selectbox('Gender', ['Male', 'Female'], index=0, key='gender')
            senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'], index=0, key='senior_citizen')
            partner = st.selectbox('Has a partner', ['Yes', 'No'], index=0, key='partner')
            dependents = st.selectbox('Has dependents', ['Yes', 'No'], index=0, key='dependents')
            phoneservice = st.selectbox('Has phone service', ['Yes', 'No'], index=0, key='phoneservice')
            paymentmethod = st.selectbox('Payment method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                                            'Credit card (automatic)'], index=0, key='paymentmethod')

        with col2:
            monthly_charges = st.number_input('Monthly charges', min_value=1, max_value=100000, step=1000, key='monthlycharges')
            internetservice = st.selectbox('Internet service', ['DSL', 'Fiber optic', 'No'], index=0, key='internetservice')
            onlinesecurity = st.selectbox('Online security', ['Yes', 'No', 'No Internet Service'], index=0, key='onlinesecurity')
            onlinebackup = st.selectbox('Online backup', ['Yes', 'No', 'No Internet Service'], index=0, key='onlinebackup')
            deviceprotection = st.selectbox('Device protection', ['Yes', 'No', 'No Internet Service'], index=0, key='deviceprotection')
            techsupport = st.selectbox('Tech support', ['Yes', 'No', 'No Internet Service'], index=0, key='techsupport')  
        
        with col3:
            total_charges = st.number_input('Total charges', min_value=1, max_value=10000, step=100, key='totalcharges')
            streamingtv = st.selectbox('Streaming TV', ['Yes', 'No', 'No Internet Service'], index=0, key='streamingtv')
            streamingmovies = st.selectbox('Streaming movies', ['Yes', 'No', 'No Internet Service'], index=0, key='streamingmovies')
            contract = st.selectbox('Type of contract', ['Month-to-month', 'One year', 'Two year'], index=0, key='contract')
            paperlessbilling = st.selectbox('Paperless billing', ['Yes', 'No'], index=0, key='paperlessbilling')
            multiplelines = st.selectbox('Multiple lines', ['Yes', 'No', 'No Internet Service'], index=0, key='multiplelines')
        submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            make_predictions(pipeline, encoder, threshold)

if __name__ == '__main__':
    st.title('Make a prediction')
    pipeline, encoder, threshold = select_model()
    form(pipeline, encoder, threshold)

    prediction = st.session_state.get('prediction', None)
    probability = st.session_state.get('probability', None)
    
    if not prediction:
        st.markdown('No prediction made yet.')
    elif prediction == 'Yes':
        st.markdown('⚠️ The customer is likely to churn.')
        st.markdown('Probability of churn: {:.2f}%'.format(probability[0][1] * 100))
    else:
        st.markdown('✅ The customer is not likely to churn.')
        st.markdown('Probability of retention: {:.2f}%'.format(probability[0][0] * 100))
