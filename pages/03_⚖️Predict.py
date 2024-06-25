import streamlit as st
import joblib
import pandas as pd
import datetime
import os
import numpy as np
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# Function to reset session state
def reset_session_state():
    keys_to_keep = ['authentication_status', 'selected_model']
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

# Set up the predict page
if st.session_state["authentication_status"]:
    st.set_page_config(page_title="Predictions", page_icon='⚖️', layout="wide")
    st.title("Predict Customer Churn!")

    try:
        with open('config.yaml') as file:
            config = yaml.load(file, Loader=SafeLoader)

        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
            config['pre-authorized']
        )

        authenticator.logout(location='sidebar')

    except Exception as e:
        st.error(f"Error loading configuration: {e}")

    # Functions to load models and encoder
    @st.cache_resource(show_spinner='Models Loading')
    def load_xgb():
        try:
            model, threshold = joblib.load('./Model/best_gb_model_and_threshold.joblib')
            return model, threshold
        except Exception as e:
            st.error(f"Error loading XGB model: {e}")
            return None, None

    @st.cache_resource(show_spinner='Models Loading')
    def load_lr():
        try:
            model, threshold = joblib.load('./Model/best_LR_model_and_threshold.joblib')
            return model, threshold
        except Exception as e:
            st.error(f"Error loading Logistic Regression model: {e}")
            return None, None

    def select_model():
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Choose your preferred model")
            model_options = ['XGB Classifier', 'Logistic Regressor']
            model_choice = st.selectbox('Select model', ['Choose an option'] + model_options, index=0, key='selected_model')
        with col2:
            pass

        if model_choice == 'XGB Classifier':
            pipeline, threshold = load_xgb()
        elif model_choice == 'Logistic Regressor':
            pipeline, threshold = load_lr()
        else:
            pipeline, threshold = None, None

        if pipeline and threshold:
            try:
                encoder = joblib.load("./Model/encoder.joblib")
            except Exception as e:
                st.error(f"Error loading encoder: {e}")
                encoder = None
        else:
            encoder = None

        return pipeline, encoder, threshold

    if 'prediction' not in st.session_state:
        st.session_state['prediction'] = None

    if 'probability' not in st.session_state:
        st.session_state['probability'] = None

    # Function to make predictions
    def make_predictions(pipeline, encoder, threshold):
        columns = ['gender', 'seniorcitizen', 'partner', 'dependents', 'tenure',
                   'phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity',
                   'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv',
                   'streamingmovies', 'contract', 'paperlessbilling', 'paymentmethod',
                   'monthlycharges', 'totalcharges']

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

        df = pd.DataFrame(data, columns=columns)

        probability = pipeline.predict_proba(df)
        pred = (probability[:, 1] >= threshold).astype(int)
        pred = int(pred[0])
        prediction = encoder.inverse_transform([pred])[0]

        history_df = df.copy()
        now = datetime.datetime.now()
        date = now.date()
        hour = now.hour
        minute = now.minute
        history_df['Prediction Time'] = f"Time: {hour:02d}:{minute:02d} Date: {date}"
        history_df['Model Used'] = st.session_state['selected_model']
        history_df['Will Customer Churn'] = prediction
        history_df['Probability'] = np.where(pred == 0,
                                             np.round(probability[:, 0] * 100, 2),
                                             np.round(probability[:, 1] * 100, 2))

        history_df.to_csv('./data/history.csv', mode='a', header=not os.path.exists('./data/history.csv'), index=False)

        st.session_state['prediction'] = prediction
        st.session_state['probability'] = probability

        return prediction, probability

    def selectbox_with_placeholder(label, options, key):
        placeholder = 'Choose an option'
        selected = st.selectbox(label, [placeholder] + options, key=key)
        return selected if selected != placeholder else None

    def form(pipeline, encoder, threshold):
        st.write("### Enter Customer's Information")
        with st.form(key='Customer Information'):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tenure = st.number_input('Months of tenure', min_value=1, max_value=72, key='tenure')
                gender = selectbox_with_placeholder('Gender', ['Male', 'Female'], key='gender')
                senior_citizen = selectbox_with_placeholder('Senior Citizen', ['Yes', 'No'], key='senior_citizen')
                partner = selectbox_with_placeholder('Has a partner', ['Yes', 'No'], key='partner')
                dependents = selectbox_with_placeholder('Has dependents', ['Yes', 'No'], key='dependents')
                phoneservice = selectbox_with_placeholder('Has phone service', ['Yes', 'No'], key='phoneservice')
                paymentmethod = selectbox_with_placeholder('Payment method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], key='paymentmethod')

            with col2:
                monthly_charges = st.number_input('Monthly charges', min_value=18, max_value=119, step=1, key='monthlycharges')
                internetservice = selectbox_with_placeholder('Internet service', ['DSL', 'Fiber optic', 'No'], key='internetservice')
                onlinesecurity = selectbox_with_placeholder('Online security', ['Yes', 'No', 'No internet service'], key='onlinesecurity')
                onlinebackup = selectbox_with_placeholder('Online backup', ['Yes', 'No', 'No internet service'], key='onlinebackup')
                deviceprotection = selectbox_with_placeholder('Device protection', ['Yes', 'No', 'No internet service'], key='deviceprotection')
                techsupport = selectbox_with_placeholder('Tech support', ['Yes', 'No', 'No internet service'], key='techsupport')  
            
            with col3:
                total_charges = st.number_input('Total charges', min_value=18, max_value=8671, step=1, key='totalcharges')
                streamingtv = selectbox_with_placeholder('Streaming TV', ['Yes', 'No', 'No internet service'], key='streamingtv')
                streamingmovies = selectbox_with_placeholder('Streaming movies', ['Yes', 'No', 'No internet service'], key='streamingmovies')
                contract = selectbox_with_placeholder('Type of contract', ['Month-to-month', 'One year', 'Two year'], key='contract')
                paperlessbilling = selectbox_with_placeholder('Paperless billing', ['Yes', 'No'], key='paperlessbilling')
                multiplelines = selectbox_with_placeholder('Multiple lines', ['Yes', 'No', 'No phone service'], key='multiplelines')
            
            submit_button = st.form_submit_button(label='Submit')
            
            if submit_button:
                if None in [gender, senior_citizen, partner, dependents, phoneservice, internetservice, onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingtv, streamingmovies, contract, paperlessbilling, multiplelines, paymentmethod]:
                    st.warning("Please fill all the fields.")
                else:
                    make_predictions(pipeline, encoder, threshold)

    if __name__ == '__main__':
        pipeline, encoder, threshold = select_model()
        if pipeline and encoder and threshold:
            form(pipeline, encoder, threshold)

            prediction = st.session_state.get('prediction', None)
            probability = st.session_state.get('probability', None)
            reset_button = False  # Initialize reset_button to False

            if prediction == 'Yes':
                st.markdown('⚠️ The customer is likely to churn.')
                st.markdown('Probability of churn: {:.2f}%'.format(probability[0][1] * 100))
                reset_button = st.button('Reset for New Prediction')
            elif prediction == 'No':
                st.markdown('✅ The customer is not likely to churn.')
                st.markdown('Probability of retention: {:.2f}%'.format(probability[0][0] * 100))
                reset_button = st.button('Reset for New Prediction')
            else:
                st.markdown('No prediction made yet.')

            if reset_button:
                reset_session_state()
                st.experimental_rerun()
else:
    st.info('Login in the Home page to access Predict Page')
