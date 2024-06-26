import streamlit as st
import yaml
from yaml.loader import  SafeLoader
import streamlit_authenticator as stauth

# Set up Home page
st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTION APPLICATION",
    page_icon='🏠',
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

authenticator.login(location='sidebar')



if st.session_state["authentication_status"]:
    authenticator.logout(location='sidebar')
    st.write(f'Welcome *{st.session_state["name"]}*', location='sidebar')
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')
    st.info('Login to access Prediction Application')
    st.code('''
        Login Credentials for Test Account:
        Username: test
        Password: 1a2b3c''')

if st.session_state["authentication_status"]:
    st.markdown("<h1 style='color: skyblue;'>CUSTOMER CHURN PREDICTION APP</h1>", unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Attrition Insight",)
        st.write("Predict if a customer is about to churn based on known characteristics using Machine Learning.")
        
        st.write("### Key Features",)
        st.write("""
        - **Data**: Access data from Vodafone Telecom.
        - **Dashboard**: Explore interactive data visualizations for insghts.
        - **Predict**: Instantly see predictions for customer attrition.
        - **History**: See past predictions made.

        """)
        
        st.write("### Machine Learning Integration",)
        st.write("""
                - **Accurate Predictions**: Integrate advanced ML algorithms for accurate predictions.
                - **Data-Driven Decisions**: Leverage comprehensive customer data to inform strategic initiatives.
                - **Variety**: Choose between two advanced ML algorithms for predictions""")


    with col2:
        st.write("### User Benefits",)
        st.write("""
        - **Accurate Prediction**: Reduce churn rate.
        - **Data-Driven Decisions**: Inform strategic initiatives.
        - **Enhanced Insights**: Understand customer behavior.
        """)
        
        with st.expander("Need Help?", expanded=False):
            st.write("""
            Refer to our [documentation](https://github.com/murithidenisgitobu/Customer-Churn-Prediction-App) or contact support.
            
            - **Documentation**: Guides for ML integration.
            - **Support**: Contact us at iamdenis46@gmail.com.
            """)

        st.write("#### About Developer")
        st.write("""
            I'm a data analyst/business analyst specializing in leveraging data science to assist businesses.Using advanced analytics and machine learning techniques, I aim to empower businesses with actionable insights for growth and success.
            """)

        with st.expander("Developer's Portfolio", expanded=False):
            st.write("""
            - **Email**: [iamdenis46@gmail.com](mailto:iamdenis46@gmail.com)
            - **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/murithidenisgitobu/)
            - **GitHub**: [GitHub Profile](https://github.com/murithidenisgitobu/)
            - **Medium**: [Medium Profile](https://medium.com/@murithidenisgitobu)
            """)

