import streamlit as st

# Set up Home page
st.set_page_config(
    page_title="CUSTOMER CHURN PREDICTION APPLICATION",
    page_icon='🏠',
    layout="wide"
)

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Attrition Insight",)
    st.write("Predict if a customer is about to churn based on known characteristics using Machine Learning.")
    
    st.header("Key Features",)
    st.write("""
    - **Data**: Access data from Vodafone Telecom.
    - **Dashboard**: Explore interactive data visualizations for insghts.
    - **Predict**: Instantly see predictions for customer attrition.
    - **History**: See past predictions made.

    """)
    
    st.header("Machine Learning Integration",)
    st.write("""
             - **Accurate Predictions**: Integrate advanced ML algorithms for accurate predictions.
             - **Data-Driven Decisions**: Leverage comprehensive customer data to inform strategic initiatives.
             - **Variety**: Choose between two advanced ML algorithms for predictions""")


with col2:
    st.header("User Benefits",)
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

    st.subheader("About Developer")
    st.write("""
    I'm a data analyst/business analyst specializing in leveraging data science to assist businesses.Using advanced analytics and machine learning techniques, I aim to empower businesses with actionable insights for growth and success.
    """)

    with st.expander("Developer's Portfolio", expanded=False):
     st.write("""
     - **Email**: [iamdenis46@gmail.com](mailto:iamdenis46@gmail.com)
     - **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/murithidenisgitobu/)
     - **GitHub**: [GitHub Profile](https://github.com/murithidenisgitobu/)
     - **Medium**: [Medium Profile](https://medium.com/murithidenisgitobu)
     """)
