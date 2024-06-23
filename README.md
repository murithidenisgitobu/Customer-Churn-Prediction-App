# Customer Churn Prediction App

This project integrates advanced customer churn prediction machine learning models into a Streamlit-based graphical user interface (GUI). Designed for third-party stakeholders who lack the technical expertise to interpret Jupyter notebooks, this tool provides an intuitive and interactive way to predict and analyze customer churn.

## Application Pages

### 1. **Home Page**
- **Purpose**: Introduces the application and serves as the login page.
- **Features**:
  - Information about the application.
  - Navigation to the other pages.
  - Displays the username and password for access.

### 2. **Data Page**
- **Purpose**: Provides access to the data used to train the model.
- **Features**:
  - Displays data from three sources:
    1. SQL database
    2. Git repository
    3. OneDrive file

### 3. **Dashboard**
- **Purpose**: Showcases key metrics and visualizations of the data.
- **Features**:
  - **Key Indicator Dashboard**: Displays important metrics.
  - **Exploratory Data Analysis Dashboard**: Provides detailed visualizations and analysis.

### 4. **Predict Page**
- **Purpose**: Allows users to input customer information and receive churn predictions.
- **Features**:
  - Input fields for known customer information.
  - Predicts churn using two models:
    1. Logistic Regression
    2. XGBoost Classifier

### 5. **History Page**
- **Purpose**: Displays data about previous predictions.
- **Features**:
  - Historical data of past predictions for review and analysis.

This intuitive interface ensures that third-party stakeholders can effectively utilize the customer churn prediction models without needing advanced technical skills. The various pages of the application provide comprehensive access to data, predictive tools, and historical analysis.
"""