import streamlit as st
import xgboost as xgb
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer

# Load the pre-trained model, DictVectorizer, and scaler

def load_model():
    with open('churn_modell.pkl', 'rb') as model_file:
        model, dv_vectorizer, st_scaler = pickle.load(model_file)
    return model, dv_vectorizer, st_scaler

model, dv_vectorizer, st_scaler = load_model()

# Streamlit interface
st.title('Customer Churn Prediction')

# Streamlit app layout
st.markdown("---")
st.markdown("**This App will predict whether the customer will churn or not**.")
st.image("stream_image.JPG")

st.markdown("<p style='font-size:20px;'><b><i>Churn Prediction App Summary:</i></b></p>", unsafe_allow_html=True)

message_one = """
            This Streamlit app is designed to predict whether a customer is likely to churn based on their provided data. 
            The app leverages a pre-trained machine learning model using Logistic Regression, 
            and it takes user inputs through an interactive form to determine customer-specific features.
"""

message_two = """
            **User Input Form**: Users input various customer details such as gender, senior citizen status,
            tenure, internet and phone service usage, payment method, and monthly charges, among other factors.

            **Model Prediction**: Once the user submits the data, the app processes the information using
            a pre-trained machine learning model to predict the likelihood of customer churn.
            
            **Prediction Output**: After the 'Predict Churn' button is clicked, the app displays a result, 
            indicating whether the customer is predicted to churn or not (i.e., "Churn" or "Not Churn"). 
            The app provides an interactive way to explore customer behavior and supports decision-making
            around customer retention by giving insights into potential churn scenarios.
"""

st.markdown(message_one)
st.markdown("<p style='font-size:20px;'><b><i>Key Features:</i></b></p>", unsafe_allow_html=True)

st.markdown(message_two)
st.markdown("---")

# User input form
st.markdown("**Select Customer Information:**")
gender = st.selectbox('Gender', ['Male', 'Female'])
senior_citizen = st.selectbox('Senior Citizen', ['Yes', 'No'])
partner = st.selectbox('Partner', ['Yes', 'No'])
dependents = st.selectbox('Dependents', ['Yes', 'No'])
tenure = st.slider('Tenure (months)', 0, 72, 12)
phone_service = st.selectbox('Phone Service', ['Yes', 'No'])
multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.slider('Monthly Charges', 18.0, 120.0, 70.0)
total_charges = st.slider('Total Charges', 0.0, 9000.0, 1400.0)

# Collecting the inputs in a dictionary
customer_data = {
    'gender': gender,      'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,  'Partner': partner,
    'Dependents': dependents,   'tenure': tenure,    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,   'InternetService': internet_service,   'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,   'DeviceProtection': device_protection,   'TechSupport': tech_support,
    'StreamingTV': streaming_tv,   'StreamingMovies': streaming_movies,   'Contract': contract,
    'PaperlessBilling': paperless_billing,   'PaymentMethod': payment_method,   'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Preprocessing the customer data for prediction
customer_data_encoded = dv_vectorizer.transform([customer_data])
customer_data_scaled = st_scaler.transform(customer_data_encoded)

# Make the prediction
# Add a button to generate prediction
if st.button('Predict Churn'):
    # Make the prediction when the button is clicked
    prediction = model.predict(customer_data_scaled)
    
    # Interpret the prediction
    prediction_label = 'Churn' if prediction >= 0.5 else 'Not Churn'
    
    # Display the prediction result
    st.write('## Prediction:')
    st.write(f"The model predicts that the customer will: **{prediction_label}**")
