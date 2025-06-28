import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
st.title("Loan predictor App")

def load_and_preprocess_data(file_path="https://raw.githubusercontent.com/Sandeshb24/loan-predictor/refs/heads/main/loan_approval_dataset.csv"):
    """
    Loads the loan approval dataset and performs necessary preprocessing.
    """
    try:
        loan_data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Please ensure the CSV file is in the same directory.")
        st.stop() # Stop the app if the file is not found

    # Drop the 'loan_id' column as it's not a feature for prediction
    loan_data = loan_data.drop("loan_id", axis=1)

    # Define mapping for categorical features
    status_mapping_loan_status = {' Approved': 1, ' Rejected': 0}
    status_mapping_education = {' Graduate': 1, ' Not Graduate': 0}
    status_mapping_self_employed = {' Yes': 1, ' No': 0}

    # Apply mappings to create new binary columns
    loan_data["loan_status_binary"] = loan_data["loan_status"].map(status_mapping_loan_status)
    loan_data["education_binary"] = loan_data["education"].map(status_mapping_education)
    loan_data["self_employed_binary"] = loan_data["self_employed"].map(status_mapping_self_employed)

    # Drop original categorical columns and the target column for features (X)
    # The original notebook dropped 'loan_status', 'education', 'self_employed'
    # and then used the new binary columns as features.
    cleaned_loan_data = loan_data.drop(
        ["education", "self_employed", "loan_status"], axis=1
    )

    X = cleaned_loan_data.drop("loan_status_binary", axis=1)
    y = cleaned_loan_data["loan_status_binary"]

    return X, y
    
@st.cache_resource # Cache the model training
def train_model(X_train, y_train):
    """
    Trains the RandomForestClassifier model.
    """
    model = RandomForestClassifier(random_state=42) # Added random_state for reproducibility
    model.fit(X_train, y_train)
    return model

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("💰 Loan Approval Predictor")
st.markdown("### Predict if a loan will be approved based on applicant's details.")
