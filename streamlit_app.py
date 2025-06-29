import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
# Load and preprocess data
X, y = load_and_preprocess_data()

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = train_model(X_train, y_train)

# Display model performance (optional, for informational purposes)
st.sidebar.header("Model Performance & Metrics:")
test_score = model.score(X_test, y_test)
st.sidebar.write(f"Model's Accuracy on Test Set: **{test_score*100:.2f}%**")
cross_score = cross_val_score(model,X,y,cv = 20, scoring= None)
st.sidebar.write(f"Model's 20 CV score on Test Set: **{np.mean(cross_score)*100:.2f}%**")
y_preds = model.predict(X_test)
# st.sidebar.write("Confusion Matrix:")
# st.sidebar.write(ConfusionMatrixDisplay.from_estimator(estimator = model, X=X, y=y))


st.header("Enter Applicant Details:")

# Input fields for user
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.slider("Number of Dependents", 0, 5, 2)
    education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    income_annum = st.number_input("Annual Income (in ₹)", min_value=0, value=5000000, step=100000)
    loan_amount = st.number_input("Loan Amount (in ₹)", min_value=0, value=15000000, step=100000)
    residential_assets_value = st.number_input("Residential Assets Value (in ₹)", min_value=0, value=2000000, step=100000)
    luxury_assets_value = st.number_input("Luxury Assets Value (in ₹)", min_value=0, value=10000000, step=100000)


with col2:
    self_employed = st.selectbox("Self Employed", ("No", "Yes"))
    loan_term = st.slider("Loan Term (in Years)", 2, 20, 10)
    cibil_score = st.slider("CIBIL Score", 300, 900, 700)
    commercial_assets_value = st.number_input("Commercial Assets Value (in ₹)", min_value=0, value=1000000, step=100000)
    bank_asset_value = st.number_input("Bank Asset Value (in ₹)", min_value=0, value=500000, step=100000)


# Map categorical inputs to numerical for prediction
education_binary_input = 1 if education == "Graduate" else 0
self_employed_binary_input = 1 if self_employed == "Yes" else 0

# Create a DataFrame for the new input
input_data = pd.DataFrame([[
    no_of_dependents,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value,
    education_binary_input, # mapped education
    self_employed_binary_input # mapped self_employed
]], columns=[
    'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
    'cibil_score', 'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value', 'education_binary',
    'self_employed_binary'
])


# Ensure the column order matches the training data (X)
# This is crucial for correct predictions
input_data = input_data[X.columns]


if st.button("Predict Loan Status"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success(f"**Loan Status: Approved! 🎉**")
        st.write(f"Confidence (Approved): **{prediction_proba[0][1]*100:.2f}%**")
        
        st.write(f"Confidence (Rejected): {prediction_proba[0][0]*100:.2f}%")
    else:
        st.error(f"**Loan Status: Rejected 😔**")
        st.write(f"Confidence (Rejected): **{prediction_proba[0][0]*100:.2f}%**")
        st.write(f"Confidence (Approved): {prediction_proba[0][1]*100:.2f}%")

    st.markdown("---")
    st.markdown("#### Model report")
    st.markdown(classification_report(y_test,y_preds))
    # st.dataframe(input_data)


st.markdown("---")
st.markdown("#### Developed with ❤️ by Sandesh")
st.markdown("This web app is intended for educational purposes only. It should not be constituted as a real predictor or relied upon for making decisions.")
