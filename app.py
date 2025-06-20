import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('churn_dl_model.h5')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title("ChurnPred")
st.subheader("Deep Learning-Powered Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 35)
tenure = st.slider("Tenure (Years with bank)", 0, 10, 3)
balance = st.number_input("Account Balance", value=10000.0)
num_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
salary = st.number_input("Estimated Salary", value=50000.0)
geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])

# Preprocess user input
input_data = {
    'CreditScore': credit_score,
    'Gender': 1 if gender == 'Male' else 0,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': salary,
    'Geography_Germany': 1 if geography == 'Germany' else 0,
    'Geography_Spain': 1 if geography == 'Spain' else 0
}

for col in feature_columns:
    if col not in input_data:
        input_data[col] = 0

input_df = np.array([input_data[col] for col in feature_columns]).reshape(1, -1)
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prob = float(model.predict(input_scaled)[0][0])
    prediction = int(prob >= 0.5)

    if prediction == 1:
        st.error(f"The customer is likely to churn.\nProbability: {prob:.2f}")
    else:
        st.success(f"The customer is likely to stay.\nProbability: {prob:.2f}")

    if prediction == 1:
        st.error(f" The customer is **likely to churn**.\nProbability: {prob:.2f}")
    else:
        st.success(f"The customer is **likely to stay**.\nProbability: {prob:.2f}")
