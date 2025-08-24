import streamlit as st
import sys
import pandas as pd 
import numpy as np
import joblib
st.title("E-commerce Customer Purchase Prediction")

Time_on_Website = st.slider("Time on Website (in minutes)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
Time_on_App = st.slider("Time on App (in minutes)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)
Length_of_Membership = st.number_input("Length of Membership (in years)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
creatinine_level = st.number_input("Creatinine Level", min_value=0.0, max_value=800.0, value=1.0, step=0.1)
Yearly_Amount_Spent = st.number_input("Yearly Amount Spent", min_value=0.0, max_value=10000.0, value=1000.0, step=0.1)
Email = st.radio("Email", options=["Yes", "No"], index=0)
Avg_Session_Length = st.number_input("Avg. Session Length (in minutes)", min_value=0.0, max_value=100.0, value=10.0, step=0.1)

if st.button("Predict"):
    model = joblib.load("ecommerce_model.pkl")
    columns = joblib.load("ecommerce_columns.pkl")
    email_val = 1 if Email == "Yes" else 0
    input_data = pd.DataFrame([[Time_on_Website, Time_on_App, Length_of_Membership, creatinine_level, Yearly_Amount_Spent, email_val, Avg_Session_Length]], columns=columns)
    prediction = model.predict(input_data)
    st.success(f"The predicted purchase amount is: ${prediction[0]:.2f}")