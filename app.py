import streamlit as st
import pickle
import pandas as pd

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
feature_names = model_data["features_names"]
#model → trained RandomForest model

# feature_names → correct column order used during training

st.title("Customer Churn Prediction App")

st.write("Enter customer details to predict churn.")



gender = st.selectbox("Gender", ["Female", "Male"])
SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
Partner = st.selectbox("Partner", ["Yes","No"])
Dependents = st.selectbox("Dependents", ["Yes","No"])
tenure = st.slider("Tenure (Months)",0,72)
PhoneService = st.selectbox("Phone Service",["Yes","No"])
MultipleLines = st.selectbox("Multiple Lines",["No","Yes","No phone service"])
InternetService = st.selectbox("Internet Service",["DSL","Fiber optic","No"])
OnlineSecurity = st.selectbox("Online Security",["Yes","No","No internet service"])
OnlineBackup = st.selectbox("Online Backup",["Yes","No","No internet service"])
DeviceProtection = st.selectbox("Device Protection",["Yes","No","No internet service"])
TechSupport = st.selectbox("Tech Support",["Yes","No","No internet service"])
StreamingTV = st.selectbox("Streaming TV",["Yes","No","No internet service"])
StreamingMovies = st.selectbox("Streaming Movies",["Yes","No","No internet service"])
Contract = st.selectbox("Contract",["Month-to-month","One year","Two year"])
PaperlessBilling = st.selectbox("Paperless Billing",["Yes","No"])
PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
)

MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")



input_dict = {
    'gender': gender,
    'SeniorCitizen': SeniorCitizen,
    'Partner': Partner,
    'Dependents': Dependents,
    'tenure': tenure,
    'PhoneService': PhoneService,
    'MultipleLines': MultipleLines,
    'InternetService': InternetService,
    'OnlineSecurity': OnlineSecurity,
    'OnlineBackup': OnlineBackup,
    'DeviceProtection': DeviceProtection,
    'TechSupport': TechSupport,
    'StreamingTV': StreamingTV,
    'StreamingMovies': StreamingMovies,
    'Contract': Contract,
    'PaperlessBilling': PaperlessBilling,
    'PaymentMethod': PaymentMethod,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges
}

input_df = pd.DataFrame([input_dict])



for column in encoders:
    if column in input_df.columns:
        le = encoders[column]
        input_df[column] = le.transform(input_df[column])


input_df = input_df[feature_names]


if st.button("Predict Churn"):

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")