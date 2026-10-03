import streamlit as st
import joblib
import pandas as pd


model = joblib.load('../models/random_forest_fraud_model.pkl')
scaler = joblib.load('../models/scaler.pkl')
X_test = pd.read_csv('../data/processed/X_test.csv')
y_test = pd.read_csv('../data/processed/y_test.csv')

st.title("Bank Fraud Detection")
st.write("Select a transaction from the test set to check if it's fraudulent")

row_number = st.number_input("Select Transaction Number", min_value=0, max_value=len(X_test)-1, value=0, step=1)

selected_transaction = X_test.iloc[[row_number]]
actual_label = y_test.iloc[row_number, 0]

col1, col2, col3 = st.columns(3)
col1.metric("Amount", f"₹{selected_transaction['Amount'].values[0]:.2f}")
col2.metric("Time (sec)", f"{selected_transaction['Time'].values[0]:.0f}")
col3.metric("Actual Label", "Fraud" if actual_label == 1 else "Not Fraud")

st.divider()

if st.button("Predict", type="primary"):
    scaled_transaction = scaler.transform(selected_transaction)
    prediction = model.predict(scaled_transaction)[0]
    probability = model.predict_proba(scaled_transaction)[0][1]
    
    
    if prediction == 1:
        st.error(f"FRAUD DETECTED")
    else:
        st.success(f"NOT FRAUD")
    
    
    st.write("**Model Confidence (Probability of Fraud):**")
    st.progress(probability)
    st.write(f"{probability:.1%}")