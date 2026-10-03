import joblib
import pandas as pd

def load_model():
    model = joblib.load('models/random_forest_fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

def predict_fraud(model, transaction_data):
    prediction = model.predict(transaction_data)
    probability = model.predict_proba(transaction_data)[:, 1]
    return prediction, probability