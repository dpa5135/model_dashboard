# app_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

# Train model if not already saved
MODEL_FILE = 'model.pkl'

if not os.path.exists(MODEL_FILE):
    df = pd.DataFrame({
        'feature1': range(1, 101),
        'feature2': [x * 2 for x in range(1, 101)],
    })
    df['target'] = df['feature1'] * 3 + df['feature2'] * 2 + 5
    X = df[['feature1', 'feature2']]
    y = df['target']
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
else:
    model = joblib.load(MODEL_FILE)

# Streamlit UI
st.title("🔮 Simple Regression Model Dashboard")

st.write("Enter values for the features to get a prediction:")

feature1 = st.number_input("Feature 1", value=1.0)
feature2 = st.number_input("Feature 2", value=2.0)

if st.button("Predict"):
    X_input = np.array([[feature1, feature2]])
    prediction = model.predict(X_input)[0]
    st.success(f"📈 Predicted Value: **{round(prediction, 2)}**")
