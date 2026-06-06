import streamlit as st
import joblib
import pandas as pd

st.title("🤖 XGBoost Prediction")

model = joblib.load(
    "models/xgboost_model.pkl"
)

close_price = st.number_input(
    "Close Price"
)

eps = st.number_input(
    "Basic EPS"
)

der = st.number_input(
    "DER"
)

roa = st.number_input(
    "ROA"
)

roe = st.number_input(
    "ROE"
)

ma5 = st.number_input(
    "MA5"
)

ma20 = st.number_input(
    "MA20"
)

vol = st.number_input(
    "Volatility20"
)

if st.button("Predict"):

    data = pd.DataFrame(
        [[
            close_price,
            eps,
            der,
            roa,
            roe,
            ma5,
            ma20,
            vol
        ]],
        columns=[
            'Close_Price',
            'Basic EPS',
            'DER',
            'ROA',
            'ROE',
            'MA5',
            'MA20',
            'Volatility20'
        ]
    )

    pred = model.predict(data)

    st.success(
        f"Prediction: {pred[0]:.2f}"
    )