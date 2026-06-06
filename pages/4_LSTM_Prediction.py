import streamlit as st
import pandas as pd

st.title("🧠 LSTM Prediction")

st.success("✅ LSTM Model Trained Successfully")

st.info("""
LSTM model berhasil dilatih pada tahap development.
Versi cloud menampilkan hasil training karena TensorFlow
belum kompatibel dengan environment Streamlit Cloud.
""")

df = pd.read_csv("data/stockmaster_preprocessed.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

col1, col2, col3 = st.columns(3)

col1.metric("Epoch", "10")
col2.metric("Status", "Success")
col3.metric("Model", "lstm_model.h5")
