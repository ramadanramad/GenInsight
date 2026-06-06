import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="LSTM Prediction",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 LSTM Prediction")

# =====================================
# LOAD MODEL
# =====================================

try:
    model = load_model(
        "models/lstm_model.h5",
        compile=False
    )

    st.success("✅ LSTM Model Loaded Successfully")

except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# =====================================
# LOAD DATASET
# =====================================

try:
    df = pd.read_csv(
        "data/stockmaster_preprocessed.csv"
    )

    st.success("✅ Dataset Loaded Successfully")

except Exception as e:
    st.error(f"❌ Error loading dataset: {e}")
    st.stop()

# =====================================
# DATASET OVERVIEW
# =====================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Rows",
    f"{df.shape[0]:,}"
)

col2.metric(
    "Columns",
    df.shape[1]
)

col3.metric(
    "Average Target",
    f"{df['Target'].mean():,.2f}"
)

col4.metric(
    "Max Target",
    f"{df['Target'].max():,.2f}"
)

# =====================================
# DATASET PREVIEW
# =====================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# =====================================
# LSTM PREDICTION
# =====================================

st.subheader("📈 Next Target Prediction")

try:

    data = df["Target"].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    window = 60

    last_60_days = scaled_data[-window:]

    X_test = np.array([last_60_days])

    prediction = model.predict(
        X_test,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction
    )

    st.metric(
        "Predicted Next Target",
        f"{prediction[0][0]:,.2f}"
    )

except Exception as e:

    st.error(
        f"Prediction Error: {e}"
    )

# =====================================
# HISTORICAL TARGET TREND
# =====================================

st.subheader("📉 Historical Target Trend")

fig = px.line(
    df.tail(200),
    y="Target",
    title="Target Trend (Last 200 Records)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =====================================
# LAST 100 RECORDS
# =====================================

st.subheader("📊 Last 100 Target Values")

chart_data = df.tail(100)

fig2 = px.line(
    chart_data,
    y="Target",
    markers=True
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================
# MODEL INFORMATION
# =====================================

st.subheader("ℹ️ Model Information")

st.info(
    """
    Model Type : LSTM (Long Short-Term Memory)

    Window Size : 60

    Prediction Target : Target

    Purpose :
    Forecast future target values using historical sequential data.
    """
)