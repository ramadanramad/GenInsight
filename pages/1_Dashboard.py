import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Dataset
df = pd.read_csv(
    "data/stockmaster_preprocessed.csv"
)

st.title("📊 Dataset Dashboard")

# Metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Rows",
    f"{df.shape[0]:,}"
)

col2.metric(
    "Total Columns",
    df.shape[1]
)

col3.metric(
    "Missing Values",
    int(df.isnull().sum().sum())
)

col4.metric(
    "Average Target",
    round(df["Target"].mean(), 2)
)

st.divider()

st.subheader("Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)