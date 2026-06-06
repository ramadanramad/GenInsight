import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Data Analysis")

df = pd.read_csv(
    "data/stockmaster_preprocessed.csv"
)

st.subheader("Statistical Summary")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# ==========================
# Correlation Matrix
# ==========================

st.subheader("Correlation Matrix")

corr = df.select_dtypes(
    include="number"
).corr()

fig = px.imshow(
    corr,
    text_auto=True,
    aspect="auto"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================
# Histogram
# ==========================

st.subheader("Distribution")

feature = st.selectbox(
    "Select Feature",
    corr.columns
)

fig2 = px.histogram(
    df,
    x=feature,
    nbins=40
)

st.plotly_chart(
    fig2,
    use_container_width=True
)