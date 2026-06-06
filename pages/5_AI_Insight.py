import streamlit as st
import pandas as pd
from google import genai

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Insight Generator",
    page_icon="💡",
    layout="wide"
)

st.title("💡 AI Insight Generator")
st.markdown(
    """
    Generate AI-powered insights from your financial dataset using Gemini AI.
    """
)

# =====================================
# GEMINI CLIENT
# =====================================

try:

    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

    st.success("✅ Gemini Connected Successfully")

except Exception as e:

    st.error(f"❌ Gemini Connection Error: {e}")
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

    st.error(f"❌ Dataset Error: {e}")
    st.stop()

# =====================================
# DATASET OVERVIEW
# =====================================

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

with col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Average Target",
        f"{df['Target'].mean():,.2f}"
    )

with col4:
    st.metric(
        "Max Target",
        f"{df['Target'].max():,.2f}"
    )

# =====================================
# DATA PREVIEW
# =====================================

st.subheader("📋 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# =====================================
# GENERATE AI INSIGHT
# =====================================

st.divider()

st.subheader("🤖 AI Insight Generator")

analysis_type = st.selectbox(
    "Select Analysis Type",
    [
        "General Business Insight",
        "Trend Analysis",
        "Risk Analysis",
        "Investment Recommendation",
        "Executive Summary"
    ]
)

if st.button("🚀 Generate AI Insight", use_container_width=True):

    with st.spinner("Gemini is analyzing your dataset..."):

        try:

            summary_stats = df.describe().to_string()

            sample_data = df.head(20).to_string()

            prompt = f"""
You are a senior financial analyst and business intelligence expert.

Analyze the following financial dataset.

Analysis Type:
{analysis_type}

Dataset Information:

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Columns:
{list(df.columns)}

Statistical Summary:
{summary_stats}

Sample Data:
{sample_data}

Generate a detailed analysis including:

1. Dataset Overview
2. Key Findings
3. Trend Analysis
4. Potential Risks
5. Business Insights
6. Strategic Recommendations

Use professional business language.
Use markdown formatting.
"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.success("✅ Insight Generated Successfully")

            st.subheader("📊 AI Generated Insight")

            st.markdown(response.text)

        except Exception as e:

            st.error(f"❌ Insight Generation Error: {e}")

# =====================================
# QUICK STATISTICS
# =====================================

st.divider()

st.subheader("📈 Quick Statistics")

st.dataframe(
    df.describe(),
    use_container_width=True
)

# =====================================
# COLUMN INFORMATION
# =====================================

st.divider()

st.subheader("📑 Dataset Columns")

column_df = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(
    column_df,
    use_container_width=True
)

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "GenInsight • AI-Based Data Analysis & Automated Insight Generation Dashboard"
)