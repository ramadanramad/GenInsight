import streamlit as st
import pandas as pd

from chatbot.report_generator import generate_report

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Report Generator",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Report Generator")

st.markdown(
    """
Generate a professional business report automatically using Gemini AI.
"""
)

# =====================================
# LOAD DATASET
# =====================================

try:

    df = pd.read_csv(
        "data/stockmaster_preprocessed.csv"
    )

    st.success("✅ Dataset Loaded Successfully")

except Exception as e:

    st.error(f"Dataset Error: {e}")
    st.stop()

# =====================================
# DATASET INFO
# =====================================

st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

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
    round(df["Target"].mean(), 2)
)

# =====================================
# REPORT TYPE
# =====================================

report_type = st.selectbox(
    "Select Report Type",
    [
        "Executive Report",
        "Business Intelligence Report",
        "Investment Analysis Report",
        "Risk Assessment Report"
    ]
)

# =====================================
# GENERATE REPORT
# =====================================

if st.button(
    "📄 Generate AI Report",
    use_container_width=True
):

    with st.spinner("Generating report..."):

        try:

            summary = df.describe().to_string()

            prompt = f"""
You are a senior business analyst.

Create a professional report.

Report Type:
{report_type}

Dataset Statistics:
{summary}

Generate:

1. Executive Summary
2. Dataset Overview
3. Key Findings
4. Model Performance Discussion
5. Risk Assessment
6. Strategic Recommendations
7. Conclusion

Use markdown format.
Use professional language.
"""

            report = generate_report(
                prompt
            )

            st.success(
                "✅ Report Generated Successfully"
            )

            st.subheader(
                "📑 Generated Report"
            )

            st.markdown(
                report
            )

            st.download_button(
                label="⬇ Download Report (.txt)",
                data=report,
                file_name="geninsight_report.txt",
                mime="text/plain"
            )

        except Exception as e:

            st.error(
                f"Report Error: {e}"
            )