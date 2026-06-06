import streamlit as st
import pandas as pd
from google import genai
import time

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="GenInsight Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GenInsight AI Assistant")

# =====================================
# GEMINI CLIENT
# =====================================

try:

    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

except Exception as e:

    st.error(f"Gemini Error: {e}")
    st.stop()

# =====================================
# LOAD DATASET CONTEXT
# =====================================

try:

    df = pd.read_csv(
        "data/stockmaster_preprocessed.csv"
    )

except Exception:

    df = None

# =====================================
# CHAT HISTORY
# =====================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================
# USER INPUT
# =====================================

prompt = st.chat_input(
    "Ask something about the dataset..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        try:

            dataset_context = ""

            if df is not None:

                stats_summary = (
                    df.describe()
                    .round(2)
                    .to_string()
                )

                dataset_context = f"""
Dataset Information

Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column Names:
{list(df.columns)}

Statistics:
{stats_summary}
"""

            full_prompt = f"""
You are GenInsight AI Assistant.

You are an expert in:

- Financial Analysis
- Business Intelligence
- XGBoost Prediction
- LSTM Forecasting
- Risk Analysis
- Stock Market Analysis

Dataset Context:

{dataset_context}

User Question:

{prompt}

Provide a clear, concise, and professional answer.
"""

            response = None

            for attempt in range(3):

                try:

                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=full_prompt
                    )

                    break

                except Exception:

                    if attempt < 2:
                        time.sleep(3)
                    else:
                        raise

            answer = response.text

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            st.error(
                f"Chatbot Error: {e}"
            )
