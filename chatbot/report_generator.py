from google import genai
import streamlit as st
import time

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

def generate_report(prompt):

    for attempt in range(3):

        try:

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )

            return response.text

        except Exception as e:

            if attempt < 2:
                time.sleep(5)
            else:
                raise e
