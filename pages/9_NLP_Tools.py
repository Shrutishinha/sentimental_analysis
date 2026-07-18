import streamlit as st
import requests

st.title("📝 NLP Tools")

text_input = st.text_input("Enter text fragment to compute parameters:", "Hello NLP sentiment analyzer!")

if st.button("Compute Stats"):
    if text_input.strip():
        try:
            res = requests.post("http://localhost:8000/analyze", json={"text": text_input}).json()
            st.write("### Analysis Summary:")
            st.write("- **Total Characters:**", res["characters"])
            st.write("- **Total Words:**", res["words"])
            st.write("- **Vocabulary Count:**", res["vocabulary"])
            st.write("- **Sentence Count:**", res["sentences"])
        except Exception as e:
            st.error(f"Connection failed: {e}")