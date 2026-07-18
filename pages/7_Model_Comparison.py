import streamlit as st
import requests

st.title("📈 Model Comparison")

try:
    res = requests.get("http://localhost:8000/metrics").json()
    if res["model_loaded"]:
        st.write("### Active Classification Pipeline Details")
        st.write("- **Algorithm:**", res["algorithm"])
        st.write("- **Feature Extraction:**", res["vectorizer"])
        st.write("- **Output Dimensions:**", res["features"])
        st.write("#### Hyperparameters:")
        st.json(res["parameters"])
    else:
        st.warning("No trained model loaded. Train a model first.")
except Exception as e:
    st.error(f"Failed to fetch model comparison details: {e}")