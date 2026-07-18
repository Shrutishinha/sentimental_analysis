import streamlit as st
import requests

st.set_page_config(page_title="NLP Sentiment Studio", page_icon="💬")

st.title("💬 NLP Sentiment Analysis Studio")

st.markdown("""
An end-to-end NLP application built with **Python, Streamlit, Scikit-Learn, FastAPI and Machine Learning**.
""")

# Backend Status
try:
    response = requests.get("http://localhost:8000/").json()

    if response["status"] == "healthy":
        st.success("🟢 Backend Connected")
        st.info(f"Model Loaded : {response['model_loaded']}")
except:
    st.error("🔴 Backend Offline")
    st.code("uvicorn main:app --reload")

st.divider()

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric("🤖 ML Models","5+")

with col2:
    st.metric("📚 NLP Techniques","10+")

with col3:
    st.metric("📊 Visualizations","Interactive")

with col4:
    st.metric("⚡ Prediction","Real-Time")

st.divider()

st.subheader("📖 Project Overview")

st.write("""
This application demonstrates the complete NLP pipeline:

- 📂 Dataset Exploration
- 🧹 Text Preprocessing
- 🤖 Model Training
- 🔍 Sentiment Prediction
- 📈 Model Comparison
- ☁️ Word Cloud Generation
- 📑 Batch CSV Prediction
""")