import streamlit as st
import requests

st.title("💬 Real-Time Prediction")

text_input = st.text_input("Enter a review for sentiment prediction:", "This movie was absolutely fantastic. The acting and music were amazing.")

if st.button("Predict Sentiment"):
    if text_input.strip():
        with st.spinner("Classifying sentiment..."):
            try:
                res = requests.post("http://localhost:8000/predict", json={"text": text_input}).json()
                
                # Display sentiment outcome
                st.write("### Result")
                pred = res["prediction"].upper()
                emoji = res["emoji"]
                confidence = res["confidence"]
                
                if pred == "POSITIVE":
                    st.success(f"Sentiment: **{pred}** {emoji} (Confidence: {confidence * 100:.2f}%)")
                else:
                    st.error(f"Sentiment: **{pred}** {emoji} (Confidence: {confidence * 100:.2f}%)")
                
                # Class probabilities chart
                st.write("#### Probabilities:")
                st.json(res["probabilities"])
            except Exception as e:
                st.error(f"Error predicting: {e}")