import streamlit as st
import pandas as pd
import requests
import io

st.title("📂 Batch Prediction")

uploaded_file = st.file_uploader("Upload a CSV file containing reviews:", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Preview")
    st.dataframe(df.head())
    
    text_column = st.selectbox("Select Review Text Column:", df.columns)
    
    if st.button("Run Batch Prediction"):
        with st.spinner("Processing file..."):
            try:
                # Prepare file for multipart upload
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                params = {"text_column": text_column}
                
                res = requests.post("http://localhost:8000/predict/file", files=files, params=params).json()
                
                res_df = pd.DataFrame(res)
                st.success("✅ Batch Prediction Finished!")
                st.write("### Prediction Results")
                st.dataframe(res_df)
                
                # Download button
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name="sentiment_predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Batch processing failed: {e}")