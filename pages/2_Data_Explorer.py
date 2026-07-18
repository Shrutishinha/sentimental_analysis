import streamlit as st
import pandas as pd
import os

st.title("📊 Data Explorer")

csv_path = "data/sample_reviews.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.write("### Dataset Preview")
    st.dataframe(df)
    
    st.write("### Dataset Details")
    st.write(f"Total Rows: **{len(df)}**")
    st.write("#### Class Distribution:")
    st.write(df['sentiment'].value_counts())
else:
    st.info("No data found. Please go to **Model Training** page and train a model first to generate the dataset.")