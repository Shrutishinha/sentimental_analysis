import streamlit as st
import requests

st.title("🧹 Text Preprocessing")

text_input = st.text_area("Enter review to preprocess:", "I absolutely loved this movie! The acting was amazing and the story was wonderful.")

if st.button("Analyze and Preprocess"):
    if text_input.strip():
        with st.spinner("Analyzing text..."):
            try:
                res = requests.post("http://localhost:8001/analyze", json={"text": text_input}).json()
                
                # Metadata Metrics
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Characters", res["characters"])
                c2.metric("Words", res["words"])
                c3.metric("Sentences", res["sentences"])
                c4.metric("Vocabulary Size", res["vocabulary"])
                
                # Tokenization steps
                st.write("### Token Stages")
                st.write("**Cleaned Text:**", res["cleaned"])
                st.write("**Tokens:**", res["tokens"])
                st.write("**Without Stopwords:**", res["without_stopwords"])
                st.write("**Stemmed Tokens:**", res["stemmed"])
                st.write("**Lemmatized Tokens:**", res["lemmatized"])
                
                # POS / NER Tagging
                st.write("### POS & Named Entity Recognition")
                st.write("**POS Tags:**", res["pos_tags"])
                st.write("**Named Entities:**", res["entities"])
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")