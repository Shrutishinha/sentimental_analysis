import streamlit as st
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.title("☁ Word Cloud")

text_input = st.text_area("Enter text payload to generate Word Cloud:", "Insert reviews text block here...")

if st.button("Generate Word Cloud"):
    if text_input.strip():
        with st.spinner("Generating frequencies..."):
            try:
                res = requests.post("http://localhost:8000/wordcloud", json={"text": text_input}).json()
                freqs = res["frequencies"]
                
                if freqs:
                    wc = WordCloud(width=800, height=450, background_color="white", colormap="viridis").generate_from_frequencies(freqs)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("No matching vocabulary words to draw.")
            except Exception as e:
                st.error(f"Failed to generate: {e}")