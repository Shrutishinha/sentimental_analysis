import streamlit as st
from PIL import Image
import base64

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="NLP Sentiment Analysis Studio",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

.main{
    background-color:#F7F9FC;
}

.hero{
    padding:40px;
    border-radius:22px;
    background:linear-gradient(135deg,#2563EB,#4F46E5,#7C3AED);
    color:white;
    box-shadow:0 15px 35px rgba(0,0,0,.25);
}
.feature-card{
    background:linear-gradient(135deg,#ffffff,#F8FAFC);
    color:#111827;
    padding:22px;
    border-radius:18px;
    box-shadow:0 10px 25px rgba(0,0,0,.15);
    transition:.3s;
}

.feature-card h3{
    color:#2563EB;
}

.feature-card li{
    color:#374151;
}

.feature-card:hover{
    transform:translateY(-6px);
}

.metric{
    background:#ffffff;
    padding:20px;
    border-radius:18px;
    box-shadow:0 10px 25px rgba(0,0,0,.15);
    text-align:center;
    color:#111827;
}

.metric h2{
    color:#2563EB;
    font-size:40px;
    font-weight:bold;
    margin-bottom:8px;
}

.metric p{
    color:#374151;
    font-size:18px;
    font-weight:600;
}

.footer{
    text-align:center;
    color:gray;
    padding:30px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2103/2103832.png",
    width=120,
)

st.sidebar.title("📚 NLP Studio")

st.sidebar.info(
"""
### Navigation

Use the pages from the sidebar.

🏠 Home

📊 Data Explorer

🧹 Text Preprocessing

🤖 Model Training

💬 Prediction

📂 Batch Prediction

📈 Model Comparison

☁ Word Cloud

📝 NLP Tools

ℹ About
"""
)

st.sidebar.success("Version 1.0")

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero">

<h1>🤖 NLP Sentiment Analysis Studio</h1>
<h4>
🚀 Interactive NLP Dashboard with Machine Learning, Deep Learning,
FastAPI, Streamlit and Data Visualization.
</h4>

</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Metrics
# -----------------------------
col1,col2,col3,col4=st.columns(4)

with col1:
    st.markdown("""
    <div class="metric">
   <h1>🤖</h1>
<h2>5+</h2>
<p>ML Models</p>
    </div>
    """,unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric">
   <h1>📚</h1>
            
<h2>10+</h2>
<p>NLP Techniques</p>
    </div>
    """,unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric">
   <h1>📈</h1>
<h2>Interactive</h2>
<p>Visualizations</p>
    </div>
    """,unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric">
    <h1>⚡</h1>
<h2>Real-Time</h2>
<p>Prediction</p>
    </div>
    """,unsafe_allow_html=True)

st.write("")

# -----------------------------
# About Project
# -----------------------------
st.header("📖 Project Overview")

st.write("""

The **NLP Sentiment Analysis Studio** is a comprehensive Natural Language
Processing application that demonstrates the complete workflow of
sentiment analysis using Machine Learning and Deep Learning.

The project covers every stage of the NLP pipeline—from raw text
preprocessing to feature extraction, model training, prediction, and
interactive visualization.

This application is designed as an educational and research-oriented
tool suitable for students, researchers, and professionals interested
in sentiment analysis and text mining.

""")

# -----------------------------
# Workflow
# -----------------------------
st.header("⚙ NLP Workflow")

st.graphviz_chart("""
digraph{
rankdir=LR
Input->Cleaning
Cleaning->Tokenization
Tokenization->Lemmatization
Lemmatization->TFIDF
TFIDF->Model
Model->Prediction
}
""")

# -----------------------------
# Features
# -----------------------------
st.header("🚀 Features")

c1,c2,c3=st.columns(3)

with c1:

    st.markdown("""
    <div class="feature-card">

### 📊 Data Explorer

- View Dataset
- Search Reviews
- Missing Values
- Statistics
- Class Distribution

</div>
    """,unsafe_allow_html=True)

with c2:

    st.markdown("""
    <div class="feature-card">

### 🧹 NLP Processing

- Cleaning
- Tokenization
- Stopwords
- Lemmatization
- Stemming
- POS Tagging

</div>
    """,unsafe_allow_html=True)

with c3:

    st.markdown("""
    <div class="feature-card">

### 🤖 Machine Learning

- Logistic Regression
- Naive Bayes
- SVM
- Random Forest
- Decision Tree

</div>
    """,unsafe_allow_html=True)

st.write("")

c4,c5,c6=st.columns(3)

with c4:

    st.markdown("""
    <div class="feature-card">

### 💬 Prediction

- Real-time Prediction
- Confidence Score
- Positive/Negative
- Explainable Output

</div>
    """,unsafe_allow_html=True)

with c5:

    st.markdown("""
    <div class="feature-card">

### 📈 Visualization

- Pie Chart
- Histogram
- ROC Curve
- Confusion Matrix
- WordCloud

</div>
    """,unsafe_allow_html=True)

with c6:

    st.markdown("""
    <div class="feature-card">

### ☁ Batch Prediction

- Upload CSV
- Predict Reviews
- Download Results
- Export CSV

</div>
    """,unsafe_allow_html=True)

# -----------------------------
# Technologies
# -----------------------------
st.header("🛠 Technologies Used")

tech1,tech2,tech3=st.columns(3)

with tech1:
    st.success("""
✅ Python

✅ Streamlit

✅ Pandas

✅ NumPy
""")

with tech2:
    st.success("""
✅ NLTK

✅ Scikit-Learn

✅ SpaCy

✅ TextBlob
""")

with tech3:
    st.success("""
✅ TensorFlow

✅ Plotly

✅ Matplotlib

✅ WordCloud
""")

# -----------------------------
# Model Pipeline
# -----------------------------
st.header("🧠 Sentiment Analysis Pipeline")

st.info("""

Input Review

⬇

Text Cleaning

⬇

Tokenization

⬇

Stopword Removal

⬇

Lemmatization

⬇

TF-IDF Vectorization

⬇

Machine Learning Classifier

⬇

Positive / Negative Prediction

""")

# -----------------------------
# Future Scope
# -----------------------------
st.header("🔮 Future Scope")

st.write("""

✔ Deep Learning Models (LSTM, BiLSTM)

✔ Transformer Models (BERT)

✔ Multilingual Sentiment Analysis

✔ Aspect-Based Sentiment Analysis

✔ Explainable AI (SHAP/LIME)

✔ Emotion Detection

✔ Fake Review Detection

✔ Live Twitter/X Sentiment Analysis

""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">

Made with ❤️ using Streamlit | NLP Sentiment Analysis Studio

</div>
""",unsafe_allow_html=True)