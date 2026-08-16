<div align="center">

<!-- Animated Header -->

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&height=280&color=0:020617,50:0F172A,100:1E293B&text=Sentiment%20Analysis&fontSize=52&fontColor=FFFFFF&animation=fadeIn&fontAlignY=38&desc=Machine%20Learning%20%E2%80%A2%20Natural%20Language%20Processing%20%E2%80%A2%20Python&descAlignY=60&descSize=18"/>

<br>

<!-- Typing Animation -->

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=600&size=24&duration=3000&pause=800&color=0F172A&center=true&vCenter=true&width=850&lines=Analyze+Human+Language+with+AI;Detect+Positive+%7C+Neutral+%7C+Negative+Sentiment;TF-IDF+%2B+Machine+Learning;Real-Time+Sentiment+Prediction;Built+with+Python+%26+NLP"/>

<br><br>

<!-- Tech Stack -->

<img src="https://skillicons.dev/icons?i=python,sklearn,flask,pandas,numpy,git,github&perline=7"/>

<br><br>

<img src="https://img.shields.io/github/stars/Shrutisinha/sentiment-analysis?style=for-the-badge&logo=github&label=Stars"/>
<img src="https://img.shields.io/github/forks/Shrutisinha/sentiment-analysis?style=for-the-badge&logo=github&label=Forks"/>
<img src="https://img.shields.io/github/last-commit/Shrutisinha/sentiment-analysis?style=for-the-badge&logo=git&label=Updated"/>

</div>

---

<!-- Animated Illustration -->

<p align="center">
<img src="https://user-images.githubusercontent.com/74038190/212750228-0d3d7d82-b0d2-4df1-a4d4-ff4c2e0c3d4f.gif" width="80%"/>
</p>

## 🧠 Sentiment Analysis

> **Turning human language into actionable intelligence using Machine Learning and Natural Language Processing.**

This project analyzes textual data and automatically classifies sentiment into:

<p align="center">

😊 **Positive**    •   
😐 **Neutral**    •   
😔 **Negative**

</p>

The system combines **NLP preprocessing**, **TF-IDF feature extraction**, and **Machine Learning classification** to provide fast and reusable sentiment predictions.

---

## ✨ Features

<table>
<tr>
<td align="center" width="33%">

### 🧹

**NLP Pipeline**

Text cleaning, tokenization and stopword removal.

</td>

<td align="center" width="33%">

### 📊

**TF-IDF**

Converts textual information into numerical features.

</td>

<td align="center" width="33%">

### 🤖

**ML Classification**

Predicts Positive, Neutral or Negative sentiment.

</td>
</tr>

<tr>
<td align="center">

### ⚡

**Real-Time**

Instant sentiment prediction.

</td>

<td align="center">

### 💾

**Model Persistence**

Models stored using Joblib.

</td>

<td align="center">

### 🌐

**Web Application**

Interactive Flask interface.

</td>
</tr>
</table>

---

## 🔄 ML Pipeline

```mermaid
flowchart LR
    A["📝 Raw Text"]
    B["🧹 Cleaning"]
    C["🔤 Tokenization"]
    D["🚫 Stopword Removal"]
    E["📊 TF-IDF"]
    F["🤖 ML Model"]
    G{"🎯 Sentiment"}

    A --> B --> C --> D --> E --> F --> G

    G --> H["😊 Positive"]
    G --> I["😐 Neutral"]
    G --> J["😔 Negative"]
```

---

## 🛠️ Tech Stack

<p align="center">

<img src="https://skillicons.dev/icons?i=python,flask,git,github"/>

</p>

<p align="center">

<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-0F172A?style=flat-square&logo=scikitlearn"/>
<img src="https://img.shields.io/badge/NLTK-NLP-0F172A?style=flat-square"/>
<img src="https://img.shields.io/badge/TF--IDF-Feature%20Extraction-0F172A?style=flat-square"/>
<img src="https://img.shields.io/badge/Joblib-Model%20Persistence-0F172A?style=flat-square"/>

</p>

---

## 📂 Project Structure

```text
sentiment-analysis/
│
├── 📁 static/
│   ├── css/
│   └── js/
│
├── 📁 templates/
│   └── index.html
│
├── 📄 app.py
├── 📄 train.py
├── 📄 predict.py
├── 📄 preprocess.py
├── 📄 requirements.txt
│
├── 📦 model.pkl
├── 📦 vectorizer.pkl
├── 📊 dataset.csv
│
├── 📓 sentiment_analysis.ipynb
└── 📖 README.md
```

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/Shrutisinha/sentiment-analysis.git
cd sentiment-analysis
```

### 2. Create Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Resources

```bash
python -m nltk.downloader stopwords punkt
```

---

## 🚀 Run the Project

### Train Model

```bash
python train.py
```

### Start Web Application

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

---

## 💻 Example

```python
from predict import predict_sentiment

result = predict_sentiment(
    "This product exceeded my expectations!"
)

print(result)
```

**Output:**

```text
Positive 😊
```

---

## 📊 Model Performance

| Model                  | Accuracy | F1-Score |
| :--------------------- | :------: | :------: |
| Logistic Regression    |   ~88%   |   0.87   |
| Naive Bayes            |   ~85%   |   0.84   |
| Support Vector Machine | **~89%** | **0.88** |
| Random Forest          |   ~86%   |   0.85   |
| Gradient Boosting      |   ~87%   |   0.86   |

### 🏆 Best Performance

**Support Vector Machine — ~89% Accuracy**

---

## 🔮 Future Enhancements

* [ ] 🎯 Add prediction confidence scores
* [ ] 😊 Add emotion detection
* [ ] 📈 Add sentiment analytics dashboard
* [ ] 🧠 Experiment with Deep Learning models
* [ ] 🌐 Deploy the application online
* [ ] 🔌 Add REST API
* [ ] 📊 Add interactive visualizations
* [ ] 🗣️ Improve sarcasm detection
* [ ] 🔄 Automated model retraining

---

## 📈 Project Workflow

```text
          📝 USER TEXT
               │
               ▼
        🧹 PREPROCESSING
               │
               ▼
        📊 TF-IDF FEATURES
               │
               ▼
          🤖 ML MODEL
               │
               ▼
       🎯 SENTIMENT RESULT
          /      |      \
         /       |       \
        ▼        ▼        ▼
      😊        😐        😔
   Positive   Neutral   Negative
```

---

## 📊 GitHub Activity

<div align="center">

<img height="180" src="https://github-readme-stats.vercel.app/api?username=Shrutisinha&show_icons=true&hide_border=true&theme=transparent&rank_icon=github"/>

<img height="180" src="https://streak-stats.demolab.com?user=Shrutisinha&hide_border=true&theme=transparent"/>

</div>

<br>

<!-- Contribution Animation -->

<p align="center">
<img src="https://raw.githubusercontent.com/Platane/snk/output/github-contribution-grid-snake.svg" width="90%"/>
</p>

---

## 👩‍💻 Author

<div align="center">

### **Shruti Sinha**

**B.Tech CSE | Machine Learning | Data Analytics | NLP**

<br>

<a href="https://github.com/Shrutisinha">
<img src="https://img.shields.io/badge/GitHub-Shrutisinha-0F172A?style=for-the-badge&logo=github"/>
</a>

</div>

---

<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&weight=500&size=18&duration=3500&pause=1000&color=0F172A&center=true&vCenter=true&width=700&lines=Turning+Text+into+Intelligence+%F0%9F%A4%96;One+Sentence+at+a+Time+%F0%9F%92%AC;Powered+by+Machine+Learning+%F0%9F%A7%A0"/>

<br><br>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&section=footer&height=150&color=0:1E293B,50:0F172A,100:020617&animation=fadeIn"/>

</div>
