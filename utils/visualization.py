"""
=========================================================
Visualization Utilities
=========================================================

Functions Included
------------------
✓ Sentiment Distribution
✓ Pie Chart
✓ Word Cloud
✓ Top Frequent Words
✓ N-Grams
✓ Confusion Matrix
✓ ROC Curve
✓ Precision-Recall Curve
✓ Feature Importance
✓ Prediction Probability
✓ Text Length Distribution

=========================================================
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

from sklearn.feature_extraction.text import CountVectorizer


# --------------------------------------------------------
# Sentiment Distribution
# --------------------------------------------------------

def sentiment_distribution(df, column="sentiment"):

    counts = df[column].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        text="Count",
        title="Sentiment Distribution"
    )

    fig.update_layout(height=450)

    return fig


# --------------------------------------------------------
# Pie Chart
# --------------------------------------------------------

def sentiment_pie(df, column="sentiment"):

    counts = df[column].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]

    fig = px.pie(
        counts,
        values="Count",
        names="Sentiment",
        title="Sentiment Distribution"
    )

    return fig


# --------------------------------------------------------
# Word Cloud
# --------------------------------------------------------

def create_wordcloud(text):

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        max_words=300
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")

    return fig


# --------------------------------------------------------
# Top Words
# --------------------------------------------------------

def top_words(texts, top_n=20):

    vectorizer = CountVectorizer(stop_words="english")

    X = vectorizer.fit_transform(texts)

    freq = np.asarray(X.sum(axis=0)).ravel()

    words = vectorizer.get_feature_names_out()

    df = pd.DataFrame({
        "Word": words,
        "Frequency": freq
    })

    df = df.sort_values(
        by="Frequency",
        ascending=False
    ).head(top_n)

    fig = px.bar(
        df,
        x="Frequency",
        y="Word",
        orientation="h",
        title=f"Top {top_n} Words"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )

    return fig


# --------------------------------------------------------
# N-Grams
# --------------------------------------------------------

def top_ngrams(texts, n=2, top_k=20):

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(n, n)
    )

    X = vectorizer.fit_transform(texts)

    freq = np.asarray(X.sum(axis=0)).ravel()

    grams = vectorizer.get_feature_names_out()

    df = pd.DataFrame({
        "NGram": grams,
        "Frequency": freq
    })

    df = df.sort_values(
        by="Frequency",
        ascending=False
    ).head(top_k)

    fig = px.bar(
        df,
        x="Frequency",
        y="NGram",
        orientation="h",
        title=f"Top {top_k} {n}-Grams"
    )

    fig.update_layout(
        yaxis=dict(autorange="reversed")
    )

    return fig


# --------------------------------------------------------
# Text Length Distribution
# --------------------------------------------------------

def text_length_distribution(df, column="review"):

    lengths = df[column].str.len()

    fig = px.histogram(
        x=lengths,
        nbins=40,
        title="Review Length Distribution"
    )

    fig.update_layout(
        xaxis_title="Characters",
        yaxis_title="Count"
    )

    return fig


# --------------------------------------------------------
# Confusion Matrix
# --------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(
            x="Predicted",
            y="Actual",
            color="Count"
        ),
        title="Confusion Matrix"
    )

    return fig


# --------------------------------------------------------
# ROC Curve
# --------------------------------------------------------

def plot_roc_curve(y_true, probabilities):

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    roc_auc = auc(fpr, tpr)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {roc_auc:.3f}"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    return fig


# --------------------------------------------------------
# Precision Recall Curve
# --------------------------------------------------------

def plot_precision_recall(y_true, probabilities):

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines"
        )
    )

    fig.update_layout(
        title="Precision Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    return fig


# --------------------------------------------------------
# Feature Importance
# --------------------------------------------------------

def feature_importance(model, vectorizer, top_n=20):

    if not hasattr(model, "coef_"):
        return None

    coef = model.coef_[0]

    words = vectorizer.get_feature_names_out()

    df = pd.DataFrame({
        "Word": words,
        "Importance": coef
    })

    df = df.sort_values(
        by="Importance",
        ascending=False
    )

    top = pd.concat([
        df.head(top_n),
        df.tail(top_n)
    ])

    fig = px.bar(
        top,
        x="Importance",
        y="Word",
        orientation="h",
        title="Most Important Features"
    )

    return fig


# --------------------------------------------------------
# Prediction Probability
# --------------------------------------------------------

def probability_chart(prob_positive):

    prob_negative = 1 - prob_positive

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Negative"],
            y=[prob_negative]
        )
    )

    fig.add_trace(
        go.Bar(
            x=["Positive"],
            y=[prob_positive]
        )
    )

    fig.update_layout(
        title="Prediction Confidence",
        yaxis_title="Probability",
        barmode="group"
    )

    return fig


# --------------------------------------------------------
# KPI Cards
# --------------------------------------------------------

def dataset_statistics(df):

    stats = {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum())
    }

    return stats


# --------------------------------------------------------
# Sample Visualization Test
# --------------------------------------------------------

if __name__ == "__main__":

    sample = pd.DataFrame({

        "review": [
            "Great movie",
            "Worst movie",
            "Amazing acting",
            "Terrible film",
            "Loved it"
        ],

        "sentiment": [
            "positive",
            "negative",
            "positive",
            "negative",
            "positive"
        ]

    })

    print(dataset_statistics(sample))

    fig = sentiment_distribution(sample)

    fig.show()