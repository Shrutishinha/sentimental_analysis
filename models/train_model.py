"""
=========================================================
Model Training Script
=========================================================
Trains a sentiment analysis model using TF-IDF and Logistic Regression.
Saves the trained models as pickle files.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.preprocess import get_processed_text

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
DATA_PATH = "data/IMDB_Dataset.csv"

def generate_synthetic_data():
    """Generates synthetic positive and negative reviews if no dataset exists."""
    pos_reviews = [
        "I absolutely loved this movie! The acting was brilliant.",
        "An absolute masterpiece. Highly recommended for everyone.",
        "Great story, wonderful characters, and a very satisfying ending.",
        "The cinematography was breathtaking and the acting was top-notch.",
        "One of the best movies of the year. Captivating and emotional.",
        "Really enjoyed the plot and the background score was perfect.",
        "A beautiful story told with great sensitivity and humor.",
        "Fantastic performance by the lead actors. Extremely engaging.",
        "It was a delightful experience. I will definitely watch it again.",
        "A highly entertaining and fast-paced adventure film.",
    ]
    
    neg_reviews = [
        "This movie was a complete waste of time. Terribly slow.",
        "Poor acting and a predictable storyline. Do not watch.",
        "I was extremely disappointed. The plot made no sense.",
        "Worst movie of the year. The acting was painful to watch.",
        "Boring, uninspiring, and completely lacks any coherent direction.",
        "The cinematography was decent, but the script was awful.",
        "I could not even finish the movie. It was that bad.",
        "A waste of talent and resources. Very badly written.",
        "Completely ridiculous plot lines and flat acting.",
        "Very disappointing experience. It fell short of expectations.",
    ]
    
    # Generate variations to avoid exact duplicates and prevent data leakage
    variations = ["", " Definitely.", " Truly.", " Indeed.", " Absolutely.", " Overall, a must-mention.", " Highly relevant.", " No doubt about it.", " Just my opinion.", " In my honest view."]
    
    reviews = []
    sentiments = []
    
    for var in variations:
        for pos in pos_reviews:
            reviews.append(pos + var)
            sentiments.append("positive")
        for neg in neg_reviews:
            reviews.append(neg + var)
            sentiments.append("negative")
            
    return pd.DataFrame({"review": reviews, "sentiment": sentiments})

def train_and_save_model():
    """Trains the sentiment model and saves components to pickle files."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Load dataset
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0:
        print("Loading dataset from local IMDB CSV...")
        df = pd.read_csv(DATA_PATH)
        # Limit to 10,000 reviews for faster interactive training
        if len(df) > 10000:
            print("Dataset is large. Sampling 10,000 reviews for faster training...")
            df = df.sample(n=10000, random_state=42).reset_index(drop=True)
        # Write to sample_reviews.csv so Data Explorer can display the loaded real dataset
        df.to_csv("data/sample_reviews.csv", index=False)
    else:
        print("Dataset empty or missing. Generating synthetic data...")
        df = generate_synthetic_data()
        df.to_csv("data/sample_reviews.csv", index=False)
    
    # Check columns
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns.")
        
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess texts
    print("Preprocessing review texts...")
    df["processed_review"] = df["review"].astype(str).apply(get_processed_text)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df["processed_review"], 
        df["sentiment"], 
        test_size=0.2, 
        random_state=42,
        stratify=df["sentiment"]
    )
    
    # Vectorizer
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Label Encoder
    print("Encoding labels...")
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    
    # Model
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train_enc)
    
    # Evaluate
    preds = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test_enc, preds)
    precision = precision_score(y_test_enc, preds, average="weighted")
    recall = recall_score(y_test_enc, preds, average="weighted")
    f1 = f1_score(y_test_enc, preds, average="weighted")
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    
    # Save files
    print("\nSaving model assets...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print("Assets saved successfully.")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "dataset_size": len(df)
    }

if __name__ == "__main__":
    train_and_save_model()
