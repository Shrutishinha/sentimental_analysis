"""
=========================================================
Prediction Utilities
=========================================================

Functions
---------
✔ Load Saved Model
✔ Load TF-IDF Vectorizer
✔ Load Label Encoder
✔ Single Prediction
✔ Batch Prediction
✔ Prediction Probability
✔ Confidence Score
=========================================================
"""

import os
import joblib
import pandas as pd

from utils.preprocess import get_processed_text

# -------------------------------------------------------
# Model Paths
# -------------------------------------------------------

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# -------------------------------------------------------
# Load Model Files
# -------------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    encoder = joblib.load(ENCODER_PATH)

except Exception:

    model = None
    vectorizer = None
    encoder = None


# -------------------------------------------------------
# Check Model
# -------------------------------------------------------

def model_loaded():

    return (
        model is not None and
        vectorizer is not None and
        encoder is not None
    )


# -------------------------------------------------------
# Vectorize Text
# -------------------------------------------------------

def vectorize(text):

    processed = get_processed_text(text)

    return vectorizer.transform([processed])


# -------------------------------------------------------
# Predict Sentiment
# -------------------------------------------------------

def predict_sentiment(text):

    if not model_loaded():
        raise Exception(
            "Model not found. Train model first."
        )

    processed = get_processed_text(text)

    vector = vectorizer.transform([processed])

    prediction = model.predict(vector)[0]

    label = encoder.inverse_transform([prediction])[0]

    confidence = None

    if hasattr(model, "predict_proba"):

        confidence = float(
            model.predict_proba(vector).max()
        )

    return {

        "text": text,

        "processed_text": processed,

        "prediction": label,

        "confidence": confidence

    }


# -------------------------------------------------------
# Predict Probability
# -------------------------------------------------------

def predict_probability(text):

    if not hasattr(model, "predict_proba"):

        return None

    processed = get_processed_text(text)

    vector = vectorizer.transform([processed])

    probs = model.predict_proba(vector)[0]

    labels = encoder.classes_

    probability = {}

    for label, value in zip(labels, probs):

        probability[label] = round(
            float(value),
            4
        )

    return probability


# -------------------------------------------------------
# Batch Prediction
# -------------------------------------------------------

def batch_predict(df, text_column="review"):

    if not model_loaded():
        raise Exception(
            "Train model first."
        )

    reviews = df[text_column].astype(str)

    processed = reviews.apply(get_processed_text)

    vectors = vectorizer.transform(processed)

    predictions = model.predict(vectors)

    labels = encoder.inverse_transform(predictions)

    result = df.copy()

    result["Prediction"] = labels

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(vectors)

        result["Confidence"] = probs.max(axis=1)

    return result


# -------------------------------------------------------
# Predict CSV File
# -------------------------------------------------------

def predict_csv(csv_path, column="review"):

    df = pd.read_csv(csv_path)

    return batch_predict(df, column)


# -------------------------------------------------------
# Predict List of Reviews
# -------------------------------------------------------

def predict_reviews(review_list):

    processed = [

        get_processed_text(text)

        for text in review_list

    ]

    vectors = vectorizer.transform(processed)

    predictions = model.predict(vectors)

    labels = encoder.inverse_transform(predictions)

    return labels.tolist()


# -------------------------------------------------------
# Get Emoji
# -------------------------------------------------------

def sentiment_emoji(label):

    if label.lower() == "positive":

        return "😊"

    elif label.lower() == "negative":

        return "😞"

    return "😐"


# -------------------------------------------------------
# Display Result
# -------------------------------------------------------

def prediction_summary(text):

    result = predict_sentiment(text)

    summary = {

        "Review": text,

        "Processed": result["processed_text"],

        "Prediction": result["prediction"],

        "Emoji": sentiment_emoji(
            result["prediction"]
        ),

        "Confidence": result["confidence"]

    }

    return summary


# -------------------------------------------------------
# Sample Test
# -------------------------------------------------------

if __name__ == "__main__":

    sample = """
    This movie was absolutely fantastic.
    The acting and music were amazing.
    """

    if model_loaded():

        output = prediction_summary(sample)

        print("=" * 50)

        for key, value in output.items():

            print(f"{key} : {value}")

        print("=" * 50)

    else:

        print("Model files not found.")
        print("Run models/train_model.py first.")