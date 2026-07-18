"""
=========================================================
FastAPI Interactive Backend for Sentiment Analysis Studio
=========================================================
This file implements the complete REST API backend. It provides endpoints for
single prediction, batch prediction, text preprocessing, word cloud frequency 
calculation, model evaluation, and training.

Run the server with:
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
"""

import io
import os
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing NLP utilities
from utils.preprocess import analyze_text, clean_text, tokenize, remove_stopwords
from utils.predict import (
    model_loaded,
    predict_sentiment,
    predict_probability,
    sentiment_emoji,
    prediction_summary,
    batch_predict
)
import utils.predict as predict_utils
from models.train_model import train_and_save_model

# -------------------------------------------------------
# Initialize FastAPI App
# -------------------------------------------------------
app = FastAPI(
    title="NLP Sentiment Analysis API",
    description=(
        "Interactive REST API backend for text preprocessing, "
        "sentiment prediction, model training, and performance metrics."
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc"     # ReDoc UI
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Pydantic Schemas for Requests and Responses
# -------------------------------------------------------
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    reviews: List[str]

class PredictionResponse(BaseModel):
    text: str
    processed_text: str
    prediction: str
    confidence: Optional[float]
    emoji: str
    probabilities: Optional[Dict[str, float]]

# -------------------------------------------------------
# Model Reload Helper
# -------------------------------------------------------
def reload_model_assets() -> bool:
    """Reloads the joblib model assets dynamically in prediction utilities."""
    try:
        if os.path.exists(predict_utils.MODEL_PATH) and os.path.exists(predict_utils.VECTORIZER_PATH) and os.path.exists(predict_utils.ENCODER_PATH):
            predict_utils.model = joblib.load(predict_utils.MODEL_PATH)
            predict_utils.vectorizer = joblib.load(predict_utils.VECTORIZER_PATH)
            predict_utils.encoder = joblib.load(predict_utils.ENCODER_PATH)
            return True
    except Exception as e:
        print(f"Error reloading model assets: {e}")
    return False

# Reload models at startup
reload_model_assets()

# -------------------------------------------------------
# API Endpoints
# -------------------------------------------------------

@app.get("/", tags=["System Status"])
def health_check():
    """Checks backend status and checks if the model is currently trained and loaded."""
    loaded = model_loaded()
    return {
        "status": "healthy",
        "app_name": "NLP Sentiment Analysis Studio Backend",
        "model_loaded": loaded,
        "message": "Model is loaded and ready." if loaded else "Model not found. Please train model using /train endpoint."
    }

@app.post("/analyze", tags=["Text Preprocessing"])
def analyze_text_endpoint(input_data: TextInput):
    """
    Performs full text preprocessing and linguistic analysis:
    - Lowercasing, HTML/URL removal, cleaning.
    - Tokenization.
    - Stopword removal.
    - Stemming & Lemmatization.
    - POS Tagging.
    - Named Entity Recognition (NER).
    - Basic metadata statistics (word/sentence count, character counts).
    """
    try:
        analysis = analyze_text(input_data.text)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse, tags=["Sentiment Prediction"])
def predict_sentiment_endpoint(input_data: TextInput):
    """
    Predicts sentiment for a single review.
    Returns:
    - Cleaned and tokenized review text.
    - Predicted label (Positive/Negative).
    - Prediction confidence score.
    - Sentiment emoji.
    - Raw class probability values.
    """
    if not model_loaded():
        # Attempt reload in case files were generated outside the API runtime
        if not reload_model_assets():
            raise HTTPException(
                status_code=400,
                detail="Sentiment model files not loaded. Please run model training using the '/train' endpoint first."
            )

    try:
        summary = prediction_summary(input_data.text)
        probabilities = predict_probability(input_data.text)
        
        return PredictionResponse(
            text=summary["Review"],
            processed_text=summary["Processed"],
            prediction=summary["Prediction"],
            confidence=summary["Confidence"],
            emoji=summary["Emoji"],
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", tags=["Sentiment Prediction"])
def predict_batch_endpoint(input_data: BatchInput):
    """
    Predicts sentiment for a list of reviews.
    Returns:
    - List of objects containing text, prediction, confidence, and emoji.
    """
    if not model_loaded():
        if not reload_model_assets():
            raise HTTPException(status_code=400, detail="Sentiment model files not loaded. Train first.")

    try:
        results = []
        for review in input_data.reviews:
            summary = prediction_summary(review)
            results.append({
                "review": review,
                "prediction": summary["Prediction"],
                "confidence": summary["Confidence"],
                "emoji": summary["Emoji"]
            })
        return {"predictions": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/file", tags=["Sentiment Prediction"])
async def predict_file_endpoint(file: UploadFile = File(...), text_column: str = "review"):
    """
    Uploads a CSV file containing reviews and predicts sentiments.
    Returns the table containing predictions, confidence scores, and emojis.
    """
    if not model_loaded():
        if not reload_model_assets():
            raise HTTPException(status_code=400, detail="Sentiment model files not loaded. Train first.")

    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if text_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV file must contain the specified column '{text_column}'"
            )
            
        result_df = batch_predict(df, text_column=text_column)
        
        # Add sentiment emojis
        result_df["Emoji"] = result_df["Prediction"].apply(sentiment_emoji)
        
        # Convert NaN values to None for clean JSON response
        result_df = result_df.replace({float('nan'): None})
        
        return result_df.to_dict(orient="records")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV Prediction failed: {str(e)}")

@app.post("/train", tags=["Model Training"])
def train_model_endpoint():
    """
    Trains a Logistic Regression + TF-IDF model on local data.
    If the IMDB dataset is empty, it generates a synthetic balanced dataset to run training instantly.
    Saves new pickle files and reloads models in-memory.
    """
    try:
        metrics = train_and_save_model()
        reload_success = reload_model_assets()
        
        return {
            "status": "success",
            "message": "Model trained and loaded successfully.",
            "metrics": metrics,
            "reloaded_in_memory": reload_success
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.get("/metrics", tags=["Model Training"])
def get_metrics_endpoint():
    """
    Retrieves evaluation details for the current model.
    Checks if a model is loaded, and returns cross-validation estimation parameters if available.
    """
    if not model_loaded():
        return {
            "model_loaded": False,
            "message": "No model loaded. Model metrics are not available."
        }
        
    return {
        "model_loaded": True,
        "algorithm": "Logistic Regression Classifier",
        "vectorizer": "TF-IDF Vectorizer (ngram_range=(1,2), max_features=5000)",
        "features": ["Sentiment Classification (positive/negative)"],
        "parameters": {
            "C": 1.0,
            "max_iter": 1000,
            "penalty": "l2"
        }
    }

@app.post("/wordcloud", tags=["Text Preprocessing"])
def wordcloud_frequencies(input_data: TextInput):
    """
    Tokenizes text, cleans it, removes punctuation & stopwords,
    and returns sorted raw word frequencies for generating word clouds.
    """
    try:
        cleaned = clean_text(input_data.text)
        tokens = tokenize(cleaned)
        filtered_tokens = remove_stopwords(tokens)
        
        # Count frequencies
        freq = {}
        for token in filtered_tokens:
            if len(token) > 1: # Ignore single letter noises
                freq[token] = freq.get(token, 0) + 1
                
        # Sort by frequency descending
        sorted_freq = dict(sorted(freq.items(), key=lambda item: item[1], reverse=True))
        return {"frequencies": sorted_freq, "unique_words": len(sorted_freq)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word frequency calculation failed: {str(e)}")
