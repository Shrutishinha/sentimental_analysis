import streamlit as st
import requests

st.title("🤖 Model Training")

st.write("Train a sentiment model (TF-IDF + model comparison) on your IMDB reviews or a synthetic dataset.")

API_URL = "http://localhost:8001/train"  # matches the port your FastAPI server actually runs on

if st.button("Train Model"):
    with st.spinner("Training model in progress..."):
        try:
            response = requests.post(API_URL, timeout=300)
            response.raise_for_status()
            res = response.json()

            if res.get("status") == "success":
                st.success("✅ Model training completed successfully!")

                metrics = res["metrics"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
                col2.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
                col3.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
                col4.metric("F1-Score", f"{metrics['f1_score'] * 100:.1f}%")

                st.write(f"Model saved to disk! Loaded dataset size: **{metrics['dataset_size']}** reviews.")

                if metrics['accuracy'] >= 0.999:
                    st.warning(
                        "⚠️ Accuracy is suspiciously close to 100%. "
                        "This usually means the training data has near-duplicate rows "
                        "leaking between train/test splits (common with small synthetic datasets). "
                        "Consider using the real IMDB dataset or diversifying synthetic samples."
                    )
            else:
                st.error(f"Training failed: {res.get('message', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to backend at {API_URL}. Is the FastAPI server running?")
        except requests.exceptions.Timeout:
            st.error("Training request timed out.")
        except Exception as e:
            st.error(f"Error communicating with backend: {e}")