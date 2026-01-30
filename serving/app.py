import os
import logging
from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import torch
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
BUCKET_NAME = os.getenv("BUCKET_NAME")
MODEL_GCS_PATH = os.getenv("MODEL_GCS_PATH", "model_output")
LOCAL_MODEL_DIR = "./trained_model"

def download_model(bucket_name, source_blob_prefix, destination_dir):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=source_blob_prefix)
    found_files = False
    for blob in blobs:
        found_files = True
        # Remove the prefix from the path to save locally
        if blob.name.startswith(source_blob_prefix):
             relative_path = blob.name[len(source_blob_prefix):].lstrip('/')
        else:
            relative_path = blob.name
            
        local_path = os.path.join(destination_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded {blob.name} to {local_path}")
    
    if not found_files:
        logger.error(f"No files found in gs://{bucket_name}/{source_blob_prefix}")
        raise FileNotFoundError(f"Model not found in GCS bucket {bucket_name} at {source_blob_prefix}")

# Initialize model and tokenizer
model = None
tokenizer = None

def load_model_artifacts():
    global model, tokenizer
    try:
        if BUCKET_NAME:
            logger.info(f"Downloading model from gs://{BUCKET_NAME}/{MODEL_GCS_PATH}...")
            download_model(BUCKET_NAME, MODEL_GCS_PATH, LOCAL_MODEL_DIR)
            model_path = LOCAL_MODEL_DIR
        else:
            logger.info("BUCKET_NAME not set, expecting model in 'trained_model' directory or using default pre-trained.")
            # Fallback to pre-trained if local dir doesn't exist
            if os.path.exists(LOCAL_MODEL_DIR) and os.listdir(LOCAL_MODEL_DIR):
                model_path = LOCAL_MODEL_DIR
            else:
                 model_path = "distilbert-base-uncased"
                 logger.warning(f"Local model not found. Using default: {model_path}")

        logger.info(f"Loading model from {model_path}...")
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval() # Set to evaluation mode
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # We don't exit here so the container keeps running and might be debuggable, 
        # but readiness check should fail.

# Load on startup
load_model_artifacts()

@app.route("/health", methods=["GET"])
def health():
    if model is not None and tokenizer is not None:
        return jsonify({"status": "healthy"}), 200
    return jsonify({"status": "unhealthy", "reason": "Model not loaded"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid input. 'text' field required."}), 400
        
        text = data.get("text")
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()

        # Map class ID to sentiment (0: negative, 1: positive)
        # Verify label mapping with your specific training data
        sentiment = "positive" if predicted_class_id == 1 else "negative"

        return jsonify({"sentiment": sentiment, "score": predicted_class_id})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
