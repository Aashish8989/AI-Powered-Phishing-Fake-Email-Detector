import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- Configuration ---
MODEL_PATH = "./model"

# --- Initialize Flask App ---
app = Flask(__name__)
app.debug = True # Enable debug mode for easier troubleshooting

# --- Global Variables for Model and Tokenizer ---
tokenizer = None
model = None
device = None

# --- Load Model Function ---
def load_model_and_tokenizer():
    global tokenizer, model, device
    print(f"Loading model from {MODEL_PATH}...")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, "pytorch_model.bin")):
        print(f"FATAL ERROR: Model files not found in {MODEL_PATH}")
        print("Please run 'python train.py' first to train and save the model.")
        # In a real app, you might want to exit or handle this differently
        # For Flask, we'll let it run but predictions will fail.
        return False

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval() # Set the model to evaluation mode
        print(f"Device set to use {device}")
        print("Model loaded successfully. Server is ready.")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives text, predicts using the model, and returns JSON."""
    global tokenizer, model, device

    if not model or not tokenizer:
        return jsonify({'error': 'Model not loaded correctly. Check server logs.'}), 500

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided in request.'}), 400

        text_to_predict = data['text']

        # --- THIS IS THE FIX ---
        # Explicitly tokenize and truncate the input text
        # to match the model's maximum input size (512 tokens).
        print(f"Received text for prediction (length: {len(text_to_predict)}). Tokenizing...")
        inputs = tokenizer(
            text_to_predict,
            return_tensors="pt", # Return PyTorch tensors
            truncation=True,     # TRUNCATE long text
            padding=True,        # Pad shorter text (though usually not needed for single prediction)
            max_length=512       # Max length the model accepts
        ).to(device) # Send tensors to the correct device (CPU or GPU)
        # --- END OF FIX ---

        print("Making prediction...")
        # Make prediction directly using the model (bypassing pipeline for more control)
        with torch.no_grad(): # No need to calculate gradients for prediction
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the prediction probabilities and the final prediction label
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities).item()

        # Get the confidence score for the predicted class
        confidence_score = probabilities[0][predicted_class_id].item()

        # Map class ID to label name (0=Legitimate, 1=Phishing)
        # Assumes model.config.id2label exists and is correct, otherwise define manually
        if hasattr(model.config, 'id2label'):
             prediction_label = model.config.id2label[predicted_class_id]
        else:
            # Fallback if id2label is missing (adjust if your labels differ)
             prediction_label = "Legitimate" if predicted_class_id == 0 else "Phishing"
        print(f"Prediction: {prediction_label}, Confidence: {confidence_score:.4f}")

        # Return the result as JSON
        return jsonify({
            'prediction': prediction_label,
            'confidence': confidence_score
        })

    except Exception as e:
        print(f"!!! Error during prediction: {e}")
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during prediction. Check server logs.'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # Load the model when the script starts
    load_model_and_tokenizer()
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000) # Makes it accessible on your local network too

