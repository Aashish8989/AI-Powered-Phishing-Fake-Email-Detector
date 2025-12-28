import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

# --- Configuration ---
MODEL_PATH = "./model"

# --- 1. Load the Fine-Tuned Model & Tokenizer ---

# Check if model directory exists
if not os.path.exists(MODEL_PATH) or not os.listdir(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    print("Please run 'python train.py' first to train and save the model.")
    exit()

print(f"Loading model from {MODEL_PATH}...")

# Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # --- 2. Create a Text-Classification Pipeline ---
    # The pipeline handles all the tokenization and inference steps for us.
    # We specify device=0 for CUDA, or device=-1 for CPU.
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1
    )

    print("Model loaded successfully. Ready for predictions.")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Ensure the model was trained correctly and files exist in the 'model' directory.")
    exit()


# --- 3. Prediction Loop ---

def predict_email(text):
    """Classifies a single piece of text."""
    # The model outputs a list of dictionaries
    result = classifier(text)[0]
    
    # The model was trained with 0=Legit, 1=Phishing.
    # The pipeline labels are "LABEL_0" and "LABEL_1".
    label = result['label']
    score = result['score']
    
    if label == "LABEL_1":
        return "Phishing", score
    else:
        return "Legitimate", score

# Loop to get user input
while True:
    try:
        email_text = input("\nEnter email text (or 'quit' to exit): \n> ")
    except EOFError:
        print("\nExiting.")
        break
        
    if email_text.lower() == 'quit':
        print("Exiting.")
        break
    
    if not email_text.strip():
        print("Please enter some text.")
        continue
        
    prediction, confidence = predict_email(email_text)
    
    print(f"\nPrediction: {prediction}, Confidence: {confidence:.4f}")