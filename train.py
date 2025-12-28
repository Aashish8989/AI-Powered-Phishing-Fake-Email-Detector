import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os # Import os to create the model path

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased"
DATA_FILE = "./data/emails.csv"
MODEL_PATH = "./model"
BATCH_SIZE = 8 # Batch size for both training and evaluation

# --- 1. Load and Check Dataset ---
try:
    # Try reading with default utf-8 encoding
    df = pd.read_csv(DATA_FILE)
except UnicodeDecodeError:
    try:
        # If utf-8 fails, try 'latin1' (common for older datasets)
        print("UTF-8 failed, trying 'latin1' encoding...")
        df = pd.read_csv(DATA_FILE, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        print("Could not read the CSV file. Please ensure it's saved as UTF-8 or Latin1.")
        exit()
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_FILE}")
    print("Please make sure 'data/emails.csv' exists.")
    exit()

print("--- DATA CHECKER ---")
print("Running checks on your new 'data/emails.csv' file...")

# 1. Check if the file is empty
if df.empty:
    print("FATAL ERROR: Your 'data/emails.csv' file is empty.")
    exit()

# 2. Check the column names
print(f"\nYour columns are: {list(df.columns)}")
if 'text' not in df.columns or 'label' not in df.columns:
    print("\nFATAL ERROR: Columns not named correctly.")
    print("Your CSV file *must* have one column named 'text' and one column named 'label'.")
    print("Please rename your columns and re-run.")
    exit()

# 3. Check for missing values in 'text'
if df['text'].isnull().any():
    print("\nWARNING: Your 'text' column has missing values. Dropping them...")
    df = df.dropna(subset=['text'])

# 4. Check the labels (for 'ham'/'spam' or '0'/'1')
print(f"\nYour 'label' column looks like this:\n{df['label'].head()}")

if df['label'].dtype == 'object':
    print("\nFATAL ERROR: Your 'label' column is text (e.g., 'ham' or 'spam').")
    print("The script needs numbers: '0' for legitimate and '1' for phishing.")
    print("Please open your CSV and use 'Find and Replace' to fix this:")
    print(" - Replace all 'ham' (or 'legitimate') with '0'")
    print(" - Replace all 'spam' (or 'phishing') with '1'")
    exit()
    
# 5. Check the balance of your data
print("\nChecking data balance (0 = Legitimate, 1 = Phishing):")
print(df['label'].value_counts())
print("--- DATA CHECKER COMPLETE ---")

# --- 2. Prepare Dataset (Continued) ---
print("\nData checks passed. Starting to process data...")

# Ensure labels are integers (0 or 1)
df['label'] = df['label'].astype(int)

# Drop any rows with missing data
df = df.dropna(subset=['text', 'label'])

# Convert text to string (in case some are numbers)
df['text'] = df['text'].astype(str)

# Split the dataset
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# --- 3. Tokenization ---
# Load the tokenizer for our model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # Tokenizes text. `padding="max_length"` and `truncation=True`
    # ensure all inputs are the same size (512 tokens).
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Apply tokenization to the datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# --- 4. Format Dataset for Training ---
# Rename the 'label' column to 'labels' which the model expects
# THIS IS THE FIX: Renaming on the 'tokenized' datasets
tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
tokenized_val_dataset = tokenized_val_dataset.rename_column("label", "labels")

# Set the format to torch and specify only the columns the model needs.
tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# --- 5. Load Model ---
# We have 2 labels: 0 (Legitimate) and 1 (Phishing)
# Check if CUDA (GPU) is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)


# --- 6. Configure Training (BARE-BONES: Compatible with all old versions) ---
# We are removing ALL extra arguments that your old library does not recognize.
training_args = TrainingArguments(
    output_dir=f"{MODEL_PATH}/checkpoints", 
    num_train_epochs=3,           # Train for 3 epochs (faster, safer for large data)
    per_device_train_batch_size=BATCH_SIZE,  
    per_device_eval_batch_size=BATCH_SIZE,   
    weight_decay=0.01,            # This "penalty" prevents overfitting
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    # We remove eval_dataset and compute_metrics from here for compatibility
)

# --- 7. Train the Model ---
print("Starting model training...")
trainer.train()
print("Training complete.")

# --- 8. Evaluate the Model ---
print("Evaluating final model on validation set...")
# We evaluate manually at the end
eval_results = trainer.evaluate(eval_dataset=tokenized_val_dataset)
print(f"Evaluation Results: {eval_results}")


# --- 9. Save the Final Model (Manual Way to avoid Antivirus/safetensors error) ---
print(f"Saving model to {MODEL_PATH}...")

# Create the model directory if it doesn't exist
os.makedirs(MODEL_PATH, exist_ok=True)

# Define the file path for the model weights
model_save_path = os.path.join(MODEL_PATH, "pytorch_model.bin")

# Save the model's state (weights) using torch.save
# This bypasses safetensors and the "os error 1224"
torch.save(trainer.model.state_dict(), model_save_path)

# Save the model's configuration file (config.json)
trainer.model.config.save_pretrained(MODEL_PATH)

# Save the tokenizer
tokenizer.save_pretrained(MODEL_PATH)

print(f"Model and tokenizer saved to {MODEL_PATH}")
print("All steps complete!")

