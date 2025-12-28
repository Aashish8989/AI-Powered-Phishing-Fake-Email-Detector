AI Phishing & Fake Email Detector 🛡️

An end-to-end AI-powered application designed to detect phishing attempts and malicious intent in emails and documents using Natural Language Processing (NLP).

🚀 Project Description

Traditional phishing filters often rely on blacklisted URLs or sender addresses, which can be easily bypassed by attackers using fresh domains. This project takes a behavioral approach using Large Language Models (LLMs).

By analyzing the semantic meaning of the text, the AI identifies linguistic "red flags" such as:

Induced Urgency: Pressuring the user to act quickly to avoid negative consequences.

Financial Coercion: Unusual requests for wire transfers, gift cards, or crypto payments.

Authority Impersonation: Mimicking corporate or government tone to gain trust.

Data Harvesting: Contextual patterns that suggest a request for sensitive login credentials.

The system provides a probability score, allowing users to understand the risk level before interacting with a suspicious message.

🧠 AI/LLM Features

Model: Fine-tuned distilbert-base-uncased using the Hugging Face Transformers library.

Task: Sequence Classification (Phishing vs. Safe).

Inference: Integrated PyTorch backend for real-time text analysis.

Data Pipeline: Custom tokenization logic with handling for padding and truncation.

🏗️ Architecture

Frontend: A responsive UI for users to paste email body text or upload PDF documents.

Preprocessing: Text is cleaned, stripped of HTML/unnecessary metadata, and converted into tensors.

Inference Engine: The DistilBERT model processes the input through its 6 transformer layers to produce a classification logit.

Post-processing: A Softmax function converts logits into a human-readable confidence percentage.

🏋️ Training the Model

The model is fine-tuned on a labeled dataset of phishing and legitimate emails.

1. Dataset Preparation

Data is formatted into a CSV with two columns: text (email content) and label (0 for Safe, 1 for Phish).

We use the datasets library to load and split the data:

from datasets import load_dataset
dataset = load_dataset('csv', data_files='phishing_data.csv')
dataset = dataset['train'].train_test_split(test_size=0.2)


2. Tokenization

Using the DistilBERT tokenizer to convert raw text into input IDs and attention masks:

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


3. Fine-Tuning

The model is trained using the Hugging Face Trainer API:

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()


🛠️ Technical Stack

Frontend: HTML5, Tailwind CSS, JavaScript.

Backend: Flask (Python).

AI/ML: PyTorch, Hugging Face Transformers.

⚙️ Installation & Setup

Clone the repository:

git clone [https://github.com/your-username/phishing-detector.git](https://github.com/your-username/phishing-detector.git)
cd phishing-detector


Install Dependencies:

pip install -r requirements.txt


Run the Application:

python app.py


Developed by [Your N
