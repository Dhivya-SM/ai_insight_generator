import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import numpy as np

TRAIN_FILE = "data/training/theme_labeled_data.csv"
MODEL_SAVE_PATH = "models/theme_classifier"
BASE_MODEL = "distilbert-base-uncased"

def train_theme_classifier():
    if not os.path.exists(TRAIN_FILE):
        raise FileNotFoundError(f"❌ Training file not found: {TRAIN_FILE}")

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    print(f"🔄 Loading training data from {TRAIN_FILE}...")
    df = pd.read_csv(TRAIN_FILE)

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['theme'])

    # Save label mapping
    label_map = {i: label for i, label in enumerate(le.classes_)}
    with open(os.path.join(MODEL_SAVE_PATH, "label_map.txt"), "w") as f:
        for k, v in label_map.items():
            f.write(f"{k}\t{v}\n")

    dataset = Dataset.from_pandas(df[['clean_text', 'label']])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    def tokenize(batch):
        return tokenizer(batch['clean_text'], padding=True, truncation=True)

    dataset = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=len(le.classes_))

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset
    )

    print("🚀 Training theme classifier...")
    trainer.train()
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_theme_classifier()
