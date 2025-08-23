import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
import torch.nn.functional as F
import logging

# -------------------
# Setup logging
# -------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# -------------------
# Theme labels setup
# -------------------
theme_labels = [
    "Customer Service", "Pricing", "Hardware", "Software", "Delivery", "Billing",
    "Performance", "Product Quality", "Ordering Issues", "Account Issues",
    "Warranty/Repair", "UI/UX Feedback", "Installation", "Communication",
    "Advertising", "Compatibility", "Other/General"
]
id2label = {i: label for i, label in enumerate(theme_labels)}
label2id = {label: i for i, label in enumerate(theme_labels)}

# -------------------
# Load theme model with correct label config
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "distilbert-base-uncased"  # Use a custom path if fine-tuned model is available

logging.info(f"🔹 Loading theme model from: {model_path}")

# Load config with correct number of labels
config = AutoConfig.from_pretrained(
    model_path,
    num_labels=len(theme_labels),
    id2label=id2label,
    label2id=label2id
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config).to(device)
model.eval()

logging.info("🔹 Theme model loaded and ready for inference.")

# -------------------
# Batch classification function
# -------------------
def classify_batch(batch_texts, model, tokenizer):
    logging.debug(f"Classifying batch of {len(batch_texts)} texts.")
    encodings = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encodings)
        probs = F.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1).cpu().numpy()

    return [id2label[p] for p in preds]

# -------------------
# Theme classification for entire DataFrame
# -------------------
def classify_themes(df, batch_size=64, limit=5000, random_sample=False):
    """Classify themes in cleaned text using a transformer model."""
    if limit is not None:
        df = df.sample(n=limit, random_state=42) if random_sample else df.head(limit)
        df = df.reset_index(drop=True)

    logging.info(f"🔹 Starting theme classification for {len(df)} records.")

    themes = []
    for i in range(0, len(df), batch_size):
        batch = df["clean_text"][i:i + batch_size].tolist()
        encodings = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encodings)
            probs = F.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().numpy()
        batch_themes = [id2label[p] for p in preds]
        themes.extend(batch_themes)

        if i % (batch_size * 10) == 0:
            logging.info(f"Processed {i}/{len(df)} records...")

    df["theme"] = themes
    logging.info(f"✅ Theme classification completed for {len(df)} records.")

    # Optional: show distribution summary
    logging.info(f"📊 Theme distribution:\n{df['theme'].value_counts()}")

    return df
