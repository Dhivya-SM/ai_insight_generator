import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
)
import torch

# -------------------
# CONFIG
# -------------------
os.makedirs("models", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# HELPER FUNCTIONS
# -------------------
def load_csv_dataset(file_path, text_col, label_col=None, label_list=None):
    """Load CSV into HuggingFace Dataset, handle labels if needed."""
    df = pd.read_csv(file_path)
    if label_col and label_list:
        label2id = {label: i for i, label in enumerate(label_list)}
        df[label_col] = df[label_col].map(label2id)
    return Dataset.from_pandas(df)

def tokenize_classification(examples, tokenizer, text_col):
    return tokenizer(examples[text_col], padding="max_length", truncation=True)

def tokenize_summarization(examples, tokenizer, text_col, summary_col):
    inputs = tokenizer(examples[text_col], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples[summary_col], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

# -------------------
# 1. Sentiment Fine-tuning (DistilBERT)
# -------------------
print("🚀 Fine-tuning Sentiment Model...")
sentiment_labels = ["positive", "negative"]
sentiment_dataset = load_csv_dataset("data/training/sentiment_train.csv", "text", "label", sentiment_labels)

tokenizer_sent = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_sent = sentiment_dataset.map(lambda x: tokenize_classification(x, tokenizer_sent, "text"), batched=True)

model_sent = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(sentiment_labels)
).to(device)

training_args_sent = TrainingArguments(
    output_dir="./models/sentiment_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch", 
)


trainer_sent = Trainer(
    model=model_sent,
    args=training_args_sent,
    train_dataset=tokenized_sent,
    eval_dataset=tokenized_sent,
    tokenizer=tokenizer_sent
)

trainer_sent.train()
model_sent.save_pretrained("models/sentiment_model")
tokenizer_sent.save_pretrained("models/sentiment_model")
print("✅ Sentiment model saved!")

# -------------------
# 2. Theme Classification Fine-tuning (DistilBERT)
# -------------------
print("🚀 Fine-tuning Theme Classification Model...")
theme_labels = sorted(pd.read_csv("data/training/theme_train.csv")["label"].unique().tolist())
theme_dataset = load_csv_dataset("data/training/theme_train.csv", "text", "label", theme_labels)

tokenizer_theme = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_theme = theme_dataset.map(lambda x: tokenize_classification(x, tokenizer_theme, "text"), batched=True)

model_theme = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(theme_labels)
).to(device)

training_args_theme = TrainingArguments(
    output_dir="models/theme_model",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10
)


trainer_theme = Trainer(
    model=model_theme,
    args=training_args_theme,
    train_dataset=tokenized_theme,
    eval_dataset=tokenized_theme,
    tokenizer=tokenizer_theme
)

trainer_theme.train()
model_theme.save_pretrained("models/theme_model")
tokenizer_theme.save_pretrained("models/theme_model")
print("✅ Theme classification model saved!")

# -------------------
# 3. Summarization Fine-tuning (BART)
# -------------------
print("🚀 Fine-tuning Summarization Model...")
summ_dataset = Dataset.from_pandas(pd.read_csv("data/training/summarization_train.csv"))
tokenizer_summ = AutoTokenizer.from_pretrained("facebook/bart-base")

tokenized_summ = summ_dataset.map(
    lambda x: tokenize_summarization(x, tokenizer_summ, "text", "summary"), batched=True
)

model_summ = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)

training_args_summ = TrainingArguments(
    output_dir="models/summarization_model",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10
)


trainer_summ = Trainer(
    model=model_summ,
    args=training_args_summ,
    train_dataset=tokenized_summ,
    eval_dataset=tokenized_summ,
    tokenizer=tokenizer_summ
)

trainer_summ.train()
model_summ.save_pretrained("models/summarization_model")
tokenizer_summ.save_pretrained("models/summarization_model")
print("✅ Summarization model saved!")

print("🎯 All models fine-tuned and saved successfully!")
