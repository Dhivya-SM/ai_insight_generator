import pandas as pd
import re

def clean_text(text):
    # Remove mentions, URLs, hashtags, special characters, etc.
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)            # Remove mentions
    text = re.sub(r"http\S+|www\S+", "", text)            # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)               # Remove special chars and numbers
    text = text.lower().strip()
    return text

def preprocess_dataset(input_path, output_path):
    df = pd.read_csv(input_path)
    
    if 'Text' not in df.columns:
        print("❌ Column 'Text' not found in CSV!")
        print("📌 Available columns:", df.columns.tolist())
        return

    df['clean_text'] = df['Text'].apply(clean_text)
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    input_path = "data/raw/dell_tweets.csv"
    output_path = "data/processed/dell_sentiment.csv"
    preprocess_dataset(input_path, output_path)
