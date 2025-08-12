import os
import pandas as pd
from transformers import pipeline

INPUT_FILE = "data/processed/cleaned_data.csv"
OUTPUT_FILE = "data/training/theme_labeled_data.csv"

# Candidate labels (same as your planned taxonomy)
CANDIDATE_LABELS = [
    "Customer Service", "Pricing", "Hardware", "Software", "Delivery", "Billing",
    "Performance", "Product Quality", "Ordering Issues", "Account Issues",
    "Warranty/Repair", "UI/UX Feedback", "Installation", "Communication",
    "Advertising", "Compatibility", "Other/General"
]

def auto_tag():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print(f"🔄 Loading zero-shot classifier for auto-tagging...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    df = pd.read_csv(INPUT_FILE)
    theme_labels = []

    for text in df['clean_text']:
        if not text or text.strip() == "":
            theme_labels.append("Other/General")
            continue
        try:
            result = classifier(text, CANDIDATE_LABELS, multi_label=False)
            theme_labels.append(result['labels'][0])
        except Exception as e:
            print(f"⚠️ Error tagging: {e}")
            theme_labels.append("Other/General")

    df['theme'] = theme_labels
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Auto-tagged data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    auto_tag()
