import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import logging

# -------------------
# Setup logging
# -------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO  # Change to DEBUG for detailed logs
)

# -------------------------
# Load fine-tuned summarization model
# -------------------------
logging.info("🔄 Loading fine-tuned summarizer...")
model_path = "sshleifer/distilbart-cnn-12-6"  # path where your fine-tuned BART is saved
bart_tokenizer = AutoTokenizer.from_pretrained(model_path)
bart_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# -------------------------
# Load FLAN-T5 for action generation
# -------------------------
logging.info("🔄 Loading FLAN-T5 action generator: google/flan-t5-small")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model.to(device)
flan_model.to(device)

logging.info("🔹 Models loaded and moved to device.")

# -------------------------
# Summarize feedback using fine-tuned BART
# -------------------------
def summarize_with_bart(text: str) -> str:
    logging.info(f"🔹 Summarizing text with BART (length: {len(text)} characters)...")
    inputs = bart_tokenizer(
        [text],
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    summary_ids = bart_model.generate(
        inputs["input_ids"],
        max_length=250,          # longer summary
        min_length=80,           # ensure enough detail
        num_beams=6,             # explore more beams
        length_penalty=2.0,      # discourage very short answers
        no_repeat_ngram_size=3,  # prevent repetition
        early_stopping=True
    )

    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    logging.info("🔹 Summary generated successfully.")
    return summary

# -------------------------
# Generate action with FLAN
# -------------------------
def generate_action_with_flan(summary: str, num_actions: int = 2) -> list:
    logging.info(f"🔹 Generating actions for summary: {summary[:30]}...")

    prompt = (
        f"Based on the customer feedback summary below, suggest {num_actions} clear and actionable improvements.\n\n"
        f"Summary:\n{summary}\n\n"
        f"Actions:"
    )

    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = flan_model.generate(
        **inputs,
        max_length=200,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    actions_text = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
    actions = [a.strip("-• ") for a in actions_text.split("\n") if a.strip()]
    logging.info("🔹 Actions generated successfully.")
    return actions

# -------------------------
# Main logic for grouped suggestions
# -------------------------
def generate_bart_flan_suggestions(df: pd.DataFrame, sentiment_filter="Negative", top_n=5):
    logging.info(f"🔹 Filtering feedback with sentiment: {sentiment_filter}")
    df_filtered = df[df["sentiment"].str.lower() == sentiment_filter.lower()]

    if df_filtered.empty:
        logging.warning(f"⚠️ No feedback found for sentiment: {sentiment_filter}")
        return []

    results = []
    theme_groups = df_filtered.groupby("theme")
    sorted_themes = sorted(theme_groups, key=lambda x: len(x[1]), reverse=True)[:top_n]

    for theme, group in sorted_themes:
        combined_text = " ".join(group["Text"].dropna().tolist())[:4000]  # allow more text
        logging.info(f"🔹 Generating summary and actions for theme: {theme} with {len(group)} records.")

        bart_summary = summarize_with_bart(combined_text)
        flan_actions = generate_action_with_flan(bart_summary, num_actions=3)  # multiple actions

        results.append((theme, len(group), bart_summary, flan_actions))

    logging.info("🔹 Suggestion generation completed.")
    return results

# -------------------------
# Test Run
# -------------------------
if __name__ == "__main__":
    logging.info("🔹 Loading data...")
    try:
        df = pd.read_csv("data/processed/final_data.csv")  # Ensure this has 'Text', 'sentiment', 'theme' columns
        logging.info(f"→ Loaded {len(df)} records from final_data.csv.")
    except Exception as e:
        logging.error(f"❌ Error loading data: {e}")
        exit(1)

    logging.info("🔹 Starting suggestion generation...")
    suggestions = generate_bart_flan_suggestions(df)

    for theme, count, summary, actions in suggestions:
        logging.info(f"\n📌 {theme} ({count} mentions)\n📝 Summary: {summary}\n")
        for i, action in enumerate(actions, 1):
            logging.info(f"👉 Action {i}: {action}")
