# scripts/generate_ai_suggestions.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

# -------------------------
# Load BART for summarization
# -------------------------
print("🔄 Loading BART summarization model: facebook/bart-large-cnn")
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# -------------------------
# Load FLAN-T5 for action generation
# -------------------------
print("🔄 Loading FLAN-T5 action generator: google/flan-t5-small")
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bart_model.to(device)
flan_model.to(device)

def summarize_with_bart(text: str) -> str:
    """Summarize feedback text using BART."""
    inputs = bart_tokenizer([text], max_length=1024, truncation=True, return_tensors="pt").to(device)
    summary_ids = bart_model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def generate_action_with_flan(summary: str) -> str:
    """Generate actionable suggestion from summary using FLAN-T5."""
    prompt = f"Based on the customer feedback summary below, provide one clear, actionable improvement suggestion.\n\nSummary: {summary}\n\nAction:"
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = flan_model.generate(**inputs, max_length=100)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_bart_flan_suggestions(df: pd.DataFrame, sentiment_filter="Negative", top_n=5):
    """
    Generates actionable suggestions:
    1) Groups negative feedback by theme
    2) Summarizes feedback (BART)
    3) Generates an action suggestion (FLAN-T5)
    """
    results = []
    df_filtered = df[df["sentiment"] == sentiment_filter]

    if df_filtered.empty:
        print(f"⚠️ No records found for sentiment: {sentiment_filter}")
        return results

    theme_groups = df_filtered.groupby("theme")
    sorted_themes = sorted(theme_groups, key=lambda x: len(x[1]), reverse=True)[:top_n]

    for theme, group in sorted_themes:
        combined_text = " ".join(group["Text"].tolist())
        bart_summary = summarize_with_bart(combined_text)
        flan_action = generate_action_with_flan(bart_summary)
        results.append((theme, len(group), flan_action))

    return results

if __name__ == "__main__":
    # Quick test
    test_data = pd.DataFrame({
        "Text": [
            "Customer service took too long to respond and was unhelpful.",
            "Waited 40 minutes for a callback, still no resolution.",
            "Agents gave conflicting answers, problem unresolved."
        ],
        "sentiment": ["Negative", "Negative", "Negative"],
        "theme": ["Customer Service", "Customer Service", "Customer Service"]
    })

    suggestions = generate_bart_flan_suggestions(test_data)
    for theme, count, action in suggestions:
        print(f"\n📌 {theme} ({count} mentions):\n👉 {action}")
