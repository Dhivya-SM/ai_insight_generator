import os
import pandas as pd
import random

# Paths
FINAL_PATH = "data/processed/final_data.csv"
EVAL_DIR = "data/eval"
HUMAN_APPROVAL_PATH = os.path.join(EVAL_DIR, "human_approval.csv")
HUMAN_READABILITY_PATH = os.path.join(EVAL_DIR, "human_readability.csv")
GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "ground_truth.csv")

# Create eval directory if it doesn't exist
os.makedirs(EVAL_DIR, exist_ok=True)

# Load final data
df = pd.read_csv(FINAL_PATH)

# Use random sampling for better diversity, increase sample size a bit
sample_size = 100
sample_df = df.sample(n=sample_size, random_state=42).copy()

# ---- Human Approval (0/1) ----
# Simulate useful suggestions with 70% chance of approval for demo
approval_df = pd.DataFrame({
    "suggestion": sample_df["suggestion"] if "suggestion" in sample_df.columns else [f"Suggestion {i}" for i in range(len(sample_df))],
    "is_useful": [1 if random.random() < 0.7 else 0 for _ in range(len(sample_df))]
})
approval_df.to_csv(HUMAN_APPROVAL_PATH, index=False)

# ---- Human Readability (1–5 rating) ----
# Higher probability of mid-to-high readability (3-5)
readability_df = pd.DataFrame({
    "suggestion": approval_df["suggestion"],
    "rating": [random.choices([1, 2, 3, 4, 5], weights=[5, 10, 30, 35, 20])[0] for _ in range(len(sample_df))]
})
readability_df.to_csv(HUMAN_READABILITY_PATH, index=False)

# ---- Ground Truth (predicted vs actual labels) ----
labels = ["positive", "neutral", "negative"]

# Generate true labels randomly (could also use actual labels if available)
true_labels = [random.choice(labels) for _ in range(len(sample_df))]

# Simulate predictions: 70% chance to match true label, else random wrong label
predicted_labels = []
for tl in true_labels:
    if random.random() < 0.7:
        predicted_labels.append(tl)
    else:
        predicted_labels.append(random.choice([l for l in labels if l != tl]))

ground_truth_df = pd.DataFrame({
    "predicted_label": predicted_labels,
    "true_label": true_labels
})
ground_truth_df.to_csv(GROUND_TRUTH_PATH, index=False)

print(f"✅ Generated improved evaluation sample files in '{EVAL_DIR}'")
