import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

# --- File Paths (auto-detect project root) ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root
HUMAN_APPROVAL_PATH = os.path.join(BASE_DIR, "data", "eval", "human_approval.csv")
HUMAN_READABILITY_PATH = os.path.join(BASE_DIR, "data", "eval", "human_readability.csv")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "data", "eval", "ground_truth.csv")

def evaluate_human_approval():
    df = pd.read_csv(HUMAN_APPROVAL_PATH)
    if not {"suggestion", "is_useful"}.issubset(df.columns):
        raise ValueError("human_approval.csv must have columns: suggestion, is_useful")
    return df["is_useful"].mean()

def evaluate_readability():
    df = pd.read_csv(HUMAN_READABILITY_PATH)
    if not {"suggestion", "rating"}.issubset(df.columns):
        raise ValueError("human_readability.csv must have columns: suggestion, rating")
    return df["rating"].mean()

def evaluate_ground_truth_simple():
    df = pd.read_csv(GROUND_TRUTH_PATH)
    if not {"predicted_label", "true_label"}.issubset(df.columns):
        raise ValueError("ground_truth.csv must have columns: predicted_label, true_label")
    
    acc = accuracy_score(df["true_label"], df["predicted_label"])
    f1 = f1_score(df["true_label"], df["predicted_label"], average="weighted")
    precision = precision_score(df["true_label"], df["predicted_label"], average="weighted")
    recall = recall_score(df["true_label"], df["predicted_label"], average="weighted")
    return acc, f1, precision, recall

if __name__ == "__main__":
    ...
    # Simple ground truth evaluation without thresholds
    try:
        acc, f1, precision, recall = evaluate_ground_truth_simple()
        print(f"✅ Accuracy: {acc:.2%}")
        print(f"✅ F1 Score: {f1:.4f}")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall: {recall:.4f}")
    except FileNotFoundError:
        print("❌ ground_truth.csv not found.")
    except Exception as e:
        print(f"⚠ Error in ground truth evaluation: {e}")