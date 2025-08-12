# scripts/evaluation_metrics.py
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score

def sentiment_accuracy(pred_csv, gold_csv):
    pred_df = pd.read_csv(pred_csv)
    gold_df = pd.read_csv(gold_csv)
    merged = pred_df.merge(gold_df, on="id", suffixes=("_pred", "_gold"))
    return accuracy_score(merged["sentiment_gold"], merged["sentiment_pred"])

def theme_precision(pred_csv, gold_csv):
    pred_df = pd.read_csv(pred_csv)
    gold_df = pd.read_csv(gold_csv)
    merged = pred_df.merge(gold_df, on="id", suffixes=("_pred", "_gold"))
    return precision_score(merged["theme_gold"], merged["theme_pred"], average="weighted")

def approval_rate(eval_csv):
    df = pd.read_csv(eval_csv)
    return df["useful"].mean()

def readability_score(eval_csv):
    df = pd.read_csv(eval_csv)
    return df["rating"].mean()

def pipeline_throughput(total_records, start_time, end_time):
    minutes = (end_time - start_time) / 60
    return total_records / minutes

if __name__ == "__main__":
    print("📊 Evaluation Metrics Report")
    try:
        acc = sentiment_accuracy("data/processed/final_data.csv", "data/eval/gold_labels.csv")
        print(f"Sentiment Accuracy: {acc:.2%}")
    except Exception as e:
        print("⚠️ Sentiment Accuracy skipped:", e)

    try:
        prec = theme_precision("data/processed/final_data.csv", "data/eval/gold_labels.csv")
        print(f"Theme Precision: {prec:.2%}")
    except Exception as e:
        print("⚠️ Theme Precision skipped:", e)

    try:
        appr = approval_rate("data/eval/action_eval.csv")
        print(f"Action Suggestion Approval Rate: {appr:.2%}")
    except Exception as e:
        print("⚠️ Approval Rate skipped:", e)

    try:
        read = readability_score("data/eval/action_eval.csv")
        print(f"Readability Score: {read:.2f}/5")
    except Exception as e:
        print("⚠️ Readability Score skipped:", e)
