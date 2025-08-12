import os
import pandas as pd
from transformers import pipeline
from joblib import load

THEME_MODEL_PATH = "models/theme_classifier"

def classify_themes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loads the fine-tuned theme classifier and applies it to the dataframe.
    Assumes the model is already trained and saved.
    """
    if not os.path.exists(THEME_MODEL_PATH):
        print(f"⚠️ Theme classifier not found at {THEME_MODEL_PATH}. Skipping classification...")
        df["theme"] = "Other/General"
        return df

    print("🔹 Loading trained theme classification model...")
    classifier = load(os.path.join(THEME_MODEL_PATH, "model.joblib"))

    themes = []
    for text in df["clean_text"]:
        if not text or text.strip() == "":
            themes.append("Other/General")
            continue
        try:
            pred = classifier(text)
            themes.append(pred[0]["label"])
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            themes.append("Other/General")

    df["theme"] = themes
    return df
