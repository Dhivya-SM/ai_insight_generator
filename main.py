"""
Main pipeline runner for AI Insight Generator.

- If LIVE_MODE=True (in .env), optionally run once or run in a scheduler loop (SCHEDULE_LOOP).
- Writes fully processed CSV to: data/processed/final_data.csv
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Script utilities
from scripts.load_data import load_data
from scripts.preprocess_data import preprocess_dataset
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.theme_classification import classify_themes
from scripts.map_insights import map_insights, generate_action_items
from scripts.visualize_summary import plot_sentiment_theme_distribution
from scripts.reddit_stream import stream_reddit_comments
from scripts.generate_ai_suggestions import generate_bart_flan_suggestions

# Training imports
from scripts.auto_tag_themes import auto_tag
from scripts.train_theme_classifier import train_theme_classifier

load_dotenv()

# Paths
KAGGLE_RAW = "data/raw/dell_tweets.csv"
LIVE_RAW = "data/raw/reddit_stream.csv"
CLEANED_PATH = "data/processed/cleaned_data.csv"
SENTIMENT_PATH = "data/processed/sentiment_data.csv"
FINAL_PATH = "data/processed/final_data.csv"
THEME_MODEL_PATH = "models/theme_classifier"

# Config from .env
LIVE_MODE = os.getenv("LIVE_MODE", "False").lower() == "true"
SCHEDULE_LOOP = os.getenv("SCHEDULE_LOOP", "False").lower() == "true"
FETCH_INTERVAL_MINUTES = int(os.getenv("FETCH_INTERVAL_MINUTES", "5"))
LIVE_FETCH_LIMIT = int(os.getenv("LIVE_FETCH_LIMIT", "200"))
AI_MODEL = os.getenv("AI_MODEL", "FLAN").upper()  # FLAN or BART


def process_pipeline(raw_path: str, out_path: str) -> pd.DataFrame:
    """Full processing pipeline."""
    try:
        print(f"🔹 Loading raw data from: {raw_path}")
        load_data(raw_path)

        print("🔹 Preprocessing...")
        preprocess_dataset(raw_path, CLEANED_PATH)

        print("🔹 Sentiment analysis...")
        run_sentiment_analysis(CLEANED_PATH, SENTIMENT_PATH)

        print("🔹 Checking for trained theme classifier...")
        if not os.path.exists(THEME_MODEL_PATH):
            print("⚠️ Theme classifier not found — starting auto-tagging & training...")
            auto_tag()  # Generate labeled data automatically
            train_theme_classifier()  # Train and save model

        print("🔹 Loading sentiment file and classifying themes...")
        df = pd.read_csv(SENTIMENT_PATH)
        df = classify_themes(df)

        print(f"🔹 Saving final data to: {out_path} (records: {len(df)})")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)

        try:
            print("🔹 Generating summary plots...")
            os.makedirs("outputs", exist_ok=True)
            plot_sentiment_theme_distribution(out_path)
        except Exception as e:
            print("⚠️ Could not generate plots:", e)

        try:
            insights = map_insights(df)
            print("🔍 Insight summary:")
            for line in insights:
                print(" •", line)

            actions = generate_action_items(df)
            print("🛠 Action counts:")
            for k, v in actions.items():
                print(f" - {k}: {len(v)} items")
        except Exception:
            pass

        # AI Suggestions
        try:
            print(f"\n🔹 Step 8: AI-generated improvement ideas using {AI_MODEL}...")
            ai_suggestions = generate_bart_flan_suggestions(df)
            for theme, count, idea in ai_suggestions:
                print(f"\n📌 {theme} ({count} mentions):\n👉 {idea}")
        except Exception as e:
            print("⚠️ Could not generate AI suggestions:", e)

        return df

    except Exception as e:
        print("❌ Error in processing pipeline:", e)
        return pd.DataFrame()


def do_live_fetch_and_process(limit: int = LIVE_FETCH_LIMIT):
    """Fetch live Reddit posts and process pipeline."""
    try:
        print(f"🔍 Streaming live Reddit posts (limit={limit})...")
        df_live = stream_reddit_comments(subreddit_name="technology", limit=limit)
        if df_live is None or df_live.empty:
            print("⚠️ No live data returned from reddit_stream.")
            return pd.DataFrame()
        return process_pipeline(LIVE_RAW, FINAL_PATH)
    except Exception as e:
        print("❌ Error fetching/processing live data:", e)
        return pd.DataFrame()


def main_once():
    """One-shot run."""
    if LIVE_MODE:
        print("🌐 LIVE MODE (one-shot).")
        df = do_live_fetch_and_process(limit=LIVE_FETCH_LIMIT)
    else:
        print("📂 OFFLINE MODE (one-shot).")
        df = process_pipeline(KAGGLE_RAW, FINAL_PATH)

    if df.empty:
        print("⚠️ Pipeline completed but no records.")
    else:
        print(f"✅ Pipeline completed: {len(df)} records saved to {FINAL_PATH}")


def main_loop():
    """Scheduler loop."""
    print(f"🌐 LIVE MODE with loop every {FETCH_INTERVAL_MINUTES} minutes.")
    while True:
        start = datetime.utcnow()
        print(f"\n⏳ [{start.isoformat()}] Starting fetch+process")
        do_live_fetch_and_process(limit=LIVE_FETCH_LIMIT)
        elapsed = (datetime.utcnow() - start).total_seconds()
        wait_seconds = max(60, FETCH_INTERVAL_MINUTES * 60 - int(elapsed))
        print(f"⏲️ Sleeping {wait_seconds} seconds...")
        time.sleep(wait_seconds)


if __name__ == "__main__":
    print("🟦 AI Insight Generator — main.py")
    print(f"LIVE_MODE={LIVE_MODE}, SCHEDULE_LOOP={SCHEDULE_LOOP}, FETCH_INTERVAL_MINUTES={FETCH_INTERVAL_MINUTES}, AI_MODEL={AI_MODEL}")
    if LIVE_MODE and SCHEDULE_LOOP:
        main_loop()
    else:
        main_once()
