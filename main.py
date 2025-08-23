# main.py

import os
import time
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Custom script imports
from scripts.load_data import load_data
from scripts.preprocess_data import preprocess_dataset
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.theme_classification import classify_themes
from scripts.map_insights import map_insights, generate_action_items
from scripts.visualize_summary import plot_sentiment_theme_distribution
from scripts.reddit_stream import stream_reddit_comments
from scripts.generate_ai_suggestions import generate_bart_flan_suggestions

# Load environment variables
load_dotenv()

# Paths
KAGGLE_RAW = "data/raw/dell_tweets.csv"
LIVE_RAW = "data/raw/reddit_stream.csv"
CLEANED_PATH = "data/processed/cleaned_data.csv"
SENTIMENT_PATH = "data/processed/sentiment_data.csv"
KAGGLE_PROCESSED_PATH = "data/processed/kaggle_data.csv"
LIVE_PROCESSED_PATH = "data/processed/live_data.csv"

# Config from .env
LIVE_MODE = os.getenv("LIVE_MODE", "False").lower() == "true"
SCHEDULE_LOOP = os.getenv("SCHEDULE_LOOP", "False").lower() == "true"
FETCH_INTERVAL_MINUTES = int(os.getenv("FETCH_INTERVAL_MINUTES", "5"))
LIVE_FETCH_LIMIT = int(os.getenv("LIVE_FETCH_LIMIT", "200"))
AI_MODEL = os.getenv("AI_MODEL", "FLAN").upper()


def process_pipeline(raw_path: str, out_path: str) -> pd.DataFrame:
    """Full pipeline to process input and store final output."""
    try:
        start_time = time.time()
        print(f"🔹 Loading raw data from: {raw_path}")
        load_data(raw_path)
        print(f"→ Data load took {time.time() - start_time:.2f} seconds")

        print("🔹 Preprocessing...")
        preprocess_dataset(raw_path, CLEANED_PATH)

        print("🔹 Sentiment analysis...")
        run_sentiment_analysis(CLEANED_PATH, SENTIMENT_PATH)

        print("🔹 Classifying themes...")
        df = pd.read_csv(SENTIMENT_PATH)
        df = classify_themes(df, batch_size=64, limit=5000)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"🔹 Saved processed data to: {out_path}")

        try:
            print("🔹 Generating plots...")
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

        try:
            print(f"\n🔹 AI suggestions using {AI_MODEL}...")
            ai_suggestions = generate_bart_flan_suggestions(df)
            for theme, count, idea in ai_suggestions:
                print(f"\n📌 {theme} ({count} mentions):\n👉 {idea}")
        except Exception as e:
            print("⚠️ Could not generate AI suggestions:", e)

        print(f"✅ Pipeline completed in {time.time() - start_time:.2f} seconds")
        return df

    except Exception as e:
        print("❌ Error in processing pipeline:", e)
        return pd.DataFrame()


def do_live_fetch_and_process(limit: int = LIVE_FETCH_LIMIT, out_path: str = LIVE_PROCESSED_PATH):
    """Fetch live Reddit posts and process."""
    try:
        start_time = time.time()
        print(f"🔍 Streaming live Reddit posts (limit={limit})...")
        df_live = stream_reddit_comments(keyword="dell", limit_per_subreddit=limit)
        if df_live is None or df_live.empty:
            print("⚠️ No live data returned.")
            return pd.DataFrame()
        print(f"→ Live fetch took {time.time() - start_time:.2f} seconds")
        return process_pipeline(LIVE_RAW, out_path)
    except Exception as e:
        print("❌ Error fetching/processing live data:", e)
        return pd.DataFrame()


def main_once():
    """Run pipeline once for Kaggle and Live data."""
    start_time = time.time()

    # if not os.path.exists(KAGGLE_PROCESSED_PATH):
    #     print("📂 Processing Kaggle dataset...")
    #     df_kaggle = process_pipeline(KAGGLE_RAW, KAGGLE_PROCESSED_PATH)
    # else:
    #     print("✅ Kaggle dataset already processed.")
    #     df_kaggle = pd.read_csv(KAGGLE_PROCESSED_PATH)

    print("📂 Processing Kaggle dataset...")
    df_kaggle = process_pipeline(KAGGLE_RAW, KAGGLE_PROCESSED_PATH)

    print("\n🌐 Processing Live Reddit stream...")
    df_live = do_live_fetch_and_process(limit=LIVE_FETCH_LIMIT, out_path=LIVE_PROCESSED_PATH)

    print(f"\n✅ All processing complete in {time.time() - start_time:.2f} seconds")
    print(f"🔹 Kaggle records: {len(df_kaggle)} | Live records: {len(df_live)}")


def main_loop():
    """Continuous loop mode for live data fetch and processing."""
    print(f"🌐 LIVE MODE enabled. Loop every {FETCH_INTERVAL_MINUTES} minutes.")
    while True:
        start = datetime.utcnow()
        print(f"\n⏳ [{start.isoformat()}] Starting live fetch & process...")
        do_live_fetch_and_process(out_path=LIVE_PROCESSED_PATH)
        elapsed = (datetime.utcnow() - start).total_seconds()
        wait_seconds = max(60, FETCH_INTERVAL_MINUTES * 60 - int(elapsed))
        print(f"⏲️ Sleeping for {wait_seconds} seconds...")
        time.sleep(wait_seconds)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["once", "loop"], default="once")
    args = parser.parse_args()

    print("🟦 AI Insight Generator — main.py")
    print(f"LIVE_MODE={LIVE_MODE}, SCHEDULE_LOOP={SCHEDULE_LOOP}, AI_MODEL={AI_MODEL}")

    if args.mode == "loop" or (LIVE_MODE and SCHEDULE_LOOP):
        main_loop()
    else:
        main_once()
