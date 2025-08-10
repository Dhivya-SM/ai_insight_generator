# main.py

import os
import pandas as pd
from dotenv import load_dotenv

from scripts.load_data import load_data
from scripts.preprocess_data import preprocess_dataset
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.theme_classification import classify_themes
from scripts.map_insights import map_insights, generate_action_items
from scripts.visualize_summary import plot_sentiment_theme_distribution
from scripts.generate_ai_suggestions import generate_bart_suggestions
from scripts.reddit_stream import stream_reddit_comments

# Load environment variables
load_dotenv()

LIVE_MODE = os.getenv("LIVE_MODE", "False").lower() == "true"

def main():
    if LIVE_MODE:
        print("🌐 LIVE MODE ENABLED: Streaming Reddit data...")
        stream_reddit_comments(subreddit_name="technology", limit=50)
        raw_path = "data/raw/reddit_stream.csv"
    else:
        print("📂 OFFLINE MODE: Using Kaggle dataset...")
        raw_path = "data/raw/dell_cleaned.csv"

    # Step 1: Load Data
    print("🔹 Step 1: Loading raw data...")
    df_raw = load_data(raw_path)

    # Step 2: Preprocess
    print("🔹 Step 2: Preprocessing data...")
    cleaned_path = "data/processed/cleaned_data.csv"
    preprocess_dataset(raw_path, cleaned_path)

    # Step 3: Sentiment Analysis
    print("🔹 Step 3: Running sentiment analysis...")
    sentiment_path = "data/processed/sentiment_data.csv"
    run_sentiment_analysis(cleaned_path, sentiment_path)

    # Step 4: Theme Classification
    print("🔹 Step 4: Classifying themes...")
    df = pd.read_csv(sentiment_path)
    df = classify_themes(df)
    theme_path = "data/processed/final_data.csv"
    df.to_csv(theme_path, index=False)

    # Step 5: Insights Summary
    print("🔹 Step 5: Mapping summary insights...")
    insights = map_insights(df)
    for line in insights:
        print("•", line)

    # Step 6: Action Items
    print("\n🔹 Step 6: Generating action items...")
    actions = generate_action_items(df)
    for sentiment, items in actions.items():
        print(f"\n🧭 {sentiment} Actions:")
        for action, count in items:
            print(f"• {action} — {count} mentions")

    # Step 7: Visualizations
    print("\n🔹 Step 7: Generating visualizations...")
    plot_sentiment_theme_distribution(theme_path)

    # Step 8: AI Suggestions
    print("\n🔹 Step 8: AI-generated improvement ideas...")
    ai_suggestions = generate_bart_suggestions(df)
    for theme, count, idea in ai_suggestions:
        print(f"\n📌 {theme} ({count} mentions):\n👉 {idea}")

    print("\n✅ Pipeline completed successfully.")

if __name__ == "__main__":
    main()
