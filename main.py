from scripts.load_data import load_data
from scripts.preprocess_data import preprocess_dataset
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.theme_classification import classify_themes
from scripts.map_insights import map_insights, generate_action_items
from scripts.visualize_summary import plot_sentiment_theme_distribution
from scripts.generate_ai_suggestions import generate_summarized_suggestions  # optional
import pandas as pd


def main():
    print("🔹 Step 1: Loading raw data...")
    raw_path = "data/raw/dell_tweets.csv"
    df_raw = load_data(raw_path)

    print("🔹 Step 2: Preprocessing tweets...")
    cleaned_path = "data/processed/dell_cleaned.csv"
    preprocess_dataset(raw_path, cleaned_path)

    print("🔹 Step 3: Running sentiment analysis...")
    sentiment_path = "data/processed/dell_sentiment.csv"
    run_sentiment_analysis(cleaned_path, sentiment_path)

    print("🔹 Step 4: Classifying themes...")
    df = pd.read_csv(sentiment_path)
    df = classify_themes(df)
    theme_path = "data/processed/dell_themes.csv"
    df.to_csv(theme_path, index=False)

    print("🔹 Step 5: Mapping summary insights...")
    insights = map_insights(df)
    print("\n📊 Insight Summary:")
    for line in insights:
        print("•", line)

    print("\n🔹 Step 6: Generating action items...")
    actions = generate_action_items(df)
    for sentiment, items in actions.items():
        print(f"\n🧭 {sentiment} Actions:")
        for action, count in items:
            print(f"• {action} — {count} mentions")

    print("\n🔹 Step 7: Generating visualizations...")
    plot_sentiment_theme_distribution(theme_path)

    # Optional: AI summarization preview
    print("\n🔹 Step 8: AI-generated improvement ideas (from comments):")
    ai_suggestions = generate_summarized_suggestions(df)
    for theme, count, idea in ai_suggestions:
        print(f"\n📌 {theme} ({count} mentions):\n👉 {idea}")

    print("\n✅ Pipeline completed successfully.")


if __name__ == "__main__":
    main()
