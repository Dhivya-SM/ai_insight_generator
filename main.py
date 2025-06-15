from scripts.load_data import load_data
from scripts.preprocess_data import preprocess_dataset
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.theme_classification import classify_themes
from scripts.map_insights import map_insights, generate_action_items
from scripts.visualize_summary import plot_sentiment_theme_distribution
import pandas as pd

def main():
    # Step 1: Load raw data
    raw_path = "data/raw/dell_tweets.csv"
    df = load_data(raw_path)

    # Step 2: Preprocess and save cleaned text
    cleaned_path = "data/processed/dell_cleaned.csv"
    preprocess_dataset(raw_path, cleaned_path)

    # Step 3: Run sentiment analysis and save results
    sentiment_path = "data/processed/dell_sentiment.csv"
    run_sentiment_analysis(cleaned_path, sentiment_path)

    # Step 4: Reload and classify themes, save again
    df = pd.read_csv(sentiment_path)
    df = classify_themes(df)
    theme_path = "data/processed/dell_themes.csv"
    df.to_csv(theme_path, index=False)

    # Step 5: Map and print insights
    insights = map_insights(df)
    print("\n📊 Insight Summary:")
    for line in insights:
        print("•", line)

    # Step 6: Generate and print action items
    print("\n🛠️ Action Items:")
    actions = generate_action_items(df)
    for action in actions:
        print("•", action)

    # Step 7: Generate visualization
    plot_sentiment_theme_distribution(theme_path)

if __name__ == "__main__":
    main()
