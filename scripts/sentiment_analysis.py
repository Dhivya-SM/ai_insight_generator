import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already available
nltk.download('vader_lexicon')

def get_sentiment(text, analyzer):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def run_sentiment_analysis(input_path, output_path):
    df = pd.read_csv(input_path)

    if 'clean_text' not in df.columns:
        print("❌ 'clean_text' column not found. Run preprocessing first.")
        return

    # Drop rows with NaN or empty 'clean_text'
    df = df.dropna(subset=['clean_text'])
    df = df[df['clean_text'].str.strip() != ""]

    analyzer = SentimentIntensityAnalyzer()
    # Apply sentiment analysis on 'clean_text' column
    df['sentiment'] = df['clean_text'].apply(lambda x: get_sentiment(x, analyzer))
    
    # Save the DataFrame with sentiment results
    df.to_csv(output_path, index=False)
    print(f"✅ Sentiment analysis saved to {output_path}")

if __name__ == "__main__":
    input_file = "data/processed/dell_sentiment.csv"  # Input path to the CSV file
    output_file = "data/processed/dell_sentiment_with_sentiment.csv"  # Output path for results
    run_sentiment_analysis(input_file, output_file)
