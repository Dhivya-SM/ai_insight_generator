# scripts/visualize_summary.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_sentiment_theme_distribution(csv_path):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 5))

    # Sentiment Plot
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='sentiment', palette='Set2')
    plt.title("Sentiment Distribution")

    # Theme Plot
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='theme', palette='Set3')
    plt.title("Theme Distribution")
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig("outputs/distribution_summary.png")
    print("✅ Charts saved to outputs/distribution_summary.png")

if __name__ == "__main__":
    plot_sentiment_theme_distribution("data/processed/dell_themes.csv")
