# scripts/map_insights.py
from collections import Counter

def map_insights(df):
    sentiment_counts = df['sentiment'].value_counts().to_dict()
    theme_counts = df['theme'].value_counts().to_dict()

    summary = [
        f"Sentiment distribution: {sentiment_counts}",
        f"Theme distribution: {theme_counts}",
    ]
    return summary

def generate_action_items(df):
    actions = []

    for _, row in df.iterrows():
        sentiment = row['sentiment']
        theme = row['theme']

        if sentiment == "Negative":
            if theme == "Customer Service":
                actions.append("Escalate to support team for poor service complaint.")
            elif theme == "Hardware":
                actions.append("Notify product team about hardware malfunction.")
            elif theme == "Pricing":
                actions.append("Send to marketing to review pricing feedback.")
            else:
                actions.append("Route negative feedback to general quality control team.")
        elif sentiment == "Positive" and theme != "Other":
            actions.append(f"Promote as testimonial for {theme} area.")

    return list(Counter(actions).items())  # Optional: consolidate repeated actions
