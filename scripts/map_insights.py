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
    grouped_actions = {
        "Negative": [],
        "Positive": [],
        "Neutral": []
    }

    for _, row in df.iterrows():
        sentiment = row["sentiment"]
        theme = row["theme"]

        if sentiment == "Negative":
            if "service" in theme.lower():
                grouped_actions["Negative"].append("Escalate to support team for poor service complaint.")
            elif "hardware" in theme.lower():
                grouped_actions["Negative"].append("Notify product team about hardware malfunction.")
            elif "pricing" in theme.lower():
                grouped_actions["Negative"].append("Send to marketing to review pricing feedback.")
            elif "delivery" in theme.lower():
                grouped_actions["Negative"].append("Investigate delivery delay issues.")
            elif "software" in theme.lower():
                grouped_actions["Negative"].append("Forward to software QA for investigation.")
            else:
                grouped_actions["Negative"].append(f"Route negative feedback to {theme} team.")

        elif sentiment == "Positive":
            grouped_actions["Positive"].append(f"Promote as testimonial for {theme} area.")

        elif sentiment == "Neutral":
            grouped_actions["Neutral"].append(f"Note neutral feedback for {theme} area.")

    # Count and sort
    summarized = {}
    for sentiment, actions in grouped_actions.items():
        counts = Counter(actions)
        summarized[sentiment] = sorted(counts.items(), key=lambda x: -x[1])

    return summarized
