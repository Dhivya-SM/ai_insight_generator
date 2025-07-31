#generate_ai_suggestions

from transformers import pipeline

# Load once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_bart_suggestions(df, top_n=3):
    df = df[df['sentiment'] == "Negative"]
    top_themes = df['theme'].value_counts().nlargest(top_n).index.tolist()
    
    suggestions = []

    for theme in top_themes:
        comments = df[df['theme'] == theme]['Text'].dropna().tolist()
        if not comments:
            continue
        text_blob = " ".join(comments[:20])
        if not text_blob.strip():
            continue
        try:
            summary = summarizer(text_blob[:1024], max_length=80, min_length=30, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = "No summary generated."

        final = f"Customer is facing issues with {theme.lower()}. Take immediate action to improve and avoid similar complaints. Refer to the dashboard for live customer comments.\nInsight: {summary}"
        suggestions.append((theme, len(comments), final))

    return suggestions
