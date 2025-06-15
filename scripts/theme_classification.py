def classify_themes(df):
    def _classify(text):
        text = text.lower()
        if "service" in text or "support" in text:
            return "Customer Service"
        elif "price" in text or "discount" in text:
            return "Pricing"
        elif "battery" in text or "hardware" in text:
            return "Hardware"
        else:
            return "Other"
    df['theme'] = df['clean_text'].apply(_classify)
    return df
