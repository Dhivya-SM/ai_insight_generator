def classify_themes(df):
    def _classify(text):
        text = text.lower()

        if any(word in text for word in ["service", "support", "help", "assistance"]):
            return "Customer Service"
        elif any(word in text for word in ["price", "cost", "discount", "expensive", "pricing"]):
            return "Pricing"
        elif any(word in text for word in ["battery", "hardware", "screen", "charger", "keyboard", "laptop"]):
            return "Hardware"
        elif any(word in text for word in ["bug", "crash", "software", "app", "update", "install"]):
            return "Software"
        elif any(word in text for word in ["delivery", "shipping", "late", "courier", "arrived"]):
            return "Delivery"
        elif any(word in text for word in ["invoice", "billing", "charged", "payment", "refund"]):
            return "Billing"
        elif any(word in text for word in ["slow", "lag", "performance", "speed", "freeze"]):
            return "Performance"
        elif any(word in text for word in ["quality", "broke", "damage", "durability", "scratch"]):
            return "Product Quality"
        elif any(word in text for word in ["order", "purchase", "checkout", "bought", "buy"]):
            return "Ordering Issues"
        elif any(word in text for word in ["login", "password", "sign in", "account", "access denied"]):
            return "Account Issues"
        elif any(word in text for word in ["warranty", "repair", "replacement", "rma", "fixed"]):
            return "Warranty/Repair"
        elif any(word in text for word in ["interface", "user experience", "ux", "navigation", "confusing"]):
            return "UI/UX Feedback"
        elif any(word in text for word in ["install", "setup", "configuration"]):
            return "Installation"
        elif any(word in text for word in ["contacted", "response time", "customer care", "no reply"]):
            return "Communication"
        elif any(word in text for word in ["ad", "promotion", "marketing", "commercial"]):
            return "Advertising"
        elif any(word in text for word in ["compatible", "integration", "works with", "not working with"]):
            return "Compatibility"
        else:
            return "Other/General"

    df['theme'] = df['clean_text'].apply(_classify)
    return df
