# streamlit_app.py

import streamlit as st
import pandas as pd
from scripts.map_insights import generate_action_items
from jiraAutomation.create_jira_ticket import create_jira_ticket

st.set_page_config(layout="wide", page_title="AI-Powered Customer Insight Dashboard")

st.title("📊 Dell Tweets Insight Dashboard")

# Load processed data
DATA_PATH = "data/processed/dell_themes.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Sentiment distribution
st.subheader("Sentiment Distribution")
sent_counts = df['sentiment'].value_counts()
st.bar_chart(sent_counts)

# Theme distribution
st.subheader("Theme Distribution")
theme_counts = df['theme'].value_counts()
st.bar_chart(theme_counts)

# Filter by sentiment
sent_filter = st.selectbox("Filter tweets by sentiment", options=["All", "Positive", "Negative", "Neutral"])
filtered = df[df['sentiment'] == sent_filter] if sent_filter != "All" else df

# Filter by theme
theme_filter = st.selectbox("Filter tweets by theme", options=["All"] + sorted(df['theme'].unique()))
filtered = filtered[filtered['theme'] == theme_filter] if theme_filter != "All" else filtered

st.write(f"Showing {len(filtered)} tweets")
st.dataframe(filtered[['Text', 'sentiment', 'theme']])

# 📌 Action Item Suggestions
st.subheader("🛠️ Suggested Action Items Based on Feedback")
actions_by_sentiment = generate_action_items(df)

for sentiment, actions in actions_by_sentiment.items():
    st.markdown(f"### {sentiment} Feedback")
    action_df = pd.DataFrame(actions, columns=["Action Item", "Mentions"])
    st.table(action_df)

# 🚨 Create Jira Tickets
st.subheader("🚨 Create Jira Tickets for Top Action Items")
num_tickets = st.slider("Select number of top items to create tickets for", min_value=1, max_value=10, value=3)
sentiment_choice = st.radio("Choose sentiment to file tickets for", options=["Negative", "Positive", "Neutral"])

# Filter relevant rows and top actions
filtered_df = df[df['sentiment'] == sentiment_choice]
action_items = generate_action_items(filtered_df)
top_actions = action_items.get(sentiment_choice, [])[:num_tickets]

if st.button("📩 Create Jira Tickets"):
    for action_text, count in top_actions:
        summary = f"{sentiment_choice} Feedback: {action_text[:50]}"

        # Try to match tweets based on keywords from action text
        keywords = [word.lower() for word in action_text.split() if len(word) > 3]
        matched_rows = filtered_df[filtered_df['clean_text'].apply(
            lambda x: any(k in str(x).lower() for k in keywords)
        )]

        top_texts = matched_rows['Text'].head(3).tolist()
        if top_texts:
            tweet_samples = "\n".join([f"- {t[:150]}..." for t in top_texts])
        else:
            tweet_samples = "No direct feedback matched.\nPlease refer to the dashboard for more details."

        description = (
            f"{count} mentions.\n"
            f"Suggested Action: {action_text}\n\n"
            f"Sample Feedback:\n{tweet_samples}"
        )

        issue_key = create_jira_ticket(summary, description)
        if issue_key:
            st.success(f"✅ Jira ticket created: {issue_key}")
        else:
            st.error("❌ Failed to create Jira ticket.")
