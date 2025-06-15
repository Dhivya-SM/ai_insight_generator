# streamlit_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.map_insights import generate_action_items

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
if sent_filter != "All":
    filtered = df[df['sentiment'] == sent_filter]
else:
    filtered = df

# Filter by theme
theme_filter = st.selectbox("Filter tweets by theme", options=["All"] + sorted(df['theme'].unique()))
if theme_filter != "All":
    filtered = filtered[filtered['theme'] == theme_filter]

st.write(f"Showing {len(filtered)} tweets")
st.dataframe(filtered[['Text', 'sentiment', 'theme']])

# 📌 Action Item Suggestions
st.subheader("🛠️ Suggested Action Items Based on Feedback")

actions_by_sentiment = generate_action_items(df)

for sentiment, actions in actions_by_sentiment.items():
    st.markdown(f"### {sentiment} Feedback")
    action_df = pd.DataFrame(actions, columns=["Action Item", "Mentions"])
    st.table(action_df)

