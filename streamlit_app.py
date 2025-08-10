import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.map_insights import generate_action_items
from scripts.generate_ai_suggestions import generate_bart_suggestions
from jiraAutomation.create_jira_ticket import create_jira_ticket
from scripts.reddit_stream import stream_reddit_comments
from scripts.preprocess_data import preprocess_dataset
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.theme_classification import classify_themes
import os

st.set_page_config(layout="wide", page_title="AI-Powered Customer Insight Dashboard")

# UI Styling
st.markdown("""
    <style>
    .block-container { padding: 2rem; }
    .stButton > button, .stSelectbox, .stTextInput, .stDownloadButton > button {
        border-radius: 10px;
        padding: 0.4rem 1rem;
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.05);
    }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .stChart { background-color: #ffffff; padding: 0.5rem; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>📊 Dell Tweets Insight Dashboard</div>", unsafe_allow_html=True)

# File paths
DATA_PATH = "data/processed/dell_themes.csv"
LIVE_PATH = "data/processed/live_themes.csv"

# Sidebar source selector
st.sidebar.title("📂 Data Source")
data_source = st.sidebar.radio("Select Data Source:", ["Kaggle", "Live"])

@st.cache_data
def load_kaggle_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_live_data():
    st.info("🔍 Fetching Live Reddit data...")
    live_df = stream_reddit_comments(subreddit_name="technology", limit=100)
    if not live_df.empty:
        preprocess_dataset("data/raw/reddit_stream.csv", "data/processed/live_cleaned.csv")
        run_sentiment_analysis("data/processed/live_cleaned.csv", "data/processed/live_sentiment.csv")
        df = pd.read_csv("data/processed/live_sentiment.csv")
        df = classify_themes(df)
        df.to_csv(LIVE_PATH, index=False)
        st.success(f"✅ Live data processed ({len(df)} records)")
        return df
    else:
        st.error("⚠️ No live data fetched.")
        return pd.DataFrame()

# Load data based on source
df = load_kaggle_data() if data_source == "Kaggle" else load_live_data()

if df.empty:
    st.warning("⚠️ No data available. Please check the source.")
    st.stop()

# Checkbox to show raw data
col1, col2 = st.columns([10, 1])
with col2:
    show_data = st.checkbox("Show raw data")
if show_data:
    st.dataframe(df.head())

# Sentiment Distribution
st.subheader("Sentiment Distribution")
sent_counts = df['sentiment'].value_counts().reset_index()
sent_counts.columns = ['sentiment', 'count']
fig_sentiment = px.bar(
    sent_counts, x='count', y='sentiment', orientation='h',
    color='sentiment',
    color_discrete_map={
        'Positive': "#37df80", 'Neutral': "#f6e191", 'Negative': "#f1786b"
    },
    height=300
)
st.plotly_chart(fig_sentiment, use_container_width=True)

# Theme Distribution
st.subheader("Theme Distribution")
theme_counts = df['theme'].value_counts().reset_index()
theme_counts.columns = ['theme', 'count']
view_mode = st.radio("Select chart style for Theme Distribution", ["Gradient", "Solid"], horizontal=True)
color_style = 'Hot'
bar_colors = theme_counts['theme'].apply(lambda x: "#FF4500") if view_mode == 'Solid' else None
fig_theme_dist = px.bar(
    theme_counts.sort_values(by="count", ascending=True),
    x='count', y='theme', orientation='h',
    color='theme' if view_mode == 'Solid' else 'count',
    color_discrete_sequence=bar_colors if view_mode == 'Solid' else None,
    color_continuous_scale=color_style if view_mode == 'Gradient' else None,
    height=400
)
st.plotly_chart(fig_theme_dist, use_container_width=True)

# Theme-wise Sentiment Breakdown
st.subheader("📊 Theme-wise Sentiment Breakdown")
theme_sentiment = df.groupby(['theme', 'sentiment']).size().reset_index(name='count')
fig_theme = px.bar(
    theme_sentiment,
    x='theme', y='count', color='sentiment', barmode='group',
    color_discrete_map={
        'Positive': "#37df80", 'Neutral': "#f6e191", 'Negative': "#f1786b"
    }
)
st.plotly_chart(fig_theme, use_container_width=True)

# Filters
col_f1, col_f2 = st.columns(2)
with col_f1:
    sent_filter = st.selectbox("Filter tweets by sentiment", options=["All", "Positive", "Negative", "Neutral"])
with col_f2:
    theme_filter = st.selectbox("Filter tweets by theme", options=["All"] + sorted(df['theme'].unique()))

filtered = df.copy()
if sent_filter != "All":
    filtered = filtered[filtered['sentiment'] == sent_filter]
if theme_filter != "All":
    filtered = filtered[filtered['theme'] == theme_filter]

st.write(f"Showing {len(filtered)} tweets")
with st.expander("📂 View Filtered Tweets"):
    st.dataframe(filtered[['Text', 'sentiment', 'theme']].reset_index(drop=True), use_container_width=True)

# Suggested Action Items
st.subheader("🛠️ Suggested Action Items Based on Feedback")
actions_by_sentiment = generate_action_items(df)
for sentiment, actions in actions_by_sentiment.items():
    st.markdown(f"### {sentiment} Feedback")
    if actions:
        cleaned_actions = [(a.replace("Promote as testimonial for ", "")
                             .replace("Note neutral feedback for ", "")
                             .replace("Route negative feedback to ", "")
                             .replace("Escalate to support team for ", ""), c)
                            for a, c in actions]
        action_df = pd.DataFrame(cleaned_actions, columns=["Action Area", "Mentions"])
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(action_df.sort_values(by="Mentions", ascending=False))
        with col2:
            fig = px.pie(action_df, names="Action Area", values="Mentions", hole=0.4,
                         title=f"{sentiment} Feedback Distribution",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

# AI Suggestions
st.subheader("🤖 AI-Generated Action Suggestions (Negative)")
with st.spinner("Generating suggestions using BART model..."):
    ai_suggestions = generate_bart_suggestions(df)
if ai_suggestions:
    ai_df = pd.DataFrame(ai_suggestions, columns=["Theme", "Mentions", "Suggested Action"])
    st.dataframe(ai_df, use_container_width=True)

# Jira Ticket Creation

st.subheader("🚨 Suggested Jira Tickets (Top Negative Themes)")

if st.button("Create Test Jira Ticket"):
    issue_key = create_jira_ticket("Python Test from Streamlit", "Testing from Streamlit app")
    if issue_key:
        st.success(f"✅ Created Jira ticket: {issue_key}")
    else:
        st.error("❌ Failed to create Jira ticket")

# Export Actions
st.subheader("📥 Export Action Items")
col_exp1, col_exp2 = st.columns([1, 1])
with col_exp1:
    export_sentiment = st.selectbox("Choose sentiment to export", ["Negative", "Positive", "Neutral"])
with col_exp2:
    export_data = actions_by_sentiment.get(export_sentiment, [])
    if export_data:
        df_export = pd.DataFrame(export_data, columns=["Action", "Mentions"])
        st.download_button(label="📥 Download CSV",
                           data=df_export.to_csv(index=False),
                           file_name=f"{export_sentiment.lower()}_actions.csv",
                           mime='text/csv')
