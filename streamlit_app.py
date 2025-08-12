# streamlit_app.py
"""
Streamlit dashboard for AI Insight Generator.

- Reads the processed CSV (data/processed/final_data.csv) and renders visualizations.
- If you want to process live data from the UI (not recommended for large fetches),
  there's a 'Quick Live Fetch & Process' button that will run a single fetch+process cycle
  inside the Streamlit app (may take time and freeze UI while running).
- Preferred pattern: run main.py separately (scheduler or one-shot). Streamlit only reads the output CSV.
"""
import os
import time
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px

from scripts.map_insights import generate_action_items
from scripts.generate_ai_suggestions import generate_bart_flan_suggestions
from jiraAutomation.create_jira_ticket import create_jira_ticket
from scripts.reddit_stream import stream_reddit_comments
from scripts.preprocess_data import preprocess_dataset
from scripts.sentiment_analysis import run_sentiment_analysis
from scripts.theme_classification import classify_themes

# UI Styling (kept as your theme)
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

st.set_page_config(layout="wide", page_title="AI-Powered Customer Insight Dashboard")

# Title
st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>📊 Dell Tweets Insight Dashboard</div>", unsafe_allow_html=True)

# Paths
PROCESSED_PATH = "data/processed/final_data.csv"
KAGGLE_RAW = "data/raw/dell_tweets.csv"
LIVE_RAW = "data/raw/reddit_stream.csv"

# Sidebar: data source selection
st.sidebar.title("📂 Data Source")
data_source = st.sidebar.radio("Select Data Source:", ["Kaggle (processed)", "Live (processed)"])

# Helper: safe read CSV
def safe_read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Could not read {path}: {e}")
        return pd.DataFrame()

# Show processed file timestamp
def show_last_updated(path):
    if os.path.exists(path):
        ts = datetime.fromtimestamp(os.path.getmtime(path))
        st.caption(f"Last processed: {ts.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.caption("No processed file found. Run main.py in another terminal (recommended).")

# Load processed data according to selection
if data_source == "Kaggle (processed)":
    df = safe_read_csv(PROCESSED_PATH)
    show_last_updated(PROCESSED_PATH)
else:
    # Live processed uses same FINAL_PATH by default (main.py writes final_data.csv)
    df = safe_read_csv(PROCESSED_PATH)
    show_last_updated(PROCESSED_PATH)

# If no processed data, show helpful guidance + quick actions
if df.empty:
    st.warning("⚠️ No processed data available. Recommended: run `python main.py` in a separate terminal (LIVE_MODE or offline), then refresh this page.")
    st.markdown("**Quick options (runs inside this Streamlit app — can be slow):**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Quick Live Fetch & Process (limit=100)"):
            with st.spinner("Fetching & processing live data (this runs workload inside Streamlit)..."):
                try:
                    # fetch
                    live_df = stream_reddit_comments(subreddit_name="technology", limit=100)
                    if live_df is None or live_df.empty:
                        st.error("No posts fetched from Reddit.")
                    else:
                        # run same processing steps
                        preprocess_dataset(LIVE_RAW, "data/processed/live_cleaned.csv")
                        run_sentiment_analysis("data/processed/live_cleaned.csv", "data/processed/live_sentiment.csv")
                        df_proc = pd.read_csv("data/processed/live_sentiment.csv")
                        df_proc = classify_themes(df_proc)
                        os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
                        df_proc.to_csv(PROCESSED_PATH, index=False)
                        st.success(f"✅ Live fetched & processed ({len(df_proc)} records). You can now use the dashboard.")
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to fetch/process in-app: {e}")
    with col2:
        st.info("Recommended: run `python main.py` in a separate terminal so the dashboard remains responsive. Use scheduler mode for continuous updates.")

# If data present, render dashboard
if not df.empty:
    # Allow small cleanups: accept legacy column names 'Text' or 'text'
    if 'Text' not in df.columns and 'text' in df.columns:
        df = df.rename(columns={'text': 'Text'})

    # Top-right show/hide raw
    colA, colB = st.columns([10, 1])
    with colB:
        show_data = st.checkbox("Show raw data")
    if show_data:
        st.dataframe(df.head())

    # Sentiment Distribution
    st.subheader("Sentiment Distribution")
    if 'sentiment' in df.columns:
        sent_counts = df['sentiment'].value_counts().reset_index()
        sent_counts.columns = ['sentiment', 'count']
        fig_sentiment = px.bar(
            sent_counts, x='count', y='sentiment', orientation='h',
            color='sentiment',
            color_discrete_map={'Positive': "#37df80", 'Neutral': "#f6e191", 'Negative': "#f1786b"},
            height=300
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)
    else:
        st.info("No 'sentiment' column found. Please run the full pipeline (main.py).")

    # Theme Distribution
    st.subheader("Theme Distribution")
    if 'theme' in df.columns:
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
    else:
        st.info("No 'theme' column found. Please run the full pipeline (main.py).")

    # Theme-wise Sentiment Breakdown
    if 'theme' in df.columns and 'sentiment' in df.columns:
        st.subheader("📊 Theme-wise Sentiment Breakdown")
        theme_sentiment = df.groupby(['theme', 'sentiment']).size().reset_index(name='count')
        fig_theme = px.bar(
            theme_sentiment,
            x='theme', y='count', color='sentiment', barmode='group',
            color_discrete_map={'Positive': "#37df80", 'Neutral': "#f6e191", 'Negative': "#f1786b"}
        )
        st.plotly_chart(fig_theme, use_container_width=True)

    # Filters & Tweet view
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sent_filter = st.selectbox("Filter tweets by sentiment", options=["All"] + sorted(df['sentiment'].unique()) if 'sentiment' in df.columns else ["All"])
    with col_f2:
        theme_opts = ["All"] + sorted(df['theme'].unique()) if 'theme' in df.columns else ["All"]
        theme_filter = st.selectbox("Filter tweets by theme", options=theme_opts)

    filtered = df.copy()
    if sent_filter != "All" and 'sentiment' in filtered.columns:
        filtered = filtered[filtered['sentiment'] == sent_filter]
    if theme_filter != "All" and 'theme' in filtered.columns:
        filtered = filtered[filtered['theme'] == theme_filter]

    st.write(f"Showing {len(filtered)} tweets")
    with st.expander("📂 View Filtered Tweets"):
        show_cols = [c for c in ['Text', 'sentiment', 'theme'] if c in filtered.columns]
        st.dataframe(filtered[show_cols].reset_index(drop=True), use_container_width=True)

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

    # AI suggestions (on-demand)
    st.subheader("🤖 AI-Generated Action Suggestions (Negative)")
    if st.button("Generate AI Suggestions (may take time)"):
        with st.spinner("Generating suggestions using BART model..."):
            try:
                ai_suggestions = generate_bart_flan_suggestions(df)
                if ai_suggestions:
                    ai_df = pd.DataFrame(ai_suggestions, columns=["Theme", "Mentions", "Suggested Action"])
                    st.dataframe(ai_df, use_container_width=True)
                    st.success(f"✅ Generated {len(ai_df)} suggestions")
                else:
                    st.info("No suggestions generated.")
            except Exception as e:
                st.error(f"AI suggestion generation failed: {e}")

    # Jira Ticket Creation (test)
    st.subheader("🚨 Jira Tickets (create from suggestion)")
    st.write("Tip: test 'Create Test Jira Ticket' to verify API + token (this uses create_jira_ticket).")
    if st.button("Create Test Jira Ticket"):
        with st.spinner("Creating Jira ticket..."):
            issue_key = create_jira_ticket("Python Test from Streamlit", "Testing from Streamlit app")
            if issue_key:
                st.success(f"✅ Created Jira ticket: {issue_key}")
            else:
                st.error("❌ Failed to create Jira ticket. Check .env and Jira project permissions.")

    # Export action items
    st.subheader("📥 Export Action Items")
    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        export_sentiment = st.selectbox("Choose sentiment to export", ["Negative", "Positive", "Neutral"])
    with col_e2:
        export_data = actions_by_sentiment.get(export_sentiment, [])
        if export_data:
            df_export = pd.DataFrame(export_data, columns=["Action", "Mentions"])
            st.download_button(label="📥 Download CSV",
                               data=df_export.to_csv(index=False),
                               file_name=f"{export_sentiment.lower()}_actions.csv",
                               mime='text/csv')
