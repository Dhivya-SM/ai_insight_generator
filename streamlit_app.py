import os
import pandas as pd
import streamlit as st
import plotly.express as px

# Local imports
from scripts.map_insights import generate_action_items
from scripts.generate_ai_suggestions import generate_bart_flan_suggestions
from jiraAutomation.create_jira_ticket import create_jira_ticket

# -------------------------
# Page Config
# -------------------------
st.set_page_config(layout="wide", page_title="AI-Powered Customer Insight Dashboard")

# -------------------------
# Styling
# -------------------------
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

st.markdown("<div style='text-align: center; font-size: 32px; font-weight: bold;'>📊 Dell Tweets Insight Dashboard</div>", unsafe_allow_html=True)

# -------------------------
# Sidebar Navigation
# -------------------------
DATA_PATH =  "data/processed/kaggle_data.csv"
LIVE_PATH = "data/processed/live_data.csv"

nav = st.sidebar.radio("📂 Select Data Source", ["Kaggle Data", "Live Data"])

if nav == "Kaggle Data":
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.sidebar.success("✅ Kaggle data loaded")
    else:
        st.sidebar.error("❌ Kaggle file not found")
        st.stop()
else:
    if os.path.exists(LIVE_PATH):
        df = pd.read_csv(LIVE_PATH)
        st.sidebar.success("✅ Live data loaded")
    else:
        st.sidebar.error("❌ Live data file not found")
        st.stop()

# -------------------------
# Sentiment Distribution
# -------------------------
st.subheader("📊 Sentiment Distribution")
sent_counts = df['sentiment'].value_counts().reset_index()
sent_counts.columns = ['sentiment', 'count']
fig_sentiment = px.bar(
    sent_counts, x='count', y='sentiment', orientation='h',
    color='sentiment',
    color_discrete_map={'Positive': "#37df80", 'Neutral': "#f6e191", 'Negative': "#f1786b"},
    height=300
)
st.plotly_chart(fig_sentiment, use_container_width=True)

# -------------------------
# Theme Distribution
# -------------------------
st.subheader("📌 Theme Distribution")
theme_counts = df['theme'].value_counts().reset_index()
theme_counts.columns = ['theme', 'count']
view_mode = st.radio("Select chart style", ["Gradient", "Solid"], horizontal=True)
bar_colors = theme_counts['theme'].apply(lambda x: "#FF4500") if view_mode == 'Solid' else None
fig_theme_dist = px.bar(
    theme_counts.sort_values(by="count", ascending=True),
    x='count', y='theme', orientation='h',
    color='theme' if view_mode == 'Solid' else 'count',
    color_discrete_sequence=bar_colors if view_mode == 'Solid' else None,
    color_continuous_scale='Hot' if view_mode == 'Gradient' else None,
    height=400
)
st.plotly_chart(fig_theme_dist, use_container_width=True)

# -------------------------
# Theme-wise Sentiment Breakdown
# -------------------------
st.subheader("📊 Theme-wise Sentiment Breakdown")
theme_sentiment = df.groupby(['theme', 'sentiment']).size().reset_index(name='count')
fig_theme = px.bar(
    theme_sentiment, x='theme', y='count', color='sentiment', barmode='group',
    color_discrete_map={'Positive': "#37df80", 'Neutral': "#f6e191", 'Negative': "#f1786b"}
)
st.plotly_chart(fig_theme, use_container_width=True)

# -------------------------
# Filters
# -------------------------
col_f1, col_f2 = st.columns(2)
with col_f1:
    sent_filter = st.selectbox("Filter by sentiment", options=["All", "Positive", "Negative", "Neutral"])
with col_f2:
    theme_filter = st.selectbox("Filter by theme", options=["All"] + sorted(df['theme'].unique()))

filtered = df.copy()
if sent_filter != "All":
    filtered = filtered[filtered['sentiment'] == sent_filter]
if theme_filter != "All":
    filtered = filtered[filtered['theme'] == theme_filter]

st.write(f"📄 Showing {len(filtered)} tweets")
with st.expander("📂 View Filtered Tweets"):
    st.dataframe(filtered[['Text', 'sentiment', 'theme']].reset_index(drop=True), use_container_width=True)

# -------------------------
# Suggested Action Items
# -------------------------
st.subheader("🛠️ Suggested Action Items")
actions_by_sentiment = generate_action_items(df)

for sentiment, actions in actions_by_sentiment.items():
    st.markdown(f"### {sentiment} Feedback")
    if actions:
        cleaned_actions = [(a.replace("Promote as testimonial for ", "")
                             .replace("Note neutral feedback for ", "")
                             .replace("Route negative feedback to ", "")
                             .replace("Escalate to support team for ", ""), c) for a, c in actions]
        action_df = pd.DataFrame(cleaned_actions, columns=["Action Area", "Mentions"])
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(action_df.sort_values(by="Mentions", ascending=False))
        with col2:
            fig = px.pie(action_df, names="Action Area", values="Mentions", hole=0.4,
                         title=f"{sentiment} Feedback Distribution",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No action items available.")

# -------------------------
# 🤖 AI Suggestions
# -------------------------
st.subheader("🤖 AI-Generated Summary")

with st.spinner("Generating AI suggestions..."):
    neg_suggestions = generate_bart_flan_suggestions(df, sentiment_filter="Negative")
    pos_suggestions = generate_bart_flan_suggestions(df, sentiment_filter="Positive")
    neu_suggestions = generate_bart_flan_suggestions(df, sentiment_filter="Neutral")

# ---------- Negative ----------
st.markdown("### 🚨 Negative Impact (Action Required)")
if neg_suggestions:
    for i, (theme, count, summary, actions) in enumerate(neg_suggestions[:3]):
        st.markdown(f"**{i+1}. {theme}** — {count} mentions")
        st.markdown(f"📌 Summary: {summary}")
        for idx, act in enumerate(actions, start=1):
            st.markdown(f"👉 Action {idx}: {act}")

        if st.button(f"Create Jira Ticket for {theme}", key=f"ai_ticket_{i}"):
            summary_text = f"Negative Feedback - {theme[:40]}"
            description = summary + "\n\n" + "\n".join(actions)
            issue_key = create_jira_ticket(summary_text, description)
            if issue_key:
                st.success(f"✅ Jira ticket created: {issue_key}")
            else:
                st.error("❌ Failed to create Jira ticket.")
else:
    st.info("No major negative themes found.")

# ---------- Positive ----------
st.markdown("### 🌟 Positive Impact (For Promotion)")
if pos_suggestions:
    pos_df = pd.DataFrame(
        pos_suggestions,
        columns=["Theme", "Mentions", "Summary", "Actions"]
    )
    pos_df["Actions"] = pos_df["Actions"].apply(lambda acts: " | ".join(acts) if isinstance(acts, list) else acts)
    st.dataframe(pos_df, use_container_width=True)
    st.success("✅ These can be highlighted in campaigns or promotions.")
else:
    st.info("No strong positive themes available.")


# ---------- Neutral ----------
st.markdown("### 📝 Neutral Impact (Review Required)")
if neu_suggestions:
    neu_df = pd.DataFrame(
        neu_suggestions,
        columns=["Theme", "Mentions", "Summary", "Actions"]
    )
    neu_df["Actions"] = neu_df["Actions"].apply(lambda acts: " | ".join(acts) if isinstance(acts, list) else acts)
    st.dataframe(neu_df, use_container_width=True)
    st.warning("⚠️ These need further review/discussion.")
else:
    st.info("No neutral items needing review.")
    
# -------------------------
# 📥 Export
# -------------------------
st.subheader("📥 Export Action Items")
col_exp1, col_exp2 = st.columns([1, 1])

with col_exp1:
    export_sentiment = st.selectbox(
        "Choose category to export",
        ["Negative", "Positive", "Neutral"]
    )

with col_exp2:
    export_map = {
        "Negative": neg_suggestions,
        "Positive": pos_suggestions,
        "Neutral": neu_suggestions
    }
    export_data = export_map.get(export_sentiment, [])
    if export_data:
        df_export = pd.DataFrame(export_data, columns=["Theme", "Mentions", "Suggested Action"])
        st.download_button(
            label=f"📥 Download {export_sentiment} CSV",
            data=df_export.to_csv(index=False),
            file_name=f"{export_sentiment.lower()}_actions.csv",
            mime="text/csv"
        )
    else:
        st.info("No data to export for this category.")
