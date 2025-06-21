import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.map_insights import generate_action_items
from jiraAutomation.create_jira_ticket import create_jira_ticket
from scripts.map_insights import generate_suggestions_from_comments
# UI Styling
st.markdown("""
    <style>
    .block-container {
        padding: 2rem;
    }
    .stButton > button, .stSelectbox, .stTextInput, .stDownloadButton > button {
        border-radius: 10px;
        padding: 0.4rem 1rem;
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.05);
    }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    .stChart {
        background-color: #ffffff;
        padding: 0.5rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="AI-Powered Customer Insight Dashboard")

# ⬆️ Centered Title
st.markdown("""
<div style='text-align: center; font-size: 32px; font-weight: bold;'>📊 Dell Tweets Insight Dashboard</div>
""", unsafe_allow_html=True)

# Load data
DATA_PATH = "data/processed/dell_themes.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# 🔳 Raw data checkbox aligned right
col1, col2 = st.columns([10, 1])
with col2:
    show_data = st.checkbox("Show raw data")

if show_data:
    st.write(df.head())

# 📊 Sentiment Distribution
st.subheader("Sentiment Distribution")
sent_counts = df['sentiment'].value_counts().reset_index()
sent_counts.columns = ['sentiment', 'count']
fig_sentiment = px.bar(
    sent_counts,
    x='count',
    y='sentiment',
    orientation='h',
    color='sentiment',
    color_discrete_map={
        'Positive': "#37df80",
        'Neutral': "#f6e191",
        'Negative': "#f1786b"
    },
    title="Sentiment Distribution",
    height=300
)

st.plotly_chart(fig_sentiment, use_container_width=True)

# 📊 Theme Distribution
st.subheader("Theme Distribution")
theme_counts = df['theme'].value_counts().reset_index()
theme_counts.columns = ['theme', 'count']

view_mode = st.radio("Select chart style for Theme Distribution", ["Gradient", "Solid"], horizontal=True)

# Apply "Hot" color style in both modes
color_style = 'Hot'
bar_colors = theme_counts['theme'].apply(lambda x: "#FF4500") if view_mode == 'Solid' else None

fig_theme_dist = px.bar(
    theme_counts.sort_values(by="count", ascending=True),
    x='count',
    y='theme',
    orientation='h',
    color='theme' if view_mode == 'Solid' else 'count',
    color_discrete_sequence=bar_colors if view_mode == 'Solid' else None,
    color_continuous_scale=color_style if view_mode == 'Gradient' else None,
    title="Theme Distribution",
    height=400
)
st.plotly_chart(fig_theme_dist, use_container_width=True)

# 📊 Theme-wise Sentiment Breakdown
st.subheader("📊 Theme-wise Sentiment Breakdown")
theme_sentiment = df.groupby(['theme', 'sentiment']).size().reset_index(name='count')
fig_theme = px.bar(
    theme_sentiment,
    x='theme',
    y='count',
    color='sentiment',
    barmode='group',
    color_discrete_map={
        'Positive': "#37df80",
        'Neutral': "#f6e191",
        'Negative': "#f1786b"
    },
    title="Theme-wise Sentiment Breakdown"
)
st.plotly_chart(fig_theme, use_container_width=True)

# 🔍 Filters
col_s1, col_s2 = st.columns(2)

with col_s1:
    sent_filter = st.selectbox("Filter tweets by sentiment", options=["All", "Positive", "Negative", "Neutral"])

with col_s2:
    theme_filter = st.selectbox("Filter tweets by theme", options=["All"] + sorted(df['theme'].unique()))

# Apply filters
filtered = df[df['sentiment'] == sent_filter] if sent_filter != "All" else df
filtered = filtered[filtered['theme'] == theme_filter] if theme_filter != "All" else filtered

# Show filtered results
st.write(f"Showing {len(filtered)} tweets")

with st.expander("📂 View Filtered Tweets"):
    st.dataframe(
        filtered[['Text', 'sentiment', 'theme']].reset_index(drop=True),
        use_container_width=True
    )


# 🧠 Suggested Action Items
st.subheader("🛠️ Suggested Action Items Based on Feedback")
actions_by_sentiment = generate_action_items(df)

for sentiment, actions in actions_by_sentiment.items():
    st.markdown(f"### {sentiment} Feedback")

    if actions:
        # Clean action text
        cleaned_actions = [
            (
                action.replace("Promote as testimonial for ", "")
                      .replace("Note neutral feedback for ", "")
                      .replace("Route negative feedback to ", "")
                      .replace("Escalate to support team for ", ""),
                count
            )
            for action, count in actions
        ]

        action_df = pd.DataFrame(cleaned_actions, columns=["Action Area", "Mentions"])

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(action_df.sort_values(by="Mentions", ascending=False))
        with col2:
            fig = px.pie(
                action_df,
                names="Action Area",
                values="Mentions",
                hole=0.4,
                title=f"{sentiment} Feedback Distribution",
                height=600,
                width=600,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)


# 📥 Export Action Items
st.subheader("📥 Export Action Items")
export_sentiment = st.selectbox("Choose sentiment to export", ["Negative", "Positive", "Neutral"])
export_data = actions_by_sentiment.get(export_sentiment, [])

if export_data:
    df_export = pd.DataFrame(export_data, columns=["Action", "Mentions"])
    st.download_button(
        label="Download CSV",
        data=df_export.to_csv(index=False),
        file_name=f"{export_sentiment.lower()}_actions.csv",
        mime='text/csv'
    )

# 💡 Additional Improvement Suggestions (only for Negative)
if export_sentiment == "Negative":
    st.subheader("💡 Improvement Ideas Based on Real Comments")
    suggestion_rows = generate_suggestions_from_comments(df[df['sentiment'] == "Negative"])
    suggestion_df = pd.DataFrame(suggestion_rows, columns=["Theme", "Mentions", "Improvement Suggestion"])

    with st.expander("📂 View Suggestions from Comments", expanded=True):
        st.dataframe(suggestion_df, use_container_width=True)

    st.download_button(
        label="Download Suggestions CSV",
        data=suggestion_df.to_csv(index=False),
        file_name="negative_feedback_suggestions.csv",
        mime="text/csv"
    )

# 🛠 Jira Ticket Creation
st.subheader("🚨 Suggested Jira Tickets Based on Mentions")
AUTO_THRESHOLD = 500
auto_actions = [a for a in actions_by_sentiment["Negative"] if a[1] > AUTO_THRESHOLD]

for i, (action_text, count) in enumerate(auto_actions):
    st.markdown(f"**{i+1}. {action_text}** — {count} mentions")
    if st.button(f"Create Ticket {i+1}", key=f"ticket_{i}"):
        summary = f"Negative Feedback: {action_text[:50]}"
        filtered_df = df[df['sentiment'] == "Negative"]

        keywords = [word.lower() for word in action_text.split() if len(word) > 3]
        matched_rows = filtered_df[filtered_df['clean_text'].apply(
            lambda x: any(k in str(x).lower() for k in keywords)
        )]

        top_texts = matched_rows['Text'].head(3).tolist()
        tweet_samples = "\n".join([f"- {t[:150]}" for t in top_texts]) if top_texts else \
            "No direct feedback matched.\nPlease refer to the dashboard for more details."

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
