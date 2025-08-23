import praw
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# 🔹 Reddit API setup (shared across functions)
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

OUTPUT_PATH = "data/raw/reddit_stream.csv"

# Default subreddits if keyword search is not used
SUBREDDITS = ["technology", "worldnews", "science", "dataisbeautiful", "AskReddit"]

def stream_reddit_comments(keyword=None, limit_per_subreddit=1000):
    """
    Fetch Reddit posts and comments.
    If `keyword` is provided → site-wide search.
    If not → fetch latest from default subreddits.
    """
    all_data = []
    total_fetched = 0

    if keyword:
        print(f"🔍 Searching site-wide for '{keyword}'...")
        posts = reddit.subreddit("all").search(keyword, limit=limit_per_subreddit)
        subreddits_to_check = [(None, posts)]  # (subreddit_name, generator)
    else:
        subreddits_to_check = []
        for sub in SUBREDDITS:
            subreddit = reddit.subreddit(sub)
            posts = subreddit.new(limit=limit_per_subreddit)
            subreddits_to_check.append((sub, posts))

    # Process each subreddit / search result
    for sub_name, posts in subreddits_to_check:
        if sub_name:
            print(f"🔍 Fetching from r/{sub_name}")

        for post in posts:
            created_dt = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
            text_content = (post.title or "") + " " + (post.selftext or "")

            all_data.append({
                "Datetime": created_dt,
                "PostId": post.id,
                "Text": text_content.strip(),
                "Username": str(post.author)
            })
            total_fetched += 1

            # Fetch top-level comments
            post.comments.replace_more(limit=0)
            for comment in post.comments[:5]:  # Limit comments per post
                comment_dt = datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S')
                all_data.append({
                    "Datetime": comment_dt,
                    "PostId": f"{post.id}_{comment.id}",
                    "Text": comment.body.strip(),
                    "Username": str(comment.author)
                })
                total_fetched += 1

            time.sleep(0.2)  # Avoid hitting API rate limits

    # Save results
    df = pd.DataFrame(all_data)
    df.drop_duplicates(subset=["PostId"], inplace=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"✅ Saved {len(df)} records to {OUTPUT_PATH}")
    return df

if __name__ == "__main__":
    # Example: Fetch Dell-specific data
    stream_reddit_comments(keyword="dell", limit_per_subreddit=500)
