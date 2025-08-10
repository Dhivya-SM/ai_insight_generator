import praw
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    username=os.getenv("REDDIT_USERNAME"),
    password=os.getenv("REDDIT_PASSWORD"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

OUTPUT_PATH = "data/raw/reddit_stream.csv"

def stream_reddit_comments(subreddit_name="technology", limit=10):
    print(f"🔍 Streaming from subreddit: {subreddit_name}")
    subreddit = reddit.subreddit(subreddit_name)

    data = []
    for post in subreddit.new(limit=limit):
        created_dt = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')
        
        text_content = (post.title or "") + " " + (post.selftext or "")
        
        data.append({
            "Datetime": created_dt,
            "Tweet Id": post.id,
            "Text": text_content.strip(),
            "Username": str(post.author)
        })

    df = pd.DataFrame(data)
    
    # Save in Kaggle-compatible format
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved {len(df)} Reddit posts to {OUTPUT_PATH}")
    return df

if __name__ == "__main__":
    stream_reddit_comments()
