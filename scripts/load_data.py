#load data.py
import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records.")
    return df

if __name__ == "__main__":
    df = load_data("data/dell_tweets.csv")
