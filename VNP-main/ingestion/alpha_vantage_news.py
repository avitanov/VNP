import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables (.env should contain ALPHA_VANTAGE_API_KEY)
load_dotenv()

API_URL = "https://www.alphavantage.co/query"


def fetch_latest_news(
    symbol: str,
    api_key: str,
    topics: list[str] | None = None,
    limit: int = 1000,
    sort: str = "LATEST",
) -> pd.DataFrame:
    """Fetch the latest `limit` news‑sentiment articles for one symbol."""

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "apikey": api_key,
        "limit": limit,
        "sort": sort,  # LATEST, EARLIEST, or RELEVANCE
    }
    if topics:
        params["topics"] = ",".join(topics)

    response = requests.get(API_URL, params=params)
    response.raise_for_status()
    data = response.json()

    feed = data.get("feed", [])
    if not feed:
        return pd.DataFrame()

    df = pd.DataFrame(feed)
    df["time_published"] = pd.to_datetime(df["time_published"])
    df = df.set_index("time_published").sort_index()
    df["symbol"] = symbol
    return df


if __name__ == "__main__":
    # ------------ Config -------------
    symbols = ["AAPL"]            # list of tickers
    topics = None                 # e.g. ["technology", "earnings"]
    limit = 1000                  # max number of articles
    out_dir = "../dataset/raw"    # output directory
    # ---------------------------------

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing ALPHA_VANTAGE_API_KEY environment variable")

    os.makedirs(out_dir, exist_ok=True)

    for sym in symbols:
        print(f"Fetching latest {limit} articles for {sym} …")
        df_news = fetch_latest_news(sym, api_key, topics, limit, sort="LATEST")
        if df_news.empty:
            print(f"No news returned for {sym}.")
            continue
        csv_path = os.path.join(out_dir, f"{sym}_NEWS_LATEST.csv")
        # Append if the file exists; otherwise write with header
        mode = "a" if os.path.isfile(csv_path) else "w"
        header = not os.path.isfile(csv_path)
        df_news.to_csv(csv_path, mode=mode, header=header)
        print(f"Saved {len(df_news)} articles to {csv_path}")
