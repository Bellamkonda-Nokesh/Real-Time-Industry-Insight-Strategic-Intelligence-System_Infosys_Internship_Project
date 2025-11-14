#!/usr/bin/env python3
"""
Task 1: Fetch AI news, compute VADER sentiment + score,
save core fields to CSV, and generate three simple plots.
"""

import os
import re
import string
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# ─── Section 1: Configuration ────────────────────────────────────────────────
# Load API keys from a .env file (do not hard-code credentials)
load_dotenv()
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_KEY")
if not NEWS_API_KEY:
    raise RuntimeError("Please set NEWSAPI_KEY in your .env file")

# ─── Section 2: Text Cleaning ───────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Remove URLs, HTML tags, punctuation, and extra whitespace.
    Returns a cleaned string ready for sentiment analysis.
    """
    text = text or ""
    # Strip URLs and HTML
    text = re.sub(r"http\S+|<.*?>|\s+", " ", text)
    # Remove punctuation
    return text.translate(str.maketrans("", "", string.punctuation)).strip()

# ─── Section 3: Fetching Articles ──────────────────────────────────────────────
def fetch_articles(query="stock market", days=7, target=100) -> pd.DataFrame:
    """
    1) Query NewsAPI for up to 100 articles published in the last `days`.
    2) If fewer than `target` articles, backfill from GNews.
    3) Deduplicate by URL and return DataFrame with core columns.
    """
    # 3.1 Build NewsAPI request
    since = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"
    params = {
        "q": query,
        "from": since,
        "language": "en",
        "pageSize": 100,
        "apiKey": NEWS_API_KEY
    }
    resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10).json()
    articles = resp.get("articles", [])

    # 3.2 Fallback to GNews if needed
    if len(articles) < target and GNEWS_API_KEY:
        left = target - len(articles)
        gresp = requests.get(
            "https://gnews.io/api/v4/search",
            params={"q": query, "lang": "en", "max": left, "token": GNEWS_API_KEY},
            timeout=10
        ).json()
        articles += gresp.get("articles", [])

    # 3.3 Extract core fields and dedupe by URL
    seen = set()
    records = []
    for art in articles:
        url = art.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        records.append({
            "source": art.get("source", {}).get("name", "Unknown"),
            "title": (art.get("title") or "").strip(),
            "description": (art.get("description") or "").strip(),
            "url": url,
            "published_at": art.get("publishedAt", "")
        })
    return pd.DataFrame(records)

# ─── Section 4: Sentiment Analysis ─────────────────────────────────────────────
def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Use NLTK VADER to compute:
      - `score`: compound sentiment score
      - `sentiment`: categorical label based on score thresholds
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiments, scores = [], []

    for _, row in df.iterrows():
        text = clean_text(row["title"] + " " + row["description"])
        s = analyzer.polarity_scores(text)["compound"]
        scores.append(s)
        # Assign label
        if s >= 0.05:
            sentiments.append("Positive")
        elif s <= -0.05:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")

    df["sentiment"] = sentiments
    df["score"] = scores
    return df

# ─── Section 5: Save & Visualize ───────────────────────────────────────────────
def save_results_and_plots(df: pd.DataFrame):
    """
    1) Save a CSV with core columns:
         source, title, description, url, published_at, sentiment, score
    2) Generate three plots:
         a) Sentiment counts bar chart
         b) Average score over time line plot
         c) Word cloud of all headlines
    """
    # 5.1 Save CSV
    df.to_csv(
        "ai_news_sentiment.csv",
        index=False,
        columns=[
            "source", "title", "description",
            "url", "published_at", "sentiment", "score"
        ]
    )
    print(f"💾 Saved ai_news_sentiment.csv ({len(df)} rows)")

    # 5.2 Plot sentiment counts
    order = ["Positive", "Neutral", "Negative"]
    counts = df["sentiment"].value_counts().reindex(order).fillna(0)
    counts.plot.bar(color=["#4caf50", "#ffeb3b", "#f44336"])
    plt.title("Sentiment Counts")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig("sentiment_counts.png")
    plt.clf()

    # 5.3 Plot average score over time
    df["date"] = pd.to_datetime(df["published_at"], errors="coerce").dt.date
    trend = df.dropna(subset=["date"]).groupby("date")["score"].mean()
    trend.plot(marker="o", color="tab:blue")
    plt.title("Avg Sentiment Score Over Time")
    plt.xlabel("Date")
    plt.ylabel("Mean Compound Score")
    plt.tight_layout()
    plt.savefig("sentiment_trend.png")
    plt.clf()

    # 5.4 Word cloud of headlines
    all_text = " ".join(df["title"].dropna())
    wc = WordCloud(width=600, height=300, background_color="white").generate(all_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Headlines")
    plt.tight_layout()
    plt.savefig("headline_wordcloud.png")
    plt.clf()
    print("✅ Plots saved: sentiment_counts.png, sentiment_trend.png, headline_wordcloud.png")

# ─── Section 6: Main Execution ─────────────────────────────────────────────────
def main():
    print("▶️ Task 1: Fetching & analyzing news...")
    df = fetch_articles()
    print(f"✔️ Retrieved {len(df)} articles")
    df = analyze_sentiment(df)
    save_results_and_plots(df)

if __name__ == "__main__":
    main()