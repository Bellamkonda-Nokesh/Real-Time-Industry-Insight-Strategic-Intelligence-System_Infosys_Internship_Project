#!/usr/bin/env python3
"""
Task 2: Load sentiment CSV and send per-article news alerts to Slack
for all sentiment categories: Positive, Neutral, and Negative.
"""

import os
import pandas as pd
import requests
from dotenv import load_dotenv

# ─── Section 1: Configuration ───────────────────────────────────────────────────
load_dotenv()
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# ─── Section 2: Slack Helper ────────────────────────────────────────────────────
def send_slack_alert(message: str):
    """
    Post a simple text message to Slack via incoming webhook.
    If no webhook URL is set, skip without error.
    """
    if not SLACK_WEBHOOK_URL:
        print("⚠️ No Slack webhook URL set; skipping alert.")
        return
    payload = {"text": message}
    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if resp.status_code == 200:
            print("✅ Slack alert sent")
        else:
            print(f"❌ Failed to send Slack alert: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"⚠️ Slack alert error: {e}")

# ─── Section 3: Main Execution ─────────────────────────────────────────────────
def main():
    print("▶️ Task 2: Sending news alerts for all sentiment categories...")
    # 3.1 Load the CSV from Task 1
    df = pd.read_csv("ai_news_sentiment.csv")
    
    # 3.2 Define emoji/label mapping for sentiments
    label_map = {
        "Positive": "🟢 Positive",
        "Neutral":  "🟡 Neutral",
        "Negative": "🔴 Negative"
    }
    
    # 3.3 Iterate over each article and send an alert
    sent_count = 0
    for _, row in df.iterrows():
        sentiment = row.get("sentiment", "Unknown")
        # Map to emoji label if available
        sentiment_label = label_map.get(sentiment, sentiment)
        
        # Build a message including the sentiment label
        msg = (
            f"*News Alert*  \n"
            f"*Sentiment:* {sentiment_label}  \n"
            f"*Source:* {row.get('source')}  \n"
            f"*Title:* {row.get('title')}  \n"
            f"*Description:* {row.get('description')}  \n"
            f"*URL:* {row.get('url')}"
        )
        send_slack_alert(msg)
        sent_count += 1

    # 3.4 Summary output
    print(f"✅ Sent {sent_count} total news alerts to Slack")

if __name__ == "__main__":
    main()