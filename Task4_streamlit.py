import os
import requests
import time
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
import nltk

# Download NLTK resources up front
for pkg in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

load_dotenv(".env")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
if not NEWSAPI_KEY:
    st.error("Missing NEWSAPI_KEY in .env!")

# ----- Card helpers -----
def kpi_card(title, value, sub=None, color="#3fa", bg="#fff"):
    return f"""
    <div style='border-radius:16px;box-shadow:0 3px 14px #ddd;background:{bg};padding:18px 24px;margin-bottom:12px;'>
        <div style='font-size:2em;font-weight:600;color:{color};margin-bottom:0.2em;'>{title}</div>
        <div style='font-size:2.6em;font-weight:800;color:#222'>{value}</div>
        <div style='font-size:1em;color:#555'>{sub or ""}</div>
    </div>
    """

def info_card(title, body, bg="#f6f8fe"):
    return f"""
    <div style='border-radius:12px;box-shadow:0 3px 10px #eee;background:{bg};padding:16px 18px;margin-bottom:10px;'>
        <div style='font-size:1.5em;font-weight:700;line-height:1.2;'>{title}</div>
        <div style='font-size:1.07em;padding-top:6px;color:#444;'>{body}</div>
    </div>
    """

def section_title(text, emoj=""):
    st.markdown(f"<h2 style='color:#2747be;'>{emoj} {text}</h2>", unsafe_allow_html=True)

# ----- Data functions -----
def fetch_news(query="AI market OR stock OR finance OR tech OR business OR economy", max_articles=150):
    from_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    articles, page, seen = [], 1, set()
    while len(articles) < max_articles:
        params = {
            "q": query,
            "language": "en",
            "pageSize": min(100, max_articles - len(articles)),
            "from": from_date,
            "to": to_date,
            "page": page,
            "apiKey": NEWSAPI_KEY
        }
        try:
            r = requests.get(url, params=params, timeout=10).json()
            batch = r.get("articles", []) or []
            for a in batch:
                url_a = a.get("url", "")
                if url_a and url_a not in seen:
                    articles.append({
                        "source": a.get("source", {}).get("name", ""),
                        "title": a.get("title", ""),
                        "description": a.get("description", ""),
                        "url": url_a,
                        "publishedAt": a.get("publishedAt", "")
                    })
                    seen.add(url_a)
            if len(batch) < params["pageSize"]:
                break
        except:
            break
        page += 1
        time.sleep(0.3)
    return articles[:max_articles]

def sentiment_call(title):
    pol = TextBlob(str(title)).sentiment.polarity
    score = round(np.clip(pol * 10, -5, 5), 2)
    if score > 0.1:
        return "Positive", score
    elif score < -0.1:
        return "Negative", score
    else:
        return "Neutral", score

def batch_sentiment(texts):
    return zip(*[sentiment_call(t) for t in texts])

def is_score_ok(series):
    return series.min() > -8 and series.max() < 8 and abs(series.mean()) < 3

# ----- Streamlit setup and sidebar -----
st.set_page_config(layout="wide", page_title="AI Market Sentiment Analytics")
with st.sidebar:
    st.markdown("<h1 style='font-size:2em;'>📊 Dashboard Nav</h1>", unsafe_allow_html=True)
    sel_page = st.radio(
        "Page:",
        [
            "🚦 Market Overview",
            "🔎 Fetch & Analyze",
            "📈 Sentiment Dashboard",
            "📊 Forecast & Alerts",
            "💡 Innovation Insights"
        ],
        index=0
    )
    st.markdown("---")
    st.caption("Powered by Streamlit, NewsAPI, Plotly, Prophet, NLP")

# ----- Section: Market Overview -----
if sel_page == "🚦 Market Overview":
    section_title("Market Sentiment Analytics", "📊")
    st.markdown(
        "<div style='padding-bottom:12px;'><b>Welcome!</b> This premium dashboard extracts, visualizes, and tracks news market sentiment, trends, and forecasts using state-of-the-art analytics and vibrant UI.</div>",
        unsafe_allow_html=True
    )
    if "df" in st.session_state:
        df = st.session_state["df"]
        col1, col2, col3 = st.columns([2, 1.4, 1.5])
        pos = (df["score"] > 0.1).sum()
        neg = (df["score"] < -0.1).sum()
        neu = ((df["score"] <= 0.1) & (df["score"] >= -0.1)).sum()
        with col1:
            st.markdown(
                kpi_card("Total Articles", len(df), sub="Sources analyzed last 30 days.", color="#3453dd"),
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                kpi_card("Positive News", pos, sub="🔵 Sentiment > 0.1", color="#16c659"),
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                kpi_card("Negative News", neg, sub="🔴 Sentiment < -0.1", color="#fa4a35"),
                unsafe_allow_html=True
            )
        top_sources = df["source"].value_counts().head(5)
        st.markdown(
            info_card("Top News Sources", "<br>".join(f"<b>{k}</b>: {v}" for k, v in top_sources.items()), bg="#eef3fb"),
            unsafe_allow_html=True
        )
    else:
        st.info("No market data loaded. Go to Fetch & Analyze to start.")

# ----- Section: Fetch & Analyze -----
if sel_page == "🔎 Fetch & Analyze":
    section_title("News & Sentiment Fetcher", "🔎")
    user_query = st.text_input(
        "Enter search topic:",
        "AI market OR stock OR finance OR economy OR tech OR business",
        help="Broader queries get more timeline coverage!"
    )
    max_articles = st.slider("Max articles to fetch:", min_value=30, max_value=200, value=100)
    if st.button("Fetch & Analyze News"):
        with st.spinner("Fetching articles & calling sentiment analysis..."):
            arts = fetch_news(user_query, max_articles=max_articles)
            texts = [a["title"] for a in arts if a["title"]]
            sentiments, scores = batch_sentiment(texts)
            for i in range(len(arts)):
                arts[i]["sentiment"] = sentiments[i]
                arts[i]["score"] = scores[i]
            df = pd.DataFrame(arts)
            df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
            df = df.dropna(subset=["publishedAt"])
            st.session_state["df"] = df
            st.success(f"Fetched {len(df)} articles from {df['publishedAt'].dt.date.nunique()} unique days.")
            dates_str = ", ".join(str(d) for d in sorted(df["publishedAt"].dt.date.unique()))
            st.markdown(
                info_card("Fetched Article Dates", dates_str, bg="#eaf7ef"),
                unsafe_allow_html=True
            )
            st.dataframe(df[["title", "sentiment", "score", "publishedAt", "url"]])
            st.download_button("Download analyzed CSV", df.to_csv(index=False), file_name="market_sentiment.csv")
            from wordcloud import WordCloud
            st.subheader("☁️ Word Cloud")
            try:
                all_titles = " ".join(df["title"].dropna())
                wc = WordCloud(width=540, height=240, background_color="#fafbff").generate(all_titles)
                st.image(wc.to_array(), caption="Headlines WordCloud", use_column_width=True)
            except:
                st.info("Word cloud unavailable (try larger dataset).")

# ----- Section: Sentiment Dashboard -----
if sel_page == "📈 Sentiment Dashboard":
    section_title("Sentiment Dashboard", "📈")
    df = st.session_state.get("df", None)
    if df is None:
        st.info("No data loaded. Run Fetch & Analyze first.")
    else:
        pos = (df["score"] > 0.1).sum()
        neg = (df["score"] < -0.1).sum()
        neu = ((df["score"] <= 0.1) & (df["score"] >= -0.1)).sum()
        sentiment_text = f"<b>Positive:</b> <span style='color:#16c659'>{pos}</span> &nbsp; <b>Neutral:</b> <span style='color:#ffc900'>{neu}</span> &nbsp; <b>Negative:</b> <span style='color:#fa4a35'>{neg}</span>"
        st.markdown(
            info_card("Sentiment Distribution", sentiment_text, bg="#ffe"),
            unsafe_allow_html=True
        )
        pie_fig = px.pie(
            values=[pos, neu, neg],
            names=["Positive", "Neutral", "Negative"],
            title="Sentiment Distribution",
            color_discrete_sequence=["#16c659", "#ffc900", "#fa4a35"]
        )
        st.plotly_chart(pie_fig, use_container_width=True)
        avg_by_source = df.groupby("source")["score"].mean().sort_values()
        src_fig = px.bar(
            x=avg_by_source.index,
            y=avg_by_source.values,
            labels={"x": "Source", "y": "Avg Sentiment"},
            title="Average Sentiment by News Source",
            color_discrete_sequence=["#3453dd"]
        )
        st.plotly_chart(src_fig, use_container_width=True)
        df["date_only"] = pd.to_datetime(df["publishedAt"], errors="coerce").dt.date
        score_over_time = df.groupby("date_only")["score"].mean().reset_index()
        if len(score_over_time) < 2:
            st.warning("Not enough unique days for timeline chart. Try broader query/more data.")
        else:
            time_fig = px.line(
                x=score_over_time["date_only"],
                y=score_over_time["score"],
                title="Mean Sentiment Score Over Time",
                markers=True,
                color_discrete_sequence=["#16c659"]
            )
            st.plotly_chart(time_fig, use_container_width=True)

# ----- Section: Forecast & Alerts -----
if sel_page == "📊 Forecast & Alerts":
    section_title("Forecast & Alerts (Prophet ML)", "📊")
    df = st.session_state.get("df", None)
    if df is None or len(df) == 0:
        st.info("No data loaded. Run Fetch & Analyze first.")
    else:
        clean_df = df.dropna(subset=["score", "publishedAt"]).copy()
        clean_df["score"] = np.clip(pd.to_numeric(clean_df["score"], errors="coerce"), -5.0, 5.0)
        clean_df = clean_df.dropna(subset=["publishedAt", "score"])
        daily_df = clean_df.resample("D", on="publishedAt")["score"].mean().reset_index()
        daily_df["y"] = daily_df["score"].rolling(window=3, min_periods=1).mean()
        daily_df = daily_df.rename(columns={"publishedAt": "ds"})
        daily_df["ds"] = pd.to_datetime(daily_df["ds"]).dt.tz_localize(None)
        valid_daily_df = daily_df.dropna(subset=["y"])
        unique_days = valid_daily_df["ds"].dt.date.nunique()
        st.markdown(
            kpi_card("Unique Daily Points", unique_days, "Days for reliable ML trend", color="#3453dd", bg="#f6f8fe"),
            unsafe_allow_html=True
        )
        if unique_days < 2:
            st.error("Not enough daily points for ML forecast.")
        else:
            from prophet import Prophet
            model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            model.fit(valid_daily_df[["ds", "y"]])
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)
            forecast_trend = round(forecast["yhat"].iloc[-1], 2)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(
                    kpi_card("Next 7d Forecast", forecast_trend, "Prophet ML", color="#16c659"),
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    kpi_card("Mean Sentiment", round(clean_df["score"].mean(), 2)),
                    unsafe_allow_html=True
                )
            with col3:
                st.markdown(
                    kpi_card("Sample Size", len(clean_df)),
                    unsafe_allow_html=True
                )
            fig = px.line(
                x=valid_daily_df["ds"],
                y=valid_daily_df["y"],
                title="ML Forecast & Actuals",
                color_discrete_sequence=["#3453dd"]
            )
            fig.add_scatter(
                x=forecast["ds"],
                y=forecast["yhat"],
                mode="lines",
                name="Forecast",
                line=dict(color="#fa4a35")
            )
            st.plotly_chart(fig, use_container_width=True)
            # Alerts
            pos_alerts = clean_df[clean_df["score"] > 0.15]
            neg_alerts = clean_df[clean_df["score"] < -0.15]
            pos_alerts_text = "<br>".join(f"🟢 {t}" for t in pos_alerts["title"].head(6))
            neg_alerts_text = "<br>".join(f"🔴 {t}" for t in neg_alerts["title"].head(6))
            st.markdown(
                info_card("Top Positive Alerts", pos_alerts_text, bg="#dff"),
                unsafe_allow_html=True
            )
            st.markdown(
                info_card("Top Negative Alerts", neg_alerts_text, bg="#ffdede"),
                unsafe_allow_html=True
            )

# ----- Section: Innovation Insights -----
if sel_page == "💡 Innovation Insights":
    section_title("Innovation Insights & Summary (LLM)", "💡")
    df = st.session_state.get("df", None)
    if df is None or len(df) < 8:
        st.info("Please fetch and analyze data (at least 8 articles) first.")
    else:
        clean_df = df.dropna(subset=["score", "publishedAt"]).copy()
        clean_df["score"] = pd.to_numeric(clean_df["score"], errors="coerce")
        clean_df = clean_df.dropna(subset=["score"])
        clean_df["publishedAt"] = pd.to_datetime(clean_df["publishedAt"], errors="coerce")
        clean_df = clean_df.dropna(subset=["publishedAt"])
        daily_df = clean_df.resample("D", on="publishedAt")["score"].mean().reset_index()
        daily_df["y"] = daily_df["score"].rolling(window=3, min_periods=1).mean()
        daily_df = daily_df.rename(columns={"publishedAt": "ds"})
        daily_df["ds"] = pd.to_datetime(daily_df["ds"]).dt.tz_localize(None)
        valid_daily_df = daily_df.dropna(subset=["y"])
        unique_days = valid_daily_df["ds"].dt.date.nunique()
        st.markdown(
            kpi_card("Unique Days in Market", unique_days, color="#3453dd"),
            unsafe_allow_html=True
        )
        if unique_days < 2:
            st.error("❌ Not enough unique days for LLM summary.")
        else:
            from prophet import Prophet
            model = Prophet(daily_seasonality=True, weekly_seasonality=True)
            model.fit(valid_daily_df[["ds", "y"]])
            future = model.make_future_dataframe(periods=7)
            forecast = model.predict(future)
            avg_prev = round(valid_daily_df["y"].mean(), 3)
            avg_fore = round(forecast["yhat"].tail(7).mean(), 3)
            trend_delta = round(avg_fore - avg_prev, 3)
            # Keyword summary
            all_titles = " ".join(clean_df["title"].dropna().astype(str))
            stopwords = set(nltk.corpus.stopwords.words("english"))
            tokens = nltk.word_tokenize(all_titles)
            keywords = [w for w in tokens if w.isalpha() and w.lower() not in stopwords]
            top_keywords = [w for w, _ in Counter(keywords).most_common(7)]
            pos_headline = clean_df.loc[clean_df["score"].idxmax()]["title"] if not clean_df.empty else "-"
            neg_headline = clean_df.loc[clean_df["score"].idxmin()]["title"] if not clean_df.empty else "-"
            pos_score = clean_df["score"].max() if not clean_df.empty else "-"
            neg_score = clean_df["score"].min() if not clean_df.empty else "-"
            pos_src = clean_df.loc[clean_df["score"].idxmax()]["source"] if not clean_df.empty else "-"
            neg_src = clean_df.loc[clean_df["score"].idxmin()]["source"] if not clean_df.empty else "-"
            pos = (clean_df["score"] > 0.1).sum()
            neg = (clean_df["score"] < -0.1).sum()
            neu = ((clean_df["score"] <= 0.1) & (clean_df["score"] >= -0.1)).sum()
            keywords_text = ", ".join(top_keywords)
            summary_body = f"""
            <b>Forecast sentiment change (next 7 days): <span style='color:#fa4a35'>{trend_delta}</span></b><br>
            Most frequent keywords: <span style='color:#16c659'>{keywords_text}</span><br>
            <b>Strongest Positive:</b> <i>{pos_headline}</i> (Score: {pos_score}, Source: {pos_src})<br>
            <b>Strongest Negative:</b> <i>{neg_headline}</i> (Score: {neg_score}, Source: {neg_src})<br>
            <b>Sentiment breakdown:</b><br>
            &nbsp;&nbsp;Positive: <span style='color:#16c659'>{pos}</span><br>
            &nbsp;&nbsp;Neutral: <span style='color:#ffc900'>{neu}</span><br>
            &nbsp;&nbsp;Negative: <span style='color:#fa4a35'>{neg}</span><br>
            <b>Headline themes:</b> <span style='color:#2747be'>{keywords_text}</span>
            <hr>
            <b>Last Week Mean Sentiment:</b> <span style='color:#16c659'>{avg_prev}</span>
            &nbsp;|&nbsp; <b>Forecast Next Week:</b> <span style='color:#fa4a35'>{avg_fore}</span>
            """
            st.markdown(
                info_card("LLM Summary", summary_body, bg="#eef3fb"),
                unsafe_allow_html=True
            )

# ----- Theme -----
st.markdown(
    """
    <style>
    .reportview-container {background:#fafbff;}
    .main, .block-container {background:#fafbff;}
    </style>
    """,
    unsafe_allow_html=True
)