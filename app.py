import os
import time
import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
import streamlit as st

# =========================
# Config
# =========================
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
NEWS_FUNCTION = "NEWS_SENTIMENT"  # AlphaVantage news endpoint
PRICE_FUNCTION = "TIME_SERIES_DAILY_ADJUSTED"
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"

# =========================
# Utilities
# =========================
@st.cache_data(ttl=3600)
def fetch_sp500_tickers() -> pd.DataFrame:
    """Fetch S&P 500 tickers from public CSV repository.
    Uses pandas.read_csv for direct download, no web scraping.
    """
    try:
        df = pd.read_csv(SP500_CSV_URL)
        # Standardize column names
        df = df.rename(columns={"Symbol": "ticker", "Name": "name"})
        # Ensure we have the required columns
        if "ticker" not in df.columns or "name" not in df.columns:
            df = df.rename(columns={df.columns[0]: "ticker", df.columns[1]: "name"})
        # Some tickers have "." which AlphaVantage expects as "-" (e.g., BRK.B -> BRK-B)
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        return df[["ticker", "name"]]
    except Exception as e:
        # Fallback: minimal local list for demo reliability
        fallback = pd.DataFrame(
            {
                "ticker": [
                    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "JPM", "V",
                    "PG", "UNH", "XOM", "JNJ", "HD", "MA", "KO", "PEP", "AVGO", "COST"
                ],
                "name": [
                    "Apple Inc.", "Microsoft Corporation", "Amazon.com, Inc.", "NVIDIA Corporation",
                    "Alphabet Inc. (Class A)", "Meta Platforms, Inc.", "Tesla, Inc.", "Berkshire Hathaway Inc. (Class B)",
                    "JPMorgan Chase & Co.", "Visa Inc.", "The Procter & Gamble Company", "UnitedHealth Group Incorporated",
                    "Exxon Mobil Corporation", "Johnson & Johnson", "The Home Depot, Inc.", "Mastercard Incorporated",
                    "The Coca-Cola Company", "PepsiCo, Inc.", "Broadcom Inc.", "Costco Wholesale Corporation"
                ],
            }
        )
        st.warning(
            "Could not fetch S&P 500 constituents from CSV. "
            "Using a local fallback list for demo purposes. "
            f"Reason: {type(e).__name__}: {e}"
        )
        return fallback

def _av_get(params: dict, retry: int = 2):
    base = "https://www.alphavantage.co/query"
    params = {**params, "apikey": ALPHAVANTAGE_API_KEY}
    for i in range(retry + 1):
        r = requests.get(base, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if "Note" in data or "Information" in data:
                # Rate limit notice — backoff then retry
                time.sleep(15)
                continue
            return data
        time.sleep(2)
    return {}


def fetch_price_summary(ticker: str) -> dict:
    """Get latest close price and daily change percent from AlphaVantage."""
    if not ALPHAVANTAGE_API_KEY:
        return {"error": "Missing ALPHAVANTAGE_API_KEY"}
    data = _av_get({"function": PRICE_FUNCTION, "symbol": ticker, "outputsize": "compact"})
    if not data or "Time Series (Daily)" not in data:
        return {}
    ts = data["Time Series (Daily)"]
    # Get last two trading days to compute change
    dates = sorted(ts.keys(), reverse=True)
    if not dates:
        return {}
    latest = ts[dates[0]]
    prev_close = float(ts[dates[1]].get("4. close", 0)) if len(dates) > 1 else None
    last_close = float(latest.get("4. close", 0))
    pct_change = None
    if prev_close and prev_close != 0:
        pct_change = (last_close - prev_close) / prev_close * 100
    return {"close": last_close, "pct_change": pct_change}


def fetch_news_and_sentiment(ticker: str) -> dict:
    """Fetch news and compute sentiment score for a ticker."""
    if not ALPHAVANTAGE_API_KEY:
        return {"sentiment": 0.0, "headlines": []}
    data = _av_get({"function": NEWS_FUNCTION, "tickers": ticker, "limit": 5})
    articles = data.get("feed", [])
    sentiment_scores = []
    headlines = []
    for article in articles[:5]:
        title = article.get("title", "")
        if title:
            blob = TextBlob(title)
            sentiment_scores.append(blob.sentiment.polarity)
            headlines.append(title)
    overall_sentiment = float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
    # pick one key headline if available
    key_headline = headlines[0] if headlines else ""
    return {"sentiment": overall_sentiment, "headlines": headlines, "key_headline": key_headline}


def expert_signal(price: float, sentiment: float) -> tuple:
    """Combine price and sentiment for a trading signal."""
    if not price or price < 0.01:
        return "Unknown", "#808080"
    if sentiment > 0.5:
        return "Strong Buy", "#0b8457"
    elif sentiment > 0.1:
        return "Buy", "#1ac956"
    elif sentiment < -0.5:
        return "Strong Sell", "#d0021b"
    elif sentiment < -0.1:
        return "Sell", "#ff6b6b"
    else:
        return "Hold", "#f59e0b"


def reason_from_signal(signal: str, sentiment: float, headline: str) -> str:
    """Generate an exact reason string based on expert logic and sentiment/headline."""
    sentiment_note = f"sentiment {sentiment:+.2f}"
    logic = {
        "Strong Buy": f"High positive news momentum ({sentiment_note}) supporting upside.",
        "Buy": f"Positive tilt in news flow ({sentiment_note}) suggests moderate upside.",
        "Hold": f"Neutral news tone ({sentiment_note}); no strong edge.",
        "Sell": f"Negative tilt in news flow ({sentiment_note}) suggests downside risk.",
        "Strong Sell": f"High negative news momentum ({sentiment_note}) with elevated risk.",
        "Unknown": "Insufficient price data to form view.",
    }.get(signal, "")
    headline_note = f" Key headline: {headline}" if headline else ""
    return logic + headline_note

# =========================
# Main Streamlit App
# =========================
st.set_page_config(
    page_title="S&P 500 Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("S&P 500 Trading Bot")
st.write("AI-driven market screener powered by AlphaVantage + TextBlob NLP.")

# === Sidebar Controls ===
with st.sidebar:
    st.header("🍐 Configuration")
    max_tickers = st.slider(
        "Number of tickers to scan",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="Higher = slower but broader coverage.",
    )
    show_news = st.checkbox("Show headlines & sentiment", value=True)

st.divider()

# === Main Content ===
if st.button("🔍 Scan Market", use_container_width=True):
    sp500_df = fetch_sp500_tickers()
    max_tickers = min(max_tickers, len(sp500_df))

    rows = []
    scanned = 0
    prog = st.progress(0, text="Scanning...")

    for _, row in sp500_df.iloc[:max_tickers].iterrows():
        ticker = row["ticker"]
        name = row["name"]
        price = fetch_price_summary(ticker)
        news = fetch_news_and_sentiment(ticker)
        sentiment = news.get("sentiment", 0.0)
        label, _ = expert_signal(price.get("close", 0), sentiment)

        rows.append(
            {
                "Ticker": ticker,
                "Company": name,
                "Price": price.get("close"),
                "Change%": price.get("pct_change"),
                "Sentiment": round(sentiment, 2) if sentiment is not None else None,
                "Signal": label,
                "Headline": news.get("key_headline", ""),
            }
        )
        scanned += 1
        prog.progress(min(scanned / max_tickers, 1.0))
        time.sleep(0.2)  # gentle pacing for API

    if not rows:
        st.warning("No results to display yet. Check API key or increase max tickers.")
    else:
        df = pd.DataFrame(rows)

        # Rank signals: Strong Buy > Buy > Hold > Sell > Strong Sell
        signal_rank = {
            "Strong Buy": 5,
            "Buy": 4,
            "Hold": 3,
            "Sell": 2,
            "Strong Sell": 1,
        }
        df["_rank"] = df["Signal"].map(signal_rank).fillna(0)

        # Split into buy and sell candidates
        buys = df[df["Signal"].isin(["Strong Buy", "Buy"])].copy()
        sells = df[df["Signal"].isin(["Strong Sell", "Sell"])].copy()

        # Sort buys: rank desc, sentiment desc, Change% desc
        buys = buys.sort_values(["_rank", "Sentiment", "Change%"], ascending=[False, False, False]).head(10)
        # Sort sells: rank asc (since lower is worse), sentiment asc, Change% asc
        sells = sells.sort_values(["_rank", "Sentiment", "Change%"], ascending=[True, True, True]).head(10)

        # Prepare exact reason text
        def add_reason(frame: pd.DataFrame) -> pd.DataFrame:
            reasons = []
            for _, r in frame.iterrows():
                reasons.append(
                    reason_from_signal(
                        signal=r["Signal"],
                        sentiment=float(r["Sentiment"]) if pd.notna(r["Sentiment"]) else 0.0,
                        headline=r.get("Headline", ""),
                    )
                )
            frame["Exact Reason"] = reasons
            return frame

        buys = add_reason(buys)
        sells = add_reason(sells)

        # Display
        st.subheader("Top 10 Buy Candidates")
        if buys.empty:
            st.info("No Buy candidates in the scanned set.")
        else:
            cols = ["Ticker", "Company", "Signal", "Price", "Change%", "Sentiment", "Headline", "Exact Reason"]
            st.dataframe(buys[cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        st.subheader("Top 10 Sell Candidates")
        if sells.empty:
            st.info("No Sell candidates in the scanned set.")
        else:
            cols = ["Ticker", "Company", "Signal", "Price", "Change%", "Sentiment", "Headline", "Exact Reason"]
            st.dataframe(sells[cols].reset_index(drop=True), use_container_width=True, hide_index=True)

        if show_news:
            st.divider()
            st.subheader("Key Headlines (Top picks)")
            top_headlines = pd.concat([buys.head(5), sells.head(5)], ignore_index=True)
            for _, r in top_headlines.iterrows():
                if r.get("Headline"):
                    st.markdown(f"- [{r['Ticker']}] {r['Company']} — {r['Signal']}: {r['Headline']}")

st.caption("Tip: Set ALPHAVANTAGE_API_KEY env var for live data. No handpicked tickers — full S&P 500 auto-scan.")
