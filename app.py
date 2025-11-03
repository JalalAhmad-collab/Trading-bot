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
    """Get latest OHLC and daily change from AlphaVantage."""
    if not ALPHAVANTAGE_API_KEY:
        return {"error": "Missing ALPHAVANTAGE_API_KEY"}
    data = _av_get(
        {"function": PRICE_FUNCTION, "symbol": ticker, "outputsize": "compact"}
    )
    if not data or "Time Series (Daily)" not in data:
        return {}
    ts = data["Time Series (Daily)"]
    latest = next(iter(ts.values()))
    return {
        "close": float(latest.get("4. close", 0)),
        "pct_change": float(data.get("Meta Data", {}).get("x1. previous close", 0)) or 0,
    }
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
            headlines.append(title[:60])
    overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
    return {"sentiment": overall_sentiment, "headlines": headlines}
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
        label, color = expert_signal(price.get("close", 0), news.get("sentiment", 0.0))
        rows.append(
            {
                "Ticker": ticker,
                "Company": name,
                "Price": price.get("close"),
                "Change%": price.get("pct_change"),
                "Sentiment": round(news.get("sentiment", 0.0), 2),
                "Signal": label,
                "_color": color,
                "Headlines": " • ".join(news.get("headlines", [])) if show_news else "",
            }
        )
        scanned += 1
        prog.progress(min(scanned / max_tickers, 1.0))
        time.sleep(0.2)  # gentle pacing for API
    if not rows:
        st.warning("No results to display yet. Check API key or increase max tickers.")
    else:
        df = pd.DataFrame(rows)
        # Sort: Strong signals first
        signal_rank = {
            "Strong Buy": 5,
            "Buy": 4,
            "Hold": 3,
            "Sell": 2,
            "Strong Sell": 1,
        }
        df["_rank"] = df["Signal"].map(signal_rank).fillna(0)
        df = df.sort_values(["_rank", "Change%"], ascending=[False, False])
        # Professional, compact table styling
        def color_signal(val, row):
            # Check if '_color' exists in the row before accessing it
            if '_color' not in row.index:
                return ""
            return f"background-color: {row['_color']}; color: white; font-weight: 600; text-align:center; border-radius:6px;"
        def style_table(dataframe: pd.DataFrame):
            # Check which columns exist before dropping them
            cols_to_drop = [c for c in ["_color", "_rank"] if c in dataframe.columns]
            styler = (
                dataframe.drop(columns=cols_to_drop)
                .style.hide(axis="index")
                .format(
                    {"Price": "${:,.2f}", "Change%": "{:+.2f}%", "Sentiment": "{:+.2f}"}
                )
                .set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": "background:#0f172a; color:white; font-weight:600; text-align:center;",
                        },
                        {
                            "selector": "td",
                            "props": "text-align:center; padding:6px 10px;",
                        },
                        {
                            "selector": "table",
                            "props": "border-collapse:separate; border-spacing:0 6px;",
                        },
                    ]
                )
            )
            # Apply signal pill colors (only if '_color' column exists)
            if "_color" in dataframe.columns:
                styler = styler.apply(
                    lambda r: [
                        ""
                        if c != "Signal"
                        else color_signal(r[c], r)
                        for c in r.index
                    ],
                    axis=1,
                )
            # Color change%
            def color_change(v):
                if pd.isna(v):
                    return ""
                return "color:#0b8457;" if v >= 0 else "color:#d0021b;"
            styler = styler.applymap(color_change, subset=["Change%"])
            return styler
        st.subheader("Screened Universe")
        st.caption("Sortable and compact. Use the column headers to sort.")
        st.dataframe(
            df.drop(columns=[c for c in ["_color", "_rank"] if c in df.columns]),
            use_container_width=True,
            hide_index=True,
        )
        # Also render a styled static preview (looks more professional than default dataframe in some themes)
        st.html(style_table(df).to_html())
        if show_news:
            st.divider()
            st.subheader("Top headlines per highlighted tickers")
            top_df = df.head(12)[["Ticker", "Company", "Signal", "Headlines"]]
            for _, r in top_df.iterrows():
                if not r["Headlines"]:
                    continue
                st.markdown(
                    f"- [{r['Ticker']}] {r['Company']} — {r['Signal']}: {r['Headlines']}"
                )
st.caption("Tip: Set ALPHAVANTAGE_API_KEY env var for live data. No handpicked tickers — full S&P 500 auto-scan.")
