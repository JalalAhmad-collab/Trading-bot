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
SNP_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# =========================
# Utilities
# =========================
@st.cache_data(ttl=3600)
def fetch_sp500_tickers() -> pd.DataFrame:
    """Fetch live S&P 500 tickers and company names from Wikipedia."""
    tables = pd.read_html(SNP_WIKI_URL)
    df = tables[0]
    df = df.rename(columns={"Symbol": "ticker", "Security": "name"})
    # Some tickers have "." which AlphaVantage expects as "-" (e.g., BRK.B -> BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    return df[["ticker", "name"]]


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
    data = _av_get({"function": PRICE_FUNCTION, "symbol": ticker, "outputsize": "compact"})
    series = data.get("Time Series (Daily)", {})
    if not series:
        return {"error": f"No price data for {ticker}"}
    # Sort by date and take last two
    items = sorted(series.items(), key=lambda x: x[0])
    last_date, last_val = items[-1]
    prev_date, prev_val = items[-2] if len(items) > 1 else (None, None)
    close = float(last_val["4. close"]) if last_val else np.nan
    prev_close = float(prev_val["4. close"]) if prev_val else np.nan
    pct = ((close - prev_close) / prev_close * 100) if prev_val else 0.0
    return {
        "date": last_date,
        "close": round(close, 2),
        "prev_close": round(prev_close, 2) if prev_val else None,
        "pct_change": round(pct, 2),
        "volume": int(float(last_val.get("6. volume", 0))) if last_val else None,
    }


def analyze_text_sentiment(texts: list[str]) -> float:
    """Average polarity using TextBlob [-1,1]."""
    if not texts:
        return 0.0
    pols = []
    for t in texts[:10]:  # limit per ticker
        try:
            pols.append(TextBlob(t).sentiment.polarity)
        except Exception:
            continue
    return float(np.mean(pols)) if pols else 0.0


def fetch_news_and_sentiment(ticker: str) -> dict:
    """Fetch latest news via AlphaVantage and compute sentiment."""
    if not ALPHAVANTAGE_API_KEY:
        return {"headlines": [], "sentiment": 0.0}
    data = _av_get({
        "function": NEWS_FUNCTION,
        "tickers": ticker,
        "sort": "LATEST",
        "time_from": "20240101T0000",
        "limit": 20,
    })
    feed = data.get("feed", []) if isinstance(data, dict) else []
    headlines = [f.get("title") for f in feed if f.get("title")]
    sentiment = analyze_text_sentiment(headlines)
    top = headlines[:3]
    return {"headlines": top, "sentiment": sentiment}


def expert_signal(price: dict, sentiment: float) -> tuple[str, str]:
    """Blend technical micro-signal with AI sentiment.
    Rules:
    - Strong Buy: pct >= 0 and sentiment > 0.3, or pct < 0 and sentiment > 0.45 (positive despite dip)
    - Buy: sentiment > 0.15 and pct > -1.5
    - Hold: -0.15 <= sentiment <= 0.15 or abs(pct) < 0.5
    - Sell: sentiment < -0.15 and pct < 0.5
    - Strong Sell: sentiment < -0.35 or pct < -3 and sentiment < 0
    Returns (label, color)
    """
    pct = price.get("pct_change", 0.0) or 0.0
    if sentiment > 0.45 and pct < 0:
        return ("Strong Buy", "#0b8457")
    if sentiment > 0.3 and pct >= 0:
        return ("Strong Buy", "#0b8457")
    if sentiment > 0.15 and pct > -1.5:
        return ("Buy", "#35c759")
    if abs(pct) < 0.5 or (-0.15 <= sentiment <= 0.15):
        return ("Hold", "#8e8e93")
    if sentiment < -0.35 or (pct < -3 and sentiment < 0):
        return ("Strong Sell", "#d0021b")
    if sentiment < -0.15 and pct < 0.5:
        return ("Sell", "#ff3b30")
    return ("Hold", "#8e8e93")


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Expert Stock Screener", layout="wide")
st.title("AI Expert Screener — S&P 500")

with st.sidebar:
    st.subheader("Settings")
    max_tickers = st.slider("Max tickers to scan (rate-limit safe)", 20, 200, 60, 10)
    show_news = st.checkbox("Show top headlines", value=True)
    st.caption("Uses: Wikipedia + AlphaVantage + TextBlob")

# Load universe
sp500 = fetch_sp500_tickers()
if sp500.empty:
    st.error("Failed to load S&P 500 tickers from Wikipedia.")
else:
    # Rate-limit friendly batching
    rows = []
    scanned = 0
    prog = st.progress(0)
    for _, row in sp500.iterrows():
        if scanned >= max_tickers:
            break
        ticker = row["ticker"]
        name = row["name"]
        price = fetch_price_summary(ticker)
        if "error" in price:
            continue
        news = fetch_news_and_sentiment(ticker)
        label, color = expert_signal(price, news.get("sentiment", 0.0))
        rows.append({
            "Ticker": ticker,
            "Company": name,
            "Price": price.get("close"),
            "Change%": price.get("pct_change"),
            "Sentiment": round(news.get("sentiment", 0.0), 2),
            "Signal": label,
            "_color": color,
            "Headlines": " • ".join(news.get("headlines", [])) if show_news else "",
        })
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
            return f"background-color: {row['_color']}; color: white; font-weight: 600; text-align:center; border-radius:6px;"

        def style_table(dataframe: pd.DataFrame):
            styler = dataframe.drop(columns=["_color", "_rank"]).style \
                .hide(axis="index") \
                .format({"Price": "${:,.2f}", "Change%": "{:+.2f}%", "Sentiment": "{:+.2f}"}) \
                .set_table_styles([
                    {"selector": "th", "props": "background:#0f172a; color:white; font-weight:600; text-align:center;"},
                    {"selector": "td", "props": "text-align:center; padding:6px 10px;"},
                    {"selector": "table", "props": "border-collapse:separate; border-spacing:0 6px;"},
                ])
            # Apply signal pill colors
            styler = styler.apply(lambda r: ["" if c != "Signal" else color_signal(r[c], r) for c in r.index], axis=1)
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
            df.drop(columns=["_color", "_rank"]),
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
                st.markdown(f"- [{r['Ticker']}] {r['Company']} — {r['Signal']}: {r['Headlines']}")

st.caption("Tip: Set ALPHAVANTAGE_API_KEY env var for live data. No handpicked tickers — full S&P 500 auto-scan.")
