
import datetime as dt
from typing import Dict, Any, Optional, Tuple
import ccxt
import pandas as pd
import yfinance as yf
import streamlit as st

def fetch_crypto_price(symbol: str = "BTC/USD", exchange_name: str = "kraken") -> Dict[str, Any]:
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
        ticker = exchange.fetch_ticker(symbol)
        return {
            "symbol": symbol,
            "exchange": exchange_name,
            "current_price": ticker.get("last"),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "high": ticker.get("high"),
            "low": ticker.get("low"),
            "timestamp": ticker.get("timestamp"),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}

def fetch_crypto_ohlcv(symbol: str, exchange_name: str, timeframe: str = "1h", limit: int = 200) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df, None
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)}"

def fetch_stock_price(ticker: str, period: str = "5d") -> Dict[str, Any]:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return {"error": f"No data for {ticker}"}
        current_price = float(hist["Close"].iloc[-1])
        previous_price = float(hist["Close"].iloc[0])
        change = current_price - previous_price
        pct_change = (change / previous_price * 100) if previous_price else 0.0
        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "previous_price": round(previous_price, 2),
            "change": round(change, 2),
            "pct_change": round(pct_change, 2),
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}

def sma(series: pd.Series, window: int) -> Optional[float]:
    if len(series) < window:
        return None
    return float(pd.Series(series).rolling(window=window).mean().iloc[-1])

def buy_the_dip_logic(current_price: float, moving_average: float, dip_threshold: float = 0.97) -> Dict[str, Any]:
    if moving_average <= 0:
        return {"error": "Invalid moving average"}
    ratio = current_price / moving_average
    should_buy = ratio < dip_threshold
    return {
        "should_buy": should_buy,
        "current_price": current_price,
        "moving_average": moving_average,
        "ratio": round(ratio, 4),
        "threshold": dip_threshold,
        "reason": f"Price is {(1 - ratio) * 100:.2f}% below MA" if should_buy else "Price above threshold",
    }

def sell_the_high_logic(current_price: float, moving_average: float, high_threshold: float = 1.03) -> Dict[str, Any]:
    if moving_average <= 0:
        return {"error": "Invalid moving average"}
    ratio = current_price / moving_average
    should_sell = ratio > high_threshold
    return {
        "should_sell": should_sell,
        "current_price": current_price,
        "moving_average": moving_average,
        "ratio": round(ratio, 4),
        "threshold": high_threshold,
        "reason": f"Price is {(ratio - 1) * 100:.2f}% above MA" if should_sell else "Price below threshold",
    }

st.set_page_config(page_title="Streamlit SMA Trading App", page_icon="📈", layout="wide")
st.title("📈 Simple SMA Trading App")

with st.sidebar:
    st.header("Configuration")
    market_type = st.radio("Market", ["Crypto (ccxt)", "Stock (yfinance)"], index=0)
    if market_type == "Crypto (ccxt)":
        symbol = st.text_input("Symbol (ccxt)", value="BTC/USD")
        exchange_name = st.text_input("Exchange (ccxt)", value="kraken")
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    else:
        stock_ticker = st.text_input("Ticker (yfinance)", value="AAPL")
    candles = st.slider("Candles for SMA", min_value=20, max_value=500, value=100, step=10)
    sma_window = st.slider("SMA Window", min_value=5, max_value=100, value=20, step=5)
    buy_threshold = st.slider("Buy threshold (ratio below SMA)", min_value=0.85, max_value=0.99, value=0.97, step=0.01)
    sell_threshold = st.slider("Sell threshold (ratio above SMA)", min_value=1.01, max_value=1.15, value=1.03, step=0.01)
    refresh = st.checkbox("Auto-refresh (every 30s)", value=False)

col1, col2 = st.columns(2)

if market_type == "Crypto (ccxt)":
    price = fetch_crypto_price(symbol, exchange_name)
    if "error" in price:
        col1.error(f"Price error: {price['error']}")
    else:
        ts = price.get("timestamp")
        ts_str = dt.datetime.fromtimestamp(ts / 1000).isoformat() if isinstance(ts, (int, float)) else "N/A"
        col1.metric(
            label=f"{price['symbol']} @ {price['exchange']}",
            value=f"{price['current_price']}",
            delta=f"Bid {price['bid']} / Ask {price['ask']}, H {price['high']} L {price['low']}, ts {ts_str}"
        )
    df, err = fetch_crypto_ohlcv(symbol, exchange_name, timeframe=timeframe, limit=candles)
    if err:
        col1.error(f"OHLCV error: {err}")
        st.stop()
    with col1:
        st.subheader("OHLCV")
        st.dataframe(df.tail(10), use_container_width=True)
        st.line_chart(df.set_index("datetime")["close"], use_container_width=True)
    closes = df["close"].values
    sma_val = sma(pd.Series(closes), sma_window)
    if sma_val is None:
        col2.warning(f"Need at least {sma_window} candles for SMA.")
    else:
        col2.subheader(f"SMA({sma_window}) = {sma_val:.2f}")
        last_price = float(closes[-1])
        buy_dec = buy_the_dip_logic(last_price, sma_val, buy_threshold)
        sell_dec = sell_the_high_logic(last_price, sma_val, sell_threshold)
        col2.write("Buy-the-Dip Decision")
        col2.json(buy_dec)
        col2.write("Sell-the-High Decision")
        col2.json(sell_dec)
        if buy_dec.get("should_buy"):
            col2.success("BUY signal")
        elif sell_dec.get("should_sell"):
            col2.error("SELL signal")
        else:
            col2.info("HOLD")
else:
    st.subheader("Stock snapshot")
    sprice = fetch_stock_price(stock_ticker, period="5d")
    if "error" in sprice:
        st.error(sprice["error"])
    else:
        st.metric(label=stock_ticker, value=sprice["current_price"], delta=f"{sprice['pct_change']}% (5d)")
    try:
        hist = yf.Ticker(stock_ticker).history(period="6mo", interval="1d")
        hist = hist.dropna()
        st.line_chart(hist["Close"], use_container_width=True)
        closes = hist["Close"].values[-candles:]
        if len(closes) >= sma_window:
            sma_val = sma(pd.Series(closes), sma_window)
            st.write(f"SMA({sma_window}) = {sma_val:.2f}")
            last_price = float(closes[-1])
            buy_dec = buy_the_dip_logic(last_price, sma_val, buy_threshold)
            sell_dec = sell_the_high_logic(last_price, sma_val, sell_threshold)
            st.write("Buy-the-Dip Decision")
            st.json(buy_dec)
            st.write("Sell-the-High Decision")
            st.json(sell_dec)
        else:
            st.warning(f"Not enough candles for SMA({sma_window}).")
    except Exception as e:
        st.error(f"History error: {type(e).__name__}: {e}")

if refresh:
    st.caption("Auto-refresh is enabled. Rerun periodically.")
import yfinance as yf
import requests
import streamlit as st
from textblob import TextBlob

ALPHAVANTAGE_API_KEY = "UANG1JGU08PNMVPA"

def get_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        return None
    return None

def get_news_sentiment_alpha_vantage(ticker):
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={ALPHAVANTAGE_API_KEY}"
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return 0, []
        data = r.json()
        headlines = [d["title"] for d in data.get("feed", [])[:5]]
        scores = []
        for hl in headlines:
            tb = TextBlob(hl)
            scores.append(tb.sentiment.polarity)
        avg_sentiment = round(sum(scores)/len(scores), 2) if scores else 0
        return avg_sentiment, headlines
    except Exception:
        return 0, []

def ai_expert_signal(ticker, expert_rules):
    price = get_live_price(ticker)
    sentiment_score, news = get_news_sentiment_alpha_vantage(ticker)
    expert_score = expert_rules.get(ticker, 1)   # 1=keep, 0=avoid

    # Combined logic: Must be expert pick AND have positive sentiment
    if expert_score and sentiment_score > 0.1:
        signal = "BUY"
        reason = "Expert pick + positive news sentiment"
    elif expert_score and sentiment_score < -0.1:
        signal = "SELL"
        reason = "Expert pick but negative news sentiment"
    else:
        signal = "HOLD"
        reason = "Mixed/neutral signal"
    return {
        "ticker": ticker,
        "price": price,
        "sentiment": sentiment_score,
        "news": news,
        "signal": signal,
        "reason": reason
    }

def display_ai_expert_dashboard(tickers, expert_rules):
    st.title("🧠 AI ExpertLogic Dashboard (Live & Sentiment-Powered)")
    for t in tickers:
        result = ai_expert_signal(t, expert_rules)
        st.subheader(f"{result['ticker']}: {result['signal']}")
        st.write(f"Live Price: {result['price']}")
        st.write(f"News Sentiment Score: {result['sentiment']:.2f}")
        st.write(f"Reason: {result['reason']}")
        if result['news']:
            st.write("Recent Headlines:")
            for n in result['news']:
                st.caption(n)
        st.write("---")

# Example expert buy candidates with placeholder rules (1=expert likes, 0=avoid, customize as needed!)
# Extend or change tickers as needed per expert strategy:
watch_tickers = ['AAPL', 'NVDA', 'KO', 'BAC', 'CROX', 'STNE', 'EEFT', 'PG', 'BMBL', 'TMHC']
expert_rules = {ticker: 1 for ticker in watch_tickers}   # Set as needed per your logic!

# Call this in your Streamlit app main section:
display_ai_expert_dashboard(watch_tickers, expert_rules)
display_ai_expert_dashboard(watch_tickers, expert_rules)

