import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="NASDAQ Stock Research Dashboard",
    page_icon="📈",
    layout="wide",
)

# ----------------------------
# Custom CSS for "dark blue + pro" look
# ----------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.0rem; padding-bottom: 1.0rem;}
      .stMetric {background: rgba(11,35,66,0.55); border: 1px solid rgba(255,255,255,0.06);
                padding: 14px; border-radius: 14px;}
      div[data-testid="stMetricLabel"] > div {font-size: 0.85rem; opacity: 0.85;}
      div[data-testid="stMetricValue"] > div {font-size: 1.6rem;}
      .card {
          background: rgba(11,35,66,0.55);
          border: 1px solid rgba(255,255,255,0.06);
          padding: 16px;
          border-radius: 16px;
      }
      .small-note {opacity: 0.75; font-size: 0.85rem;}
      .title {font-size: 1.7rem; font-weight: 700; margin-bottom: 0.2rem;}
      .subtitle {opacity: 0.85; margin-top: 0rem; margin-bottom: 0.8rem;}
      hr {border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 0.8rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def fmt_money(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or np.isinf(x))):
        return "—"
    absx = abs(x)
    if absx >= 1e12:
        return f"${x/1e12:.2f}T"
    if absx >= 1e9:
        return f"${x/1e9:.2f}B"
    if absx >= 1e6:
        return f"${x/1e6:.2f}M"
    if absx >= 1e3:
        return f"${x/1e3:.2f}K"
    return f"${x:,.2f}"

def fmt_num(x, suffix=""):
    if x is None or (isinstance(x, float) and (math.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:,.2f}{suffix}"

def safe_get(d, key, default=None):
    try:
        v = d.get(key, default)
        if v is None:
            return default
        return v
    except Exception:
        return default

def compute_rsi(close: pd.Series, window: int = 14):
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = pd.Series(gain, index=close.index).rolling(window=window).mean()
    loss = pd.Series(loss, index=close.index).rolling(window=window).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(close: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

@st.cache_data(ttl=60 * 20)
def load_data(ticker: str, period: str):
    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=False)
    info = {}
    try:
        info = t.get_info()
    except Exception:
        info = {}

    # Financial statements (can be sparse for some tickers)
    try:
        fin_is = t.financials
    except Exception:
        fin_is = pd.DataFrame()

    try:
        fin_bs = t.balance_sheet
    except Exception:
        fin_bs = pd.DataFrame()

    try:
        fin_cf = t.cashflow
    except Exception:
        fin_cf = pd.DataFrame()

    return hist, info, fin_is, fin_bs, fin_cf

def make_candles(hist: pd.DataFrame, ma1=20, ma2=50, ma3=200):
    df = hist.copy()
    df["MA20"] = df["Close"].rolling(ma1).mean()
    df["MA50"] = df["Close"].rolling(ma2).mean()
    df["MA200"] = df["Close"].rolling(ma3).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        title="Price (Candles) + Moving Averages",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig

def make_volume(hist: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume"))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        title="Volume",
    )
    return fig

def make_returns_drawdown(hist: pd.DataFrame):
    df = hist.copy()
    df["ret"] = df["Close"].pct_change()
    df["equity"] = (1 + df["ret"].fillna(0)).cumprod()
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["peak"] - 1.0

    # returns histogram
    histo = go.Figure()
    r = df["ret"].dropna()
    histo.add_trace(go.Histogram(x=r, nbinsx=60, name="Daily returns"))
    histo.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        title="Daily Returns Distribution",
    )

    # drawdown chart
    dd = go.Figure()
    dd.add_trace(go.Scatter(x=df.index, y=df["drawdown"], mode="lines", name="Drawdown"))
    dd.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        title="Drawdown",
        yaxis_tickformat=".0%",
    )
    return histo, dd

def make_rsi_macd(hist: pd.DataFrame):
    df = hist.copy()
    df["RSI14"] = compute_rsi(df["Close"], 14)
    macd, sig, h = compute_macd(df["Close"])
    df["MACD"], df["SIGNAL"], df["HIST"] = macd, sig, h

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                        row_heights=[0.5, 0.5],
                        subplot_titles=("RSI (14)", "MACD"))

    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], mode="lines", name="RSI"), row=1, col=1)
    fig.add_hline(y=70, line_dash="dot", row=1, col=1)
    fig.add_hline(y=30, line_dash="dot", row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["SIGNAL"], mode="lines", name="Signal"), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["HIST"], name="Hist"), row=2, col=1)

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        title="Technical Indicators",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def clean_fin_df(df: pd.DataFrame, n=8):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [c.date().isoformat() if hasattr(c, "date") else str(c) for c in out.columns]
    out = out.iloc[:n, :]
    out.index = [str(i) for i in out.index]
    return out

# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.markdown("### 🔎 Stock Search")
ticker = st.sidebar.text_input("NASDAQ ticker (e.g., AAPL, NVDA, MSFT)", value="AAPL").strip().upper()

period = st.sidebar.selectbox(
    "Price history",
    ["6mo", "1y", "2y", "5y", "10y"],
    index=2
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Display options")
ma1 = st.sidebar.slider("MA1", 5, 60, 20)
ma2 = st.sidebar.slider("MA2", 20, 120, 50)
ma3 = st.sidebar.slider("MA3", 50, 300, 200)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="small-note">Data source: Yahoo Finance (via yfinance). Some fields can be missing for certain tickers.</div>', unsafe_allow_html=True)

# ----------------------------
# Load
# ----------------------------
if not ticker:
    st.stop()

try:
    hist, info, fin_is, fin_bs, fin_cf = load_data(ticker, period)
except Exception as e:
    st.error(f"Could not load data for {ticker}. Error: {e}")
    st.stop()

if hist is None or hist.empty:
    st.error(f"No price history returned for {ticker}. Double-check the ticker.")
    st.stop()

# ----------------------------
# Header
# ----------------------------
name = safe_get(info, "longName", ticker)
exchange = safe_get(info, "exchange", "—")
sector = safe_get(info, "sector", "—")
industry = safe_get(info, "industry", "—")

st.markdown(f'<div class="title">{name} <span style="opacity:.7; font-weight:600;">({ticker})</span></div>', unsafe_allow_html=True)
st.markdown(f'<div class="subtitle">{exchange} • {sector} • {industry}</div>', unsafe_allow_html=True)

# ----------------------------
# KPIs
# ----------------------------
last_close = float(hist["Close"].iloc[-1])
prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else np.nan
chg = last_close - prev_close if not np.isnan(prev_close) else np.nan
chg_pct = (chg / prev_close) if (not np.isnan(prev_close) and prev_close != 0) else np.nan

day_low = float(hist["Low"].iloc[-1])
day_high = float(hist["High"].iloc[-1])
vol = float(hist["Volume"].iloc[-1])

wk52_low = safe_get(info, "fiftyTwoWeekLow")
wk52_high = safe_get(info, "fiftyTwoWeekHigh")
mcap = safe_get(info, "marketCap")
pe = safe_get(info, "trailingPE")
fpe = safe_get(info, "forwardPE")
eps = safe_get(info, "trailingEps")
beta = safe_get(info, "beta")
div_yield = safe_get(info, "dividendYield")

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

kpi1.metric("Price", f"${last_close:,.2f}", f"{chg_pct*100:,.2f}%" if not np.isnan(chg_pct) else None)
kpi2.metric("Day Range", f"${day_low:,.2f} – ${day_high:,.2f}")
kpi3.metric("Volume", f"{vol:,.0f}")
kpi4.metric("Market Cap", fmt_money(mcap))
kpi5.metric("P/E (TTM) / Fwd", f"{fmt_num(pe)} / {fmt_num(fpe)}")
kpi6.metric("EPS / Beta", f"{fmt_num(eps)} / {fmt_num(beta)}")

kpi7, kpi8, kpi9 = st.columns(3)
kpi7.metric("52w Range", f"${fmt_num(wk52_low)} – ${fmt_num(wk52_high)}")
kpi8.metric("Dividend Yield", f"{(div_yield*100):.2f}%" if isinstance(div_yield, (int, float)) else "—")
kpi9.metric("Exchange", str(exchange))

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Main charts
# ----------------------------
left, right = st.columns([1.6, 1.0])

with left:
    st.plotly_chart(make_candles(hist, ma1=ma1, ma2=ma2, ma3=ma3), use_container_width=True)
    st.plotly_chart(make_volume(hist), use_container_width=True)

with right:
    histo, dd = make_returns_drawdown(hist)
    st.plotly_chart(histo, use_container_width=True)
    st.plotly_chart(dd, use_container_width=True)

st.plotly_chart(make_rsi_macd(hist), use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Fundamentals section
# ----------------------------
st.markdown("## Fundamentals")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown('<div class="card"><b>Business Summary</b><br/>', unsafe_allow_html=True)
    summary = safe_get(info, "longBusinessSummary", "—")
    st.write(summary if summary else "—")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="card"><b>Key Stats</b><br/>', unsafe_allow_html=True)
    stats = {
        "Employees": safe_get(info, "fullTimeEmployees"),
        "Country": safe_get(info, "country"),
        "Currency": safe_get(info, "currency"),
        "Website": safe_get(info, "website"),
        "Gross Margins": safe_get(info, "grossMargins"),
        "Operating Margins": safe_get(info, "operatingMargins"),
        "Profit Margins": safe_get(info, "profitMargins"),
        "ROE": safe_get(info, "returnOnEquity"),
        "ROA": safe_get(info, "returnOnAssets"),
    }
    srows = []
    for k, v in stats.items():
        if isinstance(v, (int, float)) and ("Margins" in k or k in ["ROE", "ROA"]):
            srows.append((k, f"{v*100:.2f}%"))
        else:
            srows.append((k, "—" if v is None else str(v)))
    st.dataframe(pd.DataFrame(srows, columns=["Metric", "Value"]), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="card"><b>Valuation Multiples (best effort)</b><br/>', unsafe_allow_html=True)
    multiples = {
        "P/S": safe_get(info, "priceToSalesTrailing12Months"),
        "P/B": safe_get(info, "priceToBook"),
        "EV/Revenue": safe_get(info, "enterpriseToRevenue"),
        "EV/EBITDA": safe_get(info, "enterpriseToEbitda"),
        "PEG": safe_get(info, "pegRatio"),
    }
    mrows = []
    for k, v in multiples.items():
        mrows.append((k, fmt_num(v)))
    st.dataframe(pd.DataFrame(mrows, columns=["Multiple", "Value"]), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### Financial Statements (latest available)")
is_df = clean_fin_df(fin_is, n=12)
bs_df = clean_fin_df(fin_bs, n=12)
cf_df = clean_fin_df(fin_cf, n=12)

tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

with tab1:
    if is_df.empty:
        st.info("No income statement data returned for this ticker.")
    else:
        st.dataframe(is_df, use_container_width=True)

with tab2:
    if bs_df.empty:
        st.info("No balance sheet data returned for this ticker.")
    else:
        st.dataframe(bs_df, use_container_width=True)

with tab3:
    if cf_df.empty:
        st.info("No cash flow data returned for this ticker.")
    else:
        st.dataframe(cf_df, use_container_width=True)

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Tip: add your own watchlist, alerts, and valuation models next (DCF/comp table).")
