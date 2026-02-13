import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Stock Memo Dashboard", page_icon="🧾", layout="wide")

# ----------------------------
# Theme-ish CSS (minimal premium)
# ----------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.0rem; padding-bottom: 1.2rem; max-width: 1200px;}
      .muted {opacity: 0.75;}
      .h1 {font-size: 1.65rem; font-weight: 800; margin: 0;}
      .h2 {font-size: 1.1rem; font-weight: 700; margin: 0.1rem 0 0.4rem 0;}
      .pill {display:inline-block; padding: 6px 10px; border-radius: 999px;
             background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
             font-size: 0.82rem; opacity: 0.9;}
      .card {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.08);
          padding: 14px 14px;
          border-radius: 14px;
      }
      .kpi {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.08);
          padding: 12px 14px;
          border-radius: 14px;
      }
      .kpi-label {font-size: 0.80rem; opacity: 0.75; margin-bottom: 2px;}
      .kpi-value {font-size: 1.35rem; font-weight: 800; line-height: 1.1;}
      .kpi-delta {font-size: 0.85rem; opacity: 0.85; margin-top: 4px;}
      hr {border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 0.9rem 0;}
      .small {font-size: 0.85rem;}
      .range-wrap {margin-top: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Helpers
# ----------------------------
def safe_get(d, key, default=None):
    try:
        v = d.get(key, default)
        return default if v is None else v
    except Exception:
        return default

def fmt_money(x):
    if x is None:
        return "—"
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return "—"
    except Exception:
        return "—"
    ax = abs(x)
    if ax >= 1e12: return f"${x/1e12:.2f}T"
    if ax >= 1e9:  return f"${x/1e9:.2f}B"
    if ax >= 1e6:  return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"

def fmt_num(x, digits=2):
    if x is None:
        return "—"
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return "—"
        return f"{x:,.{digits}f}"
    except Exception:
        return "—"

def fmt_pct(x, digits=2):
    if x is None:
        return "—"
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return "—"
        return f"{x*100:.{digits}f}%"
    except Exception:
        return "—"

def kpi(label, value, delta=None):
    d = f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
    if delta is not None:
        d += f'<div class="kpi-delta">{delta}</div>'
    d += "</div>"
    st.markdown(d, unsafe_allow_html=True)

def range_bar_52w(price, low, high):
    # Returns a small HTML bar showing where price sits in 52W range
    if any(v is None for v in [price, low, high]):
        return "<div class='small muted'>52W range unavailable</div>"
    try:
        price, low, high = float(price), float(low), float(high)
        if high <= low:
            return "<div class='small muted'>52W range unavailable</div>"
        pos = (price - low) / (high - low)
        pos = max(0.0, min(1.0, pos))
        pct = int(round(pos * 100))
    except Exception:
        return "<div class='small muted'>52W range unavailable</div>"

    return f"""
    <div class="range-wrap">
      <div class="small muted">52W Range: {low:,.2f} → {high:,.2f}  •  Position: {pct}%</div>
      <div style="height:10px; border-radius:999px; background:rgba(255,255,255,0.08); overflow:hidden; border:1px solid rgba(255,255,255,0.10);">
        <div style="width:{pct}%; height:100%; background:rgba(46,91,255,0.9);"></div>
      </div>
    </div>
    """

@st.cache_data(ttl=60 * 20)
def load_ticker(ticker: str, period="2y"):
    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=False)
    info = {}
    try:
        info = t.get_info()
    except Exception:
        info = {}
    return hist, info

def returns_since(hist: pd.DataFrame, days: int):
    if hist is None or hist.empty or len(hist) < days + 1:
        return None
    p0 = float(hist["Close"].iloc[-days-1])
    p1 = float(hist["Close"].iloc[-1])
    if p0 == 0:
        return None
    return (p1 / p0) - 1.0

def make_price_chart(hist: pd.DataFrame, title="Price"):
    df = hist.copy()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], mode="lines", name="MA200"))
    fig.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def make_relative_chart(main_hist: pd.DataFrame, spy_hist: pd.DataFrame, main_label: str):
    # Normalize both to 100 at start
    a = main_hist["Close"].dropna()
    b = spy_hist["Close"].dropna()
    if a.empty or b.empty:
        return None
    idx = a.index.intersection(b.index)
    a = a.loc[idx]
    b = b.loc[idx]
    if len(idx) < 10:
        return None
    a_norm = (a / a.iloc[0]) * 100
    b_norm = (b / b.iloc[0]) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=a_norm, mode="lines", name=main_label))
    fig.add_trace(go.Scatter(x=idx, y=b_norm, mode="lines", name="QQQ"))
    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Relative Performance (Normalized to 100)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

# ----------------------------
# Sidebar (minimal)
# ----------------------------
st.sidebar.markdown("### 🧾 Stock Memo")
ticker = st.sidebar.text_input("Ticker (NASDAQ)", value="AAPL").strip().upper()
period = st.sidebar.selectbox("History", ["6mo", "1y", "2y", "5y"], index=2)

st.sidebar.markdown("---")
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "MSFT", "NVDA"]

add_col1, add_col2 = st.sidebar.columns([2, 1])
with add_col1:
    add_t = st.text_input("Add to watchlist", value="", placeholder="e.g., AMZN").strip().upper()
with add_col2:
    if st.button("Add"):
        if add_t and add_t not in st.session_state.watchlist:
            st.session_state.watchlist.append(add_t)

st.sidebar.markdown("**Watchlist**")
to_remove = None
for w in st.session_state.watchlist:
    c1, c2 = st.sidebar.columns([4, 1])
    c1.write(w)
    if c2.button("✕", key=f"rm_{w}"):
        to_remove = w
if to_remove:
    st.session_state.watchlist = [x for x in st.session_state.watchlist if x != to_remove]
    st.rerun()

st.sidebar.markdown("---")
compare = st.sidebar.multiselect(
    "Compare (2–5 tickers)",
    options=st.session_state.watchlist,
    default=[t for t in st.session_state.watchlist if t in ["AAPL", "MSFT"]][:2],
)

# ----------------------------
# Load main ticker
# ----------------------------
if not ticker:
    st.stop()

main_hist, main_info = load_ticker(ticker, period=period)
if main_hist is None or main_hist.empty:
    st.error("No price data. Double-check the ticker.")
    st.stop()

# benchmark (NASDAQ feel)
qqq_hist, _ = load_ticker("QQQ", period=period)

# ----------------------------
# Header
# ----------------------------
name = safe_get(main_info, "longName", ticker)
sector = safe_get(main_info, "sector", "—")
industry = safe_get(main_info, "industry", "—")
exchange = safe_get(main_info, "exchange", "—")

last = float(main_hist["Close"].iloc[-1])
prev = float(main_hist["Close"].iloc[-2]) if len(main_hist) >= 2 else None
dchg = (last - prev) if prev else None
dchg_pct = (dchg / prev) if prev else None

st.markdown(f"<div class='h1'>{name} <span class='muted'>({ticker})</span></div>", unsafe_allow_html=True)
st.markdown(
    f"<span class='pill'>{exchange}</span> "
    f"<span class='pill'>{sector}</span> "
    f"<span class='pill'>{industry}</span>",
    unsafe_allow_html=True
)

# ----------------------------
# KPI row (clean)
# ----------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    delta_txt = f"{'+' if dchg_pct and dchg_pct>=0 else ''}{fmt_pct(dchg_pct)} today" if dchg_pct is not None else None
    kpi("Price", f"${last:,.2f}", delta_txt)

with c2:
    r5 = returns_since(main_hist, 5)
    r21 = returns_since(main_hist, 21)
    kpi("Momentum", f"{fmt_pct(r5)} / {fmt_pct(r21)}", "1W / 1M")

with c3:
    mcap = safe_get(main_info, "marketCap")
    ev = safe_get(main_info, "enterpriseValue")
    kpi("Size", f"{fmt_money(mcap)}", f"EV {fmt_money(ev)}")

with c4:
    pe = safe_get(main_info, "trailingPE")
    fpe = safe_get(main_info, "forwardPE")
    ps = safe_get(main_info, "priceToSalesTrailing12Months")
    kpi("Valuation", f"P/E {fmt_num(pe)}", f"Fwd {fmt_num(fpe)} • P/S {fmt_num(ps)}")

# 52W bar
wk52_low = safe_get(main_info, "fiftyTwoWeekLow")
wk52_high = safe_get(main_info, "fiftyTwoWeekHigh")
st.markdown(range_bar_52w(last, wk52_low, wk52_high), unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ----------------------------
# Tabs (minimal memo)
# ----------------------------
tab_overview, tab_compare, tab_snapshot = st.tabs(["Overview", "Compare", "Company Snapshot"])

with tab_overview:
    left, right = st.columns([1.35, 0.65])

    with left:
        st.plotly_chart(make_price_chart(main_hist, title="Price (MA50 / MA200)"), use_container_width=True)
        rel = make_relative_chart(main_hist, qqq_hist, ticker)
        if rel:
            st.plotly_chart(rel, use_container_width=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Quick Take</div>", unsafe_allow_html=True)

        gm = safe_get(main_info, "grossMargins")
        om = safe_get(main_info, "operatingMargins")
        pm = safe_get(main_info, "profitMargins")
        roe = safe_get(main_info, "returnOnEquity")
        beta = safe_get(main_info, "beta")
        divy = safe_get(main_info, "dividendYield")

        lines = [
            ("Gross / Op / Profit margin", f"{fmt_pct(gm)} / {fmt_pct(om)} / {fmt_pct(pm)}"),
            ("ROE / Beta", f"{fmt_pct(roe)} / {fmt_num(beta)}"),
            ("Dividend yield", fmt_pct(divy)),
        ]
        df = pd.DataFrame(lines, columns=["Metric", "Value"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Business Summary</div>", unsafe_allow_html=True)
        summary = safe_get(main_info, "longBusinessSummary", "—")
        st.markdown(f"<div class='small muted'>{summary}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab_compare:
    st.markdown("<div class='h2'>Side-by-side (clean)</div>", unsafe_allow_html=True)

    if len(compare) < 2:
        st.info("Pick at least 2 tickers in the sidebar under Compare.")
    else:
        rows = []
        hists = {}
        for t in compare[:5]:
            h, inf = load_ticker(t, period=period)
            if h is None or h.empty:
                continue
            hists[t] = h
            lastp = float(h["Close"].iloc[-1])
            prevp = float(h["Close"].iloc[-2]) if len(h) >= 2 else np.nan
            day = (lastp / prevp - 1) if prevp and not np.isnan(prevp) else np.nan
            rows.append({
                "Ticker": t,
                "Price": f"${lastp:,.2f}",
                "1D": fmt_pct(day),
                "1W": fmt_pct(returns_since(h, 5)),
                "1M": fmt_pct(returns_since(h, 21)),
                "Mkt Cap": fmt_money(safe_get(inf, "marketCap")),
                "P/E": fmt_num(safe_get(inf, "trailingPE")),
                "Fwd P/E": fmt_num(safe_get(inf, "forwardPE")),
                "P/S": fmt_num(safe_get(inf, "priceToSalesTrailing12Months")),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Normalized performance chart
        fig = go.Figure()
        for t, h in hists.items():
            s = h["Close"].dropna()
            if len(s) < 10:
                continue
            s = (s / s.iloc[0]) * 100
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=t))
        fig.update_layout(
            template="plotly_dark",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
            title="Performance (Normalized to 100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

with tab_snapshot:
    st.markdown("<div class='h2'>Clean company facts</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        facts = {
            "Website": safe_get(main_info, "website"),
            "Employees": safe_get(main_info, "fullTimeEmployees"),
            "Country": safe_get(main_info, "country"),
            "Currency": safe_get(main_info, "currency"),
            "52W Low / High": f"{fmt_num(wk52_low)} / {fmt_num(wk52_high)}",
        }
        fdf = pd.DataFrame([(k, "—" if v is None else str(v)) for k, v in facts.items()], columns=["Field", "Value"])
        st.dataframe(fdf, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Notes (manual)</div>", unsafe_allow_html=True)
        st.markdown("<div class='small muted'>Add your own thesis notes here. (We can make this persist later.)</div>", unsafe_allow_html=True)
        st.text_area("Thesis / risks / catalysts", value="", height=220, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

st.caption("Data via Yahoo Finance (yfinance). Some fields may be missing for certain tickers.")
