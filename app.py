import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Equity Memo Dashboard",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLING (clean, premium, minimal)
# =========================================================
st.markdown(
    """
<style>
:root{
  --bg:#07162B;
  --card:#0B2342;
  --line:rgba(255,255,255,.10);
  --text:#EAF2FF;
  --muted:rgba(234,242,255,.72);
  --accent:#2E5BFF;
}

.block-container{
  max-width:1200px;
  padding-top:1.0rem;
  padding-bottom:1.0rem;
}

h1,h2,h3,h4,p,span,div{
  color:var(--text);
}

.top-title{
  font-weight:800;
  font-size:1.75rem;
  line-height:1.1;
  margin-bottom:.2rem;
}

.subline{
  color:var(--muted);
  font-size:.92rem;
  margin-top:0;
}

.badge{
  display:inline-block;
  padding:4px 10px;
  border:1px solid var(--line);
  border-radius:999px;
  background:rgba(255,255,255,.03);
  margin-right:6px;
  font-size:.78rem;
  color:var(--muted);
}

.card{
  border:1px solid var(--line);
  background:rgba(255,255,255,.03);
  border-radius:14px;
  padding:12px 14px;
  height:100%;
}

.card-title{
  color:var(--muted);
  font-size:.78rem;
  margin-bottom:6px;
}

.card-value{
  font-size:1.45rem;
  font-weight:800;
  line-height:1.15;
}

.card-delta{
  color:var(--muted);
  font-size:.80rem;
  margin-top:4px;
}

.section-title{
  font-size:1.05rem;
  font-weight:700;
  margin:.25rem 0 .6rem 0;
}

.small-muted{
  color:var(--muted);
  font-size:.83rem;
}

.range-wrap{
  margin-top:.35rem;
}

.hr{
  border:none;
  border-top:1px solid var(--line);
  margin:.8rem 0;
}

div[data-testid="stDataFrame"]{
  border:1px solid var(--line);
  border-radius:12px;
  overflow:hidden;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def sget(d: dict, key: str, default=None):
    try:
        v = d.get(key, default)
        return default if v is None else v
    except Exception:
        return default


def _to_float(x):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None


def fmt_num(x, digits=2):
    x = _to_float(x)
    if x is None:
        return "—"
    return f"{x:,.{digits}f}"


def fmt_pct(x, digits=2):
    x = _to_float(x)
    if x is None:
        return "—"
    return f"{x*100:.{digits}f}%"


def fmt_price(x):
    x = _to_float(x)
    if x is None:
        return "—"
    return f"${x:,.2f}"


def fmt_money(x):
    x = _to_float(x)
    if x is None:
        return "—"
    ax = abs(x)
    if ax >= 1e12:
        return f"${x/1e12:.2f}T"
    if ax >= 1e9:
        return f"${x/1e9:.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:.2f}M"
    if ax >= 1e3:
        return f"${x/1e3:.2f}K"
    return f"${x:,.0f}"


def metric_card(label: str, value: str, delta: str = ""):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{label}</div>
          <div class="card-value">{value}</div>
          <div class="card-delta">{delta if delta else "&nbsp;"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def returns_since(close: pd.Series, n_days: int):
    if close is None or close.empty:
        return None
    if len(close) <= n_days:
        return None
    p1 = _to_float(close.iloc[-1])
    p0 = _to_float(close.iloc[-(n_days + 1)])
    if p0 in (None, 0) or p1 is None:
        return None
    return p1 / p0 - 1.0


def make_52w_bar(price, low, high):
    p = _to_float(price)
    lo = _to_float(low)
    hi = _to_float(high)
    if p is None or lo is None or hi is None or hi <= lo:
        st.markdown('<div class="small-muted">52W range unavailable.</div>', unsafe_allow_html=True)
        return

    pos = (p - lo) / (hi - lo)
    pos = min(max(pos, 0.0), 1.0)
    pct = round(pos * 100)

    st.markdown(
        f"""
        <div class="range-wrap">
          <div class="small-muted">52W Range: {fmt_price(lo)} → {fmt_price(hi)} · Position: {pct}%</div>
          <div style="height:10px;border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.05);overflow:hidden;">
             <div style="height:100%;width:{pct}%;background:var(--accent);"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def prepare_price_chart(df: pd.DataFrame, ticker: str):
    d = df.copy()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA50"] = d["Close"].rolling(50).mean()
    d["MA200"] = d["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index,
        open=d["Open"],
        high=d["High"],
        low=d["Low"],
        close=d["Close"],
        name="Price"
    ))
    fig.add_trace(go.Scatter(x=d.index, y=d["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=d.index, y=d["MA50"], mode="lines", name="MA50"))
    fig.add_trace(go.Scatter(x=d.index, y=d["MA200"], mode="lines", name="MA200"))

    fig.update_layout(
        template="plotly_dark",
        title=f"{ticker} Price",
        height=470,
        margin=dict(l=8, r=8, t=45, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def prepare_volume_chart(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
    fig.update_layout(
        template="plotly_dark",
        title="Volume",
        height=190,
        margin=dict(l=8, r=8, t=40, b=8),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def prepare_relative_chart(main_hist: pd.DataFrame, qqq_hist: pd.DataFrame, ticker: str):
    a = main_hist["Close"].dropna()
    b = qqq_hist["Close"].dropna()
    if a.empty or b.empty:
        return None

    idx = a.index.intersection(b.index)
    if len(idx) < 20:
        return None

    a = (a.loc[idx] / a.loc[idx].iloc[0]) * 100
    b = (b.loc[idx] / b.loc[idx].iloc[0]) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=a.values, mode="lines", name=ticker))
    fig.add_trace(go.Scatter(x=idx, y=b.values, mode="lines", name="QQQ"))

    fig.update_layout(
        template="plotly_dark",
        title="Relative Performance (Normalized to 100)",
        height=280,
        margin=dict(l=8, r=8, t=42, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data(ttl=60 * 15)
def load_bundle(ticker: str, period: str):
    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=False)

    info = {}
    try:
        info = t.get_info()
    except Exception:
        pass

    # financial tables
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


def tidy_fin_df(df: pd.DataFrame, n_rows: int = 12):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.columns = [
        c.date().isoformat() if hasattr(c, "date") else str(c)
        for c in out.columns
    ]
    out = out.iloc[:n_rows, :]
    out.index = [str(i) for i in out.index]
    return out


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("### Dashboard Controls")
ticker = st.sidebar.text_input("Ticker (NASDAQ)", value="AAPL").strip().upper()
period = st.sidebar.selectbox("History Window", ["6mo", "1y", "2y", "5y", "10y"], index=2)

st.sidebar.markdown("---")
st.sidebar.markdown("### Compare")
compare_text = st.sidebar.text_input(
    "Comma-separated tickers (max 5)",
    value="AAPL,MSFT,NVDA"
).strip().upper()

compare_tickers = []
if compare_text:
    compare_tickers = [x.strip() for x in compare_text.split(",") if x.strip()]
    # unique preserve order
    seen = set()
    cleaned = []
    for x in compare_tickers:
        if x not in seen:
            cleaned.append(x)
            seen.add(x)
    compare_tickers = cleaned[:5]

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span class="small-muted">Data source: Yahoo Finance (yfinance). Some fields can be unavailable depending on ticker.</span>',
    unsafe_allow_html=True
)


# =========================================================
# MAIN LOAD
# =========================================================
if not ticker:
    st.stop()

hist, info, fin_is, fin_bs, fin_cf = load_bundle(ticker, period)
if hist is None or hist.empty:
    st.error(f"No data found for {ticker}. Check ticker and try again.")
    st.stop()

# benchmark for relative chart
qqq_hist, _, _, _, _ = load_bundle("QQQ", period)

# =========================================================
# HEADER
# =========================================================
name = sget(info, "longName", ticker)
exchange = sget(info, "exchange", "—")
sector = sget(info, "sector", "—")
industry = sget(info, "industry", "—")

st.markdown(f'<div class="top-title">{name} <span style="opacity:.75;">({ticker})</span></div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <span class="badge">{exchange}</span>
    <span class="badge">{sector}</span>
    <span class="badge">{industry}</span>
    <span class="badge">Updated {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================================================
# KPI STRIP
# =========================================================
close = hist["Close"].dropna()
last = _to_float(close.iloc[-1]) if not close.empty else None
prev = _to_float(close.iloc[-2]) if len(close) > 1 else None

day_ret = None
if last is not None and prev not in (None, 0):
    day_ret = last / prev - 1.0

r_1w = returns_since(close, 5)
r_1m = returns_since(close, 21)
r_ytd = returns_since(close, 252)  # approximation

mcap = sget(info, "marketCap")
ev = sget(info, "enterpriseValue")
pe = sget(info, "trailingPE")
fpe = sget(info, "forwardPE")
ps = sget(info, "priceToSalesTrailing12Months")

k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Price", fmt_price(last), f"{fmt_pct(day_ret)} today")
with k2:
    metric_card("Momentum", f"{fmt_pct(r_1w)} / {fmt_pct(r_1m)}", "1W / 1M")
with k3:
    metric_card("Size", fmt_money(mcap), f"EV {fmt_money(ev)}")
with k4:
    metric_card("Valuation", f"P/E {fmt_num(pe)}", f"Fwd {fmt_num(fpe)} · P/S {fmt_num(ps)}")

wk52_low = sget(info, "fiftyTwoWeekLow")
wk52_high = sget(info, "fiftyTwoWeekHigh")
make_52w_bar(last, wk52_low, wk52_high)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab_overview, tab_compare, tab_financials, tab_snapshot = st.tabs(
    ["Overview", "Compare", "Financials", "Snapshot"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    left, right = st.columns([1.55, 0.95])

    with left:
        st.plotly_chart(prepare_price_chart(hist, ticker), use_container_width=True)
        st.plotly_chart(prepare_volume_chart(hist), use_container_width=True)

        rel_fig = prepare_relative_chart(hist, qqq_hist, ticker)
        if rel_fig is not None:
            st.plotly_chart(rel_fig, use_container_width=True)

    with right:
        st.markdown('<div class="section-title">Quick Investment Read</div>', unsafe_allow_html=True)

        gm = sget(info, "grossMargins")
        om = sget(info, "operatingMargins")
        pm = sget(info, "profitMargins")
        roe = sget(info, "returnOnEquity")
        roa = sget(info, "returnOnAssets")
        beta = sget(info, "beta")
        divy = sget(info, "dividendYield")

        quick_df = pd.DataFrame(
            [
                ["Gross / Op / Net Margin", f"{fmt_pct(gm)} / {fmt_pct(om)} / {fmt_pct(pm)}"],
                ["ROE / ROA", f"{fmt_pct(roe)} / {fmt_pct(roa)}"],
                ["Beta", fmt_num(beta)],
                ["Dividend Yield", fmt_pct(divy)],
                ["YTD Return (approx)", fmt_pct(r_ytd)],
            ],
            columns=["Metric", "Value"],
        )
        st.dataframe(quick_df, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">Business Summary</div>', unsafe_allow_html=True)
        summary = sget(info, "longBusinessSummary", "No summary available.")
        st.markdown(f'<div class="small-muted">{summary}</div>', unsafe_allow_html=True)


# ---------------- COMPARE ----------------
with tab_compare:
    st.markdown('<div class="section-title">Peer Comparison</div>', unsafe_allow_html=True)

    if len(compare_tickers) < 2:
        st.info("Enter at least 2 tickers in the sidebar compare field (e.g., AAPL,MSFT,NVDA).")
    else:
        rows = []
        norm_series = {}

        for t in compare_tickers:
            h, inf, _, _, _ = load_bundle(t, period)
            if h is None or h.empty:
                continue

            c = h["Close"].dropna()
            if c.empty:
                continue

            p_last = _to_float(c.iloc[-1])
            p_prev = _to_float(c.iloc[-2]) if len(c) > 1 else None
            d = None
            if p_last is not None and p_prev not in (None, 0):
                d = p_last / p_prev - 1.0

            rows.append(
                {
                    "Ticker": t,
                    "Price": fmt_price(p_last),
                    "1D": fmt_pct(d),
                    "1W": fmt_pct(returns_since(c, 5)),
                    "1M": fmt_pct(returns_since(c, 21)),
                    "Market Cap": fmt_money(sget(inf, "marketCap")),
                    "P/E": fmt_num(sget(inf, "trailingPE")),
                    "Fwd P/E": fmt_num(sget(inf, "forwardPE")),
                    "P/S": fmt_num(sget(inf, "priceToSalesTrailing12Months")),
                }
            )

            ns = (c / c.iloc[0]) * 100
            norm_series[t] = ns

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No valid comparison data returned.")

        if norm_series:
            fig = go.Figure()
            for t, s in norm_series.items():
                fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=t))
            fig.update_layout(
                template="plotly_dark",
                title="Performance (Normalized to 100)",
                height=380,
                margin=dict(l=8, r=8, t=44, b=8),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------- FINANCIALS ----------------
with tab_financials:
    st.markdown('<div class="section-title">Financial Statements</div>', unsafe_allow_html=True)

    is_df = tidy_fin_df(fin_is, n_rows=14)
    bs_df = tidy_fin_df(fin_bs, n_rows=14)
    cf_df = tidy_fin_df(fin_cf, n_rows=14)

    t1, t2, t3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])

    with t1:
        if is_df.empty:
            st.info("Income statement not available for this ticker.")
        else:
            st.dataframe(is_df, use_container_width=True)

    with t2:
        if bs_df.empty:
            st.info("Balance sheet not available for this ticker.")
        else:
            st.dataframe(bs_df, use_container_width=True)

    with t3:
        if cf_df.empty:
            st.info("Cash flow not available for this ticker.")
        else:
            st.dataframe(cf_df, use_container_width=True)


# ---------------- SNAPSHOT ----------------
with tab_snapshot:
    st.markdown('<div class="section-title">Company Snapshot</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        facts = {
            "Website": sget(info, "website"),
            "Country": sget(info, "country"),
            "Currency": sget(info, "currency"),
            "Full-time employees": sget(info, "fullTimeEmployees"),
            "52W Low / High": f"{fmt_price(wk52_low)} / {fmt_price(wk52_high)}",
            "Avg Volume (10d)": f"{_to_float(sget(info, 'averageDailyVolume10Day')):,.0f}" if _to_float(sget(info, "averageDailyVolume10Day")) is not None else "—",
            "Avg Volume (3m)": f"{_to_float(sget(info, 'averageVolume')):,.0f}" if _to_float(sget(info, "averageVolume")) is not None else "—",
        }

        facts_df = pd.DataFrame([(k, "—" if v in [None, "", "nan"] else v) for k, v in facts.items()], columns=["Field", "Value"])
        st.dataframe(facts_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown('<div class="section-title">Analyst Notes</div>', unsafe_allow_html=True)
        st.text_area(
            "Write your memo here",
            placeholder="Bull case, bear case, catalysts, risks, valuation view...",
            height=260,
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="small-muted">Tip: keep this to 5 bullets max for a clean investment memo style.</div>',
            unsafe_allow_html=True
        )

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
st.caption("Professional memo layout • Dark blue theme • Yahoo Finance data via yfinance")
