import os, time, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import base64

# Optional dependencies
HAVE_REPORTLAB, HAVE_KALEIDO, HAVE_TTS = False, False, False
PDF_FONT_NAME = "Helvetica"
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAVE_REPORTLAB = True
    for path in [
        r"C:\Windows\Fonts\DejaVuSans.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            PDF_FONT_NAME = os.path.splitext(os.path.basename(path))[0]
            pdfmetrics.registerFont(TTFont(PDF_FONT_NAME, path))
            break
except Exception:
    HAVE_REPORTLAB = False

try:
    import kaleido  # noqa
    HAVE_KALEIDO = True
except Exception:
    HAVE_KALEIDO = False

try:
    import pyttsx3
    TTS = pyttsx3.init()
    TTS.setProperty("rate", 165)
    HAVE_TTS = True
except Exception:
    HAVE_TTS = False

# ---------- CONFIG ----------
st.set_page_config(page_title="QuantEdge Terminal", page_icon="💹", layout="wide")
st.markdown("""
<style>
/* Wider, taller, clean tabs */
.stTabs [role="tablist"] {
  gap: 18px !important;
  justify-content: flex-start !important;
  border-bottom: 1px solid rgba(255,255,255,.08);
  margin: 6px 0 12px 0;
}
.stTabs [role="tab"] {
  font-size: 1.05rem;
  font-weight: 700;
  padding: 10px 18px !important;
  border-radius: 10px 10px 0 0;
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-bottom: none;
  color: #dfe7ef;
}
.stTabs [role="tab"][aria-selected="true"] {
  color: #00ffd5;
  background: rgba(0,255,213,0.07);
  box-shadow: inset 0 -2px 0 0 #00ffd5;
}
.stTabs [role="tab"]:hover {
  filter: brightness(1.1);
}
/* keep tab content separated from sidebar */
.block-container { padding-top: .8rem; }
</style>
""", unsafe_allow_html=True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "backend", "data")
MODELS_DIR = os.path.join(ROOT, "backend", "models")
POLICY_PATH = os.path.join(MODELS_DIR, "dqn_policy.h5")

# ---------- THEME ----------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] > .main {
    background: linear-gradient(180deg,#0a0d14 0%,#0d1018 100%);
}
[data-testid="stSidebar"] {
    background: #0b0e14;
}
.block-container { padding-top: 0.7rem; padding-bottom: 1.2rem; }
h1,h2,h3,h4 { color: #00ffd5; letter-spacing: .3px; }
.stMetric > div:nth-child(1) { justify-content: center; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def data_path(ticker):
    p = os.path.join(DATA_DIR, f"{ticker}_processed.csv")
    return p if os.path.exists(p) else None

@st.cache_data
def load_data(ticker, last_n):
    fp = data_path(ticker)
    if not fp:
        st.error(f"❌ Missing: backend/data/{ticker}_processed.csv")
        st.stop()
    df = pd.read_csv(fp)
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    if last_n:
        df = df.tail(last_n).reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "SMA_14" not in df.columns:
        df["SMA_14"] = df["Close"].rolling(14).mean()
    if "SMA_50" not in df.columns:
        df["SMA_50"] = df["Close"].rolling(50).mean()
    if "RSI" not in df.columns:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        df["RSI"] = 100 - (100 / (1 + rs))
    if "Momentum" not in df.columns:
        df["Momentum"] = df["Close"].diff(10)
    if "Sentiment" not in df.columns:
        df["Sentiment"] = 0.0
    return df.dropna().reset_index(drop=True)

def try_load_policy():
    try:
        import tensorflow as tf
        if os.path.exists(POLICY_PATH):
            return tf.keras.models.load_model(POLICY_PATH, compile=False)
    except Exception:
        pass
    return None

def build_states(df):
    trend = ((df["SMA_14"] - df["SMA_50"]) / (df["Close"] + 1e-9)).values
    S = np.stack([
        df["RSI"].values, df["SMA_14"].values, df["SMA_50"].values,
        df["Momentum"].values, trend, df["Sentiment"].values
    ], axis=1).astype(float)
    return MinMaxScaler().fit_transform(S)

def infer_actions_policy(df, model):
    try:
        S = build_states(df)
        q = model.predict(S, verbose=0)
        return np.argmax(q, axis=1)
    except Exception:
        return None

def infer_actions_hybrid(df):
    acts = np.zeros(len(df), dtype=int)
    s14, s50, rsi, mom = df["SMA_14"], df["SMA_50"], df["RSI"], df["Momentum"]
    for i in range(1, len(df)):
        if s14[i] > s50[i] and s14[i-1] <= s50[i-1] and mom[i] > 0 and rsi[i] < 70:
            acts[i] = 1
        elif s14[i] < s50[i] and s14[i-1] >= s50[i-1] and mom[i] < 0 and rsi[i] > 30:
            acts[i] = 2
    return acts

def backtest(df, actions, initial_cash, voice=False):
    cash, shares = float(initial_cash), 0
    pv, trades = [], []
    for i in range(len(df)):
        price = float(df.loc[i, "Close"])
        a = int(actions[i])
        if a == 1 and cash >= price:
            qty = int(cash // price)
            if qty > 0:
                shares += qty
                cash -= qty * price
                trades.append(("BUY", df.loc[i, "Date"], price, qty))
        elif a == 2 and shares > 0:
            cash += shares * price
            trades.append(("SELL", df.loc[i, "Date"], price, int(shares)))
            shares = 0
        pv.append(cash + shares * price)
    return np.array(pv), trades

def perf_metrics(pv):
    if pv is None or len(pv) < 2:
        return {}
    daily = pd.Series(pv).pct_change().dropna()
    ret = pv[-1]/pv[0] - 1
    vol = daily.std() * math.sqrt(252)
    sharpe = (daily.mean()/daily.std() * math.sqrt(252)) if daily.std() > 1e-12 else 0
    dd = ((pd.Series(pv)/pd.Series(pv).cummax()) - 1).min()
    return {"Total Return": f"{ret*100:.2f}%", "Volatility": f"{vol*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}", "Max Drawdown": f"{dd*100:.2f}%"}

def project_prices(df, horizon=10):
    close = df["Close"].values
    ema = pd.Series(close).ewm(span=10).mean().values
    slope = ema[-1] - ema[-2] if len(ema) > 1 else 0.0
    preds = []
    base = close[-1]
    for _ in range(horizon):
        base += slope
        preds.append(base)
    dates = [df["Date"].iloc[-1] + timedelta(days=i+1) for i in range(horizon)]
    return dates, preds

def candle_fig(df, buys=None, sells=None, proj=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Price",
                                 increasing_line_color="#00cc96", decreasing_line_color="#ef553b"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_14"], name="SMA 14", line=dict(color="#32b3ff", width=1.2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50", line=dict(color="#ffaa00", width=1.2)))
    if buys:
        fig.add_trace(go.Scatter(x=df.loc[buys, "Date"], y=df.loc[buys, "Close"], mode="markers",
                                 name="BUY", marker=dict(symbol="triangle-up", size=10, color="#00ff88")))
    if sells:
        fig.add_trace(go.Scatter(x=df.loc[sells, "Date"], y=df.loc[sells, "Close"], mode="markers",
                                 name="SELL", marker=dict(symbol="triangle-down", size=10, color="#ff6b6b")))
    if proj:
        d, p = proj
        fig.add_trace(go.Scatter(x=d, y=p, name="Projection", line=dict(width=2, dash="dot")))
    fig.update_layout(template="plotly_dark", height=540, margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False)
    return fig

def save_plot_png(fig, out_path):
    """PNG export with kaleido, else fallback returns False."""
    try:
        import kaleido  # noqa
        fig.write_image(out_path, scale=2)
        return True
    except Exception:
        return False
def embed_pdf(path: str, height: int = 760):
    """Show a PDF inline in Streamlit (base64 iframe)."""
    if not os.path.exists(path):
        st.warning("Report file not found.")
        return
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    html = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="{height}" type="application/pdf"></iframe>'
    st.markdown(html, unsafe_allow_html=True)

def build_pdf(path, header, metrics, figs, summary_text):
    if not HAVE_REPORTLAB:
        return False

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

    W, H = A4
    doc = SimpleDocTemplate(path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # ---- Title Page ----
    title_style = ParagraphStyle(
        'Title', fontSize=20, leading=28,
        textColor=colors.HexColor("#00ffd5"),
        alignment=1, spaceAfter=20
    )
    story.append(Paragraph("QuantEdge AI Investment Report", title_style))
    story.append(Spacer(1, 12))

    for line in header:
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 20))

    # ---- Performance Metrics ----
    story.append(Paragraph("<b>Performance Metrics Summary</b>", styles["Heading2"]))
    data = [["Metric", "Value"]] + [[k, v] for k, v in metrics.items()]
    table = Table(data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00ffd5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#10131a")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # ---- Executive Summary ----
    story.append(Paragraph("<b>Executive Summary</b>", styles["Heading2"]))
    story.append(Paragraph(summary_text, styles["Normal"]))
    story.append(Spacer(1, 20))

    # ---- Insert Images ----
    for img in figs:
        if os.path.exists(img):
            story.append(Spacer(1, 16))
            story.append(Image(img, width=W - 4*cm, height=H / 2.2))
            story.append(Spacer(1, 12))

    # ---- Footer ----
    footer_text = Paragraph(
        "<font size=8 color='#888888'>QuantEdge Research © "
        f"{datetime.now().year} — Generated by AI Trading Suite</font>",
        styles["Normal"]
    )
    story.append(Spacer(1, 12))
    story.append(footer_text)

    # ---- Build PDF ----
    doc.build(story)
    return True

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Controls")
ticker = st.sidebar.selectbox("Ticker", ["TCS.NS", "TATAMOTORS.NS"], index=0)
last_n = st.sidebar.slider("Last N Days", 60, 365, 180, step=10)
investment = st.sidebar.number_input("Initial Investment (₹)", min_value=1000, max_value=2_000_000, value=10_000, step=1000)
proj_days = st.sidebar.slider("Projection Days", 5, 20, 10)
st.sidebar.caption("Data folder: backend/data\nModel: backend/models/dqn_policy.h5")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🎥 Simulation", "📊 Performance", "📄 Report"])

df = load_data(ticker, last_n)
policy = try_load_policy()
actions = infer_actions_policy(df, policy) if policy is not None else infer_actions_hybrid(df)
buys = np.where(actions == 1)[0].tolist()
sells = np.where(actions == 2)[0].tolist()
pv, trades = backtest(df, actions, investment)
metrics = perf_metrics(pv)
proj = project_prices(df, proj_days)

# ---------- Tab 1: Overview ----------
with tab1:
    st.subheader("Market Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last Close", f"₹{df['Close'].iloc[-1]:,.2f}")
    c2.metric("SMA(14)-SMA(50)", f"{(df['SMA_14'].iloc[-1]-df['SMA_50'].iloc[-1]):.2f}")
    c3.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
    c4.metric("Total Trades", f"{len(trades)}")
    fig = candle_fig(df, buys, sells, proj)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 2: Simulation ----------
with tab2:
    st.subheader("Trading Simulation")
    sim_btn = st.button("▶ Run Simulation")
    placeholder = st.empty()
    if sim_btn:
        sim_pv = []
        cash, shares = float(investment), 0
        for i in range(len(df)):
            price = float(df.loc[i, "Close"])
            a = int(actions[i])
            if a == 1 and cash >= price:
                qty = int(cash // price)
                if qty > 0:
                    shares += qty
                    cash -= qty * price
            elif a == 2 and shares > 0:
                cash += shares * price
                shares = 0
            sim_pv.append(cash + shares * price)
            sim_fig = go.Figure()
            sim_fig.add_trace(go.Scatter(x=df["Date"][:i+1], y=df["Close"][:i+1], name="Price", line=dict(color="#32b3ff")))
            sim_fig.add_trace(go.Scatter(x=df["Date"][:i+1], y=sim_pv, name="Portfolio", line=dict(color="#ffaa00")))
            sim_fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10, r=10, t=30, b=10))
            placeholder.plotly_chart(sim_fig, use_container_width=True)
            time.sleep(0.05)

# ---------- Tab 3: Performance ----------
with tab3:
    st.subheader("Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    for (k, v), col in zip(metrics.items(), [m1, m2, m3, m4]):
        col.metric(k, v)
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(x=df["Date"], y=pv, fill="tozeroy", name="Portfolio Value", line=dict(color="#00cc96")))
    eq_fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(eq_fig, use_container_width=True)
    st.subheader("Trade Log")
    if trades:
        tdf = pd.DataFrame(trades, columns=["Action", "Date", "Price", "Qty"])
        tdf["Date"] = pd.to_datetime(tdf["Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(tdf, use_container_width=True, height=250)
    else:
        st.info("No trades executed.")

# ---------- Tab 4: Report ----------
with tab4:
    st.subheader("📄 Report")

    export_dir = os.path.join(MODELS_DIR, "exports")
    os.makedirs(export_dir, exist_ok=True)

    # Save figures (use kaleido if available; if not, we still build PDF without images)
    candle_png = os.path.join(export_dir, f"{ticker}_candles.png")
    equity_png = os.path.join(export_dir, f"{ticker}_equity.png")
    _c_ok = save_plot_png(fig, candle_png)
    _e_ok = save_plot_png(eq_fig, equity_png)

    # ---- Build narrative text (executive summary) ----
    try:
        ret = float(metrics["Total Return"].replace("%",""))
        sharpe = float(metrics["Sharpe Ratio"])
        dd = float(metrics["Max Drawdown"].replace("%",""))
        if ret > 15:
            tone = "The portfolio delivered strong profitability with stable trend conformity."
        elif ret > 0:
            tone = "The strategy achieved modest positive returns with balanced risk exposure."
        else:
            tone = "A net drawdown was observed, driven by trend reversals and elevated volatility."
        if sharpe > 1:
            tone += " Risk-adjusted performance was efficient across the window."
        elif sharpe < 0:
            tone += " Negative Sharpe indicates under-performance vs. risk-free benchmark."
        if dd <= -20:
            tone += " Drawdowns were material; tighter risk limits are recommended."
        else:
            tone += " Drawdowns remained within acceptable thresholds for this profile."
        perf_summary = (
            "This report summarizes simulated trading using QuantEdge signals that blend RL policy outputs "
            "with technical crossovers. " + tone + " Future work: dynamic position sizing and sentiment-weighted rebalancing."
        )
    except Exception:
        perf_summary = "Performance summary unavailable due to incomplete metrics."

    # ---- Build PDF ----
    pdf_path = os.path.join(export_dir, f"{ticker}_report.pdf")
    header = [
        f"Ticker: {ticker}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Investment: ₹{investment:,.0f}",
        f"Total Trades: {len(trades)}"
    ]
    # reuse your existing build_pdf(...) that accepts perf_summary OR
    # if you used my last message version, call:
    ok_pdf = build_pdf(pdf_path, header, metrics, [p for p in [candle_png, equity_png] if os.path.exists(p)], perf_summary)

    # ---- UI: preview + download ----
    cols = st.columns(2)
    if ok_pdf and os.path.exists(pdf_path):
        st.markdown("#### Preview")
        embed_pdf(pdf_path, height=760)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download QuantEdge Report (PDF)", f,
                               file_name=os.path.basename(pdf_path),
                               mime="application/pdf")
        st.success("Report generated.")
    else:
        st.warning("Could not create PDF. Install `reportlab` (and `kaleido` for chart images).")

st.markdown("<div style='text-align:center;color:#7f8c9a;'>QuantEdge Terminal • Neo-Quant • © 2025</div>", unsafe_allow_html=True)
