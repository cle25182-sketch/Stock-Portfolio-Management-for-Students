import io
import time
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.optimize import minimize

st.set_page_config(
    page_title="QuantLab — ระบบจำลองและจัดพอร์ตหุ้นเชิงปริมาณ",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_TICKERS_ALLOWED = 10   # จำกัดจำนวนหุ้นสูงสุดเพื่อความรวดเร็วในการประมวลผล
DEFAULT_CAPITAL = 5000     # เงินตั้งต้นสำหรับนักเรียน

# ─────────────────────────────  Design tokens  ─────────────────────────────
CREAM, CREAM_2, PANEL = "#F7F3E8", "#EFE8D4", "#FBF8EE"
OLIVE, FOREST, INK, HONEY, LINE = "#7B8750", "#1F5A4A", "#34332A", "#FDCB7A", "#D9D2BA"

STRAT_ORDER = ["equal", "markowitz", "mcap", "shrinkage"]
STRAT_DISPLAY_NAMES = {
    "equal": "A: Equal-Weight",
    "markowitz": "B: Markowitz",
    "mcap": "C: Market-Cap",
    "shrinkage": "D: Shrinkage",
}
STRAT_FULL_NAMES = {
    "equal": "A: Equal-weight (สัดส่วนเท่ากัน)",
    "markowitz": "B: Markowitz (สมการคณิตศาสตร์)",
    "mcap": "C: Market-cap (ตามมูลค่าบริษัท)",
    "shrinkage": "D: Markowitz-Shrinkage (ลดสัญญาณรบกวนทั้ง Σ และ μ)",
}
STRAT_COLORS = {"equal": OLIVE, "markowitz": "#D9A441", "mcap": FOREST, "shrinkage": "#9B6B9E"}

STRESS_WINDOWS = {
    "COVID-19 (ก.พ.–เม.ย. 2020)": ("2020-02-01", "2020-04-30"),
    "เงินเฟ้อ/ดอกเบี้ยขาขึ้น (2022)": ("2022-01-01", "2022-12-31"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600&family=Caveat:wght@500;600&family=Mali:wght@400;500&family=Noto+Serif+Thai:wght@400;500;600&family=Sarabun:wght@400;500;600&display=swap');

.stApp { background: #F7F3E8; font-family: 'Sarabun', 'Noto Serif Thai', sans-serif; }
.stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea { font-family: 'Sarabun', 'Noto Serif Thai', sans-serif; color: #34332A; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: 'Playfair Display', 'Noto Serif Thai', serif; color: #7B8750; font-weight: 600; letter-spacing: .01em;
}
.block-container { padding-top: 1.5rem; max-width: 1200px; }
[data-testid="stSidebar"] { background: #EFE8D4; border-right: 1px solid #D9D2BA; }
[data-testid="stSidebar"] h2 { font-size: 1.5rem; }
header[data-testid="stHeader"] { background: transparent; }

/* hero */
.stApp .hero {
    text-align: center; padding: 76px 24px 64px; border-radius: 4px; color: #fff; margin-bottom: 8px;
    background:
        linear-gradient(180deg, rgba(36,48,34,.74) 0%, rgba(92,102,64,.58) 55%, rgba(48,62,42,.82) 100%),
        radial-gradient(ellipse at 72% 18%, #d8cf9a 0%, rgba(216,207,154,0) 55%),
        #5b6a45;
}
.stApp .hero div, .stApp .hero p { color: #fff; }
.hero-title { font-family: 'Playfair Display', 'Noto Serif Thai', serif; font-size: clamp(1.9rem, 5vw, 3.3rem); line-height: 1.15; }
.hero-script { font-family: 'Caveat', 'Mali', cursive; font-size: clamp(1.9rem, 4.8vw, 3.2rem); line-height: 1.1; margin-top: 2px; }
.stApp .hero p.hero-sub { max-width: 560px; margin: 18px auto 0; font-size: .98rem; line-height: 1.7; opacity: .95; }
.hero-hint { margin-top: 26px; font-size: .8rem; letter-spacing: .08em; opacity: .9; }

/* welcome */
.welcome { text-align: center; padding: 36px 12px 8px; }
.script-olive { font-family: 'Caveat', 'Mali', cursive; color: #5E6B3C; font-size: 2rem; }
.welcome-title { font-family: 'Playfair Display', 'Noto Serif Thai', serif; color: #7B8750; font-size: clamp(2rem, 5vw, 3rem); line-height: 1.1; }
.stApp .welcome p { max-width: 560px; margin: 14px auto 0; line-height: 1.75; }
.quote { font-family: 'Caveat', 'Mali', cursive; color: #1F5A4A; font-size: clamp(1.5rem, 3.6vw, 2.1rem); line-height: 1.25; text-align: center; padding: 26px 16px 8px; }
.marquee { font-family: 'Playfair Display', 'Noto Serif Thai', serif; color: #7B8750; font-size: clamp(2rem, 6vw, 3.8rem); white-space: nowrap; overflow: hidden; border-bottom: 1px solid #7B8750; padding: 6px 0 10px; margin: 6px 0 22px; }

/* components */
.stButton > button, .stDownloadButton > button {
    width: 100%; background: #FDCB7A; color: #3a2f12; border: 0; border-radius: 3px; font-weight: 600; padding: .6rem 1rem;
}
.stButton > button:hover, .stDownloadButton > button:hover { background: #f6bd5c; color: #3a2f12; }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid #1F5A4A; outline-offset: 2px; }
[data-testid="stMetric"] { background: #FBF8EE; border: 1px solid #D9D2BA; border-radius: 3px; padding: 14px 16px; }
[data-testid="stMetricValue"] { font-family: 'Playfair Display', serif; color: #7B8750; }
[data-testid="stAlert"] { background: #FBF8EE; border: 1px solid #D9D2BA; color: #34332A; }
.stTabs [data-baseweb="tab"] { color: #34332A; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #1F5A4A; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #7B8750; }
.verdict { background: #FBF8EE; border-left: 4px solid #7B8750; padding: 14px 18px; border-radius: 3px; margin: 8px 0 18px; line-height: 1.7; }
.verdict.no { border-left-color: #D9A441; }
.stApp .verdict b { color: #1F5A4A; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(block):
    """ส่ง HTML เข้า Streamlit (ตัดการย่อหน้าและบรรทัดว่างออก เพื่อไม่ให้ถูกตีเป็น code block)"""
    st.markdown("\n".join(l.strip() for l in block.strip().splitlines() if l.strip()), unsafe_allow_html=True)


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def style_axes(fig, ax):
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(OLIVE)
    ax.grid(True, color=LINE, alpha=0.7, linewidth=0.6)


def style_legend(ax, **kw):
    leg = ax.legend(facecolor=PANEL, edgecolor=LINE, fontsize=8, **kw)
    for t in leg.get_texts():
        t.set_color(INK)


# ─────────────────────────────  Data loading  ─────────────────────────────
def get_currency_info(tickers):
    """ตรวจสอบสกุลเงินอัตโนมัติจาก Ticker Suffix"""
    if not tickers:
        return "THB", "บาท"
    if all(t.endswith(".BK") for t in tickers):
        return "THB", "บาท"
    if all(t.endswith(".T") for t in tickers):
        return "JPY", "เยน"
    if all(t.endswith(".HK") for t in tickers):
        return "HKD", "ดอลลาร์ฮ่องกง"
    if all(not t.endswith((".BK", ".T", ".HK")) for t in tickers):
        return "USD", "ดอลลาร์สหรัฐ"
    return "USD / Mixed", "สกุลเงินผสม"


# หมายเหตุ: ฟังก์ชันที่ติดแคช (_download_*, _fetch_*) จะ "raise" เมื่อดึงข้อมูลไม่ได้
# เพราะ Streamlit ไม่แคชผลที่เป็น exception — ถ้าคืนค่าว่างแล้วถูกแคชไว้ ข้อผิดพลาดชั่วคราวจะค้างอยู่ 12 ชม.
def _date_range(years_back):
    end = datetime.today()
    start = end - timedelta(days=int(years_back * 365.25))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


@st.cache_data(ttl=43200, show_spinner=False)
def _download_prices(tickers, years_back):
    import yfinance as yf
    start_date, end_date = _date_range(years_back)
    raw = yf.download(list(tickers), start=start_date, end=end_date, progress=False, auto_adjust=True)["Close"]
    if raw.empty:
        raise ValueError("ไม่พบข้อมูลราคาหุ้นสำหรับรหัสที่ระบุ")
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(tickers[0])
    if getattr(raw.index, "tz", None) is not None:
        raw.index = raw.index.tz_localize(None)
    return raw.ffill()


def load_price_data(tickers, years_back, min_obs=60):
    """ดึงราคาปิดปรับปันผล/แตกพาร์แล้ว (auto_adjust) พร้อมตัด timezone
    min_obs = จำนวนวันขั้นต่ำที่หุ้นแต่ละตัวต้องมี (ควรไม่น้อยกว่า Train+Test)"""
    try:
        raw = _download_prices(tuple(tickers), years_back)
    except Exception as e:
        return None, [], f"ไม่สามารถดึงข้อมูลจาก Yahoo Finance ได้ในขณะนี้ ({e})"
    valid = [t for t in tickers if t in raw.columns and raw[t].notna().sum() >= min_obs]
    if not valid:
        return None, [], f"ไม่พบหุ้นที่มีข้อมูลอย่างน้อย {min_obs} วันทำการ"
    return raw[valid].dropna(), valid, None


@st.cache_data(ttl=43200, show_spinner=False)
def _fetch_index(sym, years_back):
    import yfinance as yf
    start_date, end_date = _date_range(years_back)
    raw = yf.download(sym, start=start_date, end=end_date, progress=False)
    if raw.empty:
        raise ValueError(f"no data for {sym}")
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        col = "Close" if "Close" in lvl0 else ("Adj Close" if "Adj Close" in lvl0 else None)
        series = raw[col].iloc[:, 0] if col else raw.iloc[:, 0]
    else:
        col = "Close" if "Close" in raw.columns else ("Adj Close" if "Adj Close" in raw.columns else None)
        series = raw[col] if col else raw.iloc[:, 0]
    series = series.dropna()
    if len(series) <= 10:
        raise ValueError(f"too few rows for {sym}")
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_localize(None)
    return series


def load_benchmark(bench_choice, tickers, years_back):
    """ดึงดัชนีอ้างอิง (ลองสัญลักษณ์สำรองตามลำดับ)"""
    symbol_map = {
        "🇹🇭 SET Index (หุ้นไทย)": (["^SET.BK", "TDEX.BK"], "SET Index"),
        "🇺🇸 S&P 500 (หุ้นใหญ่สหรัฐฯ)": (["^GSPC", "SPY"], "S&P 500"),
        "🚀 Nasdaq 100 (หุ้นเทคฯ สหรัฐฯ)": (["^NDX", "QQQ"], "Nasdaq 100"),
        "🇯🇵 Nikkei 225 (หุ้นญี่ปุ่น)": (["^N225", "EWJ"], "Nikkei 225"),
        "🇭🇰 Hang Seng Index (หุ้นฮ่องกง)": (["^HSI", "2800.HK"], "Hang Seng"),
        "🌐 MSCI ACWI (หุ้นทั่วโลก / พอร์ตผสม)": (["ACWI", "URTH"], "MSCI ACWI"),
    }
    if bench_choice == "⚡ อัตโนมัติ (Auto-detect)":
        if all(t.endswith(".BK") for t in tickers):
            symbols, label = ["^SET.BK", "TDEX.BK"], "SET Index"
        elif all(t.endswith(".T") for t in tickers):
            symbols, label = ["^N225", "EWJ"], "Nikkei 225"
        elif all(t.endswith(".HK") for t in tickers):
            symbols, label = ["^HSI", "2800.HK"], "Hang Seng"
        elif all(not t.endswith((".BK", ".T", ".HK")) for t in tickers):
            symbols, label = ["^GSPC", "SPY"], "S&P 500"
        else:
            symbols, label = ["ACWI", "URTH", "^GSPC"], "MSCI ACWI"
    else:
        symbols, label = symbol_map.get(bench_choice, (["^SET.BK", "TDEX.BK"], "Benchmark"))

    for sym in symbols:
        try:
            return _fetch_index(sym, years_back), sym, label
        except Exception:
            continue
    return pd.Series(dtype=float), symbols[0], label


# จำนวนหุ้นจดทะเบียนสำรอง ใช้เมื่อ Yahoo ไม่ตอบ (ปล่อยว่างไว้ได้)
# ถ้าจะเติม ให้ใช้ตัวเลขที่ตรวจจากเว็บไซต์ SET/งบการเงินเอง เช่น {"PTT.BK": 28_562_996_250}
SHARES_FALLBACK = {}


@st.cache_data(ttl=43200, show_spinner=False)
def _fetch_shares_one(t):
    import yfinance as yf
    for attempt in range(3):
        try:
            tk = yf.Ticker(t)
            s = None
            try:
                info = tk.info
                s = _clean_shares(info.get("sharesOutstanding"))
                if s is None:
                    mc = info.get("marketCap")
                    price = info.get("regularMarketPrice") or info.get("previousClose")
                    if mc and price:
                        s = _clean_shares(mc / price)
            except Exception:
                s = None
            if s is None:
                s = _clean_shares(getattr(tk.fast_info, "shares", None))
            if s is not None:
                return s
        except Exception:
            pass
        time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"no shares data for {t}")


def load_shares_outstanding(tickers):
    """จำนวนหุ้นจดทะเบียน (ใช้ค่าปัจจุบันทุกรอบ — เป็นข้อจำกัดของข้อมูลฟรี)
    คืน (dict จำนวนหุ้น, รายชื่อที่ต้องใช้ค่าสำรอง)"""
    shares, used_fallback = {}, []
    for t in tickers:
        try:
            shares[t] = _fetch_shares_one(t)
        except Exception:
            fb = SHARES_FALLBACK.get(t)
            shares[t] = fb
            if fb is not None:
                used_fallback.append(t)
    return shares, used_fallback


def _clean_shares(v):
    try:
        return None if v is None or not np.isfinite(float(v)) or float(v) <= 0 else float(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────  Portfolio maths  ─────────────────────────────
def ledoit_wolf_shrinkage(X):
    """Ledoit-Wolf (2004) shrink เข้าหา μI (μ = ค่าเฉลี่ยของความแปรปรวนรายตัว)
    δ = min(b², d²) / d²  โดย d² = ||S − μI||²_F และ b² = (1/T²) Σ_t ||x_t x_tᵀ − S||²_F"""
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T
    m = np.trace(S) / N
    F = m * np.eye(N)
    d2 = np.sum((S - F) ** 2)
    if d2 <= 1e-18:
        return S, 0.0
    outer = np.einsum("ti,tj->tij", Xc, Xc)
    b2 = np.sum((outer - S) ** 2) / T ** 2
    delta = min(b2, d2) / d2
    return delta * F + (1 - delta) * S, float(delta)


def shrink_mean(mu, cov, T):
    """Bayes-Stein (Jorion, 1986): ดึงค่าเฉลี่ยผลตอบแทนเข้าหาค่าเฉลี่ยของพอร์ต Min-Variance
    λ = (N+2) / ((N+2) + T·(μ−μ₀1)ᵀΣ⁻¹(μ−μ₀1))  — ข้อมูลยิ่งน้อย/μ กระจายยิ่งมาก λ ยิ่งสูง"""
    N = len(mu)
    inv = np.linalg.pinv(cov)
    ones = np.ones(N)
    mu0 = float(ones @ inv @ mu) / float(ones @ inv @ ones)
    d = mu - mu0 * ones
    quad = float(d @ inv @ d)
    lam = (N + 2) / ((N + 2) + T * quad) if quad > 1e-18 else 1.0
    lam = float(np.clip(lam, 0.0, 1.0))
    return (1 - lam) * mu + lam * mu0 * ones, lam


def _clean_w(w, fallback):
    w = np.clip(np.asarray(w, dtype=float), 0, 1)
    s = w.sum()
    return fallback if (not np.isfinite(s) or s <= 1e-12) else w / s


def _max_sharpe(mu, cov, rf_rate, w0, rng, n_starts=8):
    """Max-Sharpe (long-only) ด้วย SLSQP หลายจุดเริ่มต้น (Sharpe ไม่ใช่ปัญหา convex) คืนผลที่ดีที่สุดหรือ None"""
    n = len(mu)

    def neg_sharpe(w):
        ret = float(mu @ w) * 252 - rf_rate
        vol = np.sqrt(max(float(w @ cov @ w), 0.0)) * np.sqrt(252)
        return -ret / vol if vol > 1e-10 else 1e6

    bounds = tuple((0, 1) for _ in range(n))
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    starts = [w0] + list(rng.dirichlet(np.ones(n), size=n_starts))
    best = None
    for s in starts:
        r = minimize(neg_sharpe, s, method="SLSQP", bounds=bounds, constraints=cons)
        if r.success and np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return best


def _min_variance(cov, w0):
    """Min-Variance (long-only) — ใช้เมื่อผลตอบแทนคาดหวังของทุกหุ้นไม่เกิน Rf (ซึ่ง Max-Sharpe ใช้ไม่ได้)"""
    n = len(w0)
    C = cov * 1e4   # สเกลให้ตัวแก้ปัญหาเห็นค่าไม่เล็กเกินไป
    r = minimize(lambda w: float(w @ C @ w), w0, jac=lambda w: 2 * C @ w, method="SLSQP",
                 bounds=tuple((0, 1) for _ in range(n)), constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1})
    return r


def get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap, rf_rate):
    """คำนวณน้ำหนักพอร์ตของแต่ละกลยุทธ์จากข้อมูลช่วง Train เท่านั้น
    คืน (weights, info) โดย info มี ok / mv_fallback / delta / lam"""
    mu, cov = train_ret.mean().values, train_ret.cov().values
    T = len(train_ret)
    w_equal = np.full(n, 1.0 / n)
    rng = np.random.default_rng(42)   # ตายตัว เพื่อให้ผลซ้ำได้
    info = {"ok": True, "mv_fallback": 0}

    def solve(mu_arr, cov_arr):
        # Max-Sharpe ต้องมีอย่างน้อยหนึ่งหุ้นที่ผลตอบแทนคาดหวังเกิน Rf
        # ไม่งั้นการ "ทำให้ Sharpe สูงสุด" จะกลายเป็นเลือกพอร์ตที่ผันผวนสุด (ตัวเศษติดลบ)
        if (mu_arr * 252 - rf_rate).max() <= 0:
            info["mv_fallback"] += 1
            r = _min_variance(cov_arr, w_equal)
            ok = bool(r.success)
        else:
            r = _max_sharpe(mu_arr, cov_arr, rf_rate, w_equal, rng)
            ok = r is not None
        info["ok"] = info["ok"] and ok
        return _clean_w(r.x, w_equal) if ok else w_equal

    w_markowitz = solve(mu, cov)

    shrunk_cov, delta = ledoit_wolf_shrinkage(train_ret.values)
    mu_s, lam = shrink_mean(mu, shrunk_cov, T)
    w_shrink = solve(mu_s, shrunk_cov)
    info["delta"], info["lam"] = delta, lam

    weights = {"equal": w_equal, "markowitz": w_markowitz}
    if use_marketcap and shares_arr is not None and len(shares_arr) == n:
        mcap = np.asarray(train_last_price, dtype=float) * np.asarray(shares_arr, dtype=float)
        weights["mcap"] = mcap / mcap.sum() if mcap.sum() > 0 else w_equal
    weights["shrinkage"] = w_shrink
    return weights, info


def evaluate(r, rf_rate):
    """ตัวชี้วัดจากอนุกรมผลตอบแทนรายวัน (หลังหักค่าธรรมเนียม)"""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return dict(total=np.nan, ann_return=np.nan, cagr=np.nan, ann_vol=np.nan, sharpe=np.nan, max_drawdown=np.nan)
    cum = np.cumprod(1 + r)
    total = cum[-1] - 1
    years = len(r) / 252
    cagr = cum[-1] ** (1 / years) - 1 if cum[-1] > 0 else np.nan
    ann_ret = r.mean() * 252
    ann_vol = r.std(ddof=1) * np.sqrt(252)
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 1e-10 else np.nan
    peak = np.maximum.accumulate(np.concatenate([[1.0], cum]))[1:]
    mdd = ((cum - peak) / peak).min()
    return dict(total=total, ann_return=ann_ret, cagr=cagr, ann_vol=ann_vol, sharpe=sharpe, max_drawdown=mdd)


def run_walk_forward(data, shares_arr, use_marketcap, train_window, test_window, rf_rate, capital, cost_pct):
    """Walk-Forward + จำลองการถือครองจริง: ซื้อครั้งแรก, ปล่อยน้ำหนักลอยตามราคา, ปรับพอร์ตทุกรอบ และหักค่าธรรมเนียมตามยอดซื้อ+ขายจริง"""
    all_returns = data.pct_change().dropna()
    n, cols = data.shape[1], list(data.columns)
    records, failed, deltas, lams, mv_fallbacks = [], 0, [], [], 0
    equity, holdings, total_cost = {}, {}, {}
    net_parts, curve_parts, weight_rows = {}, {}, {}
    start, round_num = 0, 0

    while start + train_window + test_window <= len(all_returns):
        round_num += 1
        train_ret = all_returns.iloc[start: start + train_window]
        test_ret = all_returns.iloc[start + train_window: start + train_window + test_window]
        train_last_price = data.loc[train_ret.index[-1]].values
        weights, info = get_weights(train_ret, train_last_price, shares_arr, n, use_marketcap, rf_rate)
        failed += 0 if info["ok"] else 1
        mv_fallbacks += info["mv_fallback"]
        deltas.append(info["delta"])
        lams.append(info["lam"])
        growth = np.cumprod(1 + test_ret.values, axis=0)   # T x n

        for key, w in weights.items():
            eq = equity.get(key, float(capital))
            h = holdings.get(key, np.zeros(n))
            traded = np.abs(eq * w - h).sum()               # มูลค่าที่ต้องซื้อ+ขาย
            cost = traded * cost_pct
            h0 = (eq - cost) * w
            path = growth @ h0
            net_r = path / np.concatenate([[eq], path[:-1]]) - 1
            equity[key], holdings[key] = path[-1], h0 * growth[-1]
            total_cost[key] = total_cost.get(key, 0.0) + cost

            m = evaluate(net_r, rf_rate)
            records.append({"round": round_num, "strategy": key, "test_start": test_ret.index[0],
                            "test_end": test_ret.index[-1], "total_return": path[-1] / eq - 1,
                            "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
                            "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]})
            net_parts.setdefault(key, []).append(pd.Series(net_r, index=test_ret.index))
            curve_parts.setdefault(key, []).append(pd.Series(path, index=test_ret.index))
            weight_rows.setdefault(key, []).append({"round": round_num, **dict(zip(cols, w))})
        start += test_window

    if round_num == 0:
        return None

    # สัดส่วนที่ "ใช้ได้จริงตอนนี้" คำนวณจากข้อมูล Train ล่าสุด (ไม่ใช่ของรอบทดสอบสุดท้าย)
    current_weights, _ = get_weights(all_returns.iloc[-train_window:], data.iloc[-1].values,
                                        shares_arr, n, use_marketcap, rf_rate)
    return {
        "folds": pd.DataFrame(records),
        "n_folds": round_num,
        "n_failed": failed,
        "net_daily": {k: pd.concat(v) for k, v in net_parts.items()},
        "curves": {k: pd.concat(v) for k, v in curve_parts.items()},
        "weight_history": {k: pd.DataFrame(v).set_index("round") for k, v in weight_rows.items()},
        "total_cost": total_cost,
        "avg_delta": float(np.mean(deltas)),
        "avg_lambda": float(np.mean(lams)),
        "n_mv_fallback": mv_fallbacks,
        "returns": all_returns,
        "current_weights": current_weights,
    }


def paired_tests(folds, order):
    """Paired t-test ของ Sharpe ต่อรอบ ระหว่างทุกคู่กลยุทธ์ + ช่วงความเชื่อมั่น 95% ของค่าเฉลี่ย"""
    piv = folds.pivot(index="round", columns="strategy", values="sharpe")[order]
    rows = []
    m_tests = len(order) * (len(order) - 1) // 2   # จำนวนคู่ที่เปรียบเทียบ (ใช้ปรับ Bonferroni)
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a, b = order[i], order[j]
            pair = piv[[a, b]].dropna()
            if len(pair) < 3:
                continue
            diff = pair[a] - pair[b]
            if diff.std(ddof=1) < 1e-12:
                t_stat, p_val = np.nan, 1.0
            else:
                t_stat, p_val = stats.ttest_rel(pair[a], pair[b])
            rows.append({"คู่เปรียบเทียบ": f"{STRAT_DISPLAY_NAMES[a]}  vs  {STRAT_DISPLAY_NAMES[b]}",
                         "ส่วนต่าง Sharpe เฉลี่ย": diff.mean(), "t": t_stat, "p-value": p_val,
                         "p (Bonferroni)": min(1.0, p_val * m_tests),
                         "ผล (p ดิบ < 0.05)": "ต่างกัน" if p_val < 0.05 else "ไม่ต่างกัน",
                         "ผล (หลังปรับ Bonferroni)": "ต่างกัน" if p_val * m_tests < 0.05 else "ไม่ต่างกัน"})
    ci = {}
    for k in order:
        s = piv[k].dropna()
        if len(s) >= 3:
            half = stats.t.ppf(0.975, len(s) - 1) * s.std(ddof=1) / np.sqrt(len(s))
            ci[k] = (s.mean(), half)
    return pd.DataFrame(rows), ci


def create_share_card(best_key, cagr, sharpe, max_dd, tickers, capital, curr_name, beats, infl):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(PANEL)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(OLIVE)
    ax.text(0.05, 0.88, "QUANTLAB PORTFOLIO SUMMARY", fontsize=15, fontweight="bold", color=OLIVE, family="serif")
    ticker_str = ", ".join(tickers[:6]) + ("..." if len(tickers) > 6 else "")
    ax.text(0.05, 0.80, f"Assets: {ticker_str}", fontsize=9, color=INK)
    ax.text(0.05, 0.66, "BEST STRATEGY (by Sharpe)", fontsize=9, fontweight="bold", color=INK)
    ax.text(0.05, 0.54, STRAT_DISPLAY_NAMES.get(best_key, best_key), fontsize=17, fontweight="bold", color=FOREST, family="serif")
    for x, label, val, col in [(0.05, "ANNUAL RETURN (CAGR)", f"{cagr * 100:+.1f}%", INK),
                               (0.38, "SHARPE RATIO", f"{sharpe:.2f}", "#B98318"),
                               (0.70, "MAX DRAWDOWN", f"{max_dd * 100:.1f}%", "#A34A3F")]:
        ax.text(x, 0.35, label, fontsize=8, color="#6b6a58")
        ax.text(x, 0.22, val, fontsize=16, fontweight="bold", color=col)
    verdict = f"Beats {infl * 100:.1f}% inflation" if beats else f"Below {infl * 100:.1f}% inflation"
    ax.text(0.05, 0.12, verdict, fontsize=9, color=FOREST if beats else "#A34A3F", fontweight="bold")
    ax.text(0.05, 0.05, f"Initial capital: {capital:,.0f} {curr_name}  |  Past results do not guarantee future returns",
            fontsize=7, color="#6b6a58")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


# ─────────────────────────────  Header  ─────────────────────────────
html("""
<div class="hero">
<div class="hero-title">PORTFOLIOS, RISK, &amp; EVIDENCE</div>
<div class="hero-script">for student investors</div>
<p class="hero-sub">เครื่องมือจำลองการจัดพอร์ตหุ้นสำหรับนักเรียนที่มีเงินน้อยและไม่มีเวลาเฝ้าหุ้น ทดสอบกับข้อมูลที่โมเดลไม่เคยเห็นมาก่อน</p>
<div class="hero-hint">ปรับค่าทางซ้ายมือ แล้วกด “เริ่มวิเคราะห์”</div>
</div>
""")

# ─────────────────────────────  Sidebar  ─────────────────────────────
with st.sidebar:
    st.header("ตั้งค่าการวิเคราะห์")

    if "ticker_text" not in st.session_state:
        st.session_state.ticker_text = "PTT.BK, CPALL.BK, AOT.BK, KBANK.BK, ADVANC.BK"

    ticker_input = st.text_input(
        "รหัสหุ้น (คั่นด้วยจุลภาค ,)",
        key="ticker_text",
        help="ใช้รหัสแบบ Yahoo Finance เช่น PTT.BK, AAPL, 7203.T, 0700.HK",
    )
    selected_raw = list(dict.fromkeys([t.strip().upper() for t in ticker_input.split(",") if t.strip()]))
    if len(selected_raw) > MAX_TICKERS_ALLOWED:
        st.warning(f"จำกัดสูงสุด {MAX_TICKERS_ALLOWED} หุ้น ระบบใช้ {MAX_TICKERS_ALLOWED} ตัวแรกให้")
        selected = selected_raw[:MAX_TICKERS_ALLOWED]
    else:
        selected = selected_raw

    curr_code, curr_name = get_currency_info(selected)
    capital = st.number_input(f"เงินลงทุนเริ่มต้น ({curr_name} / {curr_code})",
                              min_value=500, value=DEFAULT_CAPITAL, step=500)

    infl_pct = st.slider("เงินเฟ้อที่ใช้เป็นเป้าหมาย (%/ปี)", 0.0, 6.0, 2.5, step=0.25,
                         help="ใช้ตัดสินว่าพอร์ตชนะเงินเฟ้อหรือไม่ (โครงร่างตั้งไว้ราว 2–3%)")
    inflation = infl_pct / 100

    benchmark_option = st.selectbox(
        "ตัวเปรียบเทียบ (Benchmark)",
        ["⚡ อัตโนมัติ (Auto-detect)", "🇹🇭 SET Index (หุ้นไทย)", "🇺🇸 S&P 500 (หุ้นใหญ่สหรัฐฯ)",
         "🚀 Nasdaq 100 (หุ้นเทคฯ สหรัฐฯ)", "🇯🇵 Nikkei 225 (หุ้นญี่ปุ่น)", "🇭🇰 Hang Seng Index (หุ้นฮ่องกง)",
         "🌐 MSCI ACWI (หุ้นทั่วโลก / พอร์ตผสม)"],
        help="เลือกดัชนีอ้างอิงเพื่อเทียบกับพอร์ตของคุณ",
    )

    with st.expander("ตั้งค่าการทดสอบขั้นสูง"):
        rf_rate_pct = st.slider("อัตราผลตอบแทนปราศจากความเสี่ยง Rf (%/ปี)", 0.0, 5.0, 0.0, step=0.25) / 100
        train_window = st.slider("ช่วง Train (วันทำการ)", 126, 378, 252, step=21)
        test_window = st.slider("ช่วง Test ต่อรอบ (วันทำการ)", 21, 126, 63, step=21)
        years_back = st.slider("ข้อมูลย้อนหลัง (ปี)", 3, 25, 9,
                               help="ช่วงทดสอบจริงเริ่มหลังช่วง Train แรก (≈1 ปี) · ย้อนหลัง 9 ปีจึงครอบคลุมช่วง COVID-19 และมีรอบทดสอบมากพอสำหรับ t-test")
        cost_pct = st.slider("ค่าธรรมเนียมซื้อขาย (% ของยอดที่ซื้อ/ขาย)", 0.0, 1.0, 0.15, step=0.05,
                             help="ตั้งตามโบรกเกอร์ที่ใช้จริง (รวม VAT) · โมเดลยังไม่รวมค่าธรรมเนียมขั้นต่ำต่อรายการ") / 100
        stress_option = st.selectbox("ดูผลเฉพาะช่วงวิกฤต", ["ไม่ระบุ"] + list(STRESS_WINDOWS.keys()))

    if st.button("เริ่มวิเคราะห์", type="primary"):
        st.session_state["has_run"] = True

if not st.session_state.get("has_run"):
    html("""
    <div class="welcome">
    <div class="script-olive">Welcome, Friend!</div>
    <div class="welcome-title">QUANTLAB</div>
    <p>กำหนดสัดส่วนเงินลงทุนล่วงหน้า ปรับพอร์ตทุกไตรมาส แล้วดูว่า 4 วิธีจัดพอร์ตวิธีไหนทำผลได้ดีที่สุดกับข้อมูลที่ไม่เคยเห็นมาก่อน และชนะเงินเฟ้อหรือไม่</p>
    </div>
    <div class="quote">ลงทุนแบบเรียบง่าย ไม่ต้องเฝ้าจอ และตรวจสอบผลได้จริง</div>
    <div class="marquee">RETURN / RISK / SHARPE / DRAWDOWN / INFLATION / RETURN / RISK / SHARPE</div>
    """)
    st.info("ตั้งค่าทางซ้ายมือ แล้วกด **เริ่มวิเคราะห์** เพื่อประมวลผล (ค่าเริ่มต้นคือหุ้นไทย 5 ตัวตามโครงร่าง เงินตั้งต้น 5,000 บาท)")
    st.stop()

if len(selected) < 2:
    st.error("ใส่รหัสหุ้นอย่างน้อย 2 ตัวเพื่อจัดพอร์ตกระจายความเสี่ยง")
    st.stop()

# ─────────────────────────────  Compute  ─────────────────────────────
with st.spinner("กำลังดึงราคาหุ้นและประมวลผล..."):
    data, valid_tickers, error_msg = load_price_data(tuple(selected), years_back, train_window + test_window + 2)
    if error_msg:
        st.error(f"ดึงข้อมูลไม่สำเร็จ: {error_msg}")
        st.stop()
    dropped = [t for t in selected if t not in valid_tickers]
    if dropped:
        st.warning(f"ไม่พบข้อมูลที่พอใช้ของ: {', '.join(dropped)} (ตัดออกจากการวิเคราะห์)")
    selected = valid_tickers
    if len(selected) < 2:
        st.error("เหลือหุ้นที่ข้อมูลสมบูรณ์น้อยกว่า 2 ตัว กรุณาเปลี่ยนรหัสหุ้น")
        st.stop()

    expected_days = years_back * 252 * 0.9
    if len(data) < 0.7 * expected_days:
        st.warning(f"ข้อมูลที่ทุกหุ้นมีร่วมกันเหลือเพียง {len(data):,} วันทำการ (น้อยกว่าที่ขอย้อนหลัง {years_back} ปีมาก) "
                   "หุ้นบางตัวอาจเพิ่งจดทะเบียน ผลทดสอบจึงสั้นลง")

    shares_raw, shares_fb = load_shares_outstanding(tuple(selected))
    if shares_fb:
        st.warning(f"ดึงจำนวนหุ้นของ {', '.join(shares_fb)} จาก Yahoo ไม่ได้ จึงใช้ค่าสำรองในโค้ด (SHARES_FALLBACK)")
    shares_clean = {t: _clean_shares(shares_raw.get(t)) for t in selected}
    use_marketcap = all(v is not None for v in shares_clean.values())
    shares_arr = np.array([shares_clean[t] for t in selected]) if use_marketcap else None
    if not use_marketcap:
        missing = [t for t, v in shares_clean.items() if v is None]
        st.warning(f"ไม่มีข้อมูลจำนวนหุ้นจดทะเบียนของ {', '.join(missing)} จึงข้ามกลยุทธ์ C (Market-Cap) แทนที่จะเดาตัวเลข")

    res = run_walk_forward(data, shares_arr, use_marketcap, train_window, test_window, rf_rate_pct, capital, cost_pct)

if res is None:
    st.error("ข้อมูลย้อนหลังไม่พอสำหรับแบ่งรอบทดสอบ ลองลดช่วง Train/Test หรือเพิ่มจำนวนปีย้อนหลัง")
    st.stop()

folds, curves, net_daily = res["folds"], res["curves"], res["net_daily"]
order = [k for k in STRAT_ORDER if k in curves]
overall = {k: evaluate(net_daily[k].values, rf_rate_pct) for k in order}
best_key = max(order, key=lambda k: overall[k]["sharpe"] if np.isfinite(overall[k]["sharpe"]) else -np.inf)
best = overall[best_key]
beats_n = sum(1 for k in order if np.isfinite(overall[k]["cagr"]) and overall[k]["cagr"] > inflation)

idx = next(iter(curves.values())).index
base_date = data.index[data.index.get_loc(idx[0]) - 1]
years_oos = len(idx) / 252

bench_raw, bench_sym, bench_label = load_benchmark(benchmark_option, tuple(selected), years_back)
bench_curve, bench_metrics = None, None
if not bench_raw.empty:
    b_idx = pd.DatetimeIndex([base_date]).append(idx)
    b = bench_raw.reindex(bench_raw.index.union(b_idx)).ffill().reindex(b_idx)
    if b.notna().all():
        bench_curve = capital * b.iloc[1:] / b.iloc[0]
        bench_metrics = evaluate(b.pct_change().dropna().values, rf_rate_pct)

# ─────────────────────────────  Tabs  ─────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 สรุป", "📈 การเติบโต", "🔬 สถิติเชิงลึก", "📣 แชร์", "ℹ️ ข้อมูล"])

with tab1:
    st.subheader(f"สรุปผล ({curr_code})")
    st.caption(f"ทดสอบนอกตัวอย่าง {res['n_folds']} รอบ ช่วง {idx[0]:%d/%m/%Y} – {idx[-1]:%d/%m/%Y} (ราว {years_oos:.1f} ปี) หลังหักค่าธรรมเนียมแล้ว")
    if res["n_failed"] > 0:
        st.warning(f"คำนวณ Markowitz ไม่ลู่เข้า {res['n_failed']}/{res['n_folds']} รอบ (ใช้ Equal-Weight แทนในรอบนั้น)")
    if res["n_mv_fallback"] > 0:
        st.info(f"มี {res['n_mv_fallback']} ครั้ง (รวมกลยุทธ์ B และ D) ที่ผลตอบแทนคาดหวังของทุกหุ้นในช่วง Train ไม่เกิน Rf "
                "Max-Sharpe ใช้ไม่ได้ในกรณีนี้ ระบบจึงใช้พอร์ตความเสี่ยงต่ำสุด (Min-Variance) แทนในรอบนั้น")

    if best["cagr"] > inflation:
        html(f"""<div class="verdict">กลยุทธ์ที่ Sharpe ดีที่สุดคือ <b>{STRAT_FULL_NAMES[best_key]}</b> ให้ผลตอบแทนทบต้น <b>{pct(best['cagr'])}</b> ต่อปี
        สูงกว่าเงินเฟ้อที่ตั้งไว้ {pct(inflation)} · มี {beats_n} จาก {len(order)} กลยุทธ์ที่ชนะเงินเฟ้อ</div>""")
    else:
        html(f"""<div class="verdict no">กลยุทธ์ที่ Sharpe ดีที่สุดคือ <b>{STRAT_FULL_NAMES[best_key]}</b> แต่ผลตอบแทนทบต้น <b>{pct(best['cagr'])}</b> ต่อปี
        ยังไม่ชนะเงินเฟ้อที่ตั้งไว้ {pct(inflation)} · มี {beats_n} จาก {len(order)} กลยุทธ์ที่ชนะเงินเฟ้อ</div>""")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ผลตอบแทนทบต้น/ปี", pct(best["cagr"]))
    c2.metric("ความผันผวน/ปี", pct(best["ann_vol"]))
    c3.metric("Sharpe ratio", f"{best['sharpe']:.2f}")
    c4.metric("Max Drawdown", pct(best["max_drawdown"]))

    st.markdown("#### สัดส่วนเงินที่ใช้ได้ตอนนี้")
    alloc_key = st.selectbox("เลือกกลยุทธ์", order, index=order.index(best_key),
                             format_func=lambda k: STRAT_FULL_NAMES[k])
    last_price = data.iloc[-1]
    plan = []
    for t, w in zip(selected, res["current_weights"][alloc_key]):
        amount = capital * w
        qty = int(amount // last_price[t]) if last_price[t] > 0 else 0
        plan.append({"รหัสหุ้น": t, "สัดส่วน": pct(w, 2), f"จำนวนเงิน ({curr_code})": f"{amount:,.2f}",
                     "ราคาล่าสุด": f"{last_price[t]:,.2f}", "ซื้อได้ (หุ้น)": qty})
    st.dataframe(pd.DataFrame(plan), hide_index=True)
    st.caption("คำนวณจากข้อมูล Train ล่าสุด · ถ้า “ซื้อได้” เป็น 0 แปลว่าเงินส่วนนั้นยังไม่พอซื้อ 1 หุ้น ลองลดจำนวนหุ้นหรือเพิ่มเงินทุน")

    st.markdown("#### เปรียบเทียบทุกกลยุทธ์ (คิดจากกราฟเงินทุนต่อเนื่องทั้งช่วงทดสอบ)")
    rows = []
    for k in order:
        m = overall[k]
        rows.append({"กลยุทธ์": STRAT_FULL_NAMES[k], "ผลตอบแทนเฉลี่ย/ปี": pct(m["ann_return"]),
                     "ทบต้น/ปี (CAGR)": pct(m["cagr"]), "ความผันผวน/ปี": pct(m["ann_vol"]),
                     "Sharpe": round(m["sharpe"], 2), "Max Drawdown": pct(m["max_drawdown"]),
                     f"เงินสุดท้าย ({curr_code})": f"{capital * (1 + m['total']):,.0f}",
                     "ค่าธรรมเนียมรวม": f"{res['total_cost'][k]:,.0f}",
                     "ชนะเงินเฟ้อ": "✅" if m["cagr"] > inflation else "❌"})
    if bench_metrics:
        rows.append({"กลยุทธ์": f"Benchmark: {bench_label}", "ผลตอบแทนเฉลี่ย/ปี": pct(bench_metrics["ann_return"]),
                     "ทบต้น/ปี (CAGR)": pct(bench_metrics["cagr"]), "ความผันผวน/ปี": pct(bench_metrics["ann_vol"]),
                     "Sharpe": round(bench_metrics["sharpe"], 2), "Max Drawdown": pct(bench_metrics["max_drawdown"]),
                     f"เงินสุดท้าย ({curr_code})": f"{capital * (1 + bench_metrics['total']):,.0f}",
                     "ค่าธรรมเนียมรวม": "–", "ชนะเงินเฟ้อ": "✅" if bench_metrics["cagr"] > inflation else "❌"})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("“กลยุทธ์ที่ดีที่สุด” เลือกย้อนหลังจากผลทดสอบชุดเดียวกัน ไม่ใช่การพยากรณ์ว่าจะดีที่สุดต่อไป · "
               "ผลในอดีตไม่รับประกันผลในอนาคต และไม่ใช่คำแนะนำการลงทุน")

with tab2:
    st.subheader(f"เงินลงทุนสะสม ({curr_code}) — ทบต้นจริง หลังหักค่าธรรมเนียม")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k in order:
        ax.plot(curves[k].index, curves[k].values, label=STRAT_DISPLAY_NAMES[k], color=STRAT_COLORS[k], linewidth=1.8)
    if bench_curve is not None:
        ax.plot(bench_curve.index, bench_curve.values, label=f"{bench_label} ({bench_sym})", color=INK,
                linewidth=1.4, linestyle="--")
    days = (idx - base_date).days.values
    ax.plot(idx, capital * (1 + inflation) ** (days / 365.25), color="#B98318", linestyle=":", linewidth=1.6,
            label=f"Inflation {inflation * 100:.1f}%/yr")
    ax.axhline(capital, color="#9a9783", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Portfolio value ({curr_code})")
    style_axes(fig, ax)
    style_legend(ax, loc="upper left")
    st.pyplot(fig)
    plt.close(fig)
    if bench_curve is None:
        st.caption("ดึงข้อมูล Benchmark ไม่ได้ในช่วงนี้ จึงไม่แสดงเส้นเปรียบเทียบ")

with tab3:
    st.subheader("วิเคราะห์สถิติเชิงลึก")

    tests, ci = paired_tests(folds, order)
    st.markdown("#### Sharpe ratio ต่อรอบ พร้อมช่วงความเชื่อมั่น 95%")
    if ci:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for i, k in enumerate(order):
            if k in ci:
                mean, half = ci[k]
                ax.errorbar(i, mean, yerr=half, fmt="o", color=STRAT_COLORS[k], capsize=6, markersize=7, linewidth=2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([STRAT_DISPLAY_NAMES[k] for k in order], fontsize=8)
        ax.axhline(0, color="#9a9783", linewidth=0.8)
        ax.set_ylabel("Mean fold Sharpe (95% CI)")
        ax.set_xlim(-0.5, len(order) - 0.5)
        style_axes(fig, ax)
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("#### Paired t-test (เทียบ Sharpe ทีละรอบทดสอบ)")
    if tests.empty:
        st.info("จำนวนรอบทดสอบน้อยเกินไปสำหรับ t-test (ต้องมีอย่างน้อย 3 รอบ)")
    else:
        show = tests.copy()
        show["ส่วนต่าง Sharpe เฉลี่ย"] = show["ส่วนต่าง Sharpe เฉลี่ย"].round(3)
        show["t"] = show["t"].round(3)
        show["p-value"] = show["p-value"].round(4)
        show["p (Bonferroni)"] = show["p (Bonferroni)"].round(4)
        st.dataframe(show, hide_index=True)
        st.caption(f"n = {res['n_folds']} รอบ · เปรียบเทียบ {len(show)} คู่พร้อมกัน จึงแสดง p ที่ปรับแบบ Bonferroni (p × จำนวนคู่) คู่กับ p ดิบ · "
                   "รอบละไม่กี่เดือน ข้อมูลน้อย ผลจึงควรอ่านอย่างระมัดระวัง")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ความเข้มของการลดสัญญาณรบกวน (กลยุทธ์ D)")
        st.info(f"Ledoit-Wolf δ (ของ Σ) เฉลี่ย = **{res['avg_delta']:.4f}** · Bayes-Stein λ (ของ μ) เฉลี่ย = **{res['avg_lambda']:.4f}**\n\n"
                "ค่ายิ่งใกล้ 1 แปลว่าค่าประมาณจากข้อมูล Train มีสัญญาณรบกวนมาก ระบบจึงถ่วงเข้าหาเป้าหมายที่เรียบกว่า")
    with col_b:
        st.markdown("#### Max Drawdown ตลอดช่วงทดสอบ")
        mdd_df = pd.DataFrame({"Max Drawdown": [pct(overall[k]["max_drawdown"]) for k in order]},
                              index=[STRAT_FULL_NAMES[k] for k in order])
        st.dataframe(mdd_df)

    st.markdown("#### สัดส่วนน้ำหนักในแต่ละรอบ")
    wh_key = st.selectbox("เลือกกลยุทธ์", order, index=order.index(best_key),
                          format_func=lambda k: STRAT_FULL_NAMES[k], key="wh_key")
    wh = res["weight_history"][wh_key]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    palette = [OLIVE, "#D9A441", FOREST, "#9B6B9E", "#B8736A", "#6E9BA8", "#A8A15C", "#7A5C45", "#C6B58A", "#4F6B8A"]
    ax.stackplot(wh.index, wh.T.values, labels=list(wh.columns), colors=palette[:wh.shape[1]], alpha=0.9)
    ax.set_xlabel("Round")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    style_axes(fig, ax)
    style_legend(ax, loc="upper center", ncol=min(5, wh.shape[1]), bbox_to_anchor=(0.5, -0.18))
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### ความสัมพันธ์ของผลตอบแทนหุ้น (Correlation)")
    corr = res["returns"].corr()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8, color=INK)
    ax.set_yticklabels(corr.columns, fontsize=8, color=INK)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.patch.set_facecolor(CREAM)
    st.pyplot(fig)
    plt.close(fig)

    if stress_option != "ไม่ระบุ":
        s, e = STRESS_WINDOWS[stress_option]
        st.markdown(f"#### ผลในช่วง: {stress_option}")
        srows = []
        for k in order:
            seg = net_daily[k].loc[s:e]
            if len(seg) >= 10:
                m = evaluate(seg.values, rf_rate_pct)
                srows.append({"กลยุทธ์": STRAT_FULL_NAMES[k], "ผลตอบแทนรวมในช่วง": pct(m["total"]),
                              "Max Drawdown": pct(m["max_drawdown"]), "จำนวนวัน": len(seg)})
        if srows:
            st.dataframe(pd.DataFrame(srows), hide_index=True)
        else:
            st.info(f"ช่วงทดสอบนอกตัวอย่างของคุณเริ่ม {idx[0]:%d/%m/%Y} จึงยังไม่ครอบคลุมช่วงนี้ ลองเพิ่มข้อมูลย้อนหลังในเมนูขั้นสูง (ต้องเผื่อช่วง Train ก่อนหน้าด้วย)")

with tab4:
    st.subheader("แชร์สรุปผลของคุณ")
    card_buf = create_share_card(best_key, best["cagr"], best["sharpe"], best["max_drawdown"], selected,
                                 capital, curr_name, best["cagr"] > inflation, inflation)
    st.image(card_buf, width=600)
    st.download_button("ดาวน์โหลดการ์ดสรุปผล (PNG)", data=card_buf, file_name="quantlab_summary.png", mime="image/png")

with tab5:
    st.subheader("วิธีคำนวณและข้อจำกัด")
    st.markdown("""
- **Walk-Forward Validation:** คำนวณสัดส่วนจากช่วง Train แล้วนำไปใช้กับช่วง Test ถัดไปที่ไม่เคยใช้คำนวณ จากนั้นเลื่อนไปข้างหน้า ป้องกัน Look-ahead bias
- **ตัวชี้วัดรวม:** Sharpe, ความผันผวน และ Max Drawdown ในตารางสรุปคิดจากกราฟเงินทุนที่ต่อกันตลอดทุกรอบ ส่วน t-test ใช้ Sharpe ต่อรอบ
- **ค่าธรรมเนียม:** หักจากยอดที่ซื้อและขายจริงในแต่ละรอบ (รวมการซื้อครั้งแรก) โดยน้ำหนักลอยตามราคาระหว่างรอบ
- **ผลตอบแทนทบต้น (CAGR):** ใช้ตัดสินว่าชนะเงินเฟ้อหรือไม่ ส่วน “ผลตอบแทนเฉลี่ย/ปี” คือค่าเฉลี่ยรายวัน × 252 ตามนิยามในโครงร่าง
- **ราคาหุ้น:** ใช้ราคาปรับปันผลและแตกพาร์แล้ว (รวมปันผล) แต่ดัชนีอ้างอิงบางตัวเป็น Price Index ไม่รวมปันผล จึงเทียบได้แบบคร่าว ๆ
- **Market-Cap:** ใช้จำนวนหุ้นจดทะเบียนปัจจุบันคูณราคาปิดวันสิ้นสุด Train เพราะข้อมูลฟรีไม่มีจำนวนหุ้นย้อนหลัง อีกทั้งราคาเป็นราคาปรับปันผลแล้ว (ราคาเก่าถูกปรับลดตามปันผลสะสม) และไม่ได้ปรับ free-float เหมือนดัชนี SET50 จึงเป็นเพียงการประมาณการถ่วงตามมูลค่าตลาด
- **Markowitz (B):** หาน้ำหนักที่ทำให้ Sharpe สูงสุดแบบห้าม short ด้วย SLSQP หลายจุดเริ่มต้น (สุ่มแบบตายตัวเพื่อให้ผลซ้ำได้) · ถ้าผลตอบแทนคาดหวังของทุกหุ้นไม่เกิน Rf จะใช้ Min-Variance แทน เพราะ Max-Sharpe ไม่มีความหมายเมื่อตัวเศษติดลบ
- **Markowitz-Shrinkage (D):** ลดสัญญาณรบกวนทั้ง Σ (Ledoit-Wolf, เป้าหมาย μI) และ μ (Bayes-Stein ของ Jorion แบบง่าย) ก่อนหาน้ำหนักด้วยวิธีเดียวกับ B
- **การทดสอบทางสถิติ:** Paired t-test ของ Sharpe ต่อรอบ แสดงทั้ง p ดิบและ p ที่ปรับ Bonferroni เพราะเปรียบเทียบหลายคู่พร้อมกัน
- **กลยุทธ์ที่ดีที่สุด:** เลือกย้อนหลังจากผลทดสอบชุดเดียวกัน จึงไม่ใช่การพยากรณ์
- **สกุลเงิน:** ตรวจจากท้ายรหัสหุ้น (.BK = บาท, .T = เยน, .HK = ดอลลาร์ฮ่องกง, ไม่มี = ดอลลาร์สหรัฐ)
""")

st.divider()
st.caption("QuantLab — เครื่องมือจำลองพอร์ตเพื่อการเรียนรู้และวิจัยทางสถิติ ไม่ใช่คำแนะนำการลงทุน")
