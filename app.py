import io
import time
from html import escape as esc
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.optimize import minimize

st.set_page_config(
    page_title="QuantLab — ระบบจำลองและจัดพอร์ตหุ้นเชิงปริมาณ",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAX_TICKERS_ALLOWED = 10   # จำกัดจำนวนหุ้นสูงสุดเพื่อความรวดเร็วในการประมวลผล
DEFAULT_CAPITAL = 5000     # เงินตั้งต้นสำหรับนักเรียน

# ─────────────────────────────  Design tokens  ─────────────────────────────
# โทนสี: น้ำเงินเข้ม (navy) · ขาวครีม (cream) · ทองอ่อน (sand) — เก็บค่าจากภาพจานสีที่เลือก
NOIR, NAVY, OAT, IVORY = "#0E1429", "#161D39", "#F8F5F0", "#FDFBF7"
SAND, MIST, POWDER, BEIGE = "#E8C799", "#BDC5CF", "#DFE3EA", "#EDE6D8"
LINE, MUTED, GOLD = "#DCD3C2", "#5E647C", "#B8893F"

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
STRAT_COLORS = {"equal": "#8FA2C2", "markowitz": "#2E4A9E", "mcap": "#C99B52", "shrinkage": "#B0675A"}

STRESS_WINDOWS = {
    "COVID-19 (ก.พ.–เม.ย. 2020)": ("2020-02-01", "2020-04-30"),
    "เงินเฟ้อ/ดอกเบี้ยขาขึ้น (2022)": ("2022-01-01", "2022-12-31"),
}

# ภาพประกอบ (ฝังเป็น base64 ใน ui_assets.py) — ถ้าไม่มีไฟล์ แอปยังทำงานได้ เพียงไม่มีรูป
try:
    from ui_assets import IMAGES as IMG
except Exception:
    IMG = {}

IMAGE_CREDITS = "ภาพประกอบ: FreeImages.com และแหล่งภาพสต็อกบนอินเทอร์เน็ต ใช้เพื่อการศึกษาในโครงงานนี้ (ระบุที่มาของแต่ละภาพได้ที่ตัวแปร IMAGE_CREDITS)"


def img_tag(name, alt=""):
    uri = IMG.get(name)
    return f'<img src="{uri}" alt="{alt}">' if uri else ""


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400;1,500&family=Italiana&family=Pinyon+Script&family=Raleway:wght@300;400;500;600&family=Sarabun:wght@300;400;500;600&family=Noto+Serif+Thai:wght@300;400;500;600&display=swap');
/* ตัวเลขทั้งเว็บ: Libre Baskerville ตัวตรง (โหลดเฉพาะ 0-9 . % - +) อ่านง่ายแต่คลาสสิก ตามตัวอย่างที่เลือก */
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&text=0123456789.%25-%2B&display=swap');

:root { --noir:#0E1429; --navy:#161D39; --oat:#F8F5F0; --ivory:#FDFBF7; --sand:#E8C799; --mist:#BDC5CF;
        --powder:#DFE3EA; --beige:#EDE6D8; --line:#DCD3C2; --muted:#5E647C; --gold:#B8893F;
        --f-body: 'Libre Baskerville', 'Raleway', 'Sarabun', sans-serif;
        --f-serif: 'Libre Baskerville', 'Cormorant Garamond', 'Noto Serif Thai', serif;
        --f-disp: 'Italiana', 'Noto Serif Thai', serif;
        --grain: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 .09 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }

html, body, .stApp { background: var(--oat); font-variant-numeric: lining-nums; }
body, .stApp, [data-baseweb="popover"], [data-baseweb="menu"] { font-family: var(--f-body); }
.stApp { color: var(--navy); }
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea { font-family: var(--f-body); color: var(--navy); line-height: 1.7; }
[data-testid="stMarkdownContainer"] p { font-size: 1.06rem; }
[data-testid="stWidgetLabel"] p { font-size: 1.02rem; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: var(--f-serif); color: var(--navy); font-weight: 500; letter-spacing: 0; }
.stApp h2 { font-size: 2rem; } .stApp h3 { font-size: 1.65rem; } .stApp h4 { font-size: 1.3rem; }
.block-container { padding-top: 3.6rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
hr { border-color: var(--line); }
[data-testid="stCaptionContainer"] p { color: var(--muted); }

/* ---------- sidebar = แฟ้มสีแดงเข้ม ---------- */
[data-testid="stSidebar"] { background-color: var(--navy); border-right: none; color: var(--oat); }
[data-testid="stSidebar"] > div, [data-testid="stSidebarContent"], [data-testid="stSidebarHeader"], [data-testid="stSidebarUserContent"] { background: transparent !important; }
[data-testid="stSidebarContent"] { overflow-x: hidden; }
[data-testid="stSidebar"] :is(p, label, li, h1, h2, h3, h4, small, summary, [data-testid="stWidgetLabel"] *, [data-testid="stCaptionContainer"] *) { color: var(--oat) !important; }
[data-testid="stSidebar"] h2 { font-size: 1.55rem; margin-top: 4px; }
[data-testid="stSidebar"] [data-baseweb="input"] input, [data-testid="stSidebar"] [data-baseweb="select"] *, [data-testid="stSidebar"] [data-baseweb="base-input"] input { color: var(--navy) !important; }
[data-testid="stSidebar"] [data-baseweb="input"] > div, [data-testid="stSidebar"] [data-baseweb="select"] > div, [data-testid="stSidebar"] [data-baseweb="base-input"] { background: var(--ivory) !important; border-color: transparent !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] div { color: var(--oat) !important; }
[data-testid="stSidebar"] [data-testid="stSlider"] { filter: hue-rotate(52deg) saturate(.35) brightness(1.7); }
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] * { color: var(--mist) !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] details { background: rgba(0,0,0,.14); border: 1px solid rgba(248,245,240,.22); border-radius: 0; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: rgba(248,245,240,.06); }
[data-testid="stSidebar"] .stButton > button { background: var(--oat); color: var(--navy); border: 1px solid var(--oat); }
[data-testid="stSidebar"] .stButton > button p { color: var(--navy) !important; }
[data-testid="stSidebar"] .stButton > button:hover { background: var(--sand); border-color: var(--sand); }
[data-testid="stSidebarHeader"] button, [data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarHeader"] button * { color: var(--oat) !important; }

/* ---------- inputs / buttons (main area) ---------- */
[data-baseweb="input"] > div, [data-baseweb="select"] > div, [data-baseweb="base-input"] { background: var(--ivory) !important; border-color: var(--line) !important; border-radius: 2px !important; }
[data-baseweb="input"] input { color: var(--navy) !important; }
.stButton > button, .stDownloadButton > button { width: 100%; background: var(--navy); color: var(--ivory); border: 1px solid var(--navy); border-radius: 0;
    font-family: var(--f-body); font-weight: 500; letter-spacing: .06em; padding: .7rem 1rem; transition: background .2s, border-color .2s; }
.stButton > button p, .stDownloadButton > button p { color: var(--ivory) !important; }
.stButton > button:hover, .stDownloadButton > button:hover { background: var(--noir); border-color: var(--noir); color: var(--ivory); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 2px solid var(--gold); outline-offset: 2px; }

/* ---------- metrics / alerts / tabs ---------- */
[data-testid="stMetric"] { background: var(--ivory); border: 1px solid var(--line); border-radius: 0; padding: 16px 18px; }
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { font-family: var(--f-serif) !important; font-style: normal; font-weight: 400; color: var(--navy); }
[data-testid="stMetricValue"] { font-size: 2.4rem; }
[data-testid="stMetricLabel"] p { font-size: .9rem; letter-spacing: .03em; color: var(--muted); }
[data-testid="stAlert"] { background: var(--ivory); border: 1px solid var(--line); border-radius: 0; color: var(--navy); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { font-family: var(--f-body); font-weight: 500; font-size: 1.04rem; color: var(--muted); }
.stTabs [data-baseweb="tab"] p { color: inherit; }
.stTabs [aria-selected="true"] { color: var(--navy); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--navy); height: 2px; }

.verdict { background: var(--ivory); border: 1px solid var(--line); border-left: 3px solid var(--navy); padding: 16px 22px;
           margin: 8px 0 18px; line-height: 1.9; font-family: var(--f-serif); font-size: 1.2rem; }
.verdict.no { border-left-color: var(--gold); }
.stApp .verdict b { color: var(--navy); font-weight: 600; }

/* ---------- tables (HTML ล้วน เพื่อให้ฟอนต์ไทย/ตัวเลขเหมือนทั้งเว็บ) ---------- */
.qt-wrap { overflow-x: auto; margin: 6px 0 14px; border: 1px solid var(--line); background: var(--ivory); }
table.qt { width: 100%; border-collapse: collapse; font-size: .98rem; font-family: var(--f-body); color: var(--navy); }
.qt th { text-align: left; font-weight: 500; font-size: .82rem; letter-spacing: .04em; color: var(--muted); padding: 11px 16px; border-bottom: 1px solid var(--navy); white-space: nowrap; }
.qt td { padding: 11px 16px; border-bottom: 1px solid var(--line); white-space: nowrap; }
.qt tr:last-child td { border-bottom: 0; }
.qt td:first-child { font-weight: 500; }
.qt tbody tr:hover { background: rgba(232,199,153,.2); }
.qt tr.hl td { background: rgba(232,199,153,.28); font-weight: 500; }
.qt tr.hl td:first-child { box-shadow: inset 3px 0 0 var(--navy); }

/* ---------- infographic tiles (แรงบันดาลใจจากแผนภูมิวงกลมแบบมีภาพ) ---------- */
.infos { display: grid; grid-template-columns: repeat(3, 1fr); border: 1px solid var(--line); background: var(--ivory); margin: 20px 0 10px; }
.info { padding: 26px 24px; border-right: 1px solid var(--line); display: flex; align-items: center; gap: 20px; }
.info:last-child { border-right: 0; }
.pie { position: relative; flex: 0 0 124px; width: 124px; height: 124px; border-radius: 50%; overflow: hidden; background: var(--beige); }
.pie img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; filter: saturate(.7) contrast(1.05); }
.pie::after { content: ""; position: absolute; inset: 0; background: conic-gradient(rgba(14,20,41,.8) 0 calc(var(--p) * 1%), transparent 0); }
.info-num { font-family: var(--f-serif); font-weight: 400; font-size: clamp(2.4rem, 4vw, 3.2rem); line-height: 1.1; color: var(--navy); }
.info-num small { font-size: .45em; margin-left: 2px; }
.info-label { font-size: .7rem; letter-spacing: .1em; text-transform: uppercase; color: var(--navy); margin-top: 10px; font-weight: 600; }
.info-note { font-size: .92rem; color: var(--muted); margin-top: 3px; line-height: 1.55; }

.doodle-foot { margin: 30px 0 4px; text-align: center; }
.doodle-foot img { width: min(640px, 100%); opacity: .85; }

/* ---------- masthead / hero / about / band / banner ---------- */
.masthead { display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 6px;
            border-bottom: 1px solid var(--navy); padding: 2px 2px 10px; margin-bottom: 14px; }
.mast-logo { font-family: var(--f-disp); font-size: 1.5rem; letter-spacing: .32em; color: var(--navy); }
.mast-sub { font-size: .68rem; letter-spacing: .2em; text-transform: uppercase; color: var(--muted); }

.hero2 { display: grid; grid-template-columns: 1.25fr 1fr; background: var(--noir); color: var(--oat); min-height: 560px; overflow: hidden; }
.hero2-text { padding: 56px 48px; display: flex; flex-direction: column; justify-content: center; }
.hero2-eyebrow { font-size: .72rem; letter-spacing: .32em; color: var(--mist); text-transform: uppercase; margin-bottom: 22px; }
.hero2-title { font-family: var(--f-disp); font-size: clamp(2.4rem, 6vw, 4.6rem); line-height: 1.02; letter-spacing: .04em; color: var(--oat); }
.hero2-script { font-family: 'Pinyon Script', cursive; font-size: clamp(2.6rem, 6.4vw, 5rem); color: var(--sand); line-height: 1.05; margin: 0 0 0 .6em; }
.hero2-rule { height: 1px; background: rgba(248,245,240,.35); margin: 30px 0 22px; position: relative; max-width: 340px; }
.hero2-rule::after { content: ""; position: absolute; right: -4px; top: -3px; width: 7px; height: 7px; border-radius: 50%; background: var(--sand); }
.stApp .hero2 p.hero2-sub { color: rgba(248,245,240,.86); max-width: 460px; font-weight: 300; line-height: 1.95; font-size: 1.06rem; margin: 0; }
.hero2-hint { margin-top: 26px; font-size: .8rem; letter-spacing: .04em; color: var(--mist); }
.hero2-img { position: relative; min-height: 320px; background: linear-gradient(135deg, #1a2342, #0E1429); }
.hero2-img img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; filter: saturate(.85) contrast(1.03); }
.hero2-img::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, rgba(14,20,41,.88) 0%, rgba(14,20,41,0) 38%); }

.about { display: grid; grid-template-columns: 1fr 1.1fr; gap: 44px; padding: 64px 8px 24px; align-items: center; }
.collage { position: relative; height: 430px; }
.polaroid { position: absolute; background: #fff; padding: 8px; box-shadow: 0 10px 28px rgba(22,29,57,.2); }
.polaroid img { display: block; width: 100%; height: auto; }
.pl1 { width: 58%; left: 0; top: 36px; transform: rotate(-4deg); }
.pl2 { width: 46%; right: 0; top: 0; transform: rotate(3deg); }
.pl3 { width: 56%; left: 20%; bottom: 0; transform: rotate(-1.5deg); }
.about-eyebrow { font-size: .72rem; letter-spacing: .3em; text-transform: uppercase; color: var(--muted); margin-bottom: 10px; }
.about-title { font-family: var(--f-serif); font-size: clamp(2rem, 4.5vw, 3rem); font-weight: 400; line-height: 1.2; color: var(--navy); }
.about-title em { font-style: italic; color: #A97C32; }
.stApp .about-text p { font-weight: 300; line-height: 1.95; margin: 18px 0 0; max-width: 520px; }
.pillars { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; margin-top: 30px; border-top: 1px solid var(--line); padding-top: 20px; }
.pillar-num { font-family: var(--f-serif); font-weight: 400; font-size: 2rem; color: #A97C32; line-height: 1.1; }
.pillar-h { font-weight: 600; font-size: .74rem; letter-spacing: .14em; text-transform: uppercase; margin: 8px 0 6px; }
.pillar-p { font-size: .94rem; line-height: 1.7; color: var(--muted); }

.band { position: relative; margin: 34px 0 22px; padding: 24px 12px; border-top: 1px solid var(--navy); border-bottom: 1px solid var(--navy); overflow: hidden; }
.band img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: .2; }
.band-text { position: relative; display: flex; flex-wrap: wrap; justify-content: center; gap: 6px 26px;
             font-family: var(--f-disp); font-size: clamp(1.05rem, 2.8vw, 1.7rem); letter-spacing: .22em; color: var(--navy); }

.rbanner { position: relative; height: 128px; overflow: hidden; border: 1px solid var(--line); margin-bottom: 18px; background: var(--ivory);
           display: flex; align-items: center; padding: 0 32px; }
.rbanner img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; opacity: .45; }
.rbanner::after { content: ""; position: absolute; inset: 0; background: linear-gradient(90deg, var(--ivory) 28%, rgba(253,251,247,0) 100%); }
.rb-text { position: relative; z-index: 2; }
.rb-eyebrow { font-size: .66rem; letter-spacing: .3em; text-transform: uppercase; color: var(--muted); }
.rb-title { font-family: var(--f-disp); font-size: clamp(1.7rem, 4vw, 2.6rem); letter-spacing: .04em; color: var(--navy); line-height: 1.2; }
.rb-title .sc { font-family: 'Pinyon Script', cursive; letter-spacing: 0; font-size: 1.25em; color: #A97C32; margin-left: .15em; }

/* ---------- คำศัพท์: ป้ายแตะดูความหมาย + หน้าค้นหา ---------- */
.tip { position: relative; cursor: help; border-bottom: 1px dotted var(--gold); outline: none; }
.tip::after { content: attr(data-tip); position: absolute; left: 0; bottom: calc(100% + 10px); width: max-content; max-width: min(300px, 78vw);
    background: var(--navy); color: var(--oat); padding: 10px 14px; font-family: var(--f-body); font-size: .88rem; font-weight: 400; line-height: 1.65;
    letter-spacing: 0; text-transform: none; text-align: left; white-space: normal; z-index: 60; opacity: 0; visibility: hidden; transform: translateY(4px);
    transition: opacity .15s, transform .15s; box-shadow: 0 8px 24px rgba(22,29,57,.25); pointer-events: none; }
.tip::before { content: ""; position: absolute; left: 14px; bottom: calc(100% + 4px); border: 6px solid transparent; border-top-color: var(--navy);
    z-index: 61; opacity: 0; visibility: hidden; transition: opacity .15s; }
.tip:hover::after, .tip:focus::after { opacity: 1; visibility: visible; transform: none; }
.tip:hover::before, .tip:focus::before { opacity: 1; visibility: visible; }
.chips { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; margin: 12px 0 8px; }
.chips-h { font-size: .8rem; color: var(--muted); margin-right: 4px; }
.chips .tip { background: var(--ivory); border: 1px solid var(--line); padding: 3px 12px; border-radius: 999px; font-size: .88rem; }
.chips .tip:hover, .chips .tip:focus { border-color: var(--gold); }
.chips .tip:nth-last-child(-n+2)::after { left: auto; right: 0; }
.chips .tip:nth-last-child(-n+2)::before { left: auto; right: 14px; }

.gl-count { font-size: .9rem; color: var(--muted); margin: 6px 0 12px; }
.gl-card { background: var(--ivory); border: 1px solid var(--line); margin: 0 0 8px; }
.gl-card summary { cursor: pointer; padding: 14px 18px; list-style: none; display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; }
.gl-card summary::-webkit-details-marker { display: none; }
.gl-card summary::before { content: "+"; font-family: var(--f-serif); color: var(--gold); font-size: 1.4rem; line-height: 1; width: 1rem; }
.gl-card[open] summary::before { content: "–"; }
.gl-en { font-family: var(--f-serif); font-size: 1.22rem; font-weight: 500; color: var(--navy); }
.gl-th { color: var(--muted); font-size: 1rem; }
.gl-cat { margin-left: auto; font-size: .7rem; letter-spacing: .06em; color: var(--gold); border: 1px solid var(--line); padding: 2px 10px; white-space: nowrap; }
.gl-body { padding: 2px 22px 18px 46px; border-top: 1px solid var(--line); }
.gl-short { font-family: var(--f-serif); font-size: 1.14rem; line-height: 1.85; color: var(--navy); margin: 14px 0 6px; }
.gl-row { margin: 9px 0; line-height: 1.85; font-size: 1.02rem; }
.gl-row b { display: inline-block; font-size: .74rem; letter-spacing: .04em; color: var(--gold); border: 1px solid var(--line); padding: 0 8px; margin-right: 10px; font-weight: 600; line-height: 1.7; vertical-align: 1px; }
.gl-empty { padding: 26px; text-align: center; color: var(--muted); border: 1px dashed var(--line); background: var(--ivory); }

@media (max-width: 900px) {
    .infos { grid-template-columns: 1fr; }
    .info { border-right: 0; border-bottom: 1px solid var(--line); }
    .info:last-child { border-bottom: 0; }
}
@media (max-width: 760px) {
    .hero2 { grid-template-columns: 1fr; }
    .hero2-img { order: -1; min-height: 250px; }
    .hero2-img::after { background: linear-gradient(0deg, rgba(14,20,41,.9) 0%, rgba(14,20,41,0) 45%); }
    .hero2-text { padding: 34px 24px 40px; }
    .about { grid-template-columns: 1fr; padding-top: 40px; }
    .collage { height: 360px; }
    .pillars { grid-template-columns: 1fr; }
    .rbanner { padding: 0 18px; height: 110px; }
    .gl-body { padding-left: 22px; }
    .gl-cat { margin-left: 0; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(block):
    """ส่ง HTML เข้า Streamlit (ตัดการย่อหน้าและบรรทัดว่างออก เพื่อไม่ให้ถูกตีเป็น code block)"""
    st.markdown("\n".join(l.strip() for l in block.strip().splitlines() if l.strip()), unsafe_allow_html=True)


def show_table(df, hl=None):
    """ตาราง HTML ล้วน (แทน st.dataframe ที่เป็น canvas ปรับฟอนต์ไม่ได้) hl = ลำดับแถวที่ไฮไลต์"""
    head = "".join(f"<th>{esc(str(c))}</th>" for c in df.columns)
    rows = []
    for i, vals in enumerate(df.itertuples(index=False)):
        cls = ' class="hl"' if hl == i else ""
        rows.append(f"<tr{cls}>" + "".join(f"<td>{esc(str(v))}</td>" for v in vals) + "</tr>")
    st.markdown(f'<div class="qt-wrap"><table class="qt"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>',
                unsafe_allow_html=True)


try:
    from glossary import GLOSSARY, GLOSSARY_CATS
except Exception:
    GLOSSARY, GLOSSARY_CATS = [], []
G = {g["key"]: g for g in GLOSSARY}


def gshort(key, extra=""):
    """ความหมายสั้น ๆ ของคำ (ใช้เป็น help ของปุ่ม/การ์ด)"""
    s = G[key]["short"] if key in G else ""
    return (s + (" · " + extra if extra and s else extra)).strip()


def tip(key, label=None):
    """คำที่แตะ/เอาเมาส์ชี้แล้วขึ้นความหมายสั้น ๆ"""
    g = G.get(key)
    label = label or (g["chip"] if g else key)
    if not g:
        return esc(label)
    return f'<span class="tip" tabindex="0" data-tip="{esc(g["short"], quote=True)}">{esc(label)}</span>'


def term_chips(keys, title="ศัพท์ในส่วนนี้ · แตะหรือเอาเมาส์ชี้เพื่อดูความหมาย"):
    chips = "".join(tip(k) for k in keys if k in G)
    if chips:
        st.markdown(f'<div class="chips"><span class="chips-h">{esc(title)}</span>{chips}</div>', unsafe_allow_html=True)


def glossary_cards(items, open_all=False):
    out = []
    for g in items:
        rows = [("อธิบายง่าย ๆ", g["plain"]), ("เปรียบเทียบ", g.get("analogy")),
                ("อ่านค่ายังไง", g.get("read")), ("ในเว็บนี้", g.get("where"))]
        body = "".join(f'<div class="gl-row"><b>{lab}</b>{esc(txt)}</div>' for lab, txt in rows if txt)
        out.append(f'<details class="gl-card"{" open" if open_all else ""}><summary><span class="gl-en">{esc(g["en"])}</span>'
                   f'<span class="gl-th">{esc(g["th"])}</span><span class="gl-cat">{esc(g["cat"])}</span></summary>'
                   f'<div class="gl-body"><div class="gl-short">{esc(g["short"])}</div>{body}</div></details>')
    return "".join(out)


def render_glossary(prefix):
    """ช่องค้นหา + กรองหมวด + การ์ดความหมายแบบกดเปิด/ปิด"""
    if not GLOSSARY:
        st.info("ยังไม่พบไฟล์คำศัพท์ (glossary.py) — อัปโหลดไฟล์นี้ไว้ข้าง app.py เพื่อเปิดใช้งาน")
        return
    q = st.text_input("ค้นหาคำศัพท์ (พิมพ์ภาษาไทยหรืออังกฤษ)", key=f"{prefix}_q",
                      placeholder="เช่น Sharpe, เงินเฟ้อ, ความผันผวน, p-value")
    cat = st.radio("หมวดหมู่", ["ทั้งหมด"] + GLOSSARY_CATS, horizontal=True, key=f"{prefix}_cat")
    tokens = q.strip().lower().split()
    items = []
    for g in GLOSSARY:
        if cat != "ทั้งหมด" and g["cat"] != cat:
            continue
        names = " ".join([g["en"], g["th"], g["chip"], " ".join(g.get("aliases", []))]).lower()
        body = " ".join([g["short"], g["plain"], g.get("analogy", ""), g.get("read", ""), g.get("where", "")]).lower()
        if tokens and not all(t in names or t in body for t in tokens):
            continue
        if not tokens:
            rank = 0
        elif any(x.lower().startswith(tokens[0]) for x in (g["en"], g["chip"], g["th"])):
            rank = 0   # ชื่อขึ้นต้นด้วยคำที่พิมพ์ = ตรงที่สุด
        elif all(t in names for t in tokens):
            rank = 1
        else:
            rank = 2
        items.append((rank, g))
    items = [g for _, g in sorted(items, key=lambda x: x[0])] if tokens else [g for _, g in items]
    st.markdown(f'<div class="gl-count">พบ {len(items)} คำ · กดที่ชื่อคำเพื่อเปิดอ่าน</div>', unsafe_allow_html=True)
    if items:
        st.markdown(glossary_cards(items, open_all=bool(tokens) and len(items) <= 3), unsafe_allow_html=True)
    else:
        st.markdown('<div class="gl-empty">ไม่พบคำนี้ ลองพิมพ์ให้สั้นลง หรือเลือกหมวด “ทั้งหมด”</div>', unsafe_allow_html=True)


def pct(x, d=1):
    return "–" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def style_axes(fig, ax):
    fig.patch.set_facecolor(OAT)
    ax.set_facecolor(IVORY)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=NAVY, labelsize=8)
    ax.xaxis.label.set_color(NAVY)
    ax.yaxis.label.set_color(NAVY)
    ax.title.set_color(NAVY)
    ax.grid(True, color=LINE, alpha=0.7, linewidth=0.6)


def style_legend(ax, **kw):
    leg = ax.legend(facecolor=IVORY, edgecolor=LINE, fontsize=8, **kw)
    for t in leg.get_texts():
        t.set_color(NAVY)


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
    fig.patch.set_facecolor(NOIR)
    ax.set_facecolor(NOIR)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)   # ต้องล็อกขอบเขตก่อนวาดเส้น ไม่งั้น autoscale ทำให้ข้อความหลุดกรอบ
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#2A3560")
    ax.plot([0.05, 0.95], [0.78, 0.78], color="#2A3560", linewidth=0.8, transform=ax.transAxes)
    ax.text(0.05, 0.88, "QUANTLAB  ·  PORTFOLIO SUMMARY", fontsize=13, color=OAT, family="serif")
    ticker_str = ", ".join(tickers[:6]) + ("..." if len(tickers) > 6 else "")
    ax.text(0.05, 0.71, f"Assets: {ticker_str}", fontsize=9, color=MIST)
    ax.text(0.05, 0.60, "BEST STRATEGY (by Sharpe)", fontsize=8, color=MIST)
    ax.text(0.05, 0.48, STRAT_DISPLAY_NAMES.get(best_key, best_key), fontsize=20, color=SAND, family="serif", style="italic")
    for x, label, val, col in [(0.05, "ANNUAL RETURN (CAGR)", f"{cagr * 100:+.1f}%", OAT),
                               (0.38, "SHARPE RATIO", f"{sharpe:.2f}", MIST),
                               (0.70, "MAX DRAWDOWN", f"{max_dd * 100:.1f}%", SAND)]:
        ax.text(x, 0.31, label, fontsize=7.5, color="#8E98B8")
        ax.text(x, 0.19, val, fontsize=17, color=col, family="serif")
    verdict = f"Beats {infl * 100:.1f}% inflation" if beats else f"Below {infl * 100:.1f}% inflation"
    ax.text(0.05, 0.10, verdict, fontsize=9, color=OAT if beats else "#E8B09C")
    ax.text(0.05, 0.04, f"Initial capital: {capital:,.0f} {curr_name}  |  Past results do not guarantee future returns",
            fontsize=7, color="#8E98B8")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


# ─────────────────────────────  Header  ─────────────────────────────
html("""
<div class="masthead">
<span class="mast-logo">QUANTLAB</span>
<span class="mast-sub">Portfolio Lab · Walk-Forward Validation</span>
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
                         help=gshort("inflation", "ใช้ตัดสินว่าพอร์ตชนะเงินเฟ้อหรือไม่ (โครงร่างตั้งไว้ราว 2–3%)"))
    inflation = infl_pct / 100

    benchmark_option = st.selectbox(
        "ตัวเปรียบเทียบ (Benchmark)",
        ["⚡ อัตโนมัติ (Auto-detect)", "🇹🇭 SET Index (หุ้นไทย)", "🇺🇸 S&P 500 (หุ้นใหญ่สหรัฐฯ)",
         "🚀 Nasdaq 100 (หุ้นเทคฯ สหรัฐฯ)", "🇯🇵 Nikkei 225 (หุ้นญี่ปุ่น)", "🇭🇰 Hang Seng Index (หุ้นฮ่องกง)",
         "🌐 MSCI ACWI (หุ้นทั่วโลก / พอร์ตผสม)"],
        help=gshort("benchmark", "เลือกดัชนีอ้างอิงเพื่อเทียบกับพอร์ตของคุณ"),
    )

    with st.expander("ตั้งค่าการทดสอบขั้นสูง"):
        rf_rate_pct = st.slider("อัตราผลตอบแทนปราศจากความเสี่ยง Rf (%/ปี)", 0.0, 5.0, 0.0, step=0.25,
                                help=gshort("rf")) / 100
        train_window = st.slider("ช่วง Train (วันทำการ)", 126, 378, 252, step=21, help=gshort("train"))
        test_window = st.slider("ช่วง Test ต่อรอบ (วันทำการ)", 21, 126, 63, step=21, help=gshort("test"))
        years_back = st.slider("ข้อมูลย้อนหลัง (ปี)", 3, 25, 9,
                               help="ช่วงทดสอบจริงเริ่มหลังช่วง Train แรก (≈1 ปี) · ย้อนหลัง 9 ปีจึงครอบคลุมช่วง COVID-19 และมีรอบทดสอบมากพอสำหรับ t-test")
        cost_pct = st.slider("ค่าธรรมเนียมซื้อขาย (% ของยอดที่ซื้อ/ขาย)", 0.0, 1.0, 0.15, step=0.05,
                             help=gshort("fee", "ตั้งตามโบรกเกอร์ที่ใช้จริง (รวม VAT) · โมเดลยังไม่รวมค่าธรรมเนียมขั้นต่ำต่อรายการ")) / 100
        stress_option = st.selectbox("ดูผลเฉพาะช่วงวิกฤต", ["ไม่ระบุ"] + list(STRESS_WINDOWS.keys()))

    if st.button("เริ่มวิเคราะห์", type="primary"):
        st.session_state["has_run"] = True
    st.caption("ไม่เข้าใจคำไหน? แตะที่คำที่มีเส้นประใต้ หรือเปิดหัวข้อ “คำศัพท์” ได้เลย")

if not st.session_state.get("has_run"):
    collage = "".join(
        f'<div class="polaroid {cls}">{img_tag(key)}</div>'
        for key, cls in [("hands", "pl1"), ("dollars", "pl2"), ("news", "pl3")] if IMG.get(key))
    html(f"""
    <div class="hero2">
    <div class="hero2-text">
    <div class="hero2-eyebrow">QuantLab · for student investors</div>
    <div class="hero2-title">PORTFOLIOS</div>
    <div class="hero2-script">risk &amp; evidence</div>
    <div class="hero2-rule"></div>
    <p class="hero2-sub">เครื่องมือจำลองการจัดพอร์ตหุ้นสำหรับนักเรียนที่มีเงินน้อยและไม่มีเวลาเฝ้าหุ้น ทดสอบกับข้อมูลที่โมเดลไม่เคยเห็นมาก่อน</p>
    <div class="hero2-hint">ปรับค่าที่แถบด้านซ้าย แล้วกด “เริ่มวิเคราะห์”</div>
    </div>
    <div class="hero2-img">{img_tag("hero", "ตลาดหลักทรัพย์")}</div>
    </div>
    <div class="about">
    <div class="collage">{collage}</div>
    <div class="about-text">
    <div class="about-eyebrow">About QuantLab</div>
    <div class="about-title">ลงทุนแบบ<em>เรียบง่าย</em><br>ตรวจสอบผลได้จริง</div>
    <p>กำหนดสัดส่วนเงินลงทุนล่วงหน้า ปรับพอร์ตเป็นรอบเวลาที่ชัดเจน แล้วดูว่า 4 วิธีจัดพอร์ตวิธีไหนทำผลได้ดีที่สุดกับข้อมูลที่ไม่เคยเห็นมาก่อน และชนะเงินเฟ้อหรือไม่ ผลทั้งหมดไม่ใช่คำแนะนำการลงทุน</p>
    <div class="pillars">
    <div><div class="pillar-num">01</div><div class="pillar-h">Walk-Forward</div><div class="pillar-p">คำนวณสัดส่วนจากอดีต แล้วทดสอบกับช่วงถัดไปที่ไม่เคยเห็น</div></div>
    <div><div class="pillar-num">02</div><div class="pillar-h">Four Strategies</div><div class="pillar-p">Equal-Weight, Markowitz, Market-Cap และ Shrinkage</div></div>
    <div><div class="pillar-num">03</div><div class="pillar-h">Real Costs</div><div class="pillar-p">หักค่าธรรมเนียมซื้อขาย เทียบเงินเฟ้อและดัชนีอ้างอิง</div></div>
    </div>
    </div>
    </div>
    <div class="band">{img_tag("chart")}<div class="band-text"><span>RETURN</span><span>RISK</span><span>SHARPE</span><span>DRAWDOWN</span><span>INFLATION</span></div></div>
    <div class="doodle-foot">{img_tag("doodle")}</div>
    """)
    st.info("ตั้งค่าทางซ้ายมือ แล้วกด **เริ่มวิเคราะห์** เพื่อประมวลผล (ค่าเริ่มต้นคือหุ้นไทย 5 ตัวตามโครงร่าง เงินตั้งต้น 5,000 บาท)")
    st.subheader("คำศัพท์ที่ควรรู้ก่อนเริ่ม")
    st.caption("ไม่เคยลงทุนก็อ่านได้ — พิมพ์ค้นหาคำที่สงสัย หรือกดเปิดอ่านทีละคำ")
    render_glossary("gw")
    st.stop()

html(f"""
<div class="rbanner">{img_tag("chart")}
<div class="rb-text"><div class="rb-eyebrow">Walk-forward results</div>
<div class="rb-title">ผลการทดสอบ<span class="sc">portfolio</span></div></div>
</div>
""")

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["สรุป", "การเติบโต", "สถิติเชิงลึก", "แชร์", "ข้อมูล", "คำศัพท์"])

with tab1:
    st.subheader(f"สรุปผล ({curr_code})")
    st.caption(f"ทดสอบนอกตัวอย่าง {res['n_folds']} รอบ ช่วง {idx[0]:%d/%m/%Y} – {idx[-1]:%d/%m/%Y} (ราว {years_oos:.1f} ปี) หลังหักค่าธรรมเนียมแล้ว")
    if res["n_failed"] > 0:
        st.warning(f"คำนวณ Markowitz ไม่ลู่เข้า {res['n_failed']}/{res['n_folds']} รอบ (ใช้ Equal-Weight แทนในรอบนั้น)")
    if res["n_mv_fallback"] > 0:
        st.info(f"มี {res['n_mv_fallback']} ครั้ง (รวมกลยุทธ์ B และ D) ที่ผลตอบแทนคาดหวังของทุกหุ้นในช่วง Train ไม่เกิน Rf "
                "Max-Sharpe ใช้ไม่ได้ในกรณีนี้ ระบบจึงใช้พอร์ตความเสี่ยงต่ำสุด (Min-Variance) แทนในรอบนั้น")

    if best["cagr"] > inflation:
        html(f"""<div class="verdict">กลยุทธ์ที่ {tip('sharpe', 'Sharpe')} ดีที่สุดคือ <b>{STRAT_FULL_NAMES[best_key]}</b> ให้ผลตอบแทนทบต้น <b>{pct(best['cagr'])}</b> ต่อปี
        สูงกว่า{tip('inflation', 'เงินเฟ้อ')}ที่ตั้งไว้ {pct(inflation)} · มี {beats_n} จาก {len(order)} กลยุทธ์ที่ชนะเงินเฟ้อ</div>""")
    else:
        html(f"""<div class="verdict no">กลยุทธ์ที่ {tip('sharpe', 'Sharpe')} ดีที่สุดคือ <b>{STRAT_FULL_NAMES[best_key]}</b> แต่ผลตอบแทนทบต้น <b>{pct(best['cagr'])}</b> ต่อปี
        ยังไม่ชนะ{tip('inflation', 'เงินเฟ้อ')}ที่ตั้งไว้ {pct(inflation)} · มี {beats_n} จาก {len(order)} กลยุทธ์ที่ชนะเงินเฟ้อ</div>""")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ผลตอบแทนทบต้น/ปี", pct(best["cagr"]), help=gshort("cagr"))
    c2.metric("ความผันผวน/ปี", pct(best["ann_vol"]), help=gshort("volatility"))
    c3.metric("Sharpe ratio", f"{best['sharpe']:.2f}", help=gshort("sharpe"))
    c4.metric("Max Drawdown", pct(best["max_drawdown"]), help=gshort("max_drawdown"))

    # แผนภูมิวงกลมแบบมีภาพ (ส่วนสีดำ = สัดส่วนของตัวเลขนั้น)
    fb = folds[folds["strategy"] == best_key]
    win_n, win_total = int((fb["total_return"] > 0).sum()), len(fb)
    win_p = 100 * win_n / win_total if win_total else 0
    dd_p = min(100.0, abs(best["max_drawdown"]) * 100) if np.isfinite(best["max_drawdown"]) else 0
    cw = res["current_weights"][best_key]
    top_i = int(np.argmax(cw))

    def tile(img, p, label, note):
        return (f'<div class="info"><div class="pie" style="--p:{p:.1f}">{img_tag(img)}</div>'
                f'<div><div class="info-num">{p:.0f}<small>%</small></div>'
                f'<div class="info-label">{label}</div><div class="info-note">{note}</div></div></div>')

    html("<div class=\"infos\">" + tile("ticker", win_p, tip("quarters_profit", "Quarters in profit"), f"{win_n} จาก {win_total} รอบทดสอบที่ได้กำไร")
         + tile("coins", dd_p, tip("max_drawdown", "Max drawdown"), "ขาดทุนสูงสุดจากจุดสูงสุดของพอร์ต")
         + tile("people", 100 * cw[top_i], tip("top_holding", "Top holding"), f"หุ้นสัดส่วนสูงสุด: {selected[top_i]}") + "</div>")
    term_chips(["sharpe", "cagr", "volatility", "max_drawdown", "quarters_profit", "top_holding", "inflation", "fold"])

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
    show_table(pd.DataFrame(plan))
    term_chips(["equal_weight", "markowitz", "market_cap", "shrinkage", "rebalance"], "กลยุทธ์ · แตะหรือเอาเมาส์ชี้เพื่อดูความหมาย")
    st.caption("คำนวณจากข้อมูล Train ล่าสุด · ถ้า “ซื้อได้” เป็น 0 แปลว่าเงินส่วนนั้นยังไม่พอซื้อ 1 หุ้น ลองลดจำนวนหุ้นหรือเพิ่มเงินทุน")

    st.markdown("#### เปรียบเทียบทุกกลยุทธ์ (คิดจากกราฟเงินทุนต่อเนื่องทั้งช่วงทดสอบ)")
    rows = []
    for k in order:
        m = overall[k]
        rows.append({"กลยุทธ์": STRAT_FULL_NAMES[k], "ผลตอบแทนเฉลี่ย/ปี": pct(m["ann_return"]),
                     "ทบต้น/ปี (CAGR)": pct(m["cagr"]), "ความผันผวน/ปี": pct(m["ann_vol"]),
                     "Sharpe": f"{m['sharpe']:.2f}", "Max Drawdown": pct(m["max_drawdown"]),
                     f"เงินสุดท้าย ({curr_code})": f"{capital * (1 + m['total']):,.0f}",
                     "ค่าธรรมเนียมรวม": f"{res['total_cost'][k]:,.0f}",
                     "ชนะเงินเฟ้อ": "ชนะ" if m["cagr"] > inflation else "ไม่ชนะ"})
    if bench_metrics:
        rows.append({"กลยุทธ์": f"Benchmark: {bench_label}", "ผลตอบแทนเฉลี่ย/ปี": pct(bench_metrics["ann_return"]),
                     "ทบต้น/ปี (CAGR)": pct(bench_metrics["cagr"]), "ความผันผวน/ปี": pct(bench_metrics["ann_vol"]),
                     "Sharpe": f"{bench_metrics['sharpe']:.2f}", "Max Drawdown": pct(bench_metrics["max_drawdown"]),
                     f"เงินสุดท้าย ({curr_code})": f"{capital * (1 + bench_metrics['total']):,.0f}",
                     "ค่าธรรมเนียมรวม": "–", "ชนะเงินเฟ้อ": "ชนะ" if bench_metrics["cagr"] > inflation else "ไม่ชนะ"})
    show_table(pd.DataFrame(rows), hl=order.index(best_key))
    term_chips(["expected_return", "cagr", "volatility", "sharpe", "max_drawdown", "benchmark", "fee"])
    st.caption("“กลยุทธ์ที่ดีที่สุด” เลือกย้อนหลังจากผลทดสอบชุดเดียวกัน ไม่ใช่การพยากรณ์ว่าจะดีที่สุดต่อไป · "
               "ผลในอดีตไม่รับประกันผลในอนาคต และไม่ใช่คำแนะนำการลงทุน")

with tab2:
    st.subheader(f"เงินลงทุนสะสม ({curr_code}) — ทบต้นจริง หลังหักค่าธรรมเนียม")
    term_chips(["rebalance", "fee", "benchmark", "inflation", "cagr"])
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for k in order:
        ax.plot(curves[k].index, curves[k].values, label=STRAT_DISPLAY_NAMES[k], color=STRAT_COLORS[k], linewidth=1.8)
    if bench_curve is not None:
        ax.plot(bench_curve.index, bench_curve.values, label=f"{bench_label} ({bench_sym})", color=NAVY,
                linewidth=1.4, linestyle="--")
    days = (idx - base_date).days.values
    ax.plot(idx, capital * (1 + inflation) ** (days / 365.25), color="#8C7B6B", linestyle=":", linewidth=1.6,
            label=f"Inflation {inflation * 100:.1f}%/yr")
    ax.axhline(capital, color="#A89F94", linewidth=0.8, linestyle=":")
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
    term_chips(["mean_fold_sharpe", "sharpe", "ci95", "fold"])
    if ci:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        for i, k in enumerate(order):
            if k in ci:
                mean, half = ci[k]
                ax.errorbar(i, mean, yerr=half, fmt="o", color=STRAT_COLORS[k], capsize=6, markersize=7, linewidth=2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([STRAT_DISPLAY_NAMES[k] for k in order], fontsize=8)
        ax.axhline(0, color="#A89F94", linewidth=0.8)
        ax.set_ylabel("Mean fold Sharpe (95% CI)")
        ax.set_xlim(-0.5, len(order) - 0.5)
        style_axes(fig, ax)
        st.pyplot(fig)
        plt.close(fig)
    st.markdown("#### Paired t-test (เทียบ Sharpe ทีละรอบทดสอบ)")
    term_chips(["paired_t", "pvalue", "bonferroni"])
    if tests.empty:
        st.info("จำนวนรอบทดสอบน้อยเกินไปสำหรับ t-test (ต้องมีอย่างน้อย 3 รอบ)")
    else:
        show = tests.copy()
        show["ส่วนต่าง Sharpe เฉลี่ย"] = show["ส่วนต่าง Sharpe เฉลี่ย"].round(3)
        show["t"] = show["t"].round(3)
        show["p-value"] = show["p-value"].round(4)
        show["p (Bonferroni)"] = show["p (Bonferroni)"].round(4)
        show_table(show)
        st.caption(f"n = {res['n_folds']} รอบ · เปรียบเทียบ {len(show)} คู่พร้อมกัน จึงแสดง p ที่ปรับแบบ Bonferroni (p × จำนวนคู่) คู่กับ p ดิบ · "
                   "รอบละไม่กี่เดือน ข้อมูลน้อย ผลจึงควรอ่านอย่างระมัดระวัง")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### ความเข้มของการลดสัญญาณรบกวน (กลยุทธ์ D)")
        term_chips(["shrinkage", "ledoit_wolf", "bayes_stein"])
        st.info(f"Ledoit-Wolf δ (ของ Σ) เฉลี่ย = **{res['avg_delta']:.4f}** · Bayes-Stein λ (ของ μ) เฉลี่ย = **{res['avg_lambda']:.4f}**\n\n"
                "ค่ายิ่งใกล้ 1 แปลว่าค่าประมาณจากข้อมูล Train มีสัญญาณรบกวนมาก ระบบจึงถ่วงเข้าหาเป้าหมายที่เรียบกว่า")
    with col_b:
        st.markdown("#### Max Drawdown ตลอดช่วงทดสอบ")
        term_chips(["max_drawdown"])
        mdd_df = pd.DataFrame({"กลยุทธ์": [STRAT_FULL_NAMES[k] for k in order],
                               "Max Drawdown": [pct(overall[k]["max_drawdown"]) for k in order]})
        show_table(mdd_df)

    st.markdown("#### สัดส่วนน้ำหนักในแต่ละรอบ")
    term_chips(["weights_per_fold", "rebalance"])
    wh_key = st.selectbox("เลือกกลยุทธ์", order, index=order.index(best_key),
                          format_func=lambda k: STRAT_FULL_NAMES[k], key="wh_key")
    wh = res["weight_history"][wh_key]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    palette = ["#8FA2C2", "#2E4A9E", "#C99B52", "#B0675A", "#B9C4D8", "#4F6B8A", "#E3CFA5", "#6F7FA8", "#D9A59B", "#7A7468"]
    ax.stackplot(wh.index, wh.T.values, labels=list(wh.columns), colors=palette[:wh.shape[1]], alpha=0.9)
    ax.set_xlabel("Round")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    style_axes(fig, ax)
    style_legend(ax, loc="upper center", ncol=min(5, wh.shape[1]), bbox_to_anchor=(0.5, -0.18))
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### ความสัมพันธ์ของผลตอบแทนหุ้น (Correlation)")
    term_chips(["correlation"])
    corr = res["returns"].corr()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    cmap = LinearSegmentedColormap.from_list("qb", ["#C99B52", IVORY, "#1F2B55"])
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8, color=NAVY)
    ax.set_yticklabels(corr.columns, fontsize=8, color=NAVY)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color=IVORY if abs(corr.values[i, j]) > 0.6 else NAVY)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.patch.set_facecolor(OAT)
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
            show_table(pd.DataFrame(srows))
        else:
            st.info(f"ช่วงทดสอบนอกตัวอย่างของคุณเริ่ม {idx[0]:%d/%m/%Y} จึงยังไม่ครอบคลุมช่วงนี้ ลองเพิ่มข้อมูลย้อนหลังในเมนูขั้นสูง (ต้องเผื่อช่วง Train ก่อนหน้าด้วย)")

with tab4:
    st.subheader("แชร์สรุปผลของคุณ")
    card_buf = create_share_card(best_key, best["cagr"], best["sharpe"], best["max_drawdown"], selected,
                                 capital, curr_code, best["cagr"] > inflation, inflation)
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
    if IMAGE_CREDITS:
        st.caption(IMAGE_CREDITS)

with tab6:
    st.subheader("คำศัพท์")
    st.caption("พิมพ์ค้นหาคำที่สงสัย หรือกดเปิดอ่านทีละคำ — อธิบายเป็นภาษาคน พร้อมตัวอย่างเปรียบเทียบ")
    render_glossary("gt")

html(f"""<div class="doodle-foot">{img_tag("doodle")}</div>""")
st.divider()
st.caption("QuantLab — เครื่องมือจำลองพอร์ตเพื่อการเรียนรู้และวิจัยทางสถิติ ไม่ใช่คำแนะนำการลงทุน")
