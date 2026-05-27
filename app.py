"""
Pocket-Precise Cognitive Diagnostic Engine
==========================================
A conference-ready, multi-task cognitive and behavioural assessment battery.

Tasks implemented (all evidence-based):
  1. Informed Consent & Demographics
  2. 9-Point Gaze Calibration (oculomotor baseline)
  3. Prosaccade Task            → Fixation duration, saccade latency, saccade amplitude
  4. Anti-Saccade Task          → Inhibitory control, error rate, corrective saccade latency
  5. Visual Search Task         → Target detection RT, miss rate, distractor interference
  6. Reaction Time (simple)     → Baseline RT, intra-individual variability (IIV)
  7. Go/No-Go Task              → Response inhibition, commission errors, omission errors
  8. N-Back (1-back & 2-back)  → Working memory capacity, d-prime sensitivity
  9. Stroop Colour-Word Task    → Cognitive interference, selective attention
 10. Trail Making A & B         → Processing speed, cognitive flexibility
 11. Corsi Block Tapping        → Visuospatial working memory span
 12. Digit Span (Forward)       → Verbal working memory

Feature extraction (Python backend, I-VT + signal processing):
  F1  Mean Fixation Duration (ms)
  F2  Fixation Count
  F3  Saccade Latency (ms)
  F4  Saccade Amplitude (deg)
  F5  Saccade Peak Velocity (deg/s)
  F6  Gaze Path Entropy (bits)
  F7  Anti-Saccade Error Rate (%)
  F8  ROI Coverage (%)
  F9  Simple RT Mean (ms)
  F10 Simple RT Std Dev (ms, = IIV)
  F11 Go/No-Go Commission Error Rate (%)
  F12 Go/No-Go Omission Error Rate (%)
  F13 N-Back d-prime (sensitivity)
  F14 N-Back Response Bias (beta)
  F15 Stroop Congruent RT (ms)
  F16 Stroop Incongruent RT (ms)
  F17 Stroop Interference Score (ms delta)
  F18 Stroop Error Rate (%)
  F19 Trail Making A Time (s)
  F20 Trail Making B Time (s)
  F21 Trail Making B-A Delta (s, cognitive flexibility)
  F22 Corsi Span Score
  F23 Digit Span Forward Score
  F24 Visual Search RT Mean (ms)
  F25 Visual Search Miss Rate (%)

References embedded throughout.
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
import tempfile
import os
from math import log2
from scipy import stats as scipy_stats
from fpdf import FPDF
import datetime
import io

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Pocket-Precise · Cognitive Diagnostic Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────
# GLOBAL STYLES  (TreadWill-inspired dark slate)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base */
.stApp, .main, [data-testid="stAppViewContainer"] {
    background-color: #1E2D35 !important;
    color: #E8EDF0 !important;
    font-family: 'Inter', sans-serif !important;
}
h1, h2, h3, h4, p, span, label, li { color: #E8EDF0 !important; }
.block-container { padding-top: 2rem !important; max-width: 1100px !important; }

/* Sidebar (hidden by default) */
[data-testid="stSidebar"] { background-color: #162028 !important; }

/* Buttons */
.stButton > button {
    background-color: transparent !important;
    color: #D4A843 !important;
    border: 1.5px solid #D4A843 !important;
    border-radius: 6px !important;
    padding: 10px 22px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background-color: #D4A843 !important;
    color: #1E2D35 !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background-color: #2A3D48 !important;
    color: #E8EDF0 !important;
    border: 1px solid #4A6070 !important;
    border-radius: 6px !important;
}
.stSelectbox label, .stTextInput label, .stNumberInput label,
.stRadio label, .stCheckbox label {
    color: #A8BDC8 !important;
    font-size: 14px !important;
}

/* Cards */
.metric-card {
    background-color: #2A3D48;
    border: 1px solid #3D5565;
    padding: 18px 20px;
    border-radius: 8px;
    text-align: center;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 12px;
    color: #7A99A8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    display: block;
    margin-bottom: 8px;
}
.metric-card .value {
    font-size: 26px;
    font-weight: 600;
    color: #D4A843;
    display: block;
}
.metric-card .norm {
    font-size: 12px;
    color: #5A7A8A;
    display: block;
    margin-top: 5px;
}

/* Section dividers */
.section-divider {
    border: none;
    border-top: 1px solid #2A3D48;
    margin: 28px 0;
}

/* Info boxes */
.info-box {
    background-color: #1E3040;
    border-left: 3px solid #D4A843;
    padding: 14px 18px;
    border-radius: 0 6px 6px 0;
    margin: 16px 0;
    font-size: 14px;
    color: #A8BDC8 !important;
    line-height: 1.6;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
}
.badge-gold { background-color: #3D2E10; color: #D4A843; }
.badge-green { background-color: #0E2E1E; color: #4CAF82; }
.badge-red { background-color: #2E0E0E; color: #E07070; }
.badge-blue { background-color: #0E1E2E; color: #70A0E0; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #2A3D48;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #7A99A8 !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    color: #D4A843 !important;
    background-color: #1E2D35 !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background-color: #D4A843 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background-color: #2A3D48 !important;
    color: #A8BDC8 !important;
    border-radius: 6px !important;
}

/* Upload zone */
.stFileUploader > div {
    background-color: #2A3D48 !important;
    border: 1px dashed #4A6070 !important;
    border-radius: 8px !important;
}

/* Hide clutter */
header, #MainMenu, footer { visibility: hidden; }

/* Table */
.results-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.results-table th {
    background-color: #2A3D48;
    color: #7A99A8;
    padding: 10px 14px;
    text-align: left;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-size: 11px;
}
.results-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #2A3D48;
    color: #E8EDF0;
}
.results-table tr:hover td { background-color: #2A3D48; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "consent",          # consent | demographics | battery | upload | results
        "participant": {},
        "consent_given": False,
        "battery_done": False,
        "event_logs": None,
        "video_bytes": None,
        "results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# HELPER: Section header
# ─────────────────────────────────────────────
def section_header(title: str, subtitle: str = ""):
    st.markdown(f"<h2 style='font-size:22px; font-weight:600; color:#E8EDF0; margin-bottom:4px;'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='font-size:14px; color:#7A99A8; margin-bottom:20px;'>{subtitle}</p>", unsafe_allow_html=True)


def info_box(text: str):
    st.markdown(f"<div class='info-box'>{text}</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, norm: str = "", col=None):
    html = f"""
    <div class='metric-card'>
        <span class='label'>{label}</span>
        <span class='value'>{value}</span>
        {'<span class="norm">' + norm + '</span>' if norm else ''}
    </div>"""
    if col:
        col.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 0: INFORMED CONSENT
# ─────────────────────────────────────────────
def page_consent():
    st.markdown("<div style='max-width:720px; margin: 0 auto; padding: 30px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='display:flex; align-items:center; gap:14px; margin-bottom:32px;'>
        <div style='width:44px; height:44px; background:#D4A843; border-radius:8px; display:flex; align-items:center; justify-content:center;'>
            <span style='color:#1E2D35; font-size:22px; font-weight:700; color:#1E2D35 !important;'>P</span>
        </div>
        <div>
            <p style='margin:0; font-size:20px; font-weight:600; color:#E8EDF0 !important;'>Pocket-Precise</p>
            <p style='margin:0; font-size:13px; color:#7A99A8 !important;'>Cognitive Diagnostic Engine · v2.0</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    section_header("Participant Information Sheet")

    info_box("""
    <strong style='color:#D4A843;'>Study Purpose</strong><br>
    This assessment battery measures core cognitive and behavioural parameters including oculomotor
    control, inhibitory control, working memory, processing speed, and attentional capacity.
    It is designed for research use and produces clinically interpretable biomarkers backed by peer-reviewed literature.
    """)

    st.markdown("""
    <div style='background:#2A3D48; border-radius:8px; padding:20px 24px; margin:20px 0;'>
    <p style='font-size:14px; color:#A8BDC8 !important; line-height:1.8; margin:0;'>
    <strong style='color:#E8EDF0;'>What this involves:</strong><br>
    You will complete a series of 10 computer-based tasks taking approximately <strong style='color:#D4A843;'>25–35 minutes</strong>.
    A webcam is required to record gaze behaviour for oculomotor analysis. No data is transmitted externally —
    all processing happens locally on this machine.<br><br>
    <strong style='color:#E8EDF0;'>Data handling:</strong><br>
    Video recordings are processed frame-by-frame using facial landmark detection and then discarded.
    Only computed numerical biomarkers are stored. Raw video is downloaded to your local device only.<br><br>
    <strong style='color:#E8EDF0;'>Voluntary participation:</strong><br>
    You may stop at any time. Incomplete sessions will not be saved to the results database.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:13px; color:#5A7A8A; margin:16px 0 8px;'>TASKS IN THIS BATTERY</p>", unsafe_allow_html=True)
    tasks = [
        ("Gaze Calibration", "Oculomotor baseline", "~1 min"),
        ("Prosaccade / Anti-Saccade", "Inhibitory control & eye movement latency", "~4 min"),
        ("Visual Search", "Selective attention & target detection", "~3 min"),
        ("Simple Reaction Time", "Baseline processing speed & IIV", "~3 min"),
        ("Go / No-Go", "Response inhibition & impulsivity", "~4 min"),
        ("N-Back (1-back & 2-back)", "Working memory capacity & d-prime", "~5 min"),
        ("Stroop Colour-Word", "Cognitive interference & selective attention", "~4 min"),
        ("Trail Making A & B", "Processing speed & cognitive flexibility", "~5 min"),
        ("Corsi Block Tapping", "Visuospatial working memory", "~3 min"),
        ("Digit Span Forward", "Verbal working memory span", "~3 min"),
    ]
    for i, (name, desc, dur) in enumerate(tasks):
        st.markdown(f"""
        <div style='display:flex; align-items:center; justify-content:space-between; padding:10px 0; border-bottom:1px solid #2A3D48;'>
            <div style='display:flex; align-items:center; gap:12px;'>
                <span style='width:24px; height:24px; background:#2A3D48; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; font-size:12px; color:#D4A843 !important;'>{i+1}</span>
                <div>
                    <p style='margin:0; font-size:14px; font-weight:500; color:#E8EDF0 !important;'>{name}</p>
                    <p style='margin:0; font-size:12px; color:#7A99A8 !important;'>{desc}</p>
                </div>
            </div>
            <span style='font-size:12px; color:#5A7A8A;'>{dur}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    agree = st.checkbox("I have read and understood the information above. I consent to participate voluntarily.")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        if st.button("Continue to Demographics →", disabled=not agree):
            st.session_state["consent_given"] = True
            st.session_state["page"] = "demographics"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 1: DEMOGRAPHICS
# ─────────────────────────────────────────────
def page_demographics():
    st.markdown("<div style='max-width:720px; margin: 0 auto; padding: 30px 0;'>", unsafe_allow_html=True)
    section_header("Participant Demographics", "This information is used to normalise biomarkers against population norms.")

    col1, col2 = st.columns(2)
    with col1:
        pid = st.text_input("Participant ID *", placeholder="e.g. P001")
        age = st.number_input("Age *", min_value=18, max_value=90, value=25)
        handedness = st.selectbox("Handedness", ["Right", "Left", "Ambidextrous"])
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
        education = st.selectbox("Education level", [
            "Secondary school", "Undergraduate", "Postgraduate", "Doctoral", "Other"
        ])
        vision = st.selectbox("Corrected-to-normal vision?", ["Yes", "No – I have uncorrected impairment"])

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Pre-session Checks", "Answer honestly — this affects data quality flags only.")

    col3, col4 = st.columns(2)
    with col3:
        sleep = st.selectbox("Sleep last night (hours)", ["< 5", "5–6", "7–8", "9+"])
        caffeine = st.selectbox("Caffeine in last 2 hours?", ["No", "Yes – 1 drink", "Yes – 2+ drinks"])
    with col4:
        medications = st.selectbox("Psychoactive medication?", ["No", "Yes (stimulant)", "Yes (sedative)", "Yes (other)"])
        anxiety = st.selectbox("Current anxiety level (self-report)", ["1 – Very low", "2", "3 – Moderate", "4", "5 – Very high"])

    st.markdown("<br>", unsafe_allow_html=True)
    valid = pid.strip() != ""

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Begin Assessment Battery →", disabled=not valid):
            st.session_state["participant"] = {
                "id": pid, "age": age, "gender": gender,
                "handedness": handedness, "education": education,
                "vision": vision, "sleep": sleep, "caffeine": caffeine,
                "medications": medications, "anxiety": anxiety,
                "timestamp": datetime.datetime.now().isoformat()
            }
            st.session_state["page"] = "battery"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2: FULL JAVASCRIPT BATTERY
# ─────────────────────────────────────────────
def page_battery():
    section_header(
        "Assessment Battery",
        "Follow each task's on-screen instructions carefully. Do not close this tab."
    )

    info_box("""
    <strong style='color:#D4A843;'>Before you start:</strong>
    Sit approximately 55–65 cm from your screen. Ensure your face is well lit and clearly
    visible to the webcam. Minimise head movement during gaze tasks.
    Tasks will advance automatically — read each instruction screen fully before clicking.
    """)

    # ── The full JS battery is served as a self-contained HTML component ──
    battery_html = _build_battery_html()
    components.html(battery_html, height=720, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)

    info_box("""
    When the battery finishes, two files will download automatically:<br>
    &nbsp;&nbsp;• <strong style='color:#D4A843;'>raw_gaze_video.webm</strong> — webcam recording for oculomotor analysis<br>
    &nbsp;&nbsp;• <strong style='color:#D4A843;'>interaction_logs.json</strong> — timestamped event log for all cognitive tasks<br><br>
    Once both files have downloaded, proceed to the next step.
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("I have both files → Upload for Analysis"):
            st.session_state["battery_done"] = True
            st.session_state["page"] = "upload"
            st.rerun()


# ─────────────────────────────────────────────
# BATTERY HTML — full 10-task JS engine
# ─────────────────────────────────────────────
def _build_battery_html() -> str:
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #1E2D35;
    color: #E8EDF0;
    font-family: 'Inter', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    min-height: 700px;
    overflow: hidden;
}
#stage {
    position: relative;
    width: 800px;
    height: 560px;
    background: #253540;
    border: 1px solid #3D5565;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 16px;
}
#ui {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    text-align: center;
    z-index: 20;
}
#progress-bar-wrap {
    width: 800px;
    height: 4px;
    background: #2A3D48;
    border-radius: 2px;
    margin-top: 10px;
}
#progress-bar { height: 4px; background: #D4A843; border-radius: 2px; width: 0%; transition: width 0.5s; }
#task-label { font-size: 11px; color: #5A7A8A; letter-spacing: 1px; text-transform: uppercase; margin-top: 6px; }
h2 { font-size: 24px; font-weight: 600; margin-bottom: 10px; }
.sub { font-size: 14px; color: #7A99A8; line-height: 1.6; margin-bottom: 28px; max-width: 560px; }
.btn {
    background: transparent;
    color: #D4A843;
    border: 1.5px solid #D4A843;
    border-radius: 6px;
    padding: 11px 28px;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    font-family: inherit;
}
.btn:hover { background: #D4A843; color: #1E2D35; }
.btn:disabled { opacity: 0.3; cursor: not-allowed; }

/* Gaze dot */
#dot {
    position: absolute;
    width: 18px; height: 18px;
    background: #D4A843;
    border-radius: 50%;
    transform: translate(-50%, -50%);
    display: none;
    z-index: 10;
    box-shadow: 0 0 0 4px rgba(212,168,67,0.25);
    transition: left 0.3s ease, top 0.3s ease;
}

/* Crosshair */
#cross {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    font-size: 42px;
    color: #4A6070;
    display: none;
    z-index: 10;
    font-weight: 300;
    line-height: 1;
}

/* Stimulus overlay */
#stim {
    position: absolute;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 15;
    flex-direction: column;
    gap: 16px;
}
#stim-box {
    background: #1E3040;
    border: 1.5px solid #D4A843;
    border-radius: 10px;
    padding: 30px 50px;
    text-align: center;
    min-width: 300px;
}
#stim-text { font-size: 48px; font-weight: 700; }
#stim-label { font-size: 13px; color: #7A99A8; margin-top: 8px; }

/* N-Back grid */
#nback-grid {
    display: none;
    position: absolute;
    inset: 0;
    align-items: center;
    justify-content: center;
    z-index: 15;
}
.nb-cell {
    width: 90px; height: 90px;
    border: 1px solid #3D5565;
    border-radius: 6px;
    background: #1E2D35;
    display: flex; align-items: center; justify-content: center;
    font-size: 36px; font-weight: 700;
    color: transparent;
    transition: all 0.1s;
}
.nb-cell.active { background: #D4A843; color: #1E2D35; }

/* Trail Making canvas */
#trail-canvas { position: absolute; inset: 0; display: none; z-index: 15; }

/* Corsi grid */
#corsi-area {
    position: absolute;
    inset: 0;
    display: none;
    z-index: 15;
}
.corsi-block {
    position: absolute;
    width: 60px; height: 60px;
    background: #2A3D48;
    border: 1.5px solid #4A6070;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.15s;
}
.corsi-block.lit { background: #D4A843; border-color: #D4A843; }
.corsi-block.correct { background: #4CAF82; border-color: #4CAF82; }
.corsi-block.wrong { background: #E07070; border-color: #E07070; }

/* Digit span */
#digitspan-area {
    position: absolute;
    inset: 0;
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 15;
}
#digit-display { font-size: 72px; font-weight: 700; color: #D4A843; display: none; }
#digit-input-wrap { display: none; flex-direction: column; align-items: center; gap: 14px; }
#digit-input {
    background: #1E2D35;
    border: 1.5px solid #4A6070;
    color: #E8EDF0;
    font-size: 28px;
    text-align: center;
    border-radius: 8px;
    padding: 12px 20px;
    width: 260px;
    font-family: inherit;
}
</style>
</head>
<body>

<div id="stage">
    <div id="ui">
        <h2 id="title">Battery Ready</h2>
        <p class="sub" id="sub">Ensure your webcam is available and you are seated ~60 cm from the screen in a well-lit environment. Click below to begin.</p>
        <button class="btn" id="btn" onclick="initBattery()">Start Battery</button>
    </div>
    <div id="dot"></div>
    <div id="cross">+</div>
    <div id="stim" style="display:none;"><div id="stim-box"><div id="stim-text">Go</div><div id="stim-label"></div></div></div>
    <div id="nback-grid" style="display:none;">
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:8px;">
            <div class="nb-cell" id="nb-0"></div><div class="nb-cell" id="nb-1"></div><div class="nb-cell" id="nb-2"></div>
            <div class="nb-cell" id="nb-3"></div><div class="nb-cell" id="nb-4"></div><div class="nb-cell" id="nb-5"></div>
            <div class="nb-cell" id="nb-6"></div><div class="nb-cell" id="nb-7"></div><div class="nb-cell" id="nb-8"></div>
        </div>
    </div>
    <canvas id="trail-canvas" width="800" height="560"></canvas>
    <div id="corsi-area"></div>
    <div id="digitspan-area">
        <p style="color:#7A99A8; font-size:14px; margin-bottom:12px;" id="ds-prompt">Memorise the following sequence</p>
        <div id="digit-display"></div>
        <div id="digit-input-wrap">
            <input id="digit-input" type="text" placeholder="Enter digits in order" autocomplete="off">
            <button class="btn" onclick="submitDigitSpan()">Submit</button>
        </div>
    </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar"></div></div>
<div id="task-label">Task 0 / 10</div>

<script>
const logs = [];
const rtData = {};
let mediaRecorder, stream;
let taskIndex = 0;
const TOTAL_TASKS = 10;
let keyHandler = null;

function log(event, details) {
    logs.push({ timestamp_ms: performance.now().toFixed(2), event, details });
}

function setProgress(n) {
    document.getElementById('progress-bar').style.width = ((n / TOTAL_TASKS) * 100) + '%';
    document.getElementById('task-label').innerText = `Task ${n} / ${TOTAL_TASKS}`;
}

function showUI(title, sub, btnLabel, onclick) {
    const ui = document.getElementById('ui');
    document.getElementById('title').innerText = title;
    document.getElementById('sub').innerText = sub;
    const btn = document.getElementById('btn');
    btn.innerText = btnLabel;
    btn.onclick = onclick;
    btn.disabled = false;
    ui.style.display = 'flex';
}

function hideUI() { document.getElementById('ui').style.display = 'none'; }

function hideAll() {
    document.getElementById('stim').style.display = 'none';
    document.getElementById('nback-grid').style.display = 'none';
    document.getElementById('trail-canvas').style.display = 'none';
    document.getElementById('corsi-area').style.display = 'none';
    document.getElementById('digitspan-area').style.display = 'none';
    document.getElementById('dot').style.display = 'none';
    document.getElementById('cross').style.display = 'none';
    if (keyHandler) { document.removeEventListener('keydown', keyHandler); keyHandler = null; }
}

async function initBattery() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', frameRate: 30 }, audio: false });
        const opts = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? { mimeType: 'video/webm;codecs=vp9' } : { mimeType: 'video/webm' };
        const chunks = [];
        mediaRecorder = new MediaRecorder(stream, opts);
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
        mediaRecorder.onstop = () => exportAll(new Blob(chunks, { type: 'video/webm' }));
        mediaRecorder.start(100);
        log('SYSTEM_START', 'Camera active, battery initiated');
        task_calibration();
    } catch(e) {
        showUI('Camera Access Denied', 'Grant webcam permission and reload. The gaze tasks require camera access.', 'Retry', initBattery);
    }
}

// ══════════════════════════════════════════
// TASK 1 — 9-POINT CALIBRATION
// Reference: Salvucci & Goldberg (2000), Proceedings of ETRA
// Purpose: Establish corneal-reflection mapping for I-VT parsing
// ══════════════════════════════════════════
function task_calibration() {
    setProgress(1);
    showUI(
        'Task 1 · Gaze Calibration',
        'A gold dot will appear at 9 positions. Follow it smoothly with your eyes. Do not move your head.',
        'Begin Calibration',
        () => {
            hideUI();
            log('TASK_START', 'Calibration');
            const coords = [[10,10],[50,10],[90,10],[10,50],[50,50],[90,50],[10,90],[50,90],[90,90]];
            const dot = document.getElementById('dot');
            dot.style.display = 'block';
            let i = 0;
            function step() {
                if (i >= coords.length) { dot.style.display = 'none'; log('TASK_END','Calibration'); task_prosaccade(); return; }
                dot.style.left = coords[i][0] + '%';
                dot.style.top = coords[i][1] + '%';
                log('CALIB_POINT', `x:${coords[i][0]},y:${coords[i][1]}`);
                i++; setTimeout(step, 1400);
            }
            step();
        }
    );
}

// ══════════════════════════════════════════
// TASK 2 — PROSACCADE & ANTI-SACCADE
// Reference: Hutton & Ettinger (2006), Neuropsychology Review
// Measures: Voluntary saccade latency, inhibitory control, error rate
// Features: F1-F5, F7
// ══════════════════════════════════════════
function task_prosaccade() {
    setProgress(2);
    showUI(
        'Task 2 · Saccade Tasks (Part A)',
        'Look AT the gold dot as fast as possible each time it appears. Fix on the centre cross (+) between trials.',
        'Begin Prosaccade',
        () => {
            hideUI();
            log('TASK_START', 'Prosaccade');
            const positions = [20,80,20,80,50,20,80,50,20,80]; // % horizontal
            const dot = document.getElementById('dot');
            const cross = document.getElementById('cross');
            let i = 0;
            function runTrial() {
                if (i >= positions.length) { cross.style.display='none'; dot.style.display='none'; task_antisaccade(); return; }
                cross.style.display = 'block';
                setTimeout(() => {
                    cross.style.display = 'none';
                    dot.style.left = positions[i] + '%'; dot.style.top = '50%';
                    dot.style.display = 'block';
                    const t0 = performance.now();
                    log('PROSAC_STIM', `pos:${positions[i]},t:${t0.toFixed(2)}`);
                    i++;
                    setTimeout(() => { dot.style.display = 'none'; setTimeout(runTrial, 600); }, 1000);
                }, 900 + Math.random() * 600); // jittered SOA 900-1500ms, ref: Hutton 2006
            }
            runTrial();
        }
    );
}

function task_antisaccade() {
    showUI(
        'Task 2 · Saccade Tasks (Part B)',
        'A dot will flash. Look to the OPPOSITE side of the screen immediately. Inhibit the reflexive look toward the dot.',
        'Begin Anti-Saccade',
        () => {
            hideUI();
            log('TASK_START', 'AntiSaccade');
            const positions = [15,85,15,85,15,85,15,85]; // alternating L/R
            const dot = document.getElementById('dot');
            const cross = document.getElementById('cross');
            let i = 0;
            function runTrial() {
                if (i >= positions.length) { cross.style.display='none'; dot.style.display='none'; log('TASK_END','AntiSaccade'); task_visualsearch(); return; }
                cross.style.display = 'block';
                setTimeout(() => {
                    cross.style.display = 'none';
                    const side = positions[i] < 50 ? 'LEFT' : 'RIGHT';
                    dot.style.left = positions[i] + '%'; dot.style.top = '50%';
                    dot.style.display = 'block';
                    log('ANTISAC_STIM', `side:${side},t:${performance.now().toFixed(2)}`);
                    i++;
                    setTimeout(() => { dot.style.display = 'none'; setTimeout(runTrial, 700); }, 250); // Brief flash, ref: Munoz & Everling 2004
                }, 900 + Math.random() * 600);
            }
            runTrial();
        }
    );
}

// ══════════════════════════════════════════
// TASK 3 — VISUAL SEARCH
// Reference: Treisman & Gelade (1980), Cognitive Psychology
// Measures: Conjunction search RT, miss rate, distractor interference
// Features: F24, F25
// ══════════════════════════════════════════
function task_visualsearch() {
    setProgress(3);
    showUI(
        'Task 3 · Visual Search',
        'Find the orange circle (○) among blue squares. Press SPACE when you see it, or N if it is absent. Respond as fast as possible.',
        'Begin Visual Search',
        () => {
            hideUI();
            log('TASK_START', 'VisualSearch');
            const stage = document.getElementById('stage');
            const stim = document.getElementById('stim');
            stim.style.display = 'flex';
            // 20 trials: 14 target-present, 6 target-absent
            const trials = [];
            for (let k=0; k<14; k++) trials.push(true);
            for (let k=0; k<6; k++) trials.push(false);
            trials.sort(() => Math.random()-0.5);

            let trialIdx = 0;
            let canvas = document.createElement('canvas');
            canvas.width = 800; canvas.height = 560;
            canvas.style.cssText = 'position:absolute;inset:0;z-index:15;';
            stage.appendChild(canvas);
            const ctx = canvas.getContext('2d');
            let t0;
            let responded = false;

            function drawSearch(hasTarget) {
                ctx.clearRect(0,0,800,560);
                const positions = [];
                function noOverlap(x,y) { return positions.every(p => Math.hypot(p[0]-x,p[1]-y) > 55); }
                for (let d=0; d<11; d++) {
                    let x,y,tries=0;
                    do { x=60+Math.random()*680; y=60+Math.random()*440; tries++; } while (!noOverlap(x,y) && tries<50);
                    positions.push([x,y]);
                    ctx.fillStyle='#3A8FD4'; ctx.strokeStyle='#5AAAE8'; ctx.lineWidth=2;
                    ctx.beginPath(); ctx.rect(x-20,y-20,40,40); ctx.fill(); ctx.stroke();
                }
                if (hasTarget) {
                    let x,y,tries=0;
                    do { x=60+Math.random()*680; y=60+Math.random()*440; tries++; } while (!noOverlap(x,y) && tries<50);
                    ctx.fillStyle='#D4A843'; ctx.strokeStyle='#E8C070'; ctx.lineWidth=2;
                    ctx.beginPath(); ctx.arc(x,y,22,0,Math.PI*2); ctx.fill(); ctx.stroke();
                }
                t0 = performance.now();
                responded = false;
            }

            function nextTrial() {
                if (trialIdx >= trials.length) {
                    ctx.clearRect(0,0,800,560); canvas.remove();
                    stim.style.display='none';
                    log('TASK_END','VisualSearch');
                    task_rt();
                    return;
                }
                const hasTarget = trials[trialIdx];
                drawSearch(hasTarget);
                log('VS_TRIAL', `trial:${trialIdx},target:${hasTarget}`);
                trialIdx++;
            }

            function handleKey(e) {
                if (responded) return;
                const hasTarget = trials[trialIdx-1];
                if (e.code === 'Space' || e.key === 'n' || e.key === 'N') {
                    responded = true;
                    const rt = performance.now() - t0;
                    const resp = e.code === 'Space' ? 'PRESENT' : 'ABSENT';
                    const correct = (resp === 'PRESENT') === hasTarget;
                    log('VS_RESPONSE', `resp:${resp},correct:${correct},rt:${rt.toFixed(2)}`);
                    setTimeout(nextTrial, 300);
                }
            }
            document.addEventListener('keydown', handleKey);
            keyHandler = handleKey;
            stim.innerHTML = '<div id="stim-box"><p style="font-size:13px;color:#7A99A8;">Press <strong style="color:#D4A843;">SPACE</strong> = target present &nbsp;|&nbsp; <strong style="color:#D4A843;">N</strong> = target absent</p></div>';
            stim.style.alignItems = 'flex-end';
            stim.style.paddingBottom = '16px';
            setTimeout(nextTrial, 500);
        }
    );
}

// ══════════════════════════════════════════
// TASK 4 — SIMPLE REACTION TIME
// Reference: Luce (1986), Response Times, Oxford UP
// Measures: Mean RT, SD (intra-individual variability, IIV)
// Features: F9, F10
// ══════════════════════════════════════════
function task_rt() {
    setProgress(4);
    showUI(
        'Task 4 · Simple Reaction Time',
        'Press SPACE as fast as possible when the gold circle appears. Wait — do not press early!',
        'Begin Reaction Time Test',
        () => {
            hideUI();
            log('TASK_START', 'SimpleRT');
            const dot = document.getElementById('dot');
            dot.style.left = '50%'; dot.style.top = '50%';
            let trials = 0, t0;
            const N = 20;

            function nextTrial() {
                if (trials >= N) { dot.style.display='none'; log('TASK_END','SimpleRT'); task_gonogo(); return; }
                const delay = 1200 + Math.random() * 2000; // 1.2–3.2s jitter prevents anticipation
                setTimeout(() => {
                    dot.style.display = 'block';
                    t0 = performance.now();
                    log('RT_STIMULUS', `trial:${trials},t:${t0.toFixed(2)}`);
                }, delay);
            }

            keyHandler = function(e) {
                if (e.code !== 'Space') return;
                if (dot.style.display === 'none') {
                    log('RT_FALSE_START', `trial:${trials}`);
                    return;
                }
                const rt = performance.now() - t0;
                dot.style.display = 'none';
                log('RT_RESPONSE', `trial:${trials},rt:${rt.toFixed(2)}`);
                trials++;
                setTimeout(nextTrial, 400);
            };
            document.addEventListener('keydown', keyHandler);
            nextTrial();
        }
    );
}

// ══════════════════════════════════════════
// TASK 5 — GO / NO-GO
// Reference: Donders (1869), cited in Luce 1986; Aron (2007), TICS
// Measures: Commission errors (false alarms), omission errors, inhibition RT
// Features: F11, F12
// ══════════════════════════════════════════
function task_gonogo() {
    setProgress(5);
    showUI(
        'Task 5 · Go / No-Go',
        'Press SPACE for a GREEN circle (Go). Do NOT press for a RED circle (No-Go). Respond quickly!',
        'Begin Go/No-Go',
        () => {
            hideUI();
            log('TASK_START', 'GoNoGo');
            const stim = document.getElementById('stim');
            stim.style.display = 'flex'; stim.style.alignItems = 'center';
            const stimText = document.getElementById('stim-text');
            const stimLabel = document.getElementById('stim-label');

            // 75% Go, 25% No-Go (standard ratio, ref: Aron 2007)
            const sequence = [];
            for (let k=0; k<30; k++) sequence.push(Math.random() < 0.75 ? 'GO' : 'NOGO');

            let idx = 0, t0, responded;

            function runTrial() {
                if (idx >= sequence.length) {
                    stim.style.display='none'; log('TASK_END','GoNoGo'); task_nback();
                    return;
                }
                const type = sequence[idx];
                stimText.style.color = type === 'GO' ? '#4CAF82' : '#E07070';
                stimText.innerText = '●';
                stimLabel.innerText = type === 'GO' ? 'GO — press SPACE' : 'NO-GO — do not press';
                t0 = performance.now(); responded = false;
                log('GNG_STIM', `trial:${idx},type:${type},t:${t0.toFixed(2)}`);
                idx++;

                const timeout = setTimeout(() => {
                    if (!responded && type === 'GO') log('GNG_OMISSION', `trial:${idx-1}`);
                    stimText.innerText = '';
                    setTimeout(runTrial, 400);
                }, 800); // 800ms window, ref: Donders paradigm

                const resp = function(e) {
                    if (e.code !== 'Space') return;
                    responded = true;
                    clearTimeout(timeout);
                    document.removeEventListener('keydown', resp);
                    const rt = performance.now() - t0;
                    const correct = type === 'GO';
                    log('GNG_RESPONSE', `trial:${idx-1},type:${type},correct:${correct},rt:${rt.toFixed(2)}`);
                    if (!correct) log('GNG_COMMISSION', `trial:${idx-1}`);
                    stimText.innerText = '';
                    setTimeout(runTrial, 300);
                };
                document.addEventListener('keydown', resp);
            }
            runTrial();
        }
    );
}

// ══════════════════════════════════════════
// TASK 6 — N-BACK (1-back then 2-back)
// Reference: Kirchner (1958), J Exp Psych; Jaeggi et al. (2008), PNAS
// Measures: Working memory capacity, d-prime (hits, false alarms)
// Features: F13, F14
// ══════════════════════════════════════════
function task_nback() {
    setProgress(6);
    function runNBack(n, onDone) {
        const letters = 'BDFGHJKLMNPQRSTVWXZ'.split('');
        const seq = [];
        for (let k=0; k<20+n; k++) {
            if (k >= n && Math.random() < 0.3) seq.push(seq[k-n]); // 30% targets
            else seq.push(letters[Math.floor(Math.random()*letters.length)]);
        }
        log(`NBACK_START`, `n:${n}`);
        const grid = document.getElementById('nback-grid');
        grid.style.display = 'flex';
        const cells = Array.from(document.querySelectorAll('.nb-cell'));

        let idx = 0;
        function showItem() {
            if (idx >= seq.length) {
                cells.forEach(c => { c.classList.remove('active'); c.innerText=''; });
                grid.style.display = 'none';
                document.removeEventListener('keydown', keyResp);
                log(`NBACK_END`, `n:${n}`);
                onDone();
                return;
            }
            const isTarget = idx >= n && seq[idx] === seq[idx-n];
            // Display in random cell
            cells.forEach(c => { c.classList.remove('active'); c.innerText=''; });
            const randCell = Math.floor(Math.random()*9);
            cells[randCell].classList.add('active');
            cells[randCell].innerText = seq[idx];
            log('NBACK_STIM', `idx:${idx},letter:${seq[idx]},target:${isTarget},t:${performance.now().toFixed(2)}`);

            let responded = false;
            const timeout = setTimeout(() => {
                if (!responded && isTarget) log('NBACK_MISS', `idx:${idx}`);
                cells.forEach(c => { c.classList.remove('active'); c.innerText=''; });
                idx++;
                setTimeout(showItem, 400); // 400ms ISI, ref: Jaeggi 2008
            }, 1600); // 1600ms display

            function keyResp(e) {
                if (e.code !== 'Space') return;
                responded = true;
                clearTimeout(timeout);
                document.removeEventListener('keydown', keyResp);
                const rt = performance.now();
                log('NBACK_RESPONSE', `idx:${idx},target:${isTarget},rt:${rt.toFixed(2)}`);
                if (!isTarget) log('NBACK_FALSE_ALARM', `idx:${idx}`);
                cells.forEach(c => { c.classList.remove('active'); c.innerText=''; });
                idx++;
                setTimeout(showItem, 400);
            }
            document.addEventListener('keydown', keyResp);
            keyHandler = keyResp;
        }
        showItem();
    }

    showUI(
        'Task 6 · N-Back Working Memory',
        '1-Back: Press SPACE if the current letter matches the one shown one step before.\n2-Back (next): match two steps back.\n\nRespond quickly and accurately.',
        'Begin 1-Back',
        () => {
            hideUI();
            runNBack(1, () => {
                showUI('Task 6 · Part B', 'Now 2-back: press SPACE when the letter matches the one shown TWO steps ago.', 'Begin 2-Back', () => {
                    hideUI();
                    runNBack(2, () => {
                        log('TASK_END','NBack');
                        task_stroop();
                    });
                });
            });
        }
    );
}

// ══════════════════════════════════════════
// TASK 7 — STROOP COLOUR-WORD
// Reference: Stroop (1935), J Exp Psych; MacLeod (1991), Psych Bull
// Measures: RT congruent/incongruent, interference score, error rate
// Features: F15–F18
// ══════════════════════════════════════════
function task_stroop() {
    setProgress(7);
    showUI(
        'Task 7 · Stroop Colour-Word',
        'Name the INK COLOR of each word (ignore what the word says). Press R = Red, G = Green, B = Blue, Y = Yellow.',
        'Begin Stroop',
        () => {
            hideUI();
            log('TASK_START','Stroop');
            const stage = document.getElementById('stage');
            const canvas = document.createElement('canvas');
            canvas.width = 800; canvas.height = 560;
            canvas.style.cssText = 'position:absolute;inset:0;z-index:15;';
            stage.appendChild(canvas);
            const ctx = canvas.getContext('2d');

            const COLORS = ['RED','GREEN','BLUE','YELLOW'];
            const INK = {'RED':'#E07070','GREEN':'#4CAF82','BLUE':'#70A0E0','YELLOW':'#D4A843'};
            const KEYS = {'r':'RED','g':'GREEN','b':'BLUE','y':'YELLOW'};

            // 32 trials: 16 congruent, 16 incongruent
            const trials = [];
            for (let k=0; k<16; k++) {
                const c = COLORS[k%4]; trials.push({word:c,ink:c,congruent:true});
            }
            for (let k=0; k<16; k++) {
                const word = COLORS[k%4];
                const ink = COLORS.filter(x=>x!==word)[k%3];
                trials.push({word,ink,congruent:false});
            }
            trials.sort(() => Math.random()-0.5);

            let idx=0, t0, responded;

            function nextTrial() {
                if (idx >= trials.length) {
                    ctx.clearRect(0,0,800,560); canvas.remove();
                    document.removeEventListener('keydown', kh);
                    log('TASK_END','Stroop');
                    task_trail();
                    return;
                }
                const tr = trials[idx];
                ctx.clearRect(0,0,800,560);
                ctx.font = 'bold 52px Inter, sans-serif';
                ctx.fillStyle = INK[tr.ink];
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText(tr.word, 400, 260);
                ctx.font = '14px Inter, sans-serif';
                ctx.fillStyle = '#5A7A8A';
                ctx.fillText('R=Red  G=Green  B=Blue  Y=Yellow', 400, 480);
                t0 = performance.now(); responded = false;
                log('STROOP_STIM', `word:${tr.word},ink:${tr.ink},cong:${tr.congruent},t:${t0.toFixed(2)}`);
                idx++;
            }

            const kh = function(e) {
                if (responded) return;
                const key = e.key.toLowerCase();
                if (!KEYS[key]) return;
                responded = true;
                const rt = performance.now() - t0;
                const resp = KEYS[key];
                const correct = resp === trials[idx-1].ink;
                log('STROOP_RESPONSE', `resp:${resp},correct:${correct},rt:${rt.toFixed(2)},cong:${trials[idx-1].congruent}`);
                setTimeout(nextTrial, 200);
            };
            document.addEventListener('keydown', kh);
            keyHandler = kh;
            setTimeout(nextTrial, 300);
        }
    );
}

// ══════════════════════════════════════════
// TASK 8 — TRAIL MAKING A & B
// Reference: Reitan (1958), Perceptual & Motor Skills
// Measures: Processing speed (A), cognitive flexibility (B), B-A delta
// Features: F19, F20, F21
// ══════════════════════════════════════════
function task_trail() {
    setProgress(8);
    function runTrail(part, onDone) {
        const canvas = document.getElementById('trail-canvas');
        canvas.style.display = 'block';
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0,0,800,560);

        const N = 10;
        // Scatter nodes with minimum spacing
        const nodes = [];
        function noOverlap(x,y) { return nodes.every(p=>Math.hypot(p.x-x,p.y-y)>70); }
        for (let k=0; k<N; k++) {
            let x,y,t=0;
            do { x=60+Math.random()*680; y=60+Math.random()*440; t++; } while (!noOverlap(x,y)&&t<200);
            let label = part==='A' ? String(k+1) : (k%2===0 ? String(k/2+1) : String.fromCharCode(65+Math.floor(k/2)));
            nodes.push({x,y,label,visited:false});
        }

        // Determine correct order for Part B: 1,A,2,B,3,C,...
        let correctOrder;
        if (part==='A') {
            correctOrder = nodes.slice().sort((a,b)=>parseInt(a.label)-parseInt(b.label));
        } else {
            correctOrder = [...nodes].sort((a,b) => {
                const rank = n => isNaN(n.label) ? (n.label.charCodeAt(0)-64)*2 : parseInt(n.label)*2-1;
                return rank(a)-rank(b);
            });
        }

        function draw(lastNode=null) {
            ctx.clearRect(0,0,800,560);
            if (lastNode) {
                ctx.strokeStyle='#D4A843'; ctx.lineWidth=2;
                for (let k=1; k<correctOrder.length; k++) {
                    if (correctOrder[k-1].visited && correctOrder[k].visited) {
                        ctx.beginPath();
                        ctx.moveTo(correctOrder[k-1].x, correctOrder[k-1].y);
                        ctx.lineTo(correctOrder[k].x, correctOrder[k].y);
                        ctx.stroke();
                    }
                }
            }
            nodes.forEach(n => {
                ctx.beginPath(); ctx.arc(n.x,n.y,22,0,Math.PI*2);
                ctx.fillStyle = n.visited ? '#2A3D48' : (n===correctOrder[nodes.filter(x=>x.visited).length] ? '#D4A843' : '#2A3D48');
                ctx.strokeStyle = n.visited ? '#4CAF82' : '#4A6070'; ctx.lineWidth=2;
                ctx.fill(); ctx.stroke();
                ctx.fillStyle = n.visited ? '#4CAF82' : '#E8EDF0';
                ctx.font='bold 14px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
                ctx.fillText(n.label, n.x, n.y);
            });
            ctx.font='13px Inter,sans-serif'; ctx.fillStyle='#5A7A8A'; ctx.textAlign='left';
            ctx.fillText(`Trail Making Part ${part} — connect in order: ${part==='A'?'1→2→3...':'1→A→2→B→3...'}`, 16, 24);
        }

        let nextIdx = 0;
        const t0 = performance.now();
        log(`TRAIL_START`, `part:${part}`);
        draw();

        canvas.onclick = function(e) {
            const rect = canvas.getBoundingClientRect();
            const mx = (e.clientX-rect.left)*(800/rect.width);
            const my = (e.clientY-rect.top)*(560/rect.height);
            const target = correctOrder[nextIdx];
            if (Math.hypot(target.x-mx, target.y-my) < 28) {
                target.visited = true;
                log('TRAIL_CLICK', `part:${part},idx:${nextIdx},node:${target.label},t:${performance.now().toFixed(2)}`);
                nextIdx++;
                draw(target);
                if (nextIdx >= N) {
                    const elapsed = (performance.now()-t0)/1000;
                    log('TRAIL_END', `part:${part},time_s:${elapsed.toFixed(3)}`);
                    canvas.onclick = null;
                    canvas.style.display='none';
                    onDone();
                }
            } else {
                // Wrong node — log error
                const clicked = nodes.find(n=>Math.hypot(n.x-mx,n.y-my)<28);
                if (clicked) log('TRAIL_ERROR', `part:${part},idx:${nextIdx},clicked:${clicked.label}`);
            }
        };
    }

    showUI(
        'Task 8 · Trail Making',
        'Part A: Click circles 1→2→3→... in order.\nPart B (next): Alternate numbers and letters: 1→A→2→B→3...',
        'Begin Trail Making A',
        () => {
            hideUI();
            runTrail('A', () => {
                showUI('Trail Making Part B', 'Now alternate: 1→A→2→B→3→C... Click as fast as possible.', 'Begin Part B', () => {
                    hideUI();
                    runTrail('B', () => {
                        log('TASK_END','TrailMaking');
                        task_corsi();
                    });
                });
            });
        }
    );
}

// ══════════════════════════════════════════
// TASK 9 — CORSI BLOCK TAPPING
// Reference: Corsi (1972), Milner (1971), Neuropsychologia
// Measures: Visuospatial working memory span
// Feature: F22
// ══════════════════════════════════════════
function task_corsi() {
    setProgress(9);
    showUI(
        'Task 9 · Corsi Block Tapping',
        'Blocks will light up in a sequence. Tap them back in the same order. The sequence gets longer.',
        'Begin Corsi',
        () => {
            hideUI();
            log('TASK_START','Corsi');
            const area = document.getElementById('corsi-area');
            area.style.display = 'block';
            area.innerHTML = '';

            // 9 blocks at fixed positions (Milner layout)
            const positions = [
                [120,160],[240,80],[380,200],[520,100],[660,180],
                [100,320],[280,360],[440,300],[600,340]
            ];
            const blocks = positions.map((pos,i) => {
                const b = document.createElement('div');
                b.className = 'corsi-block';
                b.style.left = pos[0]+'px'; b.style.top = pos[1]+'px';
                b.dataset.id = i;
                area.appendChild(b);
                return b;
            });

            let span = 2, consecutiveFails = 0, maxSpan = 0;

            function lightUp(seq, onDone) {
                let i=0;
                function step() {
                    if (i>=seq.length) { onDone(); return; }
                    blocks.forEach(b=>b.className='corsi-block');
                    blocks[seq[i]].classList.add('lit');
                    setTimeout(()=>{ blocks[seq[i]].classList.remove('lit'); i++; setTimeout(step,400); },700);
                }
                step();
            }

            function runTrial() {
                if (consecutiveFails>=2 || span>9) {
                    area.style.display='none';
                    log('CORSI_END', `max_span:${maxSpan}`);
                    log('TASK_END','Corsi');
                    task_digitspan();
                    return;
                }
                const seq = Array.from({length:span}, ()=>Math.floor(Math.random()*9));
                log('CORSI_SEQ', `span:${span},seq:${seq.join(',')}`);
                let clickSeq=[], clickIdx=0;
                blocks.forEach(b=>{
                    b.onclick=function(){
                        if(clickIdx>=span) return;
                        const id=parseInt(b.dataset.id);
                        clickSeq.push(id);
                        b.classList.add(id===seq[clickIdx]?'correct':'wrong');
                        setTimeout(()=>b.className='corsi-block',300);
                        clickIdx++;
                        if(clickIdx===span){
                            const correct=clickSeq.every((v,i)=>v===seq[i]);
                            log('CORSI_RESPONSE',`span:${span},correct:${correct},response:${clickSeq.join(',')}`);
                            blocks.forEach(b2=>b2.onclick=null);
                            if(correct){ if(span>maxSpan) maxSpan=span; consecutiveFails=0; span++; }
                            else { consecutiveFails++; }
                            setTimeout(runTrial,800);
                        }
                    };
                });
                lightUp(seq, ()=>{});
            }

            lightUp([0,1], runTrial); // demo sequence
        }
    );
}

// ══════════════════════════════════════════
// TASK 10 — DIGIT SPAN FORWARD
// Reference: Wechsler (1997), WAIS-III; Drachman & Arbit (1966)
// Measures: Verbal working memory span
// Feature: F23
// ══════════════════════════════════════════
function task_digitspan() {
    setProgress(10);
    showUI(
        'Task 10 · Digit Span',
        'A sequence of digits will appear one at a time. When they stop, type them in order and press Submit.',
        'Begin Digit Span',
        () => {
            hideUI();
            log('TASK_START','DigitSpan');
            const area = document.getElementById('digitspan-area');
            area.style.display = 'flex';
            const disp = document.getElementById('digit-display');
            const inputWrap = document.getElementById('digit-input-wrap');
            const inp = document.getElementById('digit-input');

            let span=3, fails=0, maxSpan=0;

            function runTrial() {
                if(fails>=2||span>9){
                    area.style.display='none';
                    log('DIGITSPAN_END',`max_span:${maxSpan}`);
                    log('TASK_END','DigitSpan');
                    finishBattery();
                    return;
                }
                const seq=Array.from({length:span},()=>Math.floor(Math.random()*10));
                log('DIGITSPAN_SEQ',`span:${span},seq:${seq.join('')}`);
                inp.value=''; inputWrap.style.display='none'; disp.style.display='block';
                let i=0;
                function showDigit(){
                    if(i>=seq.length){ disp.style.display='none'; inputWrap.style.display='flex'; inp.focus(); return; }
                    disp.innerText=seq[i]; i++;
                    setTimeout(()=>{ disp.innerText=''; setTimeout(showDigit,300); },800); // 800ms each, ref: Wechsler 1997
                }
                showDigit();
                window._dsSeq=seq;
            }

            window.submitDigitSpan=function(){
                const resp=inp.value.trim().split('').map(Number);
                const correct=resp.every((v,i)=>v===window._dsSeq[i])&&resp.length===window._dsSeq.length;
                log('DIGITSPAN_RESPONSE',`span:${span},correct:${correct},response:${resp.join('')}`);
                if(correct){ if(span>maxSpan) maxSpan=span; fails=0; span++; }
                else fails++;
                setTimeout(runTrial,400);
            };
            runTrial();
        }
    );
}

// ══════════════════════════════════════════
// EXPORT
// ══════════════════════════════════════════
function finishBattery() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    else exportAll(null);
}

function exportAll(videoBlob) {
    if (videoBlob) {
        const a=document.createElement('a');
        a.href=URL.createObjectURL(videoBlob);
        a.download='raw_gaze_video.webm'; a.click();
    }
    const jBlob=new Blob([JSON.stringify(logs,null,2)],{type:'application/json'});
    const b=document.createElement('a');
    b.href=URL.createObjectURL(jBlob);
    b.download='interaction_logs.json'; b.click();

    document.getElementById('progress-bar').style.width='100%';
    document.getElementById('task-label').innerText='All tasks complete';
    showUI(
        'Battery Complete',
        'Both files have downloaded. Return to the main window and click "Upload for Analysis" to generate your report.',
        '—',
        ()=>{}
    );
    document.getElementById('btn').disabled=true;
}
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# PAGE 3: UPLOAD & ANALYSIS
# ─────────────────────────────────────────────
def page_upload():
    section_header("Upload Assessment Data", "Upload both files exported by the battery to compute all 25 biomarkers.")

    info_box("""
    The video is analysed frame-by-frame using MediaPipe Face Mesh to extract iris centre coordinates
    (landmarks 468 & 473). The I-VT velocity threshold algorithm (Salvucci & Goldberg, 2000) then
    classifies each inter-frame movement as a fixation or saccade.
    JSON logs provide ground-truth event timestamps for all cognitive tasks.
    """)

    col1, col2 = st.columns(2)
    with col1:
        video_file = st.file_uploader("raw_gaze_video.webm", type=["webm", "mp4", "avi"])
    with col2:
        log_file = st.file_uploader("interaction_logs.json", type=["json"])

    if video_file and log_file:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Compute All 25 Biomarkers"):
                with st.spinner("Running computer vision pipeline and extracting features..."):
                    logs_data = json.load(log_file)
                    results = run_analysis(video_file, logs_data)
                    st.session_state["results"] = results
                    st.session_state["event_logs"] = logs_data
                    st.session_state["page"] = "results"
                    st.rerun()


# ─────────────────────────────────────────────
# ANALYSIS ENGINE
# ─────────────────────────────────────────────
def run_analysis(video_file, logs: list) -> dict:
    """
    Full feature extraction pipeline.
    Returns dict with all 25 features + metadata.
    """
    # ── 1. VIDEO PROCESSING ──────────────────
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tfile.write(video_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    gaze_stream = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # Iris landmarks 468 (left) and 473 (right) — MediaPipe Iris model
            # Reference: Kartynnik et al. (2019), arXiv:1907.06724
            lx, ly = lm[468].x, lm[468].y
            rx, ry = lm[473].x, lm[473].y
            avg_x = (lx + rx) / 2.0
            avg_y = (ly + ry) / 2.0
            gaze_stream.append({
                't': (frame_idx / fps) * 1000.0,
                'x': avg_x,
                'y': avg_y
            })
        frame_idx += 1

    cap.release()
    face_mesh.close()
    os.unlink(tfile.name)

    # ── 2. I-VT CLASSIFICATION ───────────────
    # Velocity-based fixation/saccade parsing
    # Reference: Salvucci & Goldberg (2000), Proceedings of ETRA, pp. 71-78
    # Threshold: 100 deg/s for saccade onset (standard value, ref: Rayner 1998)
    VELOCITY_THRESH = 100.0  # deg/s
    D_CM = 60.0              # standard viewing distance, cm
    PX_PER_CM = 1920 / 34.5  # ~55.6 px/cm for a 27" 1080p monitor at 60cm

    fixations_raw = []
    saccades = []

    for i in range(1, len(gaze_stream)):
        t1, x1, y1 = gaze_stream[i-1]['t'], gaze_stream[i-1]['x'], gaze_stream[i-1]['y']
        t2, x2, y2 = gaze_stream[i]['t'], gaze_stream[i]['x'], gaze_stream[i]['y']
        dt = (t2 - t1) / 1000.0
        if dt <= 0:
            continue
        dist_px = np.sqrt(((x2 - x1) * 1920) ** 2 + ((y2 - y1) * 1080) ** 2)
        dist_cm = dist_px / PX_PER_CM
        dist_deg = np.degrees(np.arctan(dist_cm / D_CM))
        velocity = dist_deg / dt

        if velocity >= VELOCITY_THRESH:
            saccades.append({
                't_start': t1, 't_end': t2,
                'amp': dist_deg,
                'velocity': velocity,
                'x_end': x2, 'y_end': y2
            })
        else:
            fixations_raw.append({
                't_start': t1, 't_end': t2,
                'x': (x1 + x2) / 2,
                'y': (y1 + y2) / 2,
                'duration': dt * 1000
            })

    # Merge adjacent fixations within 50ms gap (noise reduction)
    # Minimum fixation duration 100ms — ref: Salvucci & Goldberg 2000
    MIN_FIX_DUR = 100.0
    fixations = []
    buf = []
    for f in fixations_raw:
        if not buf or (f['t_start'] - buf[-1]['t_end']) < 50:
            buf.append(f)
        else:
            dur = sum(b['duration'] for b in buf)
            if dur >= MIN_FIX_DUR:
                fixations.append({
                    't_start': buf[0]['t_start'],
                    'duration': dur,
                    'x': float(np.mean([b['x'] for b in buf])),
                    'y': float(np.mean([b['y'] for b in buf]))
                })
            buf = [f]

    # ── 3. OCULOMOTOR FEATURES ───────────────
    # F1: Mean Fixation Duration (ms) — ref: Rayner (1998), Psych Bull, 124(3)
    f1_mfd = float(np.mean([f['duration'] for f in fixations])) if fixations else 0.0

    # F2: Fixation Count
    f2_fc = len(fixations)

    # F3: Mean Saccade Latency (ms from stimulus to first saccade)
    # Computed per-event in anti-saccade section below

    # F4: Mean Saccade Amplitude (deg) — ref: Bahill et al. (1975)
    f4_sa = float(np.mean([s['amp'] for s in saccades])) if saccades else 0.0

    # F5: Saccade Peak Velocity (deg/s) — main-sequence relationship, ref: Leigh & Zee (2015)
    f5_spv = float(np.mean([s['velocity'] for s in saccades])) if saccades else 0.0

    # F6: Gaze Path Entropy (bits) — spatial distribution of attention
    # Reference: Andrews & Coppola (1999); Shannon entropy of fixation distribution
    grid = np.zeros((5, 5))
    for f in fixations:
        gx = int(np.clip(f['x'] * 5, 0, 4))
        gy = int(np.clip(f['y'] * 5, 0, 4))
        grid[gx, gy] += f['duration']
    f6_entropy = 0.0
    if grid.sum() > 0:
        pk = grid.flatten() / grid.sum()
        f6_entropy = float(-sum(p * log2(p) for p in pk if p > 0))

    # F8: ROI Coverage (%)
    f8_roi = float((np.count_nonzero(grid) / 25.0) * 100)

    # ── 4. ANTI-SACCADE FEATURES ─────────────
    # F3: Saccade latency, F7: Error rate
    # Reference: Hutton & Ettinger (2006), Neuropsychology Review
    latencies = []
    anti_errors = 0
    anti_trials = 0

    for ev in logs:
        if ev['event'] == 'ANTISAC_STIM':
            anti_trials += 1
            ev_t = float(ev['timestamp_ms'])
            side = ev['details'].split('side:')[1].split(',')[0] if 'side:' in ev['details'] else 'RIGHT'
            # First saccade after stimulus
            next_sac = next((s for s in saccades if s['t_start'] >= ev_t), None)
            if next_sac:
                lat = next_sac['t_start'] - ev_t
                if 80 < lat < 1000:  # valid latency window, ref: Munoz & Everling 2004
                    latencies.append(lat)
                went_left = next_sac['x_end'] < 0.5
                # Error = looking TOWARD stimulus instead of away
                if (side == 'LEFT' and went_left) or (side == 'RIGHT' and not went_left):
                    anti_errors += 1

    f3_sl = float(np.mean(latencies)) if latencies else 0.0
    f7_aser = float((anti_errors / anti_trials) * 100) if anti_trials > 0 else 0.0

    # ── 5. SIMPLE RT FEATURES ────────────────
    # F9: Mean RT, F10: IIV (SD)
    # Reference: Hultsch et al. (2002), Neuropsychology
    rt_vals = []
    for ev in logs:
        if ev['event'] == 'RT_RESPONSE' and 'rt:' in ev['details']:
            try:
                rt = float(ev['details'].split('rt:')[1])
                if 100 < rt < 1500:  # outlier filter
                    rt_vals.append(rt)
            except:
                pass

    f9_rt_mean = float(np.mean(rt_vals)) if rt_vals else 0.0
    f10_rt_iiv = float(np.std(rt_vals, ddof=1)) if len(rt_vals) > 1 else 0.0

    # ── 6. GO/NO-GO FEATURES ─────────────────
    # F11: Commission error rate, F12: Omission error rate
    # Reference: Aron (2007), Trends Cogn Sci; Logan (1994)
    gng_go_trials = sum(1 for e in logs if e['event'] == 'GNG_STIM' and 'GO' in e['details'] and 'NOGO' not in e['details'])
    gng_nogo_trials = sum(1 for e in logs if e['event'] == 'GNG_STIM' and 'NOGO' in e['details'])
    gng_commissions = sum(1 for e in logs if e['event'] == 'GNG_COMMISSION')
    gng_omissions = sum(1 for e in logs if e['event'] == 'GNG_OMISSION')

    f11_commission = float((gng_commissions / gng_nogo_trials) * 100) if gng_nogo_trials > 0 else 0.0
    f12_omission = float((gng_omissions / gng_go_trials) * 100) if gng_go_trials > 0 else 0.0

    # ── 7. N-BACK FEATURES ───────────────────
    # F13: d-prime (signal detection theory sensitivity)
    # F14: Response bias (beta / c)
    # Reference: Green & Swets (1966); Jaeggi et al. (2008), PNAS
    def compute_dprime(logs_subset, n_label):
        hits, misses, fas, crs = 0, 0, 0, 0
        for ev in logs_subset:
            if 'NBACK' not in ev['event']:
                continue
            if ev['event'] == 'NBACK_STIM' and f'n:{n_label}' not in str(logs_subset):
                continue
            if ev['event'] == 'NBACK_MISS':
                misses += 1
            elif ev['event'] == 'NBACK_FALSE_ALARM':
                fas += 1
            elif ev['event'] == 'NBACK_RESPONSE':
                details = ev['details']
                if 'target:True' in details:
                    hits += 1
                elif 'target:False' in details:
                    fas += 1
        total_targets = hits + misses
        total_non = max(1, len([e for e in logs_subset if e['event'] == 'NBACK_STIM' and 'target:False' in e['details']]))
        hr = np.clip(hits / max(total_targets, 1), 0.01, 0.99)
        far = np.clip(fas / max(total_non, 1), 0.01, 0.99)
        dprime = float(scipy_stats.norm.ppf(hr) - scipy_stats.norm.ppf(far))
        bias = float(-0.5 * (scipy_stats.norm.ppf(hr) + scipy_stats.norm.ppf(far)))
        return dprime, bias

    all_nback = [e for e in logs if 'NBACK' in e['event']]
    f13_dprime, f14_bias = compute_dprime(all_nback, 2)

    # ── 8. STROOP FEATURES ───────────────────
    # F15: Congruent RT, F16: Incongruent RT, F17: Interference, F18: Error rate
    # Reference: MacLeod (1991), Psych Bull 109(2): 163-203
    stroop_cong_rt, stroop_incong_rt = [], []
    stroop_errors = 0
    stroop_total = 0

    for ev in logs:
        if ev['event'] == 'STROOP_RESPONSE':
            d = ev['details']
            try:
                rt = float(d.split('rt:')[1].split(',')[0])
                correct = 'correct:True' in d
                cong = 'cong:True' in d
                if 200 < rt < 3000:
                    if cong:
                        stroop_cong_rt.append(rt)
                    else:
                        stroop_incong_rt.append(rt)
                if not correct:
                    stroop_errors += 1
                stroop_total += 1
            except:
                pass

    f15_stroop_cong = float(np.mean(stroop_cong_rt)) if stroop_cong_rt else 0.0
    f16_stroop_incong = float(np.mean(stroop_incong_rt)) if stroop_incong_rt else 0.0
    f17_stroop_interf = f16_stroop_incong - f15_stroop_cong
    f18_stroop_err = float((stroop_errors / stroop_total) * 100) if stroop_total > 0 else 0.0

    # ── 9. TRAIL MAKING FEATURES ─────────────
    # F19: TMT-A time, F20: TMT-B time, F21: B-A delta (cognitive flexibility index)
    # Reference: Reitan (1958); Lezak (2004) Neuropsychological Assessment
    def get_trail_time(part):
        start = next((e for e in logs if e['event'] == 'TRAIL_START' and f'part:{part}' in e['details']), None)
        end = next((e for e in logs if e['event'] == 'TRAIL_END' and f'part:{part}' in e['details']), None)
        if start and end:
            try:
                return float(end['details'].split('time_s:')[1])
            except:
                pass
        return 0.0

    f19_tmt_a = get_trail_time('A')
    f20_tmt_b = get_trail_time('B')
    f21_tmt_delta = f20_tmt_b - f19_tmt_a

    # ── 10. CORSI & DIGIT SPAN ───────────────
    # F22: Corsi Span, F23: Digit Span
    def get_max_span(task_prefix):
        span_ev = [e for e in logs if e['event'] == f'{task_prefix}_END']
        if span_ev:
            try:
                return int(span_ev[-1]['details'].split('max_span:')[1])
            except:
                pass
        return 0

    f22_corsi = get_max_span('CORSI')
    f23_digitspan = get_max_span('DIGITSPAN')

    # ── 11. VISUAL SEARCH FEATURES ───────────
    # F24: Mean RT, F25: Miss rate
    # Reference: Treisman & Gelade (1980), Cognitive Psychology
    vs_rts, vs_misses, vs_total = [], 0, 0
    for ev in logs:
        if ev['event'] == 'VS_RESPONSE':
            d = ev['details']
            try:
                rt = float(d.split('rt:')[1])
                correct = 'correct:True' in d
                vs_total += 1
                if 100 < rt < 5000:
                    vs_rts.append(rt)
                if not correct and 'ABSENT' in d:
                    vs_misses += 1  # missed target
            except:
                pass

    f24_vs_rt = float(np.mean(vs_rts)) if vs_rts else 0.0
    f25_vs_miss = float((vs_misses / vs_total) * 100) if vs_total > 0 else 0.0

    return {
        "F1_MFD": f1_mfd,
        "F2_FixationCount": f2_fc,
        "F3_SaccadeLatency": f3_sl,
        "F4_SaccadeAmplitude": f4_sa,
        "F5_SaccadePeakVelocity": f5_spv,
        "F6_GazeEntropy": f6_entropy,
        "F7_AntiSaccadeErrorRate": f7_aser,
        "F8_ROICoverage": f8_roi,
        "F9_RT_Mean": f9_rt_mean,
        "F10_RT_IIV": f10_rt_iiv,
        "F11_CommissionErrors": f11_commission,
        "F12_OmissionErrors": f12_omission,
        "F13_NBack_dPrime": f13_dprime,
        "F14_NBack_Bias": f14_bias,
        "F15_StroopCongruentRT": f15_stroop_cong,
        "F16_StroopIncongruentRT": f16_stroop_incong,
        "F17_StroopInterference": f17_stroop_interf,
        "F18_StroopErrorRate": f18_stroop_err,
        "F19_TMT_A": f19_tmt_a,
        "F20_TMT_B": f20_tmt_b,
        "F21_TMT_Delta": f21_tmt_delta,
        "F22_CorsiSpan": f22_corsi,
        "F23_DigitSpan": f23_digitspan,
        "F24_VisualSearchRT": f24_vs_rt,
        "F25_VisualSearchMissRate": f25_vs_miss,
    }


# ─────────────────────────────────────────────
# PAGE 4: RESULTS
# ─────────────────────────────────────────────

# Normative reference ranges (adult population, 18-40)
# Sources: published test manuals and peer-reviewed normative studies
NORMS = {
    "F1_MFD":               {"label": "Mean Fixation Duration",       "unit": "ms",    "lo": 150, "hi": 350, "domain": "Oculomotor"},
    "F2_FixationCount":     {"label": "Fixation Count",               "unit": "",      "lo": 80,  "hi": 300, "domain": "Oculomotor"},
    "F3_SaccadeLatency":    {"label": "Saccade Latency",              "unit": "ms",    "lo": 150, "hi": 350, "domain": "Oculomotor"},
    "F4_SaccadeAmplitude":  {"label": "Saccade Amplitude",            "unit": "°",     "lo": 2.0, "hi": 8.0, "domain": "Oculomotor"},
    "F5_SaccadePeakVelocity": {"label": "Saccade Peak Velocity",      "unit": "°/s",   "lo": 200, "hi": 600, "domain": "Oculomotor"},
    "F6_GazeEntropy":       {"label": "Gaze Path Entropy",            "unit": "bits",  "lo": 2.0, "hi": 4.0, "domain": "Oculomotor"},
    "F7_AntiSaccadeErrorRate": {"label": "Anti-Saccade Error Rate",   "unit": "%",     "lo": 0,   "hi": 25,  "domain": "Inhibitory Control"},
    "F8_ROICoverage":       {"label": "ROI Coverage",                 "unit": "%",     "lo": 40,  "hi": 100, "domain": "Oculomotor"},
    "F9_RT_Mean":           {"label": "Simple RT Mean",               "unit": "ms",    "lo": 200, "hi": 350, "domain": "Processing Speed"},
    "F10_RT_IIV":           {"label": "RT Intra-individual Var.",     "unit": "ms",    "lo": 10,  "hi": 60,  "domain": "Processing Speed"},
    "F11_CommissionErrors": {"label": "Go/No-Go Commission Rate",     "unit": "%",     "lo": 0,   "hi": 20,  "domain": "Inhibitory Control"},
    "F12_OmissionErrors":   {"label": "Go/No-Go Omission Rate",      "unit": "%",     "lo": 0,   "hi": 10,  "domain": "Inhibitory Control"},
    "F13_NBack_dPrime":     {"label": "N-Back d′ (sensitivity)",     "unit": "",      "lo": 1.0, "hi": 4.0, "domain": "Working Memory"},
    "F14_NBack_Bias":       {"label": "N-Back Response Bias (c)",    "unit": "",      "lo": -1.0,"hi": 1.0, "domain": "Working Memory"},
    "F15_StroopCongruentRT": {"label": "Stroop Congruent RT",        "unit": "ms",    "lo": 400, "hi": 700, "domain": "Attention"},
    "F16_StroopIncongruentRT": {"label": "Stroop Incongruent RT",    "unit": "ms",    "lo": 500, "hi": 900, "domain": "Attention"},
    "F17_StroopInterference": {"label": "Stroop Interference Score", "unit": "ms",    "lo": 0,   "hi": 200, "domain": "Attention"},
    "F18_StroopErrorRate":  {"label": "Stroop Error Rate",           "unit": "%",     "lo": 0,   "hi": 10,  "domain": "Attention"},
    "F19_TMT_A":            {"label": "Trail Making A (time)",       "unit": "s",     "lo": 15,  "hi": 45,  "domain": "Processing Speed"},
    "F20_TMT_B":            {"label": "Trail Making B (time)",       "unit": "s",     "lo": 30,  "hi": 90,  "domain": "Executive Function"},
    "F21_TMT_Delta":        {"label": "TMT B–A Delta",               "unit": "s",     "lo": 10,  "hi": 50,  "domain": "Executive Function"},
    "F22_CorsiSpan":        {"label": "Corsi Block Span",            "unit": "",      "lo": 4,   "hi": 7,   "domain": "Working Memory"},
    "F23_DigitSpan":        {"label": "Digit Span Forward",          "unit": "",      "lo": 5,   "hi": 9,   "domain": "Working Memory"},
    "F24_VisualSearchRT":   {"label": "Visual Search RT Mean",       "unit": "ms",    "lo": 400, "hi": 1200,"domain": "Attention"},
    "F25_VisualSearchMissRate": {"label": "Visual Search Miss Rate", "unit": "%",     "lo": 0,   "hi": 15,  "domain": "Attention"},
}

DOMAIN_REFS = {
    "Oculomotor":        "Rayner (1998), Psych Bull; Salvucci & Goldberg (2000), ETRA",
    "Inhibitory Control":"Hutton & Ettinger (2006), Neuropsychology Review; Aron (2007), TICS",
    "Processing Speed":  "Luce (1986), Response Times; Hultsch et al. (2002), Neuropsychology",
    "Working Memory":    "Jaeggi et al. (2008), PNAS; Wechsler (1997), WAIS-III",
    "Attention":         "MacLeod (1991), Psych Bull; Treisman & Gelade (1980), Cognitive Psychology",
    "Executive Function":"Reitan (1958), Percept Mot Skills; Lezak (2004), Neuropsychological Assessment",
}

def classify(key, val):
    """Return (colour_class, label) based on norm ranges."""
    n = NORMS.get(key)
    if not n or val == 0.0:
        return "badge-blue", "No data"
    # For error rates and latency: lower = better (inverted norms)
    inverted_keys = {"F7_AntiSaccadeErrorRate", "F11_CommissionErrors", "F12_OmissionErrors",
                     "F18_StroopErrorRate", "F25_VisualSearchMissRate",
                     "F9_RT_Mean", "F10_RT_IIV", "F15_StroopCongruentRT",
                     "F16_StroopIncongruentRT", "F17_StroopInterference",
                     "F19_TMT_A", "F20_TMT_B", "F21_TMT_Delta"}
    in_range = n["lo"] <= val <= n["hi"]
    if in_range:
        return "badge-green", "Normal"
    if key in inverted_keys:
        return ("badge-green", "Excellent") if val < n["lo"] else ("badge-red", "Elevated")
    else:
        return ("badge-red", "Below norm") if val < n["lo"] else ("badge-gold", "Above norm")


def page_results():
    r = st.session_state["results"]
    p = st.session_state["participant"]

    # ── Header ──────────────────────────────
    st.markdown(f"""
    <div style='display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:24px;'>
        <div>
            <h1 style='font-size:24px; font-weight:700; margin-bottom:4px;'>Cognitive Assessment Report</h1>
            <p style='color:#7A99A8; font-size:14px;'>Participant {p.get('id','—')} &nbsp;·&nbsp; {p.get('age','—')} y/o {p.get('gender','—')} &nbsp;·&nbsp; {p.get('timestamp','')[:10]}</p>
        </div>
        <div style='text-align:right;'>
            <span class='badge badge-gold'>25 Biomarkers Computed</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Domain summary cards ────────────────
    domains = {}
    for key, val in r.items():
        dom = NORMS[key]["domain"]
        cls, lbl = classify(key, val)
        domains.setdefault(dom, []).append((cls, lbl))

    def domain_score(items):
        reds = sum(1 for c, _ in items if c == "badge-red")
        return reds

    cols = st.columns(6)
    for i, (dom, items) in enumerate(domains.items()):
        reds = domain_score(items)
        badge = "badge-green" if reds == 0 else ("badge-gold" if reds <= 1 else "badge-red")
        label = "All normal" if reds == 0 else f"{reds} flag{'s' if reds>1 else ''}"
        cols[i % 6].markdown(f"""
        <div class='metric-card'>
            <span class='label'>{dom}</span>
            <span class='value' style='font-size:16px;'><span class='badge {badge}'>{label}</span></span>
            <span class='norm'>{len(items)} features</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Feature table by domain ─────────────
    DOMAINS_ORDER = ["Oculomotor", "Processing Speed", "Inhibitory Control",
                     "Working Memory", "Attention", "Executive Function"]

    tabs = st.tabs(DOMAINS_ORDER)
    for tab, dom in zip(tabs, DOMAINS_ORDER):
        with tab:
            dom_feats = {k: v for k, v in r.items() if NORMS[k]["domain"] == dom}
            # Metric cards (max 4 per row)
            keys = list(dom_feats.keys())
            for row_start in range(0, len(keys), 4):
                row_keys = keys[row_start:row_start+4]
                cols = st.columns(len(row_keys))
                for col, key in zip(cols, row_keys):
                    val = dom_feats[key]
                    n = NORMS[key]
                    cls, lbl = classify(key, val)
                    unit = n["unit"]
                    if unit == "s":
                        disp = f"{val:.1f} s"
                    elif unit == "ms":
                        disp = f"{val:.0f} ms"
                    elif unit == "%":
                        disp = f"{val:.1f}%"
                    elif unit == "°" or unit == "°/s":
                        disp = f"{val:.1f}{unit}"
                    elif unit == "bits":
                        disp = f"{val:.2f} bits"
                    else:
                        disp = f"{val:.2f}"
                    col.markdown(f"""
                    <div class='metric-card'>
                        <span class='label'>{n['label']}</span>
                        <span class='value'>{disp}</span>
                        <span class='norm'><span class='badge {cls}'>{lbl}</span></span>
                    </div>
                    """, unsafe_allow_html=True)

            info_box(f"<strong style='color:#D4A843;'>Reference:</strong> {DOMAIN_REFS[dom]}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Full feature table ──────────────────
    section_header("Complete Feature Matrix", "All 25 biomarkers with normative classification")
    rows = ""
    for key, val in r.items():
        n = NORMS[key]
        cls, lbl = classify(key, val)
        unit = n["unit"]
        val_str = f"{val:.2f} {unit}".strip()
        rows += f"""
        <tr>
            <td style='color:#7A99A8; font-size:12px;'>{key}</td>
            <td>{n['label']}</td>
            <td><span class='badge badge-blue' style='font-size:11px;'>{n['domain']}</span></td>
            <td style='color:#D4A843; font-weight:500;'>{val_str}</td>
            <td>Norm: {n['lo']}–{n['hi']} {unit}</td>
            <td><span class='badge {cls}'>{lbl}</span></td>
        </tr>"""

    st.markdown(f"""
    <table class='results-table'>
        <thead><tr>
            <th>Code</th><th>Feature</th><th>Domain</th>
            <th>Value</th><th>Normative Range</th><th>Status</th>
        </tr></thead>
        <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Export buttons ───────────────────────
    section_header("Export Results")
    col1, col2, col3 = st.columns(3)

    # CSV
    df = pd.DataFrame([{
        "participant_id": p.get("id"),
        "timestamp": p.get("timestamp"),
        **r
    }])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    col1.download_button("Download CSV", csv_bytes, "cognitive_results.csv", "text/csv")

    # JSON
    export_dict = {"participant": p, "results": r, "norms": {k: NORMS[k] for k in r}}
    json_bytes = json.dumps(export_dict, indent=2).encode("utf-8")
    col2.download_button("Download JSON", json_bytes, "cognitive_results.json", "application/json")

    # PDF
    pdf_bytes = generate_pdf(p, r)
    col3.download_button("Download PDF Report", pdf_bytes, "cognitive_report.pdf", "application/pdf")

    # ── New session ────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if st.button("New Participant Session"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ─────────────────────────────────────────────
# PDF GENERATION
# ─────────────────────────────────────────────
def generate_pdf(participant: dict, results: dict) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_fill_color(30, 45, 53)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(212, 168, 67)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(15, 12)
    pdf.cell(0, 10, "Pocket-Precise Cognitive Diagnostic Report", ln=True)
    pdf.set_text_color(160, 179, 188)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(15, 26)
    pdf.cell(0, 8, f"Participant: {participant.get('id','—')}  |  Age: {participant.get('age','—')}  |  Date: {participant.get('timestamp','')[:10]}", ln=True)

    # Participant demographics
    pdf.set_text_color(30, 45, 53)
    pdf.set_xy(15, 48)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Participant Information", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for key, val in participant.items():
        if key != "timestamp":
            pdf.cell(0, 6, f"  {key.capitalize()}: {val}", ln=True)

    pdf.ln(6)

    # Results table
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Cognitive Biomarker Results", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(42, 61, 72)
    pdf.set_text_color(160, 179, 188)
    pdf.cell(45, 7, "Feature", border=1, fill=True)
    pdf.cell(35, 7, "Domain", border=1, fill=True)
    pdf.cell(30, 7, "Value", border=1, fill=True)
    pdf.cell(40, 7, "Normal Range", border=1, fill=True)
    pdf.cell(30, 7, "Status", border=1, fill=True, ln=True)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 45, 53)
    for key, val in results.items():
        n = NORMS[key]
        cls, lbl = classify(key, val)
        unit = n["unit"]
        val_str = f"{val:.2f} {unit}".strip()
        norm_str = f"{n['lo']}–{n['hi']} {unit}".strip()
        # Row colour
        if cls == "badge-red":
            pdf.set_fill_color(250, 235, 235)
        elif cls == "badge-green":
            pdf.set_fill_color(235, 250, 240)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.cell(45, 6, n['label'][:30], border=1, fill=True)
        pdf.cell(35, 6, n['domain'], border=1, fill=True)
        pdf.cell(30, 6, val_str, border=1, fill=True)
        pdf.cell(40, 6, norm_str, border=1, fill=True)
        pdf.cell(30, 6, lbl, border=1, fill=True, ln=True)

    # References
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, "Key References", ln=True)
    pdf.set_font("Helvetica", "", 8)
    refs = [
        "Rayner, K. (1998). Eye movements in reading and information processing. Psychological Bulletin, 124(3), 372-422.",
        "Salvucci, D. & Goldberg, J. (2000). Identifying fixations and saccades in eye-tracking protocols. ETRA 2000, 71-78.",
        "Hutton, S.B. & Ettinger, U. (2006). The antisaccade task as a research tool in psychopathology. Neuropsychology Review.",
        "Jaeggi, S.M. et al. (2008). Improving fluid intelligence with training on working memory. PNAS, 105(19), 6829-6833.",
        "MacLeod, C.M. (1991). Half a century of research on the Stroop effect. Psychological Bulletin, 109(2), 163-203.",
        "Reitan, R.M. (1958). Validity of the Trail Making Test as an indicator of organic brain damage. Percept Mot Skills.",
        "Wechsler, D. (1997). Wechsler Adult Intelligence Scale - Third Edition (WAIS-III). The Psychological Corporation.",
        "Treisman, A. & Gelade, G. (1980). A feature-integration theory of attention. Cognitive Psychology, 12(1), 97-136.",
        "Hultsch, D.F. et al. (2002). Intraindividual variability in cognitive performance in older adults. Neuropsychology.",
        "Green, D.M. & Swets, J.A. (1966). Signal Detection Theory and Psychophysics. Wiley.",
    ]
    for ref in refs:
        pdf.multi_cell(0, 5, f"• {ref}")

    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 140, 150)
    pdf.multi_cell(0, 5, "DISCLAIMER: This report is generated for research purposes only and does not constitute a clinical diagnosis. "
                         "Results should be interpreted by a qualified professional in the context of a full clinical assessment.")

    return bytes(pdf.output())


# ─────────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────────
def main():
    page = st.session_state["page"]
    if page == "consent":
        page_consent()
    elif page == "demographics":
        page_demographics()
    elif page == "battery":
        page_battery()
    elif page == "upload":
        page_upload()
    elif page == "results":
        page_results()


if __name__ == "__main__":
    main()
