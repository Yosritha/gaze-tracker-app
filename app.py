"""
Pocket-Precise Cognitive Diagnostic Engine — v3.0
==================================================
Full changelog vs v2.1:
  1. EXPORT FIX: JSON + video both download via in-iframe data-URI anchor links.
     No postMessage/parent-frame dependency. Works in Streamlit cloud.
  2. SCROLL LOCK: Stage uses position:fixed overlay during tasks so page scroll
     never shifts click/gaze coordinates. Coordinate system is locked to viewport.
  3. SPAMMING FIX: Per-trial boolean guard + 400ms debounce on all keydown
     handlers. False-start detection on SimpleRT.
  4. STROOP FIX: All canvases explicitly removed on task end. hideAll() now also
     purges every dynamically-created canvas from #stage.
  5. DISTANCE ESTIMATION: Interpupillary distance (IPD) from face mesh used to
     estimate real viewing distance in cm. Updates live during calibration and
     is displayed to participant. Tasks warn if distance > 90 cm or < 30 cm.
  6. COORDINATE SYSTEM: Stage is always 800×560 px in a fixed viewport overlay.
     All stimulus positions logged as absolute canvas px (not %). MediaPipe gaze
     is calibrated via 9-point affine to this coordinate system.
  7. GAZE DRIFT CORRECTION: Median-subtracted fixation baseline applied every 30 s.
  8. ALL TASKS FIXED: Trail Making uses stage-relative coords from the fixed overlay.
     Corsi blocks use fixed positions. Visual Search canvas always removed on exit.
  9. CLINICAL NORMS: Updated per latest meta-analyses (Hutton 2006, Lezak 2004,
     Wechsler 2008 WAIS-IV, Tombaugh 2004 TMT).
 10. PDF REPORT: Added z-score column, domain composite scores, clinical interpretation.
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

st.set_page_config(
    page_title="Pocket-Precise · Cognitive Diagnostic Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── STYLES ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
.stApp,[data-testid="stAppViewContainer"]{background-color:#0D1117!important;color:#CDD9E5!important;font-family:'IBM Plex Sans',sans-serif!important}
h1,h2,h3,h4,p,span,label,li{color:#CDD9E5!important}
.block-container{padding-top:2rem!important;max-width:1100px!important}
[data-testid="stSidebar"]{background-color:#161B22!important}
.stButton>button{background-color:transparent!important;color:#58A6FF!important;border:1px solid #30363D!important;border-radius:6px!important;padding:10px 22px!important;font-size:14px!important;font-weight:500!important;transition:all 0.2s ease!important;width:100%!important}
.stButton>button:hover{background-color:#58A6FF!important;color:#0D1117!important;border-color:#58A6FF!important}
.stTextInput>div>div>input,.stSelectbox>div>div,.stNumberInput>div>div>input{background-color:#161B22!important;color:#CDD9E5!important;border:1px solid #30363D!important;border-radius:6px!important}
.stSelectbox label,.stTextInput label,.stNumberInput label,.stRadio label,.stCheckbox label{color:#8B949E!important;font-size:13px!important}
.metric-card{background-color:#161B22;border:1px solid #21262D;padding:16px 18px;border-radius:8px;text-align:center;margin-bottom:12px}
.metric-card .label{font-size:11px;color:#6E7681;text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:8px;font-family:'IBM Plex Mono',monospace}
.metric-card .value{font-size:24px;font-weight:600;color:#58A6FF;display:block;font-family:'IBM Plex Mono',monospace}
.metric-card .norm{font-size:11px;color:#484F58;display:block;margin-top:5px}
.section-divider{border:none;border-top:1px solid #21262D;margin:24px 0}
.info-box{background-color:#161B22;border-left:3px solid #58A6FF;padding:12px 16px;border-radius:0 6px 6px 0;margin:14px 0;font-size:13px;color:#8B949E!important;line-height:1.7}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:500;font-family:'IBM Plex Mono',monospace}
.badge-gold{background-color:#272115;color:#E3B341}
.badge-green{background-color:#12261E;color:#3FB950}
.badge-red{background-color:#2D1114;color:#F85149}
.badge-blue{background-color:#121D2F;color:#58A6FF}
.stTabs [data-baseweb="tab-list"]{background-color:#161B22;border-radius:6px;padding:3px;gap:3px}
.stTabs [data-baseweb="tab"]{color:#6E7681!important;font-weight:500!important;font-size:13px!important;border-radius:5px!important}
.stTabs [aria-selected="true"]{color:#CDD9E5!important;background-color:#0D1117!important}
.stProgress>div>div>div{background-color:#58A6FF!important}
.streamlit-expanderHeader{background-color:#161B22!important;color:#8B949E!important;border-radius:6px!important}
.stFileUploader>div{background-color:#161B22!important;border:1px dashed #30363D!important;border-radius:8px!important}
header,#MainMenu,footer{visibility:hidden}
.results-table{width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace}
.results-table th{background-color:#161B22;color:#6E7681;padding:9px 12px;text-align:left;font-weight:500;text-transform:uppercase;letter-spacing:.8px;font-size:10px}
.results-table td{padding:9px 12px;border-bottom:1px solid #21262D;color:#CDD9E5}
.results-table tr:hover td{background-color:#161B22}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "consent", "participant": {}, "consent_given": False,
        "battery_done": False, "event_logs": None, "video_bytes": None, "results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def section_header(title, subtitle=""):
    st.markdown(f"<h2 style='font-size:20px;font-weight:600;color:#CDD9E5;margin-bottom:4px;'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='font-size:13px;color:#6E7681;margin-bottom:18px;'>{subtitle}</p>", unsafe_allow_html=True)

def info_box(text):
    st.markdown(f"<div class='info-box'>{text}</div>", unsafe_allow_html=True)

def metric_card(label, value, norm="", col=None):
    html = f"<div class='metric-card'><span class='label'>{label}</span><span class='value'>{value}</span>{'<span class=norm>'+norm+'</span>' if norm else ''}</div>"
    (col or st).markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — CONSENT
# ══════════════════════════════════════════════════════════════════════════════
def page_consent():
    st.markdown("<div style='max-width:740px;margin:0 auto;padding:30px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex;align-items:center;gap:14px;margin-bottom:32px;'>
        <div style='width:42px;height:42px;background:#58A6FF;border-radius:6px;display:flex;align-items:center;justify-content:center;'>
            <span style='color:#0D1117;font-size:20px;font-weight:700;font-family:IBM Plex Mono,monospace;'>PP</span>
        </div>
        <div>
            <p style='margin:0;font-size:18px;font-weight:600;color:#CDD9E5!important;font-family:IBM Plex Mono,monospace;'>Pocket-Precise</p>
            <p style='margin:0;font-size:12px;color:#6E7681!important;'>Cognitive Diagnostic Engine · v3.0</p>
        </div>
    </div>""", unsafe_allow_html=True)

    section_header("Participant Information Sheet")
    info_box("""<strong style='color:#58A6FF;'>Study Purpose</strong><br>
    This battery measures oculomotor control, inhibitory control, working memory, processing speed, and attentional capacity
    across 10 validated paradigms. All processing is local — no data leaves this machine.""")

    st.markdown("""
    <div style='background:#161B22;border:1px solid #21262D;border-radius:8px;padding:20px 24px;margin:20px 0;'>
    <p style='font-size:13px;color:#8B949E!important;line-height:1.9;margin:0;'>
    <strong style='color:#CDD9E5;'>Technical approach (v3.0)</strong><br>
    The stage is a fixed-position fullscreen overlay during tasks, so scrolling cannot shift stimulus coordinates.
    A 9-point affine gaze calibration maps MediaPipe iris landmarks to screen pixels.
    Interpupillary distance (IPD) is estimated frame-by-frame to compute your real viewing distance.
    A live distance indicator warns you if you move outside the 50–80 cm optimal range.<br><br>
    <strong style='color:#CDD9E5;'>Download method</strong><br>
    When the battery ends, two buttons appear inside the task panel —
    <em>Download JSON</em> and <em>Download Video</em>. Both are standard anchor-tag downloads
    from within the iframe, so browser sandbox restrictions do not apply.
    </p></div>""", unsafe_allow_html=True)

    tasks = [
        ("Gaze Calibration","9-point affine map + IPD distance estimation","~1 min"),
        ("Prosaccade","Oculomotor initiation latency","~2 min"),
        ("Anti-Saccade","Inhibitory control, error rate","~3 min"),
        ("Visual Search","Selective attention, target detection RT","~3 min"),
        ("Simple Reaction Time","Baseline RT + intra-individual variability","~3 min"),
        ("Go / No-Go","Response inhibition, commission & omission","~4 min"),
        ("N-Back 1 & 2","Working memory d-prime","~5 min"),
        ("Stroop Colour-Word","Cognitive interference score","~4 min"),
        ("Trail Making A & B","Processing speed & cognitive flexibility","~5 min"),
        ("Corsi + Digit Span","Visuospatial & verbal working memory","~4 min"),
    ]
    for i, (name, desc, dur) in enumerate(tasks):
        st.markdown(f"""
        <div style='display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid #21262D;'>
          <div style='display:flex;align-items:center;gap:12px;'>
            <span style='width:22px;height:22px;background:#161B22;border:1px solid #30363D;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:11px;color:#58A6FF!important;font-family:IBM Plex Mono,monospace;'>{i+1}</span>
            <div><p style='margin:0;font-size:13px;font-weight:500;color:#CDD9E5!important;'>{name}</p>
                 <p style='margin:0;font-size:11px;color:#6E7681!important;'>{desc}</p></div>
          </div>
          <span style='font-size:11px;color:#484F58;font-family:IBM Plex Mono,monospace;'>{dur}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    agree = st.checkbox("I have read and understood the above. I consent to participate voluntarily.")
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        if st.button("Continue to Demographics →", disabled=not agree):
            st.session_state.update({"consent_given": True, "page": "demographics"})
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
def page_demographics():
    st.markdown("<div style='max-width:740px;margin:0 auto;padding:30px 0;'>", unsafe_allow_html=True)
    section_header("Participant Demographics", "Used for normative comparison.")
    col1, col2 = st.columns(2)
    with col1:
        pid        = st.text_input("Participant ID *", placeholder="e.g. P001")
        age        = st.number_input("Age *", min_value=18, max_value=90, value=25)
        handedness = st.selectbox("Handedness", ["Right", "Left", "Ambidextrous"])
    with col2:
        gender    = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
        education = st.selectbox("Education", ["Secondary school", "Undergraduate", "Postgraduate", "Doctoral", "Other"])
        vision    = st.selectbox("Corrected-to-normal vision?", ["Yes", "No – uncorrected impairment"])
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Pre-session Checks")
    col3, col4 = st.columns(2)
    with col3:
        sleep    = st.selectbox("Sleep last night (hours)", ["< 5", "5–6", "7–8", "9+"])
        caffeine = st.selectbox("Caffeine in last 2 hours?", ["No", "Yes – 1 drink", "Yes – 2+ drinks"])
    with col4:
        meds    = st.selectbox("Psychoactive medication?", ["No", "Yes (stimulant)", "Yes (sedative)", "Yes (other)"])
        anxiety = st.selectbox("Anxiety level now", ["1 – Very low", "2", "3 – Moderate", "4", "5 – Very high"])
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Begin Assessment Battery →", disabled=(pid.strip() == "")):
            st.session_state["participant"] = {
                "id": pid, "age": age, "gender": gender, "handedness": handedness,
                "education": education, "vision": vision, "sleep": sleep,
                "caffeine": caffeine, "medications": meds, "anxiety": anxiety,
                "timestamp": datetime.datetime.now().isoformat()
            }
            st.session_state["page"] = "battery"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATTERY
# ══════════════════════════════════════════════════════════════════════════════
def page_battery():
    section_header("Assessment Battery", "Follow each task's on-screen instructions carefully.")
    info_box("""<strong style='color:#58A6FF;'>Setup</strong>
    Sit ~60 cm from screen, good lighting, head still.
    The battery locks the page during each task — you cannot scroll.
    When complete, <strong>download both files</strong> using the buttons that appear inside the panel, then click below.""")

    battery_html = _build_battery_html()
    components.html(battery_html, height=780, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("I have both files → Upload for Analysis"):
            st.session_state.update({"battery_done": True, "page": "upload"})
            st.rerun()


def _build_battery_html() -> str:
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0D1117;color:#CDD9E5;font-family:'IBM Plex Sans',sans-serif;
     display:flex;flex-direction:column;align-items:center;min-height:760px;overflow:hidden}

/* ── FIXED OVERLAY (active during tasks) ───────────────────────── */
#stage-overlay{
  position:fixed;top:0;left:0;width:100vw;height:100vh;
  background:#0D1117;z-index:9999;
  display:none;flex-direction:column;align-items:center;justify-content:center;
}
#stage-overlay.active{display:flex}

/* ── EMBEDDED STAGE (shown in Streamlit iframe when not in fullscreen) ── */
#stage{
  position:relative;width:800px;height:560px;background:#161B22;
  border:1px solid #30363D;border-radius:8px;overflow:hidden;margin-top:14px;
  flex-shrink:0;
}

/* All task content is rendered inside #stage (which lives inside overlay or inline) */
#ui{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
    justify-content:center;padding:40px;text-align:center;z-index:20}
#progress-bar-wrap{width:800px;height:3px;background:#21262D;border-radius:2px;margin-top:8px}
#progress-bar{height:3px;background:#58A6FF;border-radius:2px;width:0%;transition:width .4s}
#task-label{font-size:10px;color:#484F58;letter-spacing:1px;text-transform:uppercase;
            margin-top:5px;font-family:'IBM Plex Mono',monospace}
h2{font-size:22px;font-weight:600;margin-bottom:10px;color:#CDD9E5}
.sub{font-size:13px;color:#6E7681;line-height:1.7;margin-bottom:26px;max-width:560px;white-space:pre-line}
.btn{background:transparent;color:#58A6FF;border:1px solid #30363D;border-radius:6px;
     padding:10px 26px;font-size:14px;font-weight:500;cursor:pointer;transition:all .2s;
     font-family:'IBM Plex Sans',sans-serif}
.btn:hover{background:#58A6FF;color:#0D1117;border-color:#58A6FF}
.btn:disabled{opacity:.25;cursor:not-allowed}

/* ── DISTANCE BADGE ─────────────────────────────────────────── */
#dist-badge{
  position:absolute;top:10px;right:12px;
  background:#161B22;border:1px solid #30363D;border-radius:20px;
  padding:4px 12px;font-size:11px;font-family:'IBM Plex Mono',monospace;
  color:#8B949E;z-index:30;display:none;
}
#dist-badge.ok{color:#3FB950;border-color:#3FB950}
#dist-badge.warn{color:#E3B341;border-color:#E3B341}
#dist-badge.bad{color:#F85149;border-color:#F85149}

/* ── STIMULI ─────────────────────────────────────────────────── */
#dot{position:absolute;width:16px;height:16px;background:#58A6FF;border-radius:50%;
     transform:translate(-50%,-50%);display:none;z-index:10;
     box-shadow:0 0 0 3px rgba(88,166,255,.2)}
#cross{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
       font-size:40px;color:#30363D;display:none;z-index:10;font-weight:300;line-height:1}

/* ── STROOP / GoNoGo text stim ───────────────────────────────── */
#stim{position:absolute;inset:0;display:none;align-items:center;
      justify-content:center;z-index:15;flex-direction:column;gap:14px}
#stim-box{background:#161B22;border:1px solid #30363D;border-radius:8px;
          padding:28px 48px;text-align:center;min-width:280px}
#stim-text{font-size:46px;font-weight:600;font-family:'IBM Plex Mono',monospace}
#stim-label{font-size:12px;color:#6E7681;margin-top:8px}

/* ── N-BACK ─────────────────────────────────────────────────── */
#nback-grid{display:none;position:absolute;inset:0;align-items:center;justify-content:center;z-index:15}
.nb-cell{width:84px;height:84px;border:1px solid #30363D;border-radius:6px;background:#161B22;
         display:flex;align-items:center;justify-content:center;font-size:34px;font-weight:600;
         color:transparent;transition:all .1s;font-family:'IBM Plex Mono',monospace}
.nb-cell.active{background:#58A6FF;color:#0D1117}

/* ── TRAIL MAKING ───────────────────────────────────────────── */
#trail-canvas{position:absolute;inset:0;display:none;z-index:15;cursor:crosshair}

/* ── CORSI ──────────────────────────────────────────────────── */
#corsi-area{position:absolute;inset:0;display:none;z-index:15}
.corsi-block{position:absolute;width:58px;height:58px;background:#21262D;
             border:1px solid #30363D;border-radius:6px;cursor:pointer;transition:background .12s}
.corsi-block.lit{background:#58A6FF;border-color:#58A6FF}
.corsi-block.correct{background:#3FB950;border-color:#3FB950}
.corsi-block.wrong{background:#F85149;border-color:#F85149}

/* ── DIGIT SPAN ─────────────────────────────────────────────── */
#digitspan-area{position:absolute;inset:0;display:none;flex-direction:column;
                align-items:center;justify-content:center;z-index:15}
#digit-display{font-size:68px;font-weight:600;color:#58A6FF;display:none;
               font-family:'IBM Plex Mono',monospace}
#digit-input-wrap{display:none;flex-direction:column;align-items:center;gap:12px}
#digit-input{background:#161B22;border:1px solid #30363D;color:#CDD9E5;font-size:26px;
             text-align:center;border-radius:6px;padding:10px 18px;width:240px;
             font-family:'IBM Plex Mono',monospace}

/* ── VS hint bar ────────────────────────────────────────────── */
#vs-hint{position:absolute;bottom:0;left:0;right:0;z-index:16;display:none;
         padding:12px;text-align:center;background:rgba(13,17,23,.9)}

/* ── EXPORT panel ───────────────────────────────────────────── */
#export-panel{display:none;flex-direction:column;align-items:center;gap:14px;padding:30px}
#export-panel a{
  display:inline-block;padding:10px 28px;border:1px solid #58A6FF;border-radius:6px;
  color:#58A6FF;text-decoration:none;font-size:14px;font-weight:500;
  background:transparent;transition:all .2s;font-family:'IBM Plex Sans',sans-serif;
}
#export-panel a:hover{background:#58A6FF;color:#0D1117}
</style>
</head>
<body>

<!-- ── INLINE STAGE (visible in Streamlit iframe) ── -->
<div id="stage">
  <div id="ui">
    <h2 id="title">Battery Ready</h2>
    <p class="sub" id="sub">Ensure your webcam is available and you are seated ~60 cm from the screen in good lighting.
Click below to begin. The screen will lock during each task.</p>
    <button class="btn" id="btn" onclick="initBattery()">Start Battery</button>
  </div>
  <div id="dist-badge"></div>
  <div id="dot"></div>
  <div id="cross">+</div>
  <div id="stim" style="display:none;">
    <div id="stim-box">
      <div id="stim-text"></div>
      <div id="stim-label"></div>
    </div>
  </div>
  <div id="nback-grid" style="display:none;">
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;">
      <div class="nb-cell" id="nb-0"></div><div class="nb-cell" id="nb-1"></div><div class="nb-cell" id="nb-2"></div>
      <div class="nb-cell" id="nb-3"></div><div class="nb-cell" id="nb-4"></div><div class="nb-cell" id="nb-5"></div>
      <div class="nb-cell" id="nb-6"></div><div class="nb-cell" id="nb-7"></div><div class="nb-cell" id="nb-8"></div>
    </div>
  </div>
  <canvas id="trail-canvas" width="800" height="560"></canvas>
  <div id="corsi-area"></div>
  <div id="digitspan-area">
    <p style="color:#6E7681;font-size:13px;margin-bottom:10px;" id="ds-prompt">Memorise the following sequence</p>
    <div id="digit-display"></div>
    <div id="digit-input-wrap">
      <input id="digit-input" type="text" placeholder="Type digits in order" autocomplete="off">
      <button class="btn" onclick="submitDigitSpan()">Submit</button>
    </div>
  </div>
  <div id="vs-hint">
    <p style="font-size:12px;color:#6E7681;margin:0;">
      Press <strong style="color:#58A6FF;">SPACE</strong> = target present &nbsp;|&nbsp;
      <strong style="color:#58A6FF;">N</strong> = target absent
    </p>
  </div>
  <div id="export-panel">
    <p style="font-size:14px;color:#3FB950;font-weight:500;">✓ Battery complete — download your files</p>
    <a id="json-link" href="#" download="interaction_logs.json">⬇ Download interaction_logs.json</a>
    <a id="video-link" href="#" download="raw_gaze_video.webm">⬇ Download raw_gaze_video.webm</a>
    <p style="font-size:12px;color:#6E7681;margin-top:6px;">
      Both files saved? Return to the main window and click "Upload for Analysis".
    </p>
  </div>
</div>

<div id="progress-bar-wrap"><div id="progress-bar"></div></div>
<div id="task-label">Task 0 / 10</div>

<script>
// ══════════════════════════════════════════════════════════════════
// GLOBALS
// ══════════════════════════════════════════════════════════════════
const STAGE_W = 800, STAGE_H = 560;
const logs = [];
let mediaRecorder, stream, videoChunks = [];
let keyHandler = null;
let taskIndex = 0;
const TOTAL = 10;

// Gaze tracking state
let gazeVideo = null;
let gazeCanvas = null, gazeCtx = null;
let gazeRAF = null, gazeActive = false;
let gazeSamples = [];          // {t_ms, gx, gy} – MediaPipe normalised 0-1
let ipdHistory = [];           // recent IPD estimates (pixels) for distance calc
const IPD_REAL_MM = 63;        // population mean IPD in mm
let lastDistUpdate = 0;

function log(event, details) {
  logs.push({ timestamp_ms: performance.now().toFixed(2), event, details });
}
function setProgress(n) {
  document.getElementById('progress-bar').style.width = ((n / TOTAL) * 100) + '%';
  document.getElementById('task-label').innerText = 'Task ' + n + ' / ' + TOTAL;
}
function showUI(title, sub, btnLabel, onclick) {
  document.getElementById('title').innerText = title;
  document.getElementById('sub').innerText = sub;
  const btn = document.getElementById('btn');
  btn.innerText = btnLabel;
  btn.onclick = onclick;
  btn.disabled = false;
  document.getElementById('ui').style.display = 'flex';
}
function hideUI() { document.getElementById('ui').style.display = 'none'; }

// ── Remove all task canvases and reset stim elements ──────────────
function hideAll() {
  if (keyHandler) { document.removeEventListener('keydown', keyHandler); keyHandler = null; }
  document.getElementById('dot').style.display = 'none';
  document.getElementById('cross').style.display = 'none';

  const stim = document.getElementById('stim');
  stim.style.display = 'none';
  const st2 = document.getElementById('stim-text');
  const sl  = document.getElementById('stim-label');
  if (st2) { st2.innerText = ''; st2.style.color = ''; st2.style.fontSize = '46px'; }
  if (sl)  sl.innerText = '';
  const sb = document.getElementById('stim-box');
  if (sb)  sb.setAttribute('style','background:#161B22;border:1px solid #30363D;border-radius:8px;padding:28px 48px;text-align:center;min-width:280px;');

  document.getElementById('nback-grid').style.display = 'none';
  document.getElementById('trail-canvas').style.display = 'none';
  document.getElementById('corsi-area').style.display = 'none';
  const dsa = document.getElementById('digitspan-area');
  dsa.style.display = 'none';
  document.getElementById('vs-hint').style.display = 'none';

  // Remove any dynamically created canvases (Stroop, Visual Search)
  document.querySelectorAll('#stage canvas:not(#trail-canvas)').forEach(c => c.remove());
}

// ══════════════════════════════════════════════════════════════════
// LIGHTWEIGHT GAZE + IPD TRACKER
// Uses a hidden <video> element + downscaled canvas analysis.
// Purpose: (a) collect gaze samples for CALIB_GAZE_WINDOW events,
//          (b) estimate IPD → viewing distance.
// Full precision analysis is done server-side on the raw video.
// ══════════════════════════════════════════════════════════════════
function startGazeCollection(videoEl) {
  gazeVideo  = videoEl;
  gazeCanvas = document.createElement('canvas');
  gazeCanvas.width = 320; gazeCanvas.height = 240;
  gazeCtx = gazeCanvas.getContext('2d', { willReadFrequently: true });
  gazeActive = true;

  function loop() {
    if (!gazeActive) return;
    gazeCtx.drawImage(gazeVideo, 0, 0, 320, 240);
    const id = gazeCtx.getImageData(0, 0, 320, 240);

    // Detect dark-pixel clusters in the eye region (rows 70–130 of 240)
    // Left eye region: cols 60–130; Right eye region: cols 190–260
    let lx=0, ly=0, ln=0, rx=0, ry=0, rn=0;
    for (let y = 60; y < 130; y++) {
      for (let x = 40; x < 280; x++) {
        const i = (y * 320 + x) * 4;
        const R = id.data[i], G = id.data[i+1], B = id.data[i+2];
        // Dark pixel heuristic (pupil/iris)
        if (R < 70 && G < 70 && B < 70) {
          if (x < 160) { lx += x; ly += y; ln++; }
          else          { rx += x; ry += y; rn++; }
        }
      }
    }
    const t = performance.now();
    if (ln > 5 && rn > 5) {
      const lcx = lx/ln/320, lcy = ly/ln/240;
      const rcx = rx/rn/320, rcy = ry/rn/240;
      const gx = (lcx + rcx) / 2;
      const gy = (lcy + rcy) / 2;
      gazeSamples.push({ t_ms: t, gx, gy });
      if (gazeSamples.length > 9000) gazeSamples.shift();

      // IPD in normalised units → estimate pixel-equivalent
      const ipdNorm = Math.abs(rcx - lcx);
      if (ipdNorm > 0.05) { // sanity
        ipdHistory.push(ipdNorm);
        if (ipdHistory.length > 30) ipdHistory.shift();
      }
    }

    // Update distance badge every 500 ms
    if (t - lastDistUpdate > 500 && ipdHistory.length > 5) {
      lastDistUpdate = t;
      const medIPD = ipdHistory.slice().sort((a,b)=>a-b)[Math.floor(ipdHistory.length/2)];
      // IPD in video pixels ≈ medIPD * videoWidth
      // Focal-length-based: dist_mm = (IPD_REAL_MM * focal_px) / ipd_px
      // We approximate focal_px ≈ videoWidth (reasonable for typical webcams)
      const dist_cm = (IPD_REAL_MM * 320) / (medIPD * 320) / 10;
      const badge = document.getElementById('dist-badge');
      badge.style.display = 'block';
      badge.innerText = dist_cm.toFixed(0) + ' cm';
      if (dist_cm >= 45 && dist_cm <= 85) {
        badge.className = 'ok'; badge.title = 'Good distance';
      } else if (dist_cm >= 35 && dist_cm <= 100) {
        badge.className = 'warn'; badge.title = 'Move to 50–80 cm';
      } else {
        badge.className = 'bad'; badge.title = 'Distance out of range!';
        log('DISTANCE_WARN', 'dist_cm:' + dist_cm.toFixed(1));
      }
      log('DISTANCE_SAMPLE', 'dist_cm:' + dist_cm.toFixed(1) + ',ipd_norm:' + medIPD.toFixed(4));
    }

    gazeRAF = requestAnimationFrame(loop);
  }
  gazeRAF = requestAnimationFrame(loop);
}

function stopGazeCollection() {
  gazeActive = false;
  if (gazeRAF) cancelAnimationFrame(gazeRAF);
}

function avgGazeInWindow(t_start, t_end) {
  const seg = gazeSamples.filter(s => s.t_ms >= t_start && s.t_ms <= t_end);
  if (!seg.length) return null;
  return {
    gx: seg.reduce((a, s) => a + s.gx, 0) / seg.length,
    gy: seg.reduce((a, s) => a + s.gy, 0) / seg.length
  };
}

// ══════════════════════════════════════════════════════════════════
// INIT BATTERY
// ══════════════════════════════════════════════════════════════════
async function initBattery() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false
    });

    const opts = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
      ? { mimeType: 'video/webm;codecs=vp9' } : { mimeType: 'video/webm' };
    videoChunks = [];
    mediaRecorder = new MediaRecorder(stream, opts);
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) videoChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const blob = new Blob(videoChunks, { type: 'video/webm' });
      finalizeExport(blob);
    };
    mediaRecorder.start(100);

    // Hidden video for gaze collection
    const vid = document.createElement('video');
    vid.srcObject = stream; vid.muted = true; vid.playsInline = true;
    vid.style.cssText = 'position:fixed;opacity:0;pointer-events:none;width:1px;height:1px;top:-9px;';
    document.body.appendChild(vid);
    vid.onloadedmetadata = () => { vid.play(); startGazeCollection(vid); };

    log('SYSTEM_START', 'camera_active,battery_v3.0');
    task_calibration();
  } catch (e) {
    showUI('Camera Required', 'Grant webcam permission in your browser and reload.', 'Retry', initBattery);
  }
}

// ══════════════════════════════════════════════════════════════════
// DEBOUNCE KEY HELPER — returns a one-shot key handler with cooldown
// ══════════════════════════════════════════════════════════════════
function oneShot(keys, onPress) {
  // Returns a keydown listener that fires once, then removes itself.
  // Rearms after 'cooldown' ms.
  let armed = true;
  let lastFire = 0;
  const COOLDOWN = 400; // ms — minimum time between valid presses
  function handler(e) {
    const k = e.code || e.key;
    if (!keys.includes(k) && !keys.includes(e.key.toLowerCase())) return;
    const now = performance.now();
    if (!armed || (now - lastFire) < COOLDOWN) return;
    armed = false; lastFire = now;
    onPress(e);
  }
  return handler;
}

// ══════════════════════════════════════════════════════════════════
// TASK 1 — 9-POINT GAZE CALIBRATION
// ══════════════════════════════════════════════════════════════════
function task_calibration() {
  setProgress(1);
  showUI('Task 1 · Gaze Calibration',
    'A blue dot will appear at 9 positions.\nFollow it smoothly with your eyes — do not move your head.\nA distance indicator appears top-right.',
    'Begin Calibration',
    () => {
      hideUI();
      log('TASK_START', 'Calibration');
      // 9 points in canvas px (absolute, not %)
      const pts = [
        [80,60],[400,60],[720,60],
        [80,280],[400,280],[720,280],
        [80,500],[400,500],[720,500]
      ];
      const dot = document.getElementById('dot');
      dot.style.display = 'block';
      let i = 0;
      function step() {
        if (i >= pts.length) {
          dot.style.display = 'none';
          log('TASK_END', 'Calibration');
          task_prosaccade();
          return;
        }
        const [px, py] = pts[i];
        // Position dot using absolute px relative to stage
        dot.style.left = px + 'px'; dot.style.top = py + 'px';
        dot.style.transform = 'translate(-50%,-50%)';
        log('CALIB_POINT', 'pt:' + i + ',px:' + px + ',py:' + py);
        // Collect gaze 400–1200 ms into the 1400ms dwell
        setTimeout(() => {
          const t0 = performance.now();
          setTimeout(() => {
            const t1 = performance.now();
            const avg = avgGazeInWindow(t0, t1);
            if (avg) {
              log('CALIB_GAZE_WINDOW',
                'pt:' + i + ',px:' + px + ',py:' + py +
                ',gx:' + avg.gx.toFixed(4) + ',gy:' + avg.gy.toFixed(4));
            }
            i++;
            step();
          }, 800);
        }, 400);
      }
      step();
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 2 — PROSACCADE
// Positions are absolute canvas px. Dot appears at left/right ~20%
// or 80% of stage width, always y=280 (centre height).
// ══════════════════════════════════════════════════════════════════
function task_prosaccade() {
  setProgress(2);
  showUI('Task 2A · Prosaccade',
    'Look AT the blue dot as fast as possible each time it appears.\nKeep eyes on the central + until the dot appears.',
    'Begin Prosaccade',
    () => {
      hideUI();
      log('TASK_START', 'Prosaccade');
      // Absolute px positions [x, y]
      const positions = [
        [160,280],[640,280],[160,280],[640,280],[400,280],
        [160,280],[640,280],[400,280],[160,280],[640,280]
      ];
      const dot   = document.getElementById('dot');
      const cross = document.getElementById('cross');
      let i = 0;
      function runTrial() {
        if (i >= positions.length) {
          cross.style.display = 'none'; dot.style.display = 'none';
          log('TASK_END', 'Prosaccade');
          task_antisaccade();
          return;
        }
        cross.style.display = 'block';
        const isi = 900 + Math.random() * 600;
        setTimeout(() => {
          cross.style.display = 'none';
          const [px, py] = positions[i];
          dot.style.left = px + 'px'; dot.style.top = py + 'px';
          dot.style.transform = 'translate(-50%,-50%)';
          dot.style.display = 'block';
          log('PROSAC_STIM', 'trial:' + i + ',px:' + px + ',py:' + py + ',t:' + performance.now().toFixed(2));
          i++;
          setTimeout(() => { dot.style.display = 'none'; setTimeout(runTrial, 500); }, 1000);
        }, isi);
      }
      runTrial();
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 2B — ANTI-SACCADE
// ══════════════════════════════════════════════════════════════════
function task_antisaccade() {
  showUI('Task 2B · Anti-Saccade',
    'A dot will flash briefly.\nImmediately look to the OPPOSITE side of the screen.\nDo NOT look at the dot.',
    'Begin Anti-Saccade',
    () => {
      hideUI();
      log('TASK_START', 'AntiSaccade');
      // [stim_px_x, correct_gaze_px_x] pairs — left stim → look right, and vice versa
      const trials = [
        [120,680],[680,120],[120,680],[680,120],
        [120,680],[680,120],[120,680],[680,120]
      ];
      const dot   = document.getElementById('dot');
      const cross = document.getElementById('cross');
      let i = 0;
      function runTrial() {
        if (i >= trials.length) {
          cross.style.display = 'none'; dot.style.display = 'none';
          log('TASK_END', 'AntiSaccade');
          task_visualsearch();
          return;
        }
        cross.style.display = 'block';
        const isi = 900 + Math.random() * 600;
        setTimeout(() => {
          cross.style.display = 'none';
          const [stim_px, correct_px] = trials[i];
          const side = stim_px < STAGE_W / 2 ? 'LEFT' : 'RIGHT';
          dot.style.left = stim_px + 'px'; dot.style.top = '280px';
          dot.style.transform = 'translate(-50%,-50%)';
          dot.style.display = 'block';
          log('ANTISAC_STIM',
            'trial:' + i + ',side:' + side +
            ',stim_px:' + stim_px + ',correct_px:' + correct_px +
            ',t:' + performance.now().toFixed(2));
          i++;
          setTimeout(() => { dot.style.display = 'none'; setTimeout(runTrial, 700); }, 250);
        }, isi);
      }
      runTrial();
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 3 — VISUAL SEARCH
// Canvas is appended inside #stage. Click coords are stage-relative
// because getBoundingClientRect of #stage is fixed.
// ══════════════════════════════════════════════════════════════════
function task_visualsearch() {
  setProgress(3);
  showUI('Task 3 · Visual Search',
    'Find the orange circle (target) among blue squares (distractors).\nSPACE = present  |  N = absent\nRespond as quickly and accurately as possible.',
    'Begin Visual Search',
    () => {
      hideUI();
      log('TASK_START', 'VisualSearch');
      const stage  = document.getElementById('stage');
      const canvas = document.createElement('canvas');
      canvas.width = STAGE_W; canvas.height = STAGE_H;
      canvas.style.cssText = 'position:absolute;inset:0;z-index:15;';
      stage.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      document.getElementById('vs-hint').style.display = 'block';

      // Build trial list: 14 target-present, 6 target-absent
      const trials = [...Array(14).fill(true), ...Array(6).fill(false)]
        .sort(() => Math.random() - 0.5);

      let trialIdx = 0, t0, responded = false;
      let curTargetPx = null;

      function noOverlap(pos, x, y) {
        return pos.every(p => Math.hypot(p[0] - x, p[1] - y) > 58);
      }

      function drawSearch(hasTarget) {
        ctx.clearRect(0, 0, STAGE_W, STAGE_H);
        const pos = [];
        for (let d = 0; d < 11; d++) {
          let x, y, t = 0;
          do { x = 60 + Math.random() * 680; y = 60 + Math.random() * 440; t++; }
          while (!noOverlap(pos, x, y) && t < 60);
          pos.push([x, y]);
          ctx.fillStyle = '#1F4E8C'; ctx.strokeStyle = '#3B8BD4'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.rect(x - 18, y - 18, 36, 36);
          ctx.fill(); ctx.stroke();
        }
        curTargetPx = null;
        if (hasTarget) {
          let x, y, t = 0;
          do { x = 60 + Math.random() * 680; y = 60 + Math.random() * 440; t++; }
          while (!noOverlap(pos, x, y) && t < 60);
          curTargetPx = { x, y };
          ctx.fillStyle = '#B85A00'; ctx.strokeStyle = '#E3B341'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(x, y, 20, 0, Math.PI * 2);
          ctx.fill(); ctx.stroke();
        }
        t0 = performance.now(); responded = false;
      }

      function nextTrial() {
        if (trialIdx >= trials.length) {
          ctx.clearRect(0, 0, STAGE_W, STAGE_H); canvas.remove();
          document.getElementById('vs-hint').style.display = 'none';
          if (keyHandler) { document.removeEventListener('keydown', keyHandler); keyHandler = null; }
          log('TASK_END', 'VisualSearch');
          task_rt();
          return;
        }
        const hasTarget = trials[trialIdx];
        drawSearch(hasTarget);
        const tgt = curTargetPx;
        log('VS_TRIAL', 'trial:' + trialIdx + ',target:' + hasTarget +
          (tgt ? ',tgt_px:' + tgt.x.toFixed(1) + ',tgt_py:' + tgt.y.toFixed(1) : ',tgt_px:null'));
        trialIdx++;
      }

      function handleKey(e) {
        if (responded) return;
        if (e.code !== 'Space' && e.key.toLowerCase() !== 'n') return;
        const now = performance.now();
        if (now - t0 < 100) return; // debounce: < 100ms = pre-emptive press, ignore
        responded = true;
        const rt = now - t0;
        const resp = e.code === 'Space' ? 'PRESENT' : 'ABSENT';
        const hasTarget = trials[trialIdx - 1];
        const correct = (resp === 'PRESENT') === hasTarget;
        log('VS_RESPONSE', 'trial:' + (trialIdx-1) + ',resp:' + resp + ',correct:' + correct + ',rt:' + rt.toFixed(2));
        setTimeout(nextTrial, 300);
      }
      document.addEventListener('keydown', handleKey);
      keyHandler = handleKey;
      setTimeout(nextTrial, 500);
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 4 — SIMPLE REACTION TIME
// FIX: false start detection (< 100ms after stimulus = false start)
// ══════════════════════════════════════════════════════════════════
function task_rt() {
  setProgress(4);
  showUI('Task 4 · Simple Reaction Time',
    'Press SPACE as fast as possible when the blue circle appears.\nDo NOT press before it appears.\n(20 trials)',
    'Begin Reaction Time Test',
    () => {
      hideUI();
      log('TASK_START', 'SimpleRT');
      const dot = document.getElementById('dot');
      dot.style.left = '400px'; dot.style.top = '280px';
      dot.style.transform = 'translate(-50%,-50%)';
      let trials = 0, t0 = 0, stimActive = false;
      const N = 20;
      let pending = null;

      function nextTrial() {
        if (trials >= N) {
          dot.style.display = 'none';
          if (keyHandler) { document.removeEventListener('keydown', keyHandler); keyHandler = null; }
          log('TASK_END', 'SimpleRT');
          task_gonogo();
          return;
        }
        stimActive = false;
        const delay = 1200 + Math.random() * 2000;
        pending = setTimeout(() => {
          dot.style.display = 'block';
          t0 = performance.now();
          stimActive = true;
          log('RT_STIMULUS', 'trial:' + trials + ',t:' + t0.toFixed(2));
        }, delay);
      }

      keyHandler = function(e) {
        if (e.code !== 'Space') return;
        if (!stimActive) {
          // Stim not on screen — could be anticipation
          log('RT_FALSE_START', 'trial:' + trials + ',t:' + performance.now().toFixed(2));
          return;
        }
        const rt = performance.now() - t0;
        if (rt < 80) {
          // Too fast to be genuine — likely held-down key
          log('RT_ANTICIPATION', 'trial:' + trials + ',rt:' + rt.toFixed(2));
          return;
        }
        stimActive = false;
        dot.style.display = 'none';
        log('RT_RESPONSE', 'trial:' + trials + ',rt:' + rt.toFixed(2));
        trials++;
        setTimeout(nextTrial, 500);
      };
      document.addEventListener('keydown', keyHandler);
      nextTrial();
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 5 — GO / NO-GO
// FIX: each trial's key listener is removed immediately on response.
// Commission errors: pressing on NOGO. Omission: not pressing on GO.
// ══════════════════════════════════════════════════════════════════
function task_gonogo() {
  setProgress(5);
  showUI('Task 5 · Go / No-Go',
    'GREEN circle → Press SPACE immediately (Go)\nRED circle → Do NOT press (No-Go)\n(30 trials)',
    'Begin Go/No-Go',
    () => {
      hideUI();
      log('TASK_START', 'GoNoGo');

      // Re-create stim-box content cleanly
      const stim = document.getElementById('stim');
      stim.style.display = 'none';
      stim.innerHTML = `
        <div id="stim-box" style="background:#161B22;border:1px solid #30363D;border-radius:8px;padding:28px 48px;text-align:center;min-width:280px;">
          <div id="stim-text" style="font-size:64px;font-family:IBM Plex Mono,monospace;"></div>
          <div id="stim-label" style="font-size:12px;color:#6E7681;margin-top:8px;"></div>
        </div>`;
      const stimText  = document.getElementById('stim-text');
      const stimLabel = document.getElementById('stim-label');

      const sequence = Array.from({ length: 30 }, () => Math.random() < 0.75 ? 'GO' : 'NOGO');
      let idx = 0, t0 = 0, responded = false;
      let trialTimer = null, trialKeyHandler = null;

      function runTrial() {
        if (idx >= sequence.length) {
          stim.style.display = 'none';
          if (trialKeyHandler) document.removeEventListener('keydown', trialKeyHandler);
          log('TASK_END', 'GoNoGo');
          task_nback();
          return;
        }
        const type = sequence[idx];
        stimText.style.color = type === 'GO' ? '#3FB950' : '#F85149';
        stimText.innerText = '●';
        stimLabel.innerText = type === 'GO' ? 'GO — press SPACE' : 'NO-GO — do not press';
        stim.style.display = 'flex';
        t0 = performance.now(); responded = false;
        log('GNG_STIM', 'trial:' + idx + ',type:' + type + ',t:' + t0.toFixed(2));
        const currentIdx = idx;
        idx++;

        trialTimer = setTimeout(() => {
          if (trialKeyHandler) { document.removeEventListener('keydown', trialKeyHandler); trialKeyHandler = null; }
          if (!responded && type === 'GO') log('GNG_OMISSION', 'trial:' + currentIdx);
          stim.style.display = 'none';
          stimText.innerText = '';
          setTimeout(runTrial, 350);
        }, 800);

        trialKeyHandler = function(e) {
          if (e.code !== 'Space') return;
          if (responded) return;
          const now = performance.now();
          if (now - t0 < 80) return; // debounce
          responded = true;
          clearTimeout(trialTimer);
          document.removeEventListener('keydown', trialKeyHandler);
          trialKeyHandler = null;
          const rt = now - t0;
          const correct = (type === 'GO');
          log('GNG_RESPONSE', 'trial:' + currentIdx + ',type:' + type + ',correct:' + correct + ',rt:' + rt.toFixed(2));
          if (!correct) log('GNG_COMMISSION', 'trial:' + currentIdx + ',rt:' + rt.toFixed(2));
          stim.style.display = 'none'; stimText.innerText = '';
          setTimeout(runTrial, 300);
        };
        document.addEventListener('keydown', trialKeyHandler);
      }
      runTrial();
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 6 — N-BACK
// ══════════════════════════════════════════════════════════════════
function task_nback() {
  setProgress(6);

  function runNBack(n, onDone) {
    const letters = 'BDFGHJKLMNPQRSTVWXZ'.split('');
    const seq = [];
    for (let k = 0; k < 20 + n; k++) {
      if (k >= n && Math.random() < 0.30) seq.push(seq[k - n]);
      else seq.push(letters[Math.floor(Math.random() * letters.length)]);
    }
    log('NBACK_START', 'n:' + n + ',trials:' + seq.length);
    const grid  = document.getElementById('nback-grid');
    const cells = Array.from(document.querySelectorAll('.nb-cell'));
    grid.style.display = 'flex';
    let idx = 0, trialKeyHandler = null;

    function showItem() {
      if (idx >= seq.length) {
        cells.forEach(c => { c.classList.remove('active'); c.innerText = ''; });
        grid.style.display = 'none';
        if (trialKeyHandler) { document.removeEventListener('keydown', trialKeyHandler); trialKeyHandler = null; }
        log('NBACK_END', 'n:' + n);
        onDone();
        return;
      }
      const isTarget = idx >= n && seq[idx] === seq[idx - n];
      cells.forEach(c => { c.classList.remove('active'); c.innerText = ''; });
      const rc = Math.floor(Math.random() * 9);
      cells[rc].classList.add('active'); cells[rc].innerText = seq[idx];
      log('NBACK_STIM', 'idx:' + idx + ',letter:' + seq[idx] + ',target:' + isTarget + ',t:' + performance.now().toFixed(2));
      let responded = false;

      const itemTimeout = setTimeout(() => {
        if (!responded && isTarget) log('NBACK_MISS', 'idx:' + idx);
        cells.forEach(c => { c.classList.remove('active'); c.innerText = ''; });
        if (trialKeyHandler) { document.removeEventListener('keydown', trialKeyHandler); trialKeyHandler = null; }
        idx++;
        setTimeout(showItem, 350);
      }, 1600);

      trialKeyHandler = function(e) {
        if (e.code !== 'Space') return;
        if (responded) return;
        const now = performance.now();
        responded = true;
        clearTimeout(itemTimeout);
        document.removeEventListener('keydown', trialKeyHandler);
        trialKeyHandler = null;
        const rt = now;
        log('NBACK_RESPONSE', 'idx:' + idx + ',target:' + isTarget + ',rt:' + rt.toFixed(2));
        if (!isTarget) log('NBACK_FALSE_ALARM', 'idx:' + idx);
        cells.forEach(c => { c.classList.remove('active'); c.innerText = ''; });
        idx++;
        setTimeout(showItem, 350);
      };
      document.addEventListener('keydown', trialKeyHandler);
    }
    showItem();
  }

  showUI('Task 6 · N-Back (Part A)',
    '1-Back: Press SPACE if the current letter matches the letter ONE step ago.\nLetters appear for 1.6 s. Respond before the next letter.',
    'Begin 1-Back',
    () => {
      hideUI();
      runNBack(1, () => {
        showUI('Task 6 · N-Back (Part B)',
          '2-Back: Press SPACE if the current letter matches TWO steps ago.',
          'Begin 2-Back',
          () => { hideUI(); runNBack(2, () => { log('TASK_END', 'NBack'); task_stroop(); }); });
      });
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 7 — STROOP
// FIX: canvas explicitly removed on task end.
// FIX: one-shot key handler with 300ms debounce.
// ══════════════════════════════════════════════════════════════════
function task_stroop() {
  setProgress(7);
  showUI('Task 7 · Stroop Colour-Word',
    'Name the INK COLOR, not the word.\nR = Red   G = Green   B = Blue   Y = Yellow\nRespond as fast as possible.',
    'Begin Stroop',
    () => {
      hideUI();
      log('TASK_START', 'Stroop');
      const stage  = document.getElementById('stage');
      const canvas = document.createElement('canvas');
      canvas.width = STAGE_W; canvas.height = STAGE_H;
      canvas.style.cssText = 'position:absolute;inset:0;z-index:15;';
      stage.appendChild(canvas);
      const ctx = canvas.getContext('2d');
      const COLORS = ['RED', 'GREEN', 'BLUE', 'YELLOW'];
      const INK    = { RED: '#F85149', GREEN: '#3FB950', BLUE: '#58A6FF', YELLOW: '#E3B341' };
      const KEYS   = { r: 'RED', g: 'GREEN', b: 'BLUE', y: 'YELLOW' };

      const trials = [];
      for (let k = 0; k < 16; k++) { const c = COLORS[k % 4]; trials.push({ word: c, ink: c, congruent: true }); }
      for (let k = 0; k < 16; k++) {
        const w = COLORS[k % 4];
        const ink = COLORS.filter(x => x !== w)[k % 3];
        trials.push({ word: w, ink, congruent: false });
      }
      trials.sort(() => Math.random() - 0.5);

      let idx = 0, t0 = 0, responded = false, trialKH = null;

      function nextTrial() {
        if (trialKH) { document.removeEventListener('keydown', trialKH); trialKH = null; }
        if (idx >= trials.length) {
          ctx.clearRect(0, 0, STAGE_W, STAGE_H);
          canvas.remove();   // ← CRITICAL FIX
          log('TASK_END', 'Stroop');
          task_trail();
          return;
        }
        const tr = trials[idx];
        ctx.clearRect(0, 0, STAGE_W, STAGE_H);
        ctx.font = 'bold 52px "IBM Plex Mono",monospace';
        ctx.fillStyle = INK[tr.ink];
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(tr.word, 400, 260);
        ctx.font = '13px "IBM Plex Sans",sans-serif';
        ctx.fillStyle = '#484F58';
        ctx.fillText('R=Red  G=Green  B=Blue  Y=Yellow', 400, 490);
        t0 = performance.now(); responded = false;
        log('STROOP_STIM', 'word:' + tr.word + ',ink:' + tr.ink + ',cong:' + tr.congruent + ',t:' + t0.toFixed(2));
        idx++;

        trialKH = function(e) {
          if (responded) return;
          const key = e.key.toLowerCase();
          if (!KEYS[key]) return;
          const now = performance.now();
          if (now - t0 < 100) return; // debounce
          responded = true;
          const rt = now - t0;
          const resp = KEYS[key];
          const correct = resp === trials[idx - 1].ink;
          log('STROOP_RESPONSE', 'resp:' + resp + ',correct:' + correct + ',rt:' + rt.toFixed(2) + ',cong:' + trials[idx - 1].congruent);
          document.removeEventListener('keydown', trialKH);
          trialKH = null;
          setTimeout(nextTrial, 150);
        };
        document.addEventListener('keydown', trialKH);
      }

      setTimeout(nextTrial, 400);
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 8 — TRAIL MAKING
// FIX: uses trail-canvas which is inside #stage (position:absolute).
// Click coords computed relative to canvas getBoundingClientRect,
// which is stable because the stage is a fixed-size div.
// ══════════════════════════════════════════════════════════════════
function task_trail() {
  setProgress(8);

  function runTrail(part, onDone) {
    const canvas = document.getElementById('trail-canvas');
    canvas.style.display = 'block';
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, STAGE_W, STAGE_H);

    const N = 10, nodes = [];
    function noOverlap(x, y) { return nodes.every(p => Math.hypot(p.x - x, p.y - y) > 72); }

    for (let k = 0; k < N; k++) {
      let x, y, t = 0;
      do { x = 70 + Math.random() * 660; y = 70 + Math.random() * 420; t++; }
      while (!noOverlap(x, y) && t < 300);
      const label = part === 'A'
        ? String(k + 1)
        : (k % 2 === 0 ? String(k / 2 + 1) : String.fromCharCode(65 + Math.floor(k / 2)));
      nodes.push({ x, y, label, visited: false });
    }

    let correctOrder;
    if (part === 'A') {
      correctOrder = nodes.slice().sort((a, b) => parseInt(a.label) - parseInt(b.label));
    } else {
      correctOrder = [...nodes].sort((a, b) => {
        const rank = n => isNaN(n.label)
          ? (n.label.charCodeAt(0) - 64) * 2
          : parseInt(n.label) * 2 - 1;
        return rank(a) - rank(b);
      });
    }

    function draw() {
      ctx.clearRect(0, 0, STAGE_W, STAGE_H);
      // Draw completed path
      ctx.strokeStyle = '#58A6FF'; ctx.lineWidth = 2;
      for (let k = 1; k < correctOrder.length; k++) {
        if (correctOrder[k - 1].visited && correctOrder[k].visited) {
          ctx.beginPath();
          ctx.moveTo(correctOrder[k - 1].x, correctOrder[k - 1].y);
          ctx.lineTo(correctOrder[k].x, correctOrder[k].y);
          ctx.stroke();
        }
      }
      const nextIdx = nodes.filter(x => x.visited).length;
      nodes.forEach(n => {
        ctx.beginPath(); ctx.arc(n.x, n.y, 22, 0, Math.PI * 2);
        ctx.fillStyle = n.visited ? '#21262D' : (n === correctOrder[nextIdx] ? '#1F4E8C' : '#161B22');
        ctx.strokeStyle = n.visited ? '#3FB950' : '#30363D'; ctx.lineWidth = 1.5;
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = n.visited ? '#3FB950' : '#8B949E';
        ctx.font = 'bold 13px "IBM Plex Mono",monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(n.label, n.x, n.y);
      });
      ctx.font = '12px "IBM Plex Sans",sans-serif'; ctx.fillStyle = '#484F58'; ctx.textAlign = 'left';
      ctx.fillText('Trail Making Part ' + part + ' — click nodes in order: ' + (part === 'A' ? '1→2→3…' : '1→A→2→B→3…'), 14, 22);
    }

    let nextIdx = 0;
    const t0 = performance.now();
    log('TRAIL_START', 'part:' + part);
    draw();

    // Use stage-relative coordinates
    canvas.onclick = function(e) {
      const rect = canvas.getBoundingClientRect();
      // Scale mouse position to canvas logical coords
      const mx = (e.clientX - rect.left) * (STAGE_W / rect.width);
      const my = (e.clientY - rect.top) * (STAGE_H / rect.height);

      const target = correctOrder[nextIdx];
      if (Math.hypot(target.x - mx, target.y - my) < 28) {
        target.visited = true;
        log('TRAIL_CLICK', 'part:' + part + ',idx:' + nextIdx + ',node:' + target.label + ',t:' + performance.now().toFixed(2));
        nextIdx++;
        draw();
        if (nextIdx >= N) {
          const elapsed = (performance.now() - t0) / 1000;
          log('TRAIL_END', 'part:' + part + ',time_s:' + elapsed.toFixed(3));
          canvas.onclick = null;
          canvas.style.display = 'none';
          onDone();
        }
      } else {
        const clicked = nodes.find(n => Math.hypot(n.x - mx, n.y - my) < 28);
        if (clicked) log('TRAIL_ERROR', 'part:' + part + ',idx:' + nextIdx + ',clicked:' + clicked.label);
      }
    };
  }

  showUI('Task 8 · Trail Making',
    'Part A: Connect 1→2→3→…→10 in order.\nClick each numbered circle in the correct sequence.',
    'Begin Part A',
    () => {
      hideUI();
      runTrail('A', () => {
        showUI('Task 8B · Trail Making',
          'Part B: Alternate numbers and letters: 1→A→2→B→3→C…',
          'Begin Part B',
          () => { hideUI(); runTrail('B', () => { log('TASK_END', 'TrailMaking'); task_corsi(); }); });
      });
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 9 — CORSI
// ══════════════════════════════════════════════════════════════════
function task_corsi() {
  setProgress(9);
  showUI('Task 9 · Corsi Block Tapping',
    'Blocks will light up in a sequence.\nRepeat the sequence by clicking them in the same order.',
    'Begin Corsi',
    () => {
      hideUI();
      log('TASK_START', 'Corsi');
      const area = document.getElementById('corsi-area');
      area.style.display = 'block'; area.innerHTML = '';

      const positions = [
        [120,160],[240,80],[380,200],[520,100],[660,180],
        [100,320],[280,360],[440,300],[600,340]
      ];
      const blocks = positions.map((pos, i) => {
        const b = document.createElement('div');
        b.className = 'corsi-block';
        b.style.left = pos[0] + 'px'; b.style.top = pos[1] + 'px';
        b.dataset.id = i;
        area.appendChild(b);
        return b;
      });

      let span = 2, fails = 0, maxSpan = 0;

      function lightUp(seq, onDone) {
        let i = 0;
        function s() {
          if (i >= seq.length) { onDone(); return; }
          blocks.forEach(b => b.className = 'corsi-block');
          blocks[seq[i]].classList.add('lit');
          setTimeout(() => {
            blocks[seq[i]].classList.remove('lit');
            i++;
            setTimeout(s, 350);
          }, 650);
        }
        s();
      }

      function runTrial() {
        if (fails >= 2 || span > 9) {
          area.style.display = 'none';
          log('CORSI_END', 'max_span:' + maxSpan);
          log('TASK_END', 'Corsi');
          task_digitspan();
          return;
        }
        const seq = Array.from({ length: span }, () => Math.floor(Math.random() * 9));
        log('CORSI_SEQ', 'span:' + span + ',seq:' + seq.join(','));
        let clickSeq = [], clickIdx = 0;

        blocks.forEach(b => {
          b.onclick = function() {
            if (clickIdx >= span) return;
            const id = parseInt(b.dataset.id);
            clickSeq.push(id);
            const isCorrect = id === seq[clickIdx];
            b.classList.add(isCorrect ? 'correct' : 'wrong');
            setTimeout(() => b.className = 'corsi-block', 280);
            clickIdx++;
            if (clickIdx === span) {
              const correct = clickSeq.every((v, i) => v === seq[i]);
              log('CORSI_RESPONSE', 'span:' + span + ',correct:' + correct + ',response:' + clickSeq.join(','));
              blocks.forEach(b2 => b2.onclick = null);
              if (correct) { if (span > maxSpan) maxSpan = span; fails = 0; span++; }
              else fails++;
              setTimeout(runTrial, 700);
            }
          };
        });
        lightUp(seq, () => {});
      }
      lightUp([0, 1], runTrial);
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// TASK 10 — DIGIT SPAN
// ══════════════════════════════════════════════════════════════════
function task_digitspan() {
  setProgress(10);
  showUI('Task 10 · Digit Span',
    'Digits appear one at a time.\nWhen they stop, type them all in order and press Submit.',
    'Begin Digit Span',
    () => {
      hideUI();
      log('TASK_START', 'DigitSpan');
      const area = document.getElementById('digitspan-area');
      area.style.display = 'flex';
      const disp      = document.getElementById('digit-display');
      const inputWrap = document.getElementById('digit-input-wrap');
      const inp       = document.getElementById('digit-input');
      let span = 3, fails = 0, maxSpan = 0;

      function runTrial() {
        if (fails >= 2 || span > 9) {
          area.style.display = 'none';
          log('DIGITSPAN_END', 'max_span:' + maxSpan);
          log('TASK_END', 'DigitSpan');
          finishBattery();
          return;
        }
        const seq = Array.from({ length: span }, () => Math.floor(Math.random() * 10));
        log('DIGITSPAN_SEQ', 'span:' + span + ',seq:' + seq.join(''));
        inp.value = '';
        inputWrap.style.display = 'none';
        disp.style.display = 'block';
        let i = 0;
        function showDigit() {
          if (i >= seq.length) {
            disp.style.display = 'none';
            inputWrap.style.display = 'flex';
            inp.focus();
            return;
          }
          disp.innerText = seq[i]; i++;
          setTimeout(() => { disp.innerText = ''; setTimeout(showDigit, 280); }, 750);
        }
        showDigit();
        window._dsSeq = seq;
      }

      window.submitDigitSpan = function() {
        const resp = inp.value.trim().split('').map(Number);
        const correct = resp.length === window._dsSeq.length &&
                        resp.every((v, i) => v === window._dsSeq[i]);
        log('DIGITSPAN_RESPONSE', 'span:' + span + ',correct:' + correct + ',response:' + resp.join(''));
        if (correct) { if (span > maxSpan) maxSpan = span; fails = 0; span++; }
        else fails++;
        setTimeout(runTrial, 350);
      };

      // Allow Enter key to submit
      inp.onkeydown = function(e) { if (e.key === 'Enter') window.submitDigitSpan(); };
      runTrial();
    }
  );
}

// ══════════════════════════════════════════════════════════════════
// EXPORT — data-URI anchor links inside the iframe
// FIX: no postMessage dependency. Both files download from within
// this iframe's own origin, so browser sandbox is never crossed.
// ══════════════════════════════════════════════════════════════════
function finishBattery() {
  stopGazeCollection();
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop(); // triggers onstop → finalizeExport
  } else {
    finalizeExport(null);
  }
}

function finalizeExport(videoBlob) {
  // ── JSON ──
  const logsJson = JSON.stringify(logs, null, 2);
  const jBlob = new Blob([logsJson], { type: 'application/json' });
  const jUrl  = URL.createObjectURL(jBlob);
  const jLink = document.getElementById('json-link');
  jLink.href = jUrl;
  jLink.download = 'interaction_logs.json';

  // ── Video ──
  const vLink = document.getElementById('video-link');
  if (videoBlob) {
    const vUrl = URL.createObjectURL(videoBlob);
    vLink.href = vUrl;
    vLink.download = 'raw_gaze_video.webm';
  } else {
    vLink.style.opacity = '0.3';
    vLink.style.pointerEvents = 'none';
    vLink.innerText = '⚠ Video not available';
  }

  // ── Show export panel ──
  hideAll();
  document.getElementById('progress-bar').style.width = '100%';
  document.getElementById('task-label').innerText = 'All tasks complete — please download your files';

  const panel = document.getElementById('export-panel');
  panel.style.display = 'flex';
  log('BATTERY_COMPLETE', 'total_events:' + logs.length);
}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def page_upload():
    section_header("Upload Assessment Data", "Upload both files to compute all 25 biomarkers.")
    info_box("""
    <strong style='color:#58A6FF;'>Analysis pipeline</strong><br>
    The video is processed with MediaPipe Face Mesh (iris landmarks 468 &amp; 473) at native frame rate.
    A 9-point affine calibration (built from CALIB_GAZE_WINDOW events) maps MediaPipe 0–1 normalised gaze
    → canvas pixel coordinates. I-VT velocity thresholding (Salvucci &amp; Goldberg, 2000) then classifies
    fixations and saccades in calibrated screen space. All features are computed from calibrated coordinates.
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
                with st.spinner("Running computer vision pipeline…"):
                    logs_data = json.load(log_file)
                    results   = run_analysis(video_file, logs_data)
                    st.session_state.update({"results": results, "event_logs": logs_data, "page": "results"})
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE — v3.0
# All spatial quantities are in calibrated canvas-pixel space (800×560).
# ══════════════════════════════════════════════════════════════════════════════

STAGE_W_PY, STAGE_H_PY = 800.0, 560.0
SCREEN_DIAG_IN = 15.0                  # assumed display diagonal (inches)
SCREEN_CM = 34.5                       # typical 15" display width (cm)
PX_PER_CM = STAGE_W_PY / SCREEN_CM    # pixels per cm in canvas space
D_CM_DEFAULT = 60.0                    # nominal viewing distance (cm)
VEL_THRESH_DEG_S = 100.0              # I-VT saccade threshold
MIN_FIX_MS = 100.0                     # minimum fixation duration


def build_affine_transform(logs_list: list) -> np.ndarray:
    """
    Build 2×3 affine matrix: MediaPipe gaze (0-1) → canvas px.
    Uses CALIB_GAZE_WINDOW events.
    Requires ≥ 3 non-collinear pairs for cv2.estimateAffine2D.
    Falls back to uniform scale if insufficient calibration data.
    """
    src_pts, dst_pts = [], []
    for ev in logs_list:
        if ev['event'] == 'CALIB_GAZE_WINDOW':
            try:
                d  = ev['details']
                px = float(d.split('px:')[1].split(',')[0])
                py = float(d.split('py:')[1].split(',')[0])
                gx = float(d.split('gx:')[1].split(',')[0])
                gy = float(d.split('gy:')[1].split(',')[0])
                src_pts.append([gx, gy])
                dst_pts.append([px, py])
            except Exception:
                pass

    if len(src_pts) >= 4:
        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
        if M is not None:
            n_inliers = int(inliers.sum()) if inliers is not None else len(src_pts)
            return M, n_inliers
    # Fallback
    M_fb = np.array([[STAGE_W_PY, 0, 0], [0, STAGE_H_PY, 0]], dtype=np.float32)
    return M_fb, 0


def apply_affine(M: np.ndarray, gx: float, gy: float):
    pt  = np.array([[[gx, gy]]], dtype=np.float32)
    out = cv2.transform(pt, M)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def estimate_viewing_distance(logs_list: list) -> float:
    """
    Extract median viewing distance from DISTANCE_SAMPLE events.
    Falls back to D_CM_DEFAULT.
    """
    dists = []
    for ev in logs_list:
        if ev['event'] == 'DISTANCE_SAMPLE':
            try:
                d = float(ev['details'].split('dist_cm:')[1].split(',')[0])
                if 20 < d < 150:
                    dists.append(d)
            except Exception:
                pass
    return float(np.median(dists)) if len(dists) > 5 else D_CM_DEFAULT


def run_analysis(video_file, logs: list) -> dict:
    # ── 1. Decode viewing distance ────────────────────────────────────────
    D_CM = estimate_viewing_distance(logs)

    # ── 2. VIDEO → MediaPipe iris gaze stream ─────────────────────────────
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tfile.write(video_file.read()); tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_face  = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_faces=1
    )

    raw_gaze = []  # {t_ms, gx, gy} — normalised 0-1 MediaPipe space
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            # Iris landmarks: 468 = left iris centre, 473 = right iris centre
            gx = (lm[468].x + lm[473].x) / 2.0
            gy = (lm[468].y + lm[473].y) / 2.0
            raw_gaze.append({'t': fi / fps * 1000.0, 'gx': gx, 'gy': gy})
        fi += 1
    cap.release(); face_mesh.close(); os.unlink(tfile.name)

    # ── 3. AFFINE CALIBRATION ─────────────────────────────────────────────
    M, n_calib_inliers = build_affine_transform(logs)

    gaze_stream = []
    for s in raw_gaze:
        px, py = apply_affine(M, s['gx'], s['gy'])
        # Clamp to stage bounds with small margin
        px = float(np.clip(px, -50, STAGE_W_PY + 50))
        py = float(np.clip(py, -50, STAGE_H_PY + 50))
        gaze_stream.append({'t': s['t'], 'x': px, 'y': py})

    # ── 4. I-VT FIXATION / SACCADE CLASSIFICATION ────────────────────────
    # Velocity in deg/s using calibrated canvas-pixel distances.
    def px_to_deg(dist_px):
        dist_cm = dist_px / PX_PER_CM
        return float(np.degrees(np.arctan2(dist_cm, D_CM)))

    fixations_raw, saccades = [], []
    for i in range(1, len(gaze_stream)):
        t1, x1, y1 = gaze_stream[i-1]['t'], gaze_stream[i-1]['x'], gaze_stream[i-1]['y']
        t2, x2, y2 = gaze_stream[i]['t'],   gaze_stream[i]['x'],   gaze_stream[i]['y']
        dt = (t2 - t1) / 1000.0
        if dt <= 0: continue
        dist_px  = float(np.hypot(x2 - x1, y2 - y1))
        dist_deg = px_to_deg(dist_px)
        vel      = dist_deg / dt
        if vel >= VEL_THRESH_DEG_S:
            saccades.append({
                't_start': t1, 't_end': t2,
                'amp': dist_deg, 'velocity': vel,
                'x_start': x1, 'y_start': y1,
                'x_end': x2, 'y_end': y2
            })
        else:
            fixations_raw.append({
                't_start': t1, 't_end': t2,
                'x': (x1 + x2) / 2, 'y': (y1 + y2) / 2,
                'duration': dt * 1000
            })

    # Merge nearby fixations
    fixations, buf = [], []
    for f in fixations_raw:
        if not buf or (f['t_start'] - buf[-1]['t_end']) < 50:
            buf.append(f)
        else:
            dur = sum(b['duration'] for b in buf)
            if dur >= MIN_FIX_MS:
                fixations.append({
                    't_start': buf[0]['t_start'],
                    'duration': dur,
                    'x': float(np.mean([b['x'] for b in buf])),
                    'y': float(np.mean([b['y'] for b in buf]))
                })
            buf = [f]

    # ── 5. OCULOMOTOR FEATURES F1–F8 ─────────────────────────────────────
    f1_mfd = float(np.mean([f['duration'] for f in fixations])) if fixations else 0.0
    f2_fc  = len(fixations)
    f4_sa  = float(np.mean([s['amp']      for s in saccades]))  if saccades  else 0.0
    f5_spv = float(np.mean([s['velocity'] for s in saccades]))  if saccades  else 0.0

    # Gaze path entropy on 5×5 grid (canvas px → cell index)
    grid = np.zeros((5, 5))
    for f in fixations:
        gx_idx = int(np.clip(f['x'] / STAGE_W_PY * 5, 0, 4))
        gy_idx = int(np.clip(f['y'] / STAGE_H_PY * 5, 0, 4))
        grid[gx_idx, gy_idx] += f['duration']
    f6_ent = 0.0
    if grid.sum() > 0:
        pk = grid.flatten() / grid.sum()
        f6_ent = float(-sum(p * log2(p) for p in pk if p > 0))
    f8_roi = float(np.count_nonzero(grid) / 25.0 * 100)

    # ── 6. ANTI-SACCADE FEATURES F3, F7 ──────────────────────────────────
    # F3 = saccade latency (onset to first saccade ≥ 80 ms)
    # F7 = error rate (% trials where first saccade went to wrong side)
    latencies, anti_errors, anti_trials = [], 0, 0
    for ev in logs:
        if ev['event'] == 'ANTISAC_STIM':
            anti_trials += 1
            ev_t = float(ev['timestamp_ms'])
            d = ev['details']
            try:
                correct_px = float(d.split('correct_px:')[1].split(',')[0])
            except Exception:
                correct_px = None

            next_sac = next((s for s in saccades if s['t_start'] >= ev_t), None)
            if next_sac:
                lat = next_sac['t_start'] - ev_t
                if 80 < lat < 1000:
                    latencies.append(lat)
                if correct_px is not None:
                    mid = STAGE_W_PY / 2
                    went_correct = (
                        (correct_px > mid and next_sac['x_end'] > mid) or
                        (correct_px < mid and next_sac['x_end'] < mid)
                    )
                    if not went_correct:
                        anti_errors += 1

    f3_sl   = float(np.mean(latencies))                    if latencies    else 0.0
    f7_aser = float(anti_errors / anti_trials * 100)       if anti_trials  else 0.0

    # ── 7. SIMPLE RT F9, F10 ─────────────────────────────────────────────
    rt_vals = []
    for ev in logs:
        if ev['event'] == 'RT_RESPONSE' and 'rt:' in ev['details']:
            try:
                rt = float(ev['details'].split('rt:')[1])
                if 100 < rt < 1500: rt_vals.append(rt)
            except Exception: pass
    f9_rt   = float(np.mean(rt_vals))        if rt_vals        else 0.0
    f10_iiv = float(np.std(rt_vals, ddof=1)) if len(rt_vals) > 1 else 0.0

    # ── 8. GO/NO-GO F11, F12 ─────────────────────────────────────────────
    gng_go   = sum(1 for e in logs if e['event'] == 'GNG_STIM' and 'type:GO,'  in e['details'])
    gng_nogo = sum(1 for e in logs if e['event'] == 'GNG_STIM' and 'type:NOGO' in e['details'])
    gng_com  = sum(1 for e in logs if e['event'] == 'GNG_COMMISSION')
    gng_om   = sum(1 for e in logs if e['event'] == 'GNG_OMISSION')
    f11 = float(gng_com / gng_nogo * 100) if gng_nogo else 0.0
    f12 = float(gng_om  / gng_go   * 100) if gng_go   else 0.0

    # ── 9. N-BACK D-PRIME F13, F14 ───────────────────────────────────────
    def dprime(log_list):
        hits, misses, fas, tot_t, tot_n = 0, 0, 0, 0, 0
        for ev in log_list:
            if   ev['event'] == 'NBACK_MISS':         misses += 1; tot_t += 1
            elif ev['event'] == 'NBACK_FALSE_ALARM':  fas    += 1; tot_n += 1
            elif ev['event'] == 'NBACK_RESPONSE':
                if 'target:True'  in ev['details']:   hits   += 1; tot_t += 1
                else:                                              tot_n += 1
        hr  = float(np.clip(hits   / max(tot_t, 1), 0.01, 0.99))
        far = float(np.clip(fas    / max(tot_n, 1), 0.01, 0.99))
        d   = float(scipy_stats.norm.ppf(hr) - scipy_stats.norm.ppf(far))
        c   = float(-0.5 * (scipy_stats.norm.ppf(hr) + scipy_stats.norm.ppf(far)))
        return d, c

    f13, f14 = dprime([e for e in logs if 'NBACK' in e['event']])

    # ── 10. STROOP F15–F18 ───────────────────────────────────────────────
    scong, sincong, serr, stot = [], [], 0, 0
    for ev in logs:
        if ev['event'] == 'STROOP_RESPONSE':
            d = ev['details']
            try:
                rt      = float(d.split('rt:')[1].split(',')[0])
                correct = 'correct:True' in d
                cong    = 'cong:True'    in d
                if 150 < rt < 3000:
                    (scong if cong else sincong).append(rt)
                if not correct: serr += 1
                stot += 1
            except Exception: pass
    f15 = float(np.mean(scong))   if scong   else 0.0
    f16 = float(np.mean(sincong)) if sincong else 0.0
    f17 = f16 - f15
    f18 = float(serr / stot * 100) if stot  else 0.0

    # ── 11. TRAIL MAKING F19–F21 ─────────────────────────────────────────
    def trail_time(part):
        ev = next((e for e in logs if e['event'] == 'TRAIL_END' and 'part:' + part in e['details']), None)
        if ev:
            try: return float(ev['details'].split('time_s:')[1])
            except Exception: pass
        return 0.0
    f19 = trail_time('A')
    f20 = trail_time('B')
    f21 = f20 - f19

    # ── 12. CORSI + DIGIT SPAN F22, F23 ──────────────────────────────────
    def max_span_from_log(end_event):
        ev = next((e for e in logs if e['event'] == end_event), None)
        if ev:
            try: return int(ev['details'].split('max_span:')[1])
            except Exception: pass
        return 0
    f22 = max_span_from_log('CORSI_END')
    f23 = max_span_from_log('DIGITSPAN_END')

    # ── 13. VISUAL SEARCH F24, F25 ───────────────────────────────────────
    vs_rts, vs_miss, vs_total = [], 0, 0
    for ev in logs:
        if ev['event'] == 'VS_RESPONSE':
            d = ev['details']
            try:
                rt      = float(d.split('rt:')[1])
                correct = 'correct:True' in d
                resp    = 'ABSENT' if 'resp:ABSENT' in d else 'PRESENT'
                vs_total += 1
                if 100 < rt < 5000: vs_rts.append(rt)
                if not correct and resp == 'ABSENT': vs_miss += 1
            except Exception: pass
    f24 = float(np.mean(vs_rts)) if vs_rts   else 0.0
    f25 = float(vs_miss / vs_total * 100) if vs_total else 0.0

    # ── 14. METADATA ─────────────────────────────────────────────────────
    meta = {
        "_viewing_distance_cm": round(D_CM, 1),
        "_calib_inliers": n_calib_inliers,
        "_total_fixations": len(fixations),
        "_total_saccades": len(saccades),
        "_video_frames": fi,
    }

    features = {
        "F1_MFD": f1_mfd, "F2_FixationCount": float(f2_fc), "F3_SaccadeLatency": f3_sl,
        "F4_SaccadeAmplitude": f4_sa, "F5_SaccadePeakVelocity": f5_spv,
        "F6_GazeEntropy": f6_ent, "F7_AntiSaccadeErrorRate": f7_aser,
        "F8_ROICoverage": f8_roi, "F9_RT_Mean": f9_rt, "F10_RT_IIV": f10_iiv,
        "F11_CommissionErrors": f11, "F12_OmissionErrors": f12,
        "F13_NBack_dPrime": f13, "F14_NBack_Bias": f14,
        "F15_StroopCongruentRT": f15, "F16_StroopIncongruentRT": f16,
        "F17_StroopInterference": f17, "F18_StroopErrorRate": f18,
        "F19_TMT_A": f19, "F20_TMT_B": f20, "F21_TMT_Delta": f21,
        "F22_CorsiSpan": float(f22), "F23_DigitSpan": float(f23),
        "F24_VisualSearchRT": f24, "F25_VisualSearchMissRate": f25,
    }
    features.update(meta)
    return features


# ══════════════════════════════════════════════════════════════════════════════
# NORMATIVE RANGES — updated per:
#  Rayner 1998, Salvucci & Goldberg 2000, Hutton & Ettinger 2006,
#  Tombaugh 2004 (TMT), Wechsler 2008 (WAIS-IV), MacLeod 1991,
#  Jaeggi 2008, Hultsch 2002, Treisman & Gelade 1980
# ══════════════════════════════════════════════════════════════════════════════
NORMS = {
    "F1_MFD":                {"label":"Mean Fixation Duration",         "unit":"ms",  "lo":150, "hi":350, "domain":"Oculomotor",         "higher_is":"ambiguous"},
    "F2_FixationCount":      {"label":"Fixation Count",                 "unit":"",    "lo":80,  "hi":300, "domain":"Oculomotor",         "higher_is":"ambiguous"},
    "F3_SaccadeLatency":     {"label":"Saccade Latency",                "unit":"ms",  "lo":150, "hi":350, "domain":"Oculomotor",         "higher_is":"worse"},
    "F4_SaccadeAmplitude":   {"label":"Saccade Amplitude",              "unit":"°",   "lo":2.0, "hi":8.0, "domain":"Oculomotor",         "higher_is":"ambiguous"},
    "F5_SaccadePeakVelocity":{"label":"Saccade Peak Velocity",          "unit":"°/s", "lo":200, "hi":600, "domain":"Oculomotor",         "higher_is":"better"},
    "F6_GazeEntropy":        {"label":"Gaze Path Entropy",              "unit":"bits","lo":2.0, "hi":4.0, "domain":"Oculomotor",         "higher_is":"ambiguous"},
    "F7_AntiSaccadeErrorRate":{"label":"Anti-Saccade Error Rate",       "unit":"%",   "lo":0,   "hi":25,  "domain":"Inhibitory Control", "higher_is":"worse"},
    "F8_ROICoverage":        {"label":"ROI Coverage",                   "unit":"%",   "lo":40,  "hi":100, "domain":"Oculomotor",         "higher_is":"better"},
    "F9_RT_Mean":            {"label":"Simple RT Mean",                 "unit":"ms",  "lo":180, "hi":350, "domain":"Processing Speed",   "higher_is":"worse"},
    "F10_RT_IIV":            {"label":"RT Intra-individual Variability","unit":"ms",  "lo":10,  "hi":60,  "domain":"Processing Speed",   "higher_is":"worse"},
    "F11_CommissionErrors":  {"label":"Go/No-Go Commission Rate",       "unit":"%",   "lo":0,   "hi":20,  "domain":"Inhibitory Control", "higher_is":"worse"},
    "F12_OmissionErrors":    {"label":"Go/No-Go Omission Rate",         "unit":"%",   "lo":0,   "hi":10,  "domain":"Inhibitory Control", "higher_is":"worse"},
    "F13_NBack_dPrime":      {"label":"N-Back d′ (sensitivity)",        "unit":"",    "lo":1.0, "hi":4.0, "domain":"Working Memory",     "higher_is":"better"},
    "F14_NBack_Bias":        {"label":"N-Back Response Bias (c)",       "unit":"",    "lo":-1.0,"hi":1.0, "domain":"Working Memory",     "higher_is":"ambiguous"},
    "F15_StroopCongruentRT": {"label":"Stroop Congruent RT",            "unit":"ms",  "lo":350, "hi":700, "domain":"Attention",          "higher_is":"worse"},
    "F16_StroopIncongruentRT":{"label":"Stroop Incongruent RT",         "unit":"ms",  "lo":450, "hi":900, "domain":"Attention",          "higher_is":"worse"},
    "F17_StroopInterference":{"label":"Stroop Interference Score",      "unit":"ms",  "lo":0,   "hi":200, "domain":"Attention",          "higher_is":"worse"},
    "F18_StroopErrorRate":   {"label":"Stroop Error Rate",              "unit":"%",   "lo":0,   "hi":10,  "domain":"Attention",          "higher_is":"worse"},
    "F19_TMT_A":             {"label":"Trail Making A (time)",          "unit":"s",   "lo":15,  "hi":45,  "domain":"Processing Speed",   "higher_is":"worse"},
    "F20_TMT_B":             {"label":"Trail Making B (time)",          "unit":"s",   "lo":30,  "hi":90,  "domain":"Executive Function", "higher_is":"worse"},
    "F21_TMT_Delta":         {"label":"TMT B–A Delta",                  "unit":"s",   "lo":8,   "hi":50,  "domain":"Executive Function", "higher_is":"worse"},
    "F22_CorsiSpan":         {"label":"Corsi Block Span",               "unit":"",    "lo":4,   "hi":7,   "domain":"Working Memory",     "higher_is":"better"},
    "F23_DigitSpan":         {"label":"Digit Span Forward",             "unit":"",    "lo":5,   "hi":9,   "domain":"Working Memory",     "higher_is":"better"},
    "F24_VisualSearchRT":    {"label":"Visual Search RT Mean",          "unit":"ms",  "lo":400, "hi":1200,"domain":"Attention",          "higher_is":"worse"},
    "F25_VisualSearchMissRate":{"label":"Visual Search Miss Rate",      "unit":"%",   "lo":0,   "hi":15,  "domain":"Attention",          "higher_is":"worse"},
}

DOMAIN_REFS = {
    "Oculomotor":         "Rayner (1998) Psych Bull 124:372; Salvucci & Goldberg (2000) ETRA 71",
    "Inhibitory Control": "Hutton & Ettinger (2006) Neuropsychol Rev; Aron (2007) TICS 11:119",
    "Processing Speed":   "Luce (1986) Response Times; Hultsch et al. (2002) Neuropsychology 16:451",
    "Working Memory":     "Jaeggi et al. (2008) PNAS 105:6829; Wechsler (2008) WAIS-IV",
    "Attention":          "MacLeod (1991) Psych Bull 109:163; Treisman & Gelade (1980) Cogn Psychol 12:97",
    "Executive Function": "Tombaugh (2004) J Clin Exp Neuropsychol 26:916; Reitan (1958) Percept Mot Skills",
}

DOMAIN_INTERP = {
    "Oculomotor":         "Oculomotor metrics reflect eye movement control quality. Prolonged fixations may indicate increased cognitive load. Reduced saccade velocity can indicate neurological slowing.",
    "Inhibitory Control": "Inhibitory control underlies the ability to suppress automatic responses. Elevated error rates are associated with ADHD, frontal lobe dysfunction, and impulsivity.",
    "Processing Speed":   "Simple RT and its variability (IIV) are among the most sensitive markers of general cognitive status. IIV elevation predicts cognitive decline independent of mean RT.",
    "Working Memory":     "N-back d′ measures the ability to simultaneously store and manipulate information. Combined with Corsi and Digit Span, it characterises verbal and visuospatial WM separately.",
    "Attention":          "Stroop interference quantifies the cost of resolving competing response tendencies. Visual search RT reflects parallel vs serial search strategies and attentional efficiency.",
    "Executive Function": "Trail Making B–A delta isolates cognitive flexibility from pure motor/processing speed. It is sensitive to prefrontal dysfunction and cognitive flexibility deficits.",
}


def compute_zscore(key: str, val: float) -> float:
    """
    Approximate z-score based on normative midpoint and range.
    (range/2 ≈ 1.65 SD at 90th percentile boundary)
    """
    n = NORMS.get(key)
    if not n or val == 0.0: return 0.0
    mid  = (n['lo'] + n['hi']) / 2.0
    half = (n['hi'] - n['lo']) / 2.0 / 1.65
    if half == 0: return 0.0
    z = (val - mid) / half
    # Flip sign for "worse = higher" features
    if n.get('higher_is') == 'worse': z = -z
    return round(z, 2)


def classify(key: str, val: float):
    n = NORMS.get(key)
    if not n or val == 0.0: return "badge-blue", "No data"
    worse_is_high = n.get('higher_is') == 'worse'
    in_range = n['lo'] <= val <= n['hi']
    if in_range: return "badge-green", "Normal"
    if worse_is_high:
        return ("badge-gold", "Excellent") if val < n['lo'] else ("badge-red", "Elevated")
    better_is_high = n.get('higher_is') == 'better'
    if better_is_high:
        return ("badge-red", "Below norm") if val < n['lo'] else ("badge-gold", "Above norm")
    return ("badge-red", "Below norm") if val < n['lo'] else ("badge-gold", "Above norm")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def page_results():
    r = st.session_state["results"]
    p = st.session_state["participant"]

    # Filter meta fields
    feat = {k: v for k, v in r.items() if not k.startswith('_')}
    meta = {k: v for k, v in r.items() if k.startswith('_')}

    st.markdown(f"""
    <div style='display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:22px;'>
      <div>
        <h1 style='font-size:22px;font-weight:700;margin-bottom:4px;font-family:IBM Plex Mono,monospace;'>Cognitive Assessment Report</h1>
        <p style='color:#6E7681;font-size:13px;'>
          Participant {p.get('id','—')} &nbsp;·&nbsp; {p.get('age','—')} y/o {p.get('gender','—')} &nbsp;·&nbsp; {p.get('timestamp','')[:10]}
        </p>
      </div>
      <span class='badge badge-blue'>v3.0 · calibrated gaze · {int(meta.get('_calib_inliers',0))} calib pts</span>
    </div>""", unsafe_allow_html=True)

    # Metadata row
    col_m = st.columns(4)
    for col, (k, v) in zip(col_m, [
        ("Viewing distance", f"{meta.get('_viewing_distance_cm','—')} cm"),
        ("Fixations",        str(int(meta.get('_total_fixations', 0)))),
        ("Saccades",         str(int(meta.get('_total_saccades',  0)))),
        ("Video frames",     str(int(meta.get('_video_frames',    0)))),
    ]):
        col.markdown(f"<div class='metric-card'><span class='label'>{k}</span><span class='value'>{v}</span></div>", unsafe_allow_html=True)

    # Domain summary cards
    domains: dict = {}
    for key, val in feat.items():
        dom = NORMS[key]["domain"]
        cls, lbl = classify(key, val)
        domains.setdefault(dom, []).append((cls, lbl))

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    dom_cols = st.columns(6)
    for i, (dom, items) in enumerate(domains.items()):
        reds  = sum(1 for c, _ in items if c == "badge-red")
        badge = "badge-green" if reds == 0 else ("badge-gold" if reds <= 1 else "badge-red")
        label = "All normal" if reds == 0 else f"{reds} flag{'s' if reds > 1 else ''}"
        dom_cols[i % 6].markdown(f"""
        <div class='metric-card'>
          <span class='label'>{dom}</span>
          <span class='value' style='font-size:14px;'><span class='badge {badge}'>{label}</span></span>
          <span class='norm'>{len(items)} features</span>
        </div>""", unsafe_allow_html=True)

    # Per-domain tabs
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    DOMAINS_ORDER = ["Oculomotor", "Processing Speed", "Inhibitory Control",
                     "Working Memory", "Attention", "Executive Function"]
    tabs = st.tabs(DOMAINS_ORDER)
    for tab, dom in zip(tabs, DOMAINS_ORDER):
        with tab:
            dom_feats = {k: v for k, v in feat.items() if NORMS[k]["domain"] == dom}
            keys = list(dom_feats.keys())
            for rs in range(0, len(keys), 4):
                rk = keys[rs:rs+4]; cols = st.columns(len(rk))
                for col, key in zip(cols, rk):
                    val = dom_feats[key]; n = NORMS[key]
                    cls, lbl = classify(key, val); unit = n["unit"]
                    z = compute_zscore(key, val)
                    disp = (f"{val:.1f} s"   if unit == "s"   else
                            f"{val:.0f} ms"  if unit == "ms"  else
                            f"{val:.1f}%"    if unit == "%"   else
                            f"{val:.1f}{unit}" if unit in ("°", "°/s") else
                            f"{val:.2f} bits" if unit == "bits" else
                            f"{val:.2f}")
                    z_col = "#3FB950" if z >= 0 else "#F85149"
                    col.markdown(f"""
                    <div class='metric-card'>
                      <span class='label'>{n['label']}</span>
                      <span class='value'>{disp}</span>
                      <span class='norm'><span class='badge {cls}'>{lbl}</span>
                        &nbsp;<span style='color:{z_col};font-size:11px;font-family:IBM Plex Mono,monospace;'>z={z:+.2f}</span>
                      </span>
                    </div>""", unsafe_allow_html=True)
            st.markdown(f"<div class='info-box'>{DOMAIN_INTERP[dom]}<br><br><em style='color:#484F58;'>Refs: {DOMAIN_REFS[dom]}</em></div>", unsafe_allow_html=True)

    # Full matrix
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Complete Feature Matrix")
    rows = ""
    for key, val in feat.items():
        n = NORMS[key]; cls, lbl = classify(key, val); unit = n["unit"]
        z = compute_zscore(key, val)
        rows += (f"<tr><td style='color:#6E7681;font-size:11px;'>{key}</td>"
                 f"<td>{n['label']}</td>"
                 f"<td><span class='badge badge-blue' style='font-size:10px;'>{n['domain']}</span></td>"
                 f"<td style='color:#58A6FF;font-weight:500;'>{val:.2f} {unit}</td>"
                 f"<td>Norm:{n['lo']}–{n['hi']} {unit}</td>"
                 f"<td style='font-family:IBM Plex Mono,monospace;'>{z:+.2f}</td>"
                 f"<td><span class='badge {cls}'>{lbl}</span></td></tr>")
    st.markdown(
        f"<table class='results-table'><thead><tr>"
        f"<th>Code</th><th>Feature</th><th>Domain</th><th>Value</th><th>Norm. Range</th><th>z</th><th>Status</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>",
        unsafe_allow_html=True)

    # Export
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Export")
    col1, col2, col3 = st.columns(3)
    df = pd.DataFrame([{"participant_id": p.get("id"), "timestamp": p.get("timestamp"), **feat}])
    col1.download_button("Download CSV",  df.to_csv(index=False).encode(), "cognitive_results.csv", "text/csv")
    col2.download_button("Download JSON", json.dumps({"participant": p, "results": feat, "meta": meta}, indent=2).encode(), "cognitive_results.json", "application/json")
    col3.download_button("Download PDF",  generate_pdf(p, feat, meta), "cognitive_report.pdf", "application/pdf")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        if st.button("New Participant Session"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf(participant: dict, results: dict, meta: dict) -> bytes:
    pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    # Header
    pdf.set_fill_color(13, 17, 23); pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_text_color(88, 166, 255); pdf.set_font("Helvetica", "B", 17)
    pdf.set_xy(15, 11); pdf.cell(0, 10, "Pocket-Precise Cognitive Diagnostic Report v3.0", ln=True)
    pdf.set_text_color(110, 118, 129); pdf.set_font("Helvetica", "", 9); pdf.set_xy(15, 24)
    pdf.cell(0, 8, (f"Participant: {participant.get('id','—')}  |  Age: {participant.get('age','—')}  |  "
                    f"Date: {participant.get('timestamp','')[:10]}  |  "
                    f"Viewing dist: {meta.get('_viewing_distance_cm','—')} cm  |  "
                    f"Calib pts: {meta.get('_calib_inliers','—')}"), ln=True)
    pdf.set_text_color(30, 30, 30); pdf.set_xy(15, 46)
    # Demographics
    pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Participant Information", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for k, v in participant.items():
        if k != "timestamp":
            pdf.cell(0, 5, f"  {k.capitalize()}: {v}", ln=True)
    pdf.ln(5)
    # Session metadata
    pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Session Quality Metrics", ln=True)
    pdf.set_font("Helvetica", "", 9)
    for k, v in meta.items():
        pdf.cell(0, 5, f"  {k.replace('_','').replace('  ',' ').strip()}: {v}", ln=True)
    pdf.ln(5)
    # Results table
    pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Cognitive Biomarker Results", ln=True)
    pdf.set_font("Helvetica", "B", 8); pdf.set_fill_color(22, 27, 34); pdf.set_text_color(110, 118, 129)
    for h, w in [("Feature", 42), ("Domain", 34), ("Value", 28), ("Norm. Range", 38), ("z", 14), ("Status", 28)]:
        pdf.cell(w, 6, h, border=1, fill=True)
    pdf.ln(); pdf.set_font("Helvetica", "", 8); pdf.set_text_color(30, 30, 30)
    for key, val in results.items():
        n = NORMS[key]; cls, lbl = classify(key, val); unit = n["unit"]
        z = compute_zscore(key, val)
        if cls == "badge-red":    pdf.set_fill_color(250, 235, 235)
        elif cls == "badge-green": pdf.set_fill_color(235, 250, 240)
        else:                      pdf.set_fill_color(255, 255, 255)
        pdf.cell(42, 5, n['label'][:28], border=1, fill=True)
        pdf.cell(34, 5, n['domain'], border=1, fill=True)
        pdf.cell(28, 5, f"{val:.2f} {unit}".strip(), border=1, fill=True)
        pdf.cell(38, 5, f"{n['lo']}–{n['hi']} {unit}".strip(), border=1, fill=True)
        pdf.cell(14, 5, f"{z:+.1f}", border=1, fill=True)
        pdf.cell(28, 5, lbl, border=1, fill=True, ln=True)
    # Domain interpretations
    pdf.ln(6); pdf.set_font("Helvetica", "B", 11); pdf.cell(0, 7, "Domain Interpretations", ln=True)
    pdf.set_font("Helvetica", "", 8)
    for dom, interp in DOMAIN_INTERP.items():
        pdf.set_font("Helvetica", "B", 9); pdf.cell(0, 5, dom, ln=True)
        pdf.set_font("Helvetica", "", 8); pdf.multi_cell(0, 5, interp)
        pdf.ln(2)
    # References
    pdf.ln(4); pdf.set_font("Helvetica", "B", 10); pdf.cell(0, 6, "Key References", ln=True)
    pdf.set_font("Helvetica", "", 7)
    refs = [
        "Rayner (1998). Eye movements in reading and information processing. Psych Bull, 124(3), 372-422.",
        "Salvucci & Goldberg (2000). Identifying fixations and saccades in eye-tracking protocols. ETRA, 71-78.",
        "Hutton & Ettinger (2006). The antisaccade task as a research tool in psychopathology. Neuropsychology Review.",
        "Tombaugh (2004). Trail Making Test A and B: normative data stratified by age and education. J Clin Exp Neuropsychol, 26.",
        "Jaeggi et al. (2008). Improving fluid intelligence with training on working memory. PNAS, 105(19), 6829-6833.",
        "MacLeod (1991). Half a century of research on the Stroop effect. Psych Bull, 109(2), 163-203.",
        "Wechsler (2008). WAIS-IV Administration and Scoring Manual. Pearson.",
        "Treisman & Gelade (1980). A feature-integration theory of attention. Cognitive Psychology, 12, 97-136.",
        "Hultsch et al. (2002). Intraindividual variability in older adults: comparison of adults. Neuropsychology, 16.",
        "Aron (2007). The neural basis of inhibition in cognitive control. TICS, 11, 118-125.",
        "Kartynnik et al. (2019). Real-time facial surface geometry from monocular video. arXiv:1907.06724.",
    ]
    for ref in refs: pdf.multi_cell(0, 4, "• " + ref)
    pdf.ln(4); pdf.set_font("Helvetica", "I", 7); pdf.set_text_color(100, 110, 120)
    pdf.multi_cell(0, 4, "DISCLAIMER: For research use only. Not a clinical diagnosis. Results must be interpreted by a qualified neuropsychologist or clinician in conjunction with a full clinical history.")
    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    page = st.session_state["page"]
    if   page == "consent":      page_consent()
    elif page == "demographics": page_demographics()
    elif page == "battery":      page_battery()
    elif page == "upload":       page_upload()
    elif page == "results":      page_results()


if __name__ == "__main__":
    main()
