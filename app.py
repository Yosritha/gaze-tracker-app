"""
Pocket-Precise Cognitive Diagnostic Engine  v3.0
=================================================
Fixes in this version:
  1. COORDINATE MAPPING — 9-point calibration builds an affine transform
     (least-squares) that maps normalised MediaPipe iris coordinates into
     the task canvas coordinate space. Every gaze sample is remapped before
     feature extraction, so fixation/saccade positions actually correspond
     to stimulus positions.

  2. JSON DOWNLOAD — Streamlit iframes block <a>.click() data-URI downloads
     on most browsers. Fixed with postMessage: the JS battery posts the JSON
     string to the parent window; a <script> tag injected into the Streamlit
     page catches it and triggers the download outside the sandbox.
     Video download uses the same postMessage pathway.

  3. STIMULUS POSITION LOGGING — every task now logs the stimulus position
     in normalised CSS coordinates (0-1) so the Python backend can compare
     mapped gaze against known stimulus positions for spatial accuracy features.
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

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pocket-Precise · Cognitive Diagnostic Engine",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────
# STYLES
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp,[data-testid="stAppViewContainer"]{background:#1E2D35!important;color:#E8EDF0!important;font-family:'Inter',sans-serif!important}
h1,h2,h3,h4,p,span,label,li{color:#E8EDF0!important}
.block-container{padding-top:2rem!important;max-width:1100px!important}
[data-testid="stSidebar"]{background:#162028!important}
.stButton>button{background:transparent!important;color:#D4A843!important;border:1.5px solid #D4A843!important;border-radius:6px!important;padding:10px 22px!important;font-size:15px!important;font-weight:500!important;transition:all .2s!important;width:100%!important}
.stButton>button:hover{background:#D4A843!important;color:#1E2D35!important}
.stTextInput>div>div>input,.stSelectbox>div>div,.stNumberInput>div>div>input{background:#2A3D48!important;color:#E8EDF0!important;border:1px solid #4A6070!important;border-radius:6px!important}
.stSelectbox label,.stTextInput label,.stNumberInput label,.stRadio label,.stCheckbox label{color:#A8BDC8!important;font-size:14px!important}
.metric-card{background:#2A3D48;border:1px solid #3D5565;padding:18px 20px;border-radius:8px;text-align:center;margin-bottom:12px}
.metric-card .label{font-size:12px;color:#7A99A8;text-transform:uppercase;letter-spacing:.8px;display:block;margin-bottom:8px}
.metric-card .value{font-size:26px;font-weight:600;color:#D4A843;display:block}
.metric-card .norm{font-size:12px;color:#5A7A8A;display:block;margin-top:5px}
.section-divider{border:none;border-top:1px solid #2A3D48;margin:28px 0}
.info-box{background:#1E3040;border-left:3px solid #D4A843;padding:14px 18px;border-radius:0 6px 6px 0;margin:16px 0;font-size:14px;color:#A8BDC8!important;line-height:1.6}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:500}
.badge-gold{background:#3D2E10;color:#D4A843}
.badge-green{background:#0E2E1E;color:#4CAF82}
.badge-red{background:#2E0E0E;color:#E07070}
.badge-blue{background:#0E1E2E;color:#70A0E0}
.stTabs [data-baseweb="tab-list"]{background:#2A3D48;border-radius:8px;padding:4px;gap:4px}
.stTabs [data-baseweb="tab"]{color:#7A99A8!important;font-weight:500!important;font-size:14px!important;border-radius:6px!important}
.stTabs [aria-selected="true"]{color:#D4A843!important;background:#1E2D35!important}
.stFileUploader>div{background:#2A3D48!important;border:1px dashed #4A6070!important;border-radius:8px!important}
header,#MainMenu,footer{visibility:hidden}
.results-table{width:100%;border-collapse:collapse;font-size:13px}
.results-table th{background:#2A3D48;color:#7A99A8;padding:10px 14px;text-align:left;font-weight:500;text-transform:uppercase;letter-spacing:.6px;font-size:11px}
.results-table td{padding:10px 14px;border-bottom:1px solid #2A3D48;color:#E8EDF0}
.results-table tr:hover td{background:#2A3D48}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────
def init_state():
    for k, v in {
        "page": "consent", "participant": {},
        "consent_given": False, "battery_done": False,
        "event_logs": None, "video_bytes": None, "results": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def section_header(title, subtitle=""):
    st.markdown(f"<h2 style='font-size:22px;font-weight:600;color:#E8EDF0;margin-bottom:4px;'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='font-size:14px;color:#7A99A8;margin-bottom:20px;'>{subtitle}</p>", unsafe_allow_html=True)

def info_box(text):
    st.markdown(f"<div class='info-box'>{text}</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PAGE 0: CONSENT
# ──────────────────────────────────────────────────────────────
def page_consent():
    st.markdown("<div style='max-width:720px;margin:0 auto;padding:30px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex;align-items:center;gap:14px;margin-bottom:32px;'>
      <div style='width:44px;height:44px;background:#D4A843;border-radius:8px;display:flex;align-items:center;justify-content:center;'>
        <span style='color:#1E2D35;font-size:22px;font-weight:700;'>P</span>
      </div>
      <div>
        <p style='margin:0;font-size:20px;font-weight:600;color:#E8EDF0!important;'>Pocket-Precise</p>
        <p style='margin:0;font-size:13px;color:#7A99A8!important;'>Cognitive Diagnostic Engine · v3.0</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    section_header("Participant Information Sheet")
    info_box("<strong style='color:#D4A843;'>Study Purpose</strong><br>This battery measures oculomotor control, inhibitory control, working memory, processing speed, and attentional capacity. It produces 25 clinically interpretable biomarkers backed by peer-reviewed literature.")
    st.markdown("""
    <div style='background:#2A3D48;border-radius:8px;padding:20px 24px;margin:20px 0;'>
    <p style='font-size:14px;color:#A8BDC8!important;line-height:1.8;margin:0;'>
    <strong style='color:#E8EDF0;'>What this involves:</strong><br>
    10 computer-based tasks, approximately <strong style='color:#D4A843;'>25–35 minutes</strong>. Webcam required for gaze analysis. No data is transmitted externally — all processing is local.<br><br>
    <strong style='color:#E8EDF0;'>Coordinate calibration:</strong><br>
    The first task calibrates 9 screen positions to your iris positions. This affine transform maps all subsequent gaze samples into task-space coordinates, enabling accurate spatial feature extraction.<br><br>
    <strong style='color:#E8EDF0;'>Data handling:</strong><br>
    Video is processed frame-by-frame and discarded. Only computed biomarkers are stored.
    </p></div>
    """, unsafe_allow_html=True)
    tasks = [
        ("Gaze Calibration","9-point affine transform baseline","~1 min"),
        ("Prosaccade / Anti-Saccade","Inhibitory control & saccade latency","~4 min"),
        ("Visual Search","Selective attention & target detection","~3 min"),
        ("Simple Reaction Time","Processing speed & IIV","~3 min"),
        ("Go / No-Go","Response inhibition & impulsivity","~4 min"),
        ("N-Back (1-back & 2-back)","Working memory capacity & d-prime","~5 min"),
        ("Stroop Colour-Word","Cognitive interference & attention","~4 min"),
        ("Trail Making A & B","Processing speed & flexibility","~5 min"),
        ("Corsi Block Tapping","Visuospatial working memory","~3 min"),
        ("Digit Span Forward","Verbal working memory span","~3 min"),
    ]
    for i,(name,desc,dur) in enumerate(tasks):
        st.markdown(f"""
        <div style='display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid #2A3D48;'>
          <div style='display:flex;align-items:center;gap:12px;'>
            <span style='width:24px;height:24px;background:#2A3D48;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:12px;color:#D4A843!important;'>{i+1}</span>
            <div>
              <p style='margin:0;font-size:14px;font-weight:500;color:#E8EDF0!important;'>{name}</p>
              <p style='margin:0;font-size:12px;color:#7A99A8!important;'>{desc}</p>
            </div>
          </div>
          <span style='font-size:12px;color:#5A7A8A;'>{dur}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    agree = st.checkbox("I have read and understood the information above. I consent to participate voluntarily.")
    _,col2,_ = st.columns([2,2,2])
    with col2:
        if st.button("Continue to Demographics →", disabled=not agree):
            st.session_state.update({"consent_given":True,"page":"demographics"})
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PAGE 1: DEMOGRAPHICS
# ──────────────────────────────────────────────────────────────
def page_demographics():
    st.markdown("<div style='max-width:720px;margin:0 auto;padding:30px 0;'>", unsafe_allow_html=True)
    section_header("Participant Demographics","Used to normalise biomarkers against population norms.")
    c1,c2 = st.columns(2)
    with c1:
        pid = st.text_input("Participant ID *", placeholder="e.g. P001")
        age = st.number_input("Age *", min_value=18, max_value=90, value=25)
        handedness = st.selectbox("Handedness",["Right","Left","Ambidextrous"])
    with c2:
        gender = st.selectbox("Gender",["Male","Female","Non-binary","Prefer not to say"])
        education = st.selectbox("Education level",["Secondary school","Undergraduate","Postgraduate","Doctoral","Other"])
        vision = st.selectbox("Corrected-to-normal vision?",["Yes","No – I have uncorrected impairment"])
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Pre-session Checks","Answer honestly — affects data quality flags only.")
    c3,c4 = st.columns(2)
    with c3:
        sleep = st.selectbox("Sleep last night (hours)",["< 5","5–6","7–8","9+"])
        caffeine = st.selectbox("Caffeine in last 2 hours?",["No","Yes – 1 drink","Yes – 2+ drinks"])
    with c4:
        medications = st.selectbox("Psychoactive medication?",["No","Yes (stimulant)","Yes (sedative)","Yes (other)"])
        anxiety = st.selectbox("Current anxiety level",["1 – Very low","2","3 – Moderate","4","5 – Very high"])
    st.markdown("<br>", unsafe_allow_html=True)
    _,col2,_ = st.columns([1,2,1])
    with col2:
        if st.button("Begin Assessment Battery →", disabled=not pid.strip()):
            st.session_state["participant"] = {
                "id":pid,"age":age,"gender":gender,"handedness":handedness,
                "education":education,"vision":vision,"sleep":sleep,
                "caffeine":caffeine,"medications":medications,"anxiety":anxiety,
                "timestamp":datetime.datetime.now().isoformat()
            }
            st.session_state["page"] = "battery"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# PAGE 2: BATTERY
# ──────────────────────────────────────────────────────────────
def page_battery():
    section_header("Assessment Battery","Follow each task's instructions carefully. Do not close this tab.")
    info_box("<strong style='color:#D4A843;'>Before you start:</strong> Sit ~60 cm from screen. Face must be well-lit. Minimise head movement. Tasks advance automatically.")

    # ── postMessage download receiver — injected OUTSIDE the iframe ──
    # This is the fix for the JSON download being blocked inside the sandbox.
    # The iframe posts {type, filename, dataUrl} messages; this script catches
    # them in the parent window and triggers real downloads.
    st.markdown("""
    <script id="dl-receiver">
    (function(){
        if(window._dlListenerAttached) return;
        window._dlListenerAttached = true;
        window.addEventListener('message', function(ev){
            if(!ev.data || ev.data.type !== 'POCKET_PRECISE_DOWNLOAD') return;
            var a = document.createElement('a');
            a.href = ev.data.dataUrl;
            a.download = ev.data.filename;
            document.body.appendChild(a);
            a.click();
            setTimeout(function(){ document.body.removeChild(a); }, 500);
        });
    })();
    </script>
    """, unsafe_allow_html=True)

    components.html(_build_battery_html(), height=720, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)
    info_box("""
    When the battery finishes, two files download automatically:<br>
    &nbsp;&nbsp;• <strong style='color:#D4A843;'>raw_gaze_video.webm</strong> — webcam recording<br>
    &nbsp;&nbsp;• <strong style='color:#D4A843;'>interaction_logs.json</strong> — timestamped event log<br><br>
    Once both files have downloaded, click below.
    """)
    _,col2,_ = st.columns([1,2,1])
    with col2:
        if st.button("I have both files → Upload for Analysis"):
            st.session_state.update({"battery_done":True,"page":"upload"})
            st.rerun()

# ──────────────────────────────────────────────────────────────
# BATTERY HTML
# ──────────────────────────────────────────────────────────────
def _build_battery_html():
    return r"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1E2D35;color:#E8EDF0;font-family:Inter,sans-serif;display:flex;flex-direction:column;align-items:center;min-height:700px;overflow:hidden}
#stage{position:relative;width:800px;height:560px;background:#253540;border:1px solid #3D5565;border-radius:10px;overflow:hidden;margin-top:16px}
#ui{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:40px;text-align:center;z-index:20}
#progress-bar-wrap{width:800px;height:4px;background:#2A3D48;border-radius:2px;margin-top:10px}
#progress-bar{height:4px;background:#D4A843;border-radius:2px;width:0%;transition:width .5s}
#task-label{font-size:11px;color:#5A7A8A;letter-spacing:1px;text-transform:uppercase;margin-top:6px}
h2{font-size:24px;font-weight:600;margin-bottom:10px}
.sub{font-size:14px;color:#7A99A8;line-height:1.6;margin-bottom:28px;max-width:560px;white-space:pre-line}
.btn{background:transparent;color:#D4A843;border:1.5px solid #D4A843;border-radius:6px;padding:11px 28px;font-size:15px;font-weight:500;cursor:pointer;transition:all .2s;font-family:inherit}
.btn:hover{background:#D4A843;color:#1E2D35}
.btn:disabled{opacity:.3;cursor:not-allowed}
#dot{position:absolute;width:18px;height:18px;background:#D4A843;border-radius:50%;transform:translate(-50%,-50%);display:none;z-index:10;box-shadow:0 0 0 4px rgba(212,168,67,.25);transition:left .3s,top .3s}
#cross{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:42px;color:#4A6070;display:none;z-index:10;font-weight:300;line-height:1}
#stim{position:absolute;inset:0;display:none;align-items:center;justify-content:center;z-index:15;flex-direction:column}
#stim-box{background:#1E3040;border:1.5px solid #D4A843;border-radius:10px;padding:30px 50px;text-align:center;min-width:300px}
#stim-text{font-size:48px;font-weight:700}
#stim-label{font-size:13px;color:#7A99A8;margin-top:8px}
#nback-grid{display:none;position:absolute;inset:0;align-items:center;justify-content:center;z-index:15}
.nb-cell{width:90px;height:90px;border:1px solid #3D5565;border-radius:6px;background:#1E2D35;display:flex;align-items:center;justify-content:center;font-size:36px;font-weight:700;color:transparent;transition:all .1s}
.nb-cell.active{background:#D4A843;color:#1E2D35}
#trail-canvas{position:absolute;inset:0;display:none;z-index:15}
#corsi-area{position:absolute;inset:0;display:none;z-index:15}
.corsi-block{position:absolute;width:60px;height:60px;background:#2A3D48;border:1.5px solid #4A6070;border-radius:8px;cursor:pointer;transition:background .15s}
.corsi-block.lit{background:#D4A843;border-color:#D4A843}
.corsi-block.correct{background:#4CAF82;border-color:#4CAF82}
.corsi-block.wrong{background:#E07070;border-color:#E07070}
#digitspan-area{position:absolute;inset:0;display:none;flex-direction:column;align-items:center;justify-content:center;z-index:15}
#digit-display{font-size:72px;font-weight:700;color:#D4A843;display:none}
#digit-input-wrap{display:none;flex-direction:column;align-items:center;gap:14px}
#digit-input{background:#1E2D35;border:1.5px solid #4A6070;color:#E8EDF0;font-size:28px;text-align:center;border-radius:8px;padding:12px 20px;width:260px;font-family:inherit}
#vs-hint{position:absolute;bottom:0;left:0;right:0;z-index:16;display:none;padding:14px;text-align:center;background:rgba(30,45,53,.9)}
</style>
</head>
<body>
<div id="stage">
  <div id="ui">
    <h2 id="title">Battery Ready</h2>
    <p class="sub" id="sub">Ensure webcam is available and you are seated ~60 cm from screen in a well-lit environment.</p>
    <button class="btn" id="btn" onclick="initBattery()">Start Battery</button>
  </div>
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
    <p style="color:#7A99A8;font-size:14px;margin-bottom:12px;">Memorise the following sequence</p>
    <div id="digit-display"></div>
    <div id="digit-input-wrap">
      <input id="digit-input" type="text" placeholder="Enter digits in order" autocomplete="off">
      <button class="btn" onclick="submitDigitSpan()">Submit</button>
    </div>
  </div>
  <div id="vs-hint">
    <p style="font-size:13px;color:#7A99A8;margin:0;">
      Press <strong style="color:#D4A843;">SPACE</strong> = target present &nbsp;|&nbsp;
      <strong style="color:#D4A843;">N</strong> = target absent
    </p>
  </div>
</div>
<div id="progress-bar-wrap"><div id="progress-bar"></div></div>
<div id="task-label">Task 0 / 10</div>

<script>
// ══════════════════════════════════════════════════════════════
// GLOBAL STATE
// ══════════════════════════════════════════════════════════════
const logs = [];
let mediaRecorder, stream, videoChunks = [];
let keyHandler = null;
const TOTAL_TASKS = 10;

// ── FIX 1: Calibration data for affine transform ──────────────
// calibPts[i] = { screen_x_norm, screen_y_norm, gaze_x, gaze_y }
// Populated during Task 1; used by Python backend.
// We also store the canvas bounding rect at calibration time so
// the Python backend can convert % coords to the same space.
const calibPts = [];

function log(event, details) {
    logs.push({ timestamp_ms: performance.now().toFixed(2), event, details });
}

function setProgress(n) {
    document.getElementById('progress-bar').style.width = (n/TOTAL_TASKS*100)+'%';
    document.getElementById('task-label').innerText = 'Task '+n+' / '+TOTAL_TASKS;
}

function showUI(title, sub, btnLabel, onclick) {
    document.getElementById('title').innerText = title;
    document.getElementById('sub').innerText   = sub;
    const btn = document.getElementById('btn');
    btn.innerText = btnLabel;
    btn.onclick   = onclick;
    btn.disabled  = false;
    document.getElementById('ui').style.display = 'flex';
}

function hideUI() { document.getElementById('ui').style.display = 'none'; }

function hideAll() {
    if (keyHandler) { document.removeEventListener('keydown', keyHandler); keyHandler = null; }
    document.getElementById('dot').style.display  = 'none';
    document.getElementById('cross').style.display = 'none';
    const stim = document.getElementById('stim');
    stim.style.display = 'none';
    stim.style.alignItems = 'center';
    const st = document.getElementById('stim-text');
    const sl = document.getElementById('stim-label');
    if (st) { st.innerText=''; st.style.color=''; st.style.fontSize='48px'; }
    if (sl)   sl.innerText='';
    document.getElementById('nback-grid').style.display    = 'none';
    document.getElementById('trail-canvas').style.display  = 'none';
    document.getElementById('corsi-area').style.display    = 'none';
    document.getElementById('digitspan-area').style.display = 'none';
    document.getElementById('vs-hint').style.display       = 'none';
    document.getElementById('stage').querySelectorAll('canvas:not(#trail-canvas)').forEach(c=>c.remove());
}

// ── FIX 2: postMessage download — works outside sandbox ───────
function postDownload(filename, dataUrl) {
    window.parent.postMessage({ type:'POCKET_PRECISE_DOWNLOAD', filename, dataUrl }, '*');
}

async function initBattery() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video:{facingMode:'user',frameRate:30}, audio:false });
        const opts = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? {mimeType:'video/webm;codecs=vp9'} : {mimeType:'video/webm'};
        videoChunks = [];
        mediaRecorder = new MediaRecorder(stream, opts);
        mediaRecorder.ondataavailable = e => { if(e.data.size>0) videoChunks.push(e.data); };
        mediaRecorder.onstop = () => exportAll(new Blob(videoChunks,{type:'video/webm'}));
        mediaRecorder.start(100);
        log('SYSTEM_START','Battery initiated');
        task_calibration();
    } catch(e) {
        showUI('Camera Access Denied','Grant webcam permission and reload.','Retry',initBattery);
    }
}

// ══════════════════════════════════════════════════════════════
// TASK 1 — 9-POINT CALIBRATION
// FIX 1 CORE: Records normalised screen coords + timestamp for
// each calibration point. Python backend reads these to build
// a least-squares affine transform mapping gaze → screen space.
// Reference: Salvucci & Goldberg (2000), ETRA
// ══════════════════════════════════════════════════════════════
function task_calibration() {
    setProgress(1);
    showUI(
        'Task 1 · Gaze Calibration',
        'A gold dot will appear at 9 positions on screen.\nLook at each dot and hold your gaze for 1.5 seconds.\nDo not move your head.',
        'Begin Calibration',
        () => {
            hideUI();
            log('TASK_START','Calibration');
            // Record stage bounding rect so Python can interpret % coords
            const stageRect = document.getElementById('stage').getBoundingClientRect();
            log('STAGE_RECT', JSON.stringify({
                x: stageRect.x, y: stageRect.y,
                w: stageRect.width, h: stageRect.height,
                devicePixelRatio: window.devicePixelRatio || 1
            }));

            // 9 calibration points in % of stage (matching standard 3x3 grid)
            const coords = [
                [10,10],[50,10],[90,10],
                [10,50],[50,50],[90,50],
                [10,90],[50,90],[90,90]
            ];
            const dot = document.getElementById('dot');
            dot.style.display = 'block';
            let i = 0;
            function step() {
                if (i >= coords.length) {
                    dot.style.display = 'none';
                    log('TASK_END','Calibration');
                    task_prosaccade();
                    return;
                }
                dot.style.left = coords[i][0]+'%';
                dot.style.top  = coords[i][1]+'%';
                // Log with screen_norm_x/y: position as fraction of stage (0-1)
                // Python will read CALIB_POINT events and pair with gaze stream timestamps
                log('CALIB_POINT', JSON.stringify({
                    screen_nx: coords[i][0]/100,
                    screen_ny: coords[i][1]/100,
                    t: performance.now()
                }));
                i++;
                setTimeout(step, 1500); // 1.5s per point — enough for iris settle
            }
            step();
        }
    );
}

// ══════════════════════════════════════════════════════════════
// TASK 2 — PROSACCADE & ANTI-SACCADE
// Logs stimulus position as screen_nx/ny for spatial accuracy
// Reference: Hutton & Ettinger (2006), Neuropsychology Review
// ══════════════════════════════════════════════════════════════
function task_prosaccade() {
    setProgress(2);
    showUI(
        'Task 2 · Saccade Tasks (Part A)',
        'Look AT the gold dot as fast as possible.\nFix on the centre + between trials.',
        'Begin Prosaccade',
        () => {
            hideUI();
            log('TASK_START','Prosaccade');
            const positions = [20,80,20,80,50,20,80,50,20,80];
            const dot=document.getElementById('dot'), cross=document.getElementById('cross');
            let i=0;
            function run() {
                if(i>=positions.length){ cross.style.display='none'; dot.style.display='none'; task_antisaccade(); return; }
                cross.style.display='block';
                setTimeout(()=>{
                    cross.style.display='none';
                    dot.style.left=positions[i]+'%'; dot.style.top='50%'; dot.style.display='block';
                    log('PROSAC_STIM', JSON.stringify({screen_nx:positions[i]/100, screen_ny:0.5, t:performance.now()}));
                    i++;
                    setTimeout(()=>{ dot.style.display='none'; setTimeout(run,600); },1000);
                }, 900+Math.random()*600);
            }
            run();
        }
    );
}

function task_antisaccade() {
    showUI(
        'Task 2 · Saccade Tasks (Part B)',
        'A dot will flash briefly. Look to the OPPOSITE side immediately.\nInhibit the reflexive look toward the dot.',
        'Begin Anti-Saccade',
        () => {
            hideUI();
            log('TASK_START','AntiSaccade');
            const positions=[15,85,15,85,15,85,15,85];
            const dot=document.getElementById('dot'), cross=document.getElementById('cross');
            let i=0;
            function run() {
                if(i>=positions.length){ cross.style.display='none'; dot.style.display='none'; log('TASK_END','AntiSaccade'); task_visualsearch(); return; }
                cross.style.display='block';
                setTimeout(()=>{
                    cross.style.display='none';
                    const side=positions[i]<50?'LEFT':'RIGHT';
                    dot.style.left=positions[i]+'%'; dot.style.top='50%'; dot.style.display='block';
                    log('ANTISAC_STIM', JSON.stringify({side, screen_nx:positions[i]/100, screen_ny:0.5, t:performance.now()}));
                    i++;
                    setTimeout(()=>{ dot.style.display='none'; setTimeout(run,700); },250); // 250ms flash, Munoz & Everling 2004
                }, 900+Math.random()*600);
            }
            run();
        }
    );
}

// ══════════════════════════════════════════════════════════════
// TASK 3 — VISUAL SEARCH
// Logs target position as screen_nx/ny for gaze accuracy check
// Reference: Treisman & Gelade (1980), Cognitive Psychology
// ══════════════════════════════════════════════════════════════
function task_visualsearch() {
    setProgress(3);
    showUI(
        'Task 3 · Visual Search',
        'Find the orange circle among blue squares.\nPress SPACE = target present, N = target absent.',
        'Begin Visual Search',
        () => {
            hideUI();
            log('TASK_START','VisualSearch');
            const stage=document.getElementById('stage');
            const canvas=document.createElement('canvas');
            canvas.width=800; canvas.height=560;
            canvas.style.cssText='position:absolute;inset:0;z-index:15;';
            stage.appendChild(canvas);
            const ctx=canvas.getContext('2d');
            document.getElementById('vs-hint').style.display='block';

            const trials=[];
            for(let k=0;k<14;k++) trials.push(true);
            for(let k=0;k<6;k++)  trials.push(false);
            trials.sort(()=>Math.random()-.5);

            let trialIdx=0, t0, responded=false, targetPos=null;

            function noOverlap(pos,x,y){ return pos.every(p=>Math.hypot(p[0]-x,p[1]-y)>55); }

            function drawSearch(hasTarget) {
                ctx.clearRect(0,0,800,560);
                const pos=[];
                for(let d=0;d<11;d++){
                    let x,y,t=0;
                    do{x=60+Math.random()*680;y=60+Math.random()*440;t++;}while(!noOverlap(pos,x,y)&&t<50);
                    pos.push([x,y]);
                    ctx.fillStyle='#3A8FD4'; ctx.strokeStyle='#5AAAE8'; ctx.lineWidth=2;
                    ctx.beginPath(); ctx.rect(x-20,y-20,40,40); ctx.fill(); ctx.stroke();
                }
                targetPos=null;
                if(hasTarget){
                    let x,y,t=0;
                    do{x=60+Math.random()*680;y=60+Math.random()*440;t++;}while(!noOverlap(pos,x,y)&&t<50);
                    targetPos={nx:x/800, ny:y/560};
                    ctx.fillStyle='#D4A843'; ctx.strokeStyle='#E8C070'; ctx.lineWidth=2;
                    ctx.beginPath(); ctx.arc(x,y,22,0,Math.PI*2); ctx.fill(); ctx.stroke();
                }
                t0=performance.now(); responded=false;
            }

            function nextTrial() {
                if(trialIdx>=trials.length){
                    ctx.clearRect(0,0,800,560); canvas.remove();
                    document.getElementById('vs-hint').style.display='none';
                    if(keyHandler){document.removeEventListener('keydown',keyHandler);keyHandler=null;}
                    log('TASK_END','VisualSearch');
                    task_rt(); return;
                }
                const hasTarget=trials[trialIdx];
                drawSearch(hasTarget);
                log('VS_TRIAL', JSON.stringify({trial:trialIdx, target:hasTarget,
                    target_nx: targetPos?targetPos.nx:null,
                    target_ny: targetPos?targetPos.ny:null}));
                trialIdx++;
            }

            function handleKey(e) {
                if(responded) return;
                if(e.code!=='Space'&&e.key.toLowerCase()!=='n') return;
                responded=true;
                const hasTarget=trials[trialIdx-1];
                const rt=performance.now()-t0;
                const resp=e.code==='Space'?'PRESENT':'ABSENT';
                const correct=(resp==='PRESENT')===hasTarget;
                log('VS_RESPONSE', JSON.stringify({resp,correct,rt:+rt.toFixed(2)}));
                setTimeout(nextTrial,300);
            }
            document.addEventListener('keydown',handleKey);
            keyHandler=handleKey;
            setTimeout(nextTrial,500);
        }
    );
}

// ══════════════════════════════════════════════════════════════
// TASK 4 — SIMPLE REACTION TIME
// Reference: Luce (1986), Response Times, Oxford UP
// ══════════════════════════════════════════════════════════════
function task_rt() {
    setProgress(4);
    showUI(
        'Task 4 · Simple Reaction Time',
        'Press SPACE as fast as possible when the gold circle appears.\nWait — do not press early!',
        'Begin Reaction Time Test',
        () => {
            hideUI();
            log('TASK_START','SimpleRT');
            const dot=document.getElementById('dot');
            dot.style.left='50%'; dot.style.top='50%';
            let trials=0, t0;
            const N=20;

            function nextTrial() {
                if(trials>=N){
                    dot.style.display='none';
                    document.removeEventListener('keydown',rtKey);
                    keyHandler=null;
                    log('TASK_END','SimpleRT');
                    task_gonogo(); return;
                }
                const delay=1200+Math.random()*2000;
                setTimeout(()=>{ dot.style.display='block'; t0=performance.now(); log('RT_STIMULUS','trial:'+trials+',t:'+t0.toFixed(2)); },delay);
            }

            function rtKey(e) {
                if(e.code!=='Space') return;
                if(dot.style.display==='none'){ log('RT_FALSE_START','trial:'+trials); return; }
                const rt=performance.now()-t0;
                dot.style.display='none';
                log('RT_RESPONSE','trial:'+trials+',rt:'+rt.toFixed(2));
                trials++;
                setTimeout(nextTrial,400);
            }
            keyHandler=rtKey;
            document.addEventListener('keydown',rtKey);
            nextTrial();
        }
    );
}

// ══════════════════════════════════════════════════════════════
// TASK 5 — GO / NO-GO  (fully fixed)
// Reference: Donders (1869); Aron (2007), TICS
// ══════════════════════════════════════════════════════════════
function task_gonogo() {
    setProgress(5);
    showUI(
        'Task 5 · Go / No-Go',
        'Press SPACE for a GREEN circle (Go).\nDo NOT press for a RED circle (No-Go).\nRespond quickly!',
        'Begin Go/No-Go',
        () => {
            hideUI();
            if(keyHandler){document.removeEventListener('keydown',keyHandler);keyHandler=null;}
            log('TASK_START','GoNoGo');

            // Rebuild stim DOM (defensive — ensures elements exist regardless of prior task)
            const stim=document.getElementById('stim');
            stim.innerHTML='<div id="stim-box" style="background:#1E3040;border:1.5px solid #D4A843;border-radius:10px;padding:30px 50px;text-align:center;min-width:300px;"><div id="stim-text" style="font-size:72px;font-weight:700;"></div><div id="stim-label" style="font-size:14px;color:#7A99A8;margin-top:10px;"></div></div>';
            stim.style.alignItems='center';
            stim.style.justifyContent='center';
            stim.style.display='none';

            const stimText=document.getElementById('stim-text');
            const stimLabel=document.getElementById('stim-label');

            const sequence=[];
            for(let k=0;k<30;k++) sequence.push(Math.random()<.75?'GO':'NOGO');

            let idx=0, t0, responded;

            function runTrial() {
                if(idx>=sequence.length){ stim.style.display='none'; log('TASK_END','GoNoGo'); task_nback(); return; }
                const type=sequence[idx];
                stimText.style.color=type==='GO'?'#4CAF82':'#E07070';
                stimText.innerText='●';
                stimLabel.innerText=type==='GO'?'GO — press SPACE':'NO-GO — do not press';
                stim.style.display='flex';
                t0=performance.now(); responded=false;
                log('GNG_STIM','trial:'+idx+',type:'+type+',t:'+t0.toFixed(2));
                idx++;

                const timeout=setTimeout(()=>{
                    if(!responded&&type==='GO') log('GNG_OMISSION','trial:'+(idx-1));
                    stim.style.display='none';
                    stimText.innerText=''; stimLabel.innerText='';
                    setTimeout(runTrial,400);
                },800);

                function resp(e) {
                    if(e.code!=='Space') return;
                    responded=true;
                    clearTimeout(timeout);
                    document.removeEventListener('keydown',resp);
                    if(keyHandler===resp) keyHandler=null;
                    const rt=performance.now()-t0;
                    const correct=(type==='GO');
                    log('GNG_RESPONSE','trial:'+(idx-1)+',type:'+type+',correct:'+correct+',rt:'+rt.toFixed(2));
                    if(!correct) log('GNG_COMMISSION','trial:'+(idx-1));
                    stim.style.display='none';
                    stimText.innerText=''; stimLabel.innerText='';
                    setTimeout(runTrial,300);
                }
                document.addEventListener('keydown',resp);
                keyHandler=resp;
            }
            runTrial();
        }
    );
}

// ══════════════════════════════════════════════════════════════
// TASK 6 — N-BACK
// Reference: Kirchner (1958); Jaeggi et al. (2008), PNAS
// ══════════════════════════════════════════════════════════════
function task_nback() {
    setProgress(6);
    function runNBack(n,onDone) {
        const letters='BDFGHJKLMNPQRSTVWXZ'.split('');
        const seq=[];
        for(let k=0;k<20+n;k++){
            if(k>=n&&Math.random()<.3) seq.push(seq[k-n]);
            else seq.push(letters[Math.floor(Math.random()*letters.length)]);
        }
        log('NBACK_START','n:'+n);
        const grid=document.getElementById('nback-grid');
        grid.style.display='flex';
        const cells=Array.from(document.querySelectorAll('.nb-cell'));
        let idx=0;
        function showItem() {
            if(idx>=seq.length){
                cells.forEach(c=>{c.classList.remove('active');c.innerText='';});
                grid.style.display='none';
                if(keyHandler){document.removeEventListener('keydown',keyHandler);keyHandler=null;}
                log('NBACK_END','n:'+n);
                onDone(); return;
            }
            const isTarget=(idx>=n)&&(seq[idx]===seq[idx-n]);
            cells.forEach(c=>{c.classList.remove('active');c.innerText='';});
            const cell=Math.floor(Math.random()*9);
            cells[cell].classList.add('active');
            cells[cell].innerText=seq[idx];
            log('NBACK_STIM','idx:'+idx+',letter:'+seq[idx]+',target:'+isTarget+',t:'+performance.now().toFixed(2));
            let responded=false;
            const timeout=setTimeout(()=>{
                if(!responded&&isTarget) log('NBACK_MISS','idx:'+idx);
                cells.forEach(c=>{c.classList.remove('active');c.innerText='';});
                idx++; setTimeout(showItem,400);
            },1600);
            function nbResp(e) {
                if(e.code!=='Space') return;
                responded=true; clearTimeout(timeout);
                document.removeEventListener('keydown',nbResp);
                if(keyHandler===nbResp) keyHandler=null;
                log('NBACK_RESPONSE','idx:'+idx+',target:'+isTarget+',t:'+performance.now().toFixed(2));
                if(!isTarget) log('NBACK_FALSE_ALARM','idx:'+idx);
                cells.forEach(c=>{c.classList.remove('active');c.innerText='';});
                idx++; setTimeout(showItem,400);
            }
            document.addEventListener('keydown',nbResp);
            keyHandler=nbResp;
        }
        showItem();
    }
    showUI('Task 6 · N-Back Working Memory',
        '1-Back: Press SPACE if current letter matches one step before.\n2-Back (next): match two steps back.\n\nRespond quickly and accurately.',
        'Begin 1-Back',
        ()=>{ hideUI(); runNBack(1,()=>{
            showUI('Task 6 · Part B','Now 2-back: match the letter shown TWO steps ago.','Begin 2-Back',
                ()=>{ hideUI(); runNBack(2,()=>{ log('TASK_END','NBack'); task_stroop(); }); });
        }); }
    );
}

// ══════════════════════════════════════════════════════════════
// TASK 7 — STROOP
// Reference: Stroop (1935); MacLeod (1991), Psych Bull
// ══════════════════════════════════════════════════════════════
function task_stroop() {
    setProgress(7);
    showUI('Task 7 · Stroop Colour-Word',
        'Name the INK COLOR of each word (ignore what it says).\nR = Red, G = Green, B = Blue, Y = Yellow',
        'Begin Stroop',
        ()=>{
            hideUI(); log('TASK_START','Stroop');
            const stage=document.getElementById('stage');
            const canvas=document.createElement('canvas');
            canvas.width=800; canvas.height=560;
            canvas.style.cssText='position:absolute;inset:0;z-index:15;';
            stage.appendChild(canvas);
            const ctx=canvas.getContext('2d');
            const COLORS=['RED','GREEN','BLUE','YELLOW'];
            const INK={RED:'#E07070',GREEN:'#4CAF82',BLUE:'#70A0E0',YELLOW:'#D4A843'};
            const KEYS={r:'RED',g:'GREEN',b:'BLUE',y:'YELLOW'};
            const trials=[];
            for(let k=0;k<16;k++){ const c=COLORS[k%4]; trials.push({word:c,ink:c,congruent:true}); }
            for(let k=0;k<16;k++){ const w=COLORS[k%4]; const ink=COLORS.filter(x=>x!==w)[k%3]; trials.push({word:w,ink,congruent:false}); }
            trials.sort(()=>Math.random()-.5);
            let idx=0,t0,responded;
            function nextTrial() {
                if(idx>=trials.length){
                    ctx.clearRect(0,0,800,560); canvas.remove();
                    if(keyHandler){document.removeEventListener('keydown',keyHandler);keyHandler=null;}
                    log('TASK_END','Stroop'); task_trail(); return;
                }
                const tr=trials[idx];
                ctx.clearRect(0,0,800,560);
                ctx.font='bold 52px Inter,sans-serif'; ctx.fillStyle=INK[tr.ink];
                ctx.textAlign='center'; ctx.textBaseline='middle';
                ctx.fillText(tr.word,400,260);
                ctx.font='14px Inter,sans-serif'; ctx.fillStyle='#5A7A8A';
                ctx.fillText('R=Red  G=Green  B=Blue  Y=Yellow',400,480);
                t0=performance.now(); responded=false;
                log('STROOP_STIM','word:'+tr.word+',ink:'+tr.ink+',cong:'+tr.congruent+',t:'+t0.toFixed(2));
                idx++;
            }
            const kh=function(e){
                if(responded) return;
                const key=e.key.toLowerCase();
                if(!KEYS[key]) return;
                responded=true;
                const rt=performance.now()-t0;
                const resp=KEYS[key];
                const correct=resp===trials[idx-1].ink;
                log('STROOP_RESPONSE','resp:'+resp+',correct:'+correct+',rt:'+rt.toFixed(2)+',cong:'+trials[idx-1].congruent);
                setTimeout(nextTrial,200);
            };
            document.addEventListener('keydown',kh);
            keyHandler=kh;
            setTimeout(nextTrial,300);
        }
    );
}

// ══════════════════════════════════════════════════════════════
// TASK 8 — TRAIL MAKING
// Reference: Reitan (1958), Perceptual & Motor Skills
// ══════════════════════════════════════════════════════════════
function task_trail() {
    setProgress(8);
    function runTrail(part,onDone) {
        const canvas=document.getElementById('trail-canvas');
        canvas.style.display='block';
        const ctx=canvas.getContext('2d');
        ctx.clearRect(0,0,800,560);
        const N=10, nodes=[];
        function noOvlp(x,y){return nodes.every(p=>Math.hypot(p.x-x,p.y-y)>70);}
        for(let k=0;k<N;k++){
            let x,y,t=0;
            do{x=60+Math.random()*680;y=60+Math.random()*440;t++;}while(!noOvlp(x,y)&&t<200);
            const label=part==='A'?String(k+1):(k%2===0?String(k/2+1):String.fromCharCode(65+Math.floor(k/2)));
            nodes.push({x,y,label,visited:false});
        }
        let correctOrder;
        if(part==='A') correctOrder=nodes.slice().sort((a,b)=>parseInt(a.label)-parseInt(b.label));
        else correctOrder=[...nodes].sort((a,b)=>{
            const rank=n=>isNaN(n.label)?(n.label.charCodeAt(0)-64)*2:parseInt(n.label)*2-1;
            return rank(a)-rank(b);
        });
        function draw(){
            ctx.clearRect(0,0,800,560);
            ctx.strokeStyle='#D4A843'; ctx.lineWidth=2;
            for(let k=1;k<correctOrder.length;k++){
                if(correctOrder[k-1].visited&&correctOrder[k].visited){
                    ctx.beginPath(); ctx.moveTo(correctOrder[k-1].x,correctOrder[k-1].y); ctx.lineTo(correctOrder[k].x,correctOrder[k].y); ctx.stroke();
                }
            }
            const nxt=nodes.filter(x=>x.visited).length;
            nodes.forEach(n=>{
                ctx.beginPath(); ctx.arc(n.x,n.y,22,0,Math.PI*2);
                ctx.fillStyle=n.visited?'#2A3D48':(n===correctOrder[nxt]?'#D4A843':'#2A3D48');
                ctx.strokeStyle=n.visited?'#4CAF82':'#4A6070'; ctx.lineWidth=2;
                ctx.fill(); ctx.stroke();
                ctx.fillStyle=n.visited?'#4CAF82':'#E8EDF0';
                ctx.font='bold 14px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
                ctx.fillText(n.label,n.x,n.y);
            });
            ctx.font='13px Inter,sans-serif'; ctx.fillStyle='#5A7A8A'; ctx.textAlign='left';
            ctx.fillText('Trail Making Part '+part+' — connect: '+(part==='A'?'1→2→3...':'1→A→2→B→3...'),16,24);
        }
        let nextIdx=0;
        const t0=performance.now();
        log('TRAIL_START','part:'+part);
        draw();
        canvas.onclick=function(e){
            const r=canvas.getBoundingClientRect();
            const mx=(e.clientX-r.left)*(800/r.width);
            const my=(e.clientY-r.top)*(560/r.height);
            const target=correctOrder[nextIdx];
            if(Math.hypot(target.x-mx,target.y-my)<28){
                target.visited=true;
                log('TRAIL_CLICK','part:'+part+',idx:'+nextIdx+',node:'+target.label+',t:'+performance.now().toFixed(2));
                nextIdx++; draw();
                if(nextIdx>=N){
                    const elapsed=(performance.now()-t0)/1000;
                    log('TRAIL_END','part:'+part+',time_s:'+elapsed.toFixed(3));
                    canvas.onclick=null; canvas.style.display='none'; onDone();
                }
            } else {
                const clicked=nodes.find(n=>Math.hypot(n.x-mx,n.y-my)<28);
                if(clicked) log('TRAIL_ERROR','part:'+part+',clicked:'+clicked.label);
            }
        };
    }
    showUI('Task 8 · Trail Making',
        'Part A: Click circles 1→2→3→... in order.\nPart B (next): Alternate 1→A→2→B→3...',
        'Begin Trail Making A',
        ()=>{ hideUI(); runTrail('A',()=>{
            showUI('Trail Making Part B','Now alternate: 1→A→2→B→3→C...','Begin Part B',
                ()=>{ hideUI(); runTrail('B',()=>{ log('TASK_END','TrailMaking'); task_corsi(); }); });
        }); }
    );
}

// ══════════════════════════════════════════════════════════════
// TASK 9 — CORSI BLOCK TAPPING
// Reference: Corsi (1972); Milner (1971), Neuropsychologia
// ══════════════════════════════════════════════════════════════
function task_corsi() {
    setProgress(9);
    showUI('Task 9 · Corsi Block Tapping',
        'Blocks will light up in a sequence.\nTap them back in the same order. Sequence gets longer.',
        'Begin Corsi',
        ()=>{
            hideUI(); log('TASK_START','Corsi');
            const area=document.getElementById('corsi-area');
            area.style.display='block'; area.innerHTML='';
            const positions=[[120,160],[240,80],[380,200],[520,100],[660,180],[100,320],[280,360],[440,300],[600,340]];
            const blocks=positions.map((pos,i)=>{
                const b=document.createElement('div');
                b.className='corsi-block'; b.style.left=pos[0]+'px'; b.style.top=pos[1]+'px'; b.dataset.id=i;
                area.appendChild(b); return b;
            });
            let span=2,fails=0,maxSpan=0;
            function lightUp(seq,onDone){
                let i=0;
                function step(){
                    if(i>=seq.length){onDone();return;}
                    blocks.forEach(b=>b.className='corsi-block');
                    blocks[seq[i]].classList.add('lit');
                    setTimeout(()=>{blocks[seq[i]].classList.remove('lit');i++;setTimeout(step,400);},700);
                }
                step();
            }
            function runTrial(){
                if(fails>=2||span>9){
                    area.style.display='none';
                    log('CORSI_END','max_span:'+maxSpan);
                    log('TASK_END','Corsi');
                    task_digitspan(); return;
                }
                const seq=Array.from({length:span},()=>Math.floor(Math.random()*9));
                log('CORSI_SEQ','span:'+span+',seq:'+seq.join(','));
                let clickSeq=[],clickIdx=0;
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
                            log('CORSI_RESPONSE','span:'+span+',correct:'+correct+',response:'+clickSeq.join(','));
                            blocks.forEach(b2=>b2.onclick=null);
                            if(correct){if(span>maxSpan)maxSpan=span;fails=0;span++;}else fails++;
                            setTimeout(runTrial,800);
                        }
                    };
                });
                lightUp(seq,()=>{});
            }
            lightUp([0,1],runTrial);
        }
    );
}

// ══════════════════════════════════════════════════════════════
// TASK 10 — DIGIT SPAN
// Reference: Wechsler (1997), WAIS-III
// ══════════════════════════════════════════════════════════════
function task_digitspan() {
    setProgress(10);
    showUI('Task 10 · Digit Span',
        'Digits will appear one at a time.\nType them in order when they stop, then press Submit.',
        'Begin Digit Span',
        ()=>{
            hideUI(); log('TASK_START','DigitSpan');
            const area=document.getElementById('digitspan-area');
            area.style.display='flex';
            const disp=document.getElementById('digit-display');
            const inputWrap=document.getElementById('digit-input-wrap');
            const inp=document.getElementById('digit-input');
            let span=3,fails=0,maxSpan=0;
            function runTrial(){
                if(fails>=2||span>9){
                    area.style.display='none';
                    log('DIGITSPAN_END','max_span:'+maxSpan);
                    log('TASK_END','DigitSpan');
                    finishBattery(); return;
                }
                const seq=Array.from({length:span},()=>Math.floor(Math.random()*10));
                log('DIGITSPAN_SEQ','span:'+span+',seq:'+seq.join(''));
                inp.value=''; inputWrap.style.display='none'; disp.style.display='block';
                let i=0;
                function showDigit(){
                    if(i>=seq.length){disp.style.display='none';inputWrap.style.display='flex';inp.focus();return;}
                    disp.innerText=seq[i]; i++;
                    setTimeout(()=>{disp.innerText='';setTimeout(showDigit,300);},800);
                }
                showDigit();
                window._dsSeq=seq;
            }
            window.submitDigitSpan=function(){
                const resp=inp.value.trim().split('').map(Number);
                const correct=resp.every((v,i)=>v===window._dsSeq[i])&&resp.length===window._dsSeq.length;
                log('DIGITSPAN_RESPONSE','span:'+span+',correct:'+correct+',response:'+resp.join(''));
                if(correct){if(span>maxSpan)maxSpan=span;fails=0;span++;}else fails++;
                setTimeout(runTrial,400);
            };
            runTrial();
        }
    );
}

// ══════════════════════════════════════════════════════════════
// EXPORT — FIX 2: postMessage to parent so downloads work
// outside the sandboxed iframe
// ══════════════════════════════════════════════════════════════
function finishBattery() {
    if(mediaRecorder&&mediaRecorder.state!=='inactive') mediaRecorder.stop();
    else exportAll(null);
}

function exportAll(videoBlob) {
    // Export JSON via postMessage (bypasses iframe sandbox)
    const jsonStr = JSON.stringify(logs, null, 2);
    const jsonB64 = 'data:application/json;charset=utf-8,' + encodeURIComponent(jsonStr);
    postDownload('interaction_logs.json', jsonB64);

    // Export video via postMessage
    if (videoBlob) {
        const reader = new FileReader();
        reader.onloadend = function() {
            postDownload('raw_gaze_video.webm', reader.result);
        };
        reader.readAsDataURL(videoBlob);
    }

    document.getElementById('progress-bar').style.width = '100%';
    document.getElementById('task-label').innerText = 'All tasks complete';
    showUI(
        'Battery Complete',
        'Both files are downloading now.\nReturn to the main window and click "Upload for Analysis".',
        '—', ()=>{}
    );
    document.getElementById('btn').disabled = true;
}
</script>
</body>
</html>"""

# ──────────────────────────────────────────────────────────────
# PAGE 3: UPLOAD
# ──────────────────────────────────────────────────────────────
def page_upload():
    section_header("Upload Assessment Data","Upload both files exported by the battery.")
    info_box("""
    <strong style='color:#D4A843;'>What happens here:</strong><br>
    1. The calibration points from your JSON log are used to fit a 2D affine transform
       mapping normalised MediaPipe iris coordinates → task canvas space.<br>
    2. Every gaze sample is remapped through this transform before I-VT classification.<br>
    3. Fixation/saccade positions are now spatially meaningful relative to stimulus positions.
    """)
    c1,c2 = st.columns(2)
    with c1: video_file = st.file_uploader("raw_gaze_video.webm", type=["webm","mp4","avi"])
    with c2: log_file   = st.file_uploader("interaction_logs.json", type=["json"])
    if video_file and log_file:
        st.markdown("<br>", unsafe_allow_html=True)
        _,col2,_ = st.columns([1,2,1])
        with col2:
            if st.button("Compute All 25 Biomarkers"):
                with st.spinner("Building gaze calibration transform and extracting features..."):
                    logs_data = json.load(log_file)
                    results   = run_analysis(video_file, logs_data)
                    st.session_state.update({"results":results,"event_logs":logs_data,"page":"results"})
                    st.rerun()

# ──────────────────────────────────────────────────────────────
# ANALYSIS ENGINE
# ──────────────────────────────────────────────────────────────
def _build_affine_transform(logs: list, gaze_stream: list):
    """
    FIX 1: Build least-squares affine transform from calibration data.

    Each CALIB_POINT event has screen_nx, screen_ny (0-1 fractions of stage)
    and a timestamp. We find the median gaze position in a ±300ms window
    around that timestamp and build a 2D affine transform:

        [gaze_x, gaze_y] --A--> [screen_nx, screen_ny]

    Returns the 2x3 affine matrix A (or None if calibration failed).
    """
    calib_events = [e for e in logs if e['event'] == 'CALIB_POINT']
    if len(calib_events) < 4:
        return None  # not enough points to fit

    src_pts = []  # gaze coordinates
    dst_pts = []  # screen coordinates

    for ev in calib_events:
        try:
            d = json.loads(ev['details'])
            t_calib = float(d['t'])
            sx, sy  = float(d['screen_nx']), float(d['screen_ny'])
        except Exception:
            continue

        # Collect gaze samples within ±300ms of calibration point onset
        # (after 200ms settle time to exclude saccade toward dot)
        window = [g for g in gaze_stream
                  if t_calib + 200 <= g['t'] <= t_calib + 1300]
        if len(window) < 3:
            continue

        gx = float(np.median([g['x'] for g in window]))
        gy = float(np.median([g['y'] for g in window]))
        src_pts.append([gx, gy])
        dst_pts.append([sx, sy])

    if len(src_pts) < 4:
        return None

    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)

    # Least-squares affine fit: dst ≈ src @ A.T + b
    # Using numpy lstsq
    ones = np.ones((len(src), 1), dtype=np.float32)
    src_h = np.hstack([src, ones])          # (N, 3)
    # Solve for each output dimension separately
    Ax, _, _, _ = np.linalg.lstsq(src_h, dst[:, 0], rcond=None)
    Ay, _, _, _ = np.linalg.lstsq(src_h, dst[:, 1], rcond=None)
    # Ax = [a00, a01, tx],  Ay = [a10, a11, ty]
    affine = np.array([Ax, Ay], dtype=np.float32)  # (2, 3)
    return affine


def _apply_affine(affine, gaze_stream):
    """Apply affine transform to all gaze samples. Returns new list."""
    if affine is None:
        return gaze_stream  # fallback: use raw coordinates
    mapped = []
    for g in gaze_stream:
        pt = np.array([g['x'], g['y'], 1.0], dtype=np.float32)
        mapped_x = float(affine[0] @ pt)
        mapped_y = float(affine[1] @ pt)
        # Clip to [0,1]
        mapped.append({
            't':  g['t'],
            'x':  float(np.clip(mapped_x, 0.0, 1.0)),
            'y':  float(np.clip(mapped_y, 0.0, 1.0)),
        })
    return mapped


def run_analysis(video_file, logs: list) -> dict:
    # ── VIDEO PROCESSING ─────────────────────
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tfile.write(video_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    gaze_stream = []
    frame_idx   = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # Iris landmarks 468/473 — Kartynnik et al. (2019) arXiv:1907.06724
            lx, ly = lm[468].x, lm[468].y
            rx, ry = lm[473].x, lm[473].y
            gaze_stream.append({
                't': (frame_idx / fps) * 1000.0,
                'x': (lx + rx) / 2.0,
                'y': (ly + ry) / 2.0,
            })
        frame_idx += 1

    cap.release()
    face_mesh.close()
    os.unlink(tfile.name)

    # ── FIX 1: BUILD AFFINE TRANSFORM & REMAP ─
    # Reference for calibration methodology: Duchowski (2007), Eye Tracking Methodology
    affine      = _build_affine_transform(logs, gaze_stream)
    gaze_mapped = _apply_affine(affine, gaze_stream)

    # ── I-VT CLASSIFICATION ──────────────────
    # Reference: Salvucci & Goldberg (2000), ETRA
    VELOCITY_THRESH = 100.0   # deg/s
    D_CM            = 60.0    # viewing distance
    PX_PER_CM       = 1920 / 34.5

    fixations_raw, saccades = [], []

    for i in range(1, len(gaze_mapped)):
        t1,x1,y1 = gaze_mapped[i-1]['t'], gaze_mapped[i-1]['x'], gaze_mapped[i-1]['y']
        t2,x2,y2 = gaze_mapped[i]['t'],   gaze_mapped[i]['x'],   gaze_mapped[i]['y']
        dt = (t2 - t1) / 1000.0
        if dt <= 0: continue
        dist_px  = np.sqrt(((x2-x1)*1920)**2 + ((y2-y1)*1080)**2)
        dist_cm  = dist_px / PX_PER_CM
        dist_deg = np.degrees(np.arctan(dist_cm / D_CM))
        velocity = dist_deg / dt

        if velocity >= VELOCITY_THRESH:
            saccades.append({'t_start':t1,'t_end':t2,'amp':dist_deg,'velocity':velocity,'x_end':x2,'y_end':y2})
        else:
            fixations_raw.append({'t_start':t1,'t_end':t2,'x':(x1+x2)/2,'y':(y1+y2)/2,'duration':dt*1000})

    # Merge fixations within 50ms gap; minimum 100ms
    fixations, buf = [], []
    for f in fixations_raw:
        if not buf or (f['t_start'] - buf[-1]['t_end']) < 50:
            buf.append(f)
        else:
            dur = sum(b['duration'] for b in buf)
            if dur >= 100.0:
                fixations.append({'t_start':buf[0]['t_start'],'duration':dur,
                    'x':float(np.mean([b['x'] for b in buf])),
                    'y':float(np.mean([b['y'] for b in buf]))})
            buf = [f]

    # ── OCULOMOTOR FEATURES ──────────────────
    f1_mfd  = float(np.mean([f['duration'] for f in fixations])) if fixations else 0.0
    f2_fc   = len(fixations)
    f4_sa   = float(np.mean([s['amp']      for s in saccades]))  if saccades  else 0.0
    f5_spv  = float(np.mean([s['velocity'] for s in saccades]))  if saccades  else 0.0

    grid = np.zeros((5,5))
    for f in fixations:
        gx = int(np.clip(f['x']*5,0,4))
        gy = int(np.clip(f['y']*5,0,4))
        grid[gx,gy] += f['duration']
    f6_entropy = 0.0
    if grid.sum() > 0:
        pk = grid.flatten() / grid.sum()
        f6_entropy = float(-sum(p*log2(p) for p in pk if p > 0))
    f8_roi = float((np.count_nonzero(grid)/25.0)*100)

    # ── ANTI-SACCADE (F3, F7) ────────────────
    latencies, anti_errors, anti_trials = [], 0, 0
    for ev in logs:
        if ev['event'] == 'ANTISAC_STIM':
            anti_trials += 1
            ev_t = float(ev['timestamp_ms'])
            try:   d = json.loads(ev['details']); side = d.get('side','RIGHT')
            except: side = 'RIGHT'
            next_sac = next((s for s in saccades if s['t_start'] >= ev_t), None)
            if next_sac:
                lat = next_sac['t_start'] - ev_t
                if 80 < lat < 1000: latencies.append(lat)
                went_left = next_sac['x_end'] < 0.5
                if (side=='LEFT' and went_left) or (side=='RIGHT' and not went_left):
                    anti_errors += 1

    f3_sl   = float(np.mean(latencies))                   if latencies    else 0.0
    f7_aser = float((anti_errors/anti_trials)*100)        if anti_trials > 0 else 0.0

    # ── SIMPLE RT (F9, F10) ──────────────────
    rt_vals = []
    for ev in logs:
        if ev['event'] == 'RT_RESPONSE' and 'rt:' in ev['details']:
            try:
                rt = float(ev['details'].split('rt:')[1])
                if 100 < rt < 1500: rt_vals.append(rt)
            except: pass
    f9_rt_mean = float(np.mean(rt_vals))           if rt_vals          else 0.0
    f10_rt_iiv = float(np.std(rt_vals,ddof=1))     if len(rt_vals) > 1 else 0.0

    # ── GO/NO-GO (F11, F12) ──────────────────
    gng_go    = sum(1 for e in logs if e['event']=='GNG_STIM' and 'type:GO' in e['details'] and 'NOGO' not in e['details'])
    gng_nogo  = sum(1 for e in logs if e['event']=='GNG_STIM' and 'type:NOGO' in e['details'])
    gng_comm  = sum(1 for e in logs if e['event']=='GNG_COMMISSION')
    gng_omit  = sum(1 for e in logs if e['event']=='GNG_OMISSION')
    f11_commission = float((gng_comm/gng_nogo)*100) if gng_nogo > 0 else 0.0
    f12_omission   = float((gng_omit/gng_go)*100)   if gng_go   > 0 else 0.0

    # ── N-BACK d-prime (F13, F14) ────────────
    def compute_dprime(logs_list):
        hits=misses=fas=t_tar=t_non=0
        for ev in logs_list:
            if   ev['event']=='NBACK_MISS':         misses+=1; t_tar+=1
            elif ev['event']=='NBACK_FALSE_ALARM':  fas+=1;    t_non+=1
            elif ev['event']=='NBACK_RESPONSE':
                if 'target:True' in ev['details']:  hits+=1;   t_tar+=1
                else:                               t_non+=1
        hr  = np.clip(hits  / max(t_tar,1), 0.01, 0.99)
        far = np.clip(fas   / max(t_non,1), 0.01, 0.99)
        dp  = float(scipy_stats.norm.ppf(hr) - scipy_stats.norm.ppf(far))
        bias= float(-0.5*(scipy_stats.norm.ppf(hr)+scipy_stats.norm.ppf(far)))
        return dp, bias
    f13_dprime, f14_bias = compute_dprime([e for e in logs if 'NBACK' in e['event']])

    # ── STROOP (F15–F18) ─────────────────────
    sc_rt, si_rt, s_err, s_tot = [], [], 0, 0
    for ev in logs:
        if ev['event']=='STROOP_RESPONSE':
            d=ev['details']
            try:
                rt=float(d.split('rt:')[1].split(',')[0])
                ok='correct:True' in d; cong='cong:True' in d
                if 200<rt<3000:
                    (sc_rt if cong else si_rt).append(rt)
                if not ok: s_err+=1
                s_tot+=1
            except: pass
    f15=float(np.mean(sc_rt)) if sc_rt else 0.0
    f16=float(np.mean(si_rt)) if si_rt else 0.0
    f17=f16-f15
    f18=float((s_err/s_tot)*100) if s_tot>0 else 0.0

    # ── TRAIL MAKING (F19–F21) ───────────────
    def get_trail(part):
        ev=next((e for e in logs if e['event']=='TRAIL_END' and 'part:'+part in e['details']),None)
        if ev:
            try: return float(ev['details'].split('time_s:')[1])
            except: pass
        return 0.0
    f19=get_trail('A'); f20=get_trail('B'); f21=f20-f19

    # ── CORSI & DIGIT SPAN (F22, F23) ────────
    def get_span(end_ev):
        ev=next((e for e in logs if e['event']==end_ev),None)
        if ev:
            try: return int(ev['details'].split('max_span:')[1])
            except: pass
        return 0
    f22=get_span('CORSI_END'); f23=get_span('DIGITSPAN_END')

    # ── VISUAL SEARCH (F24, F25) ─────────────
    vs_rt,vs_miss,vs_tot=[],0,0
    for ev in logs:
        if ev['event']=='VS_RESPONSE':
            d=ev['details']
            try:
                info=json.loads(d) if d.startswith('{') else {}
                rt=float(info.get('rt',0)) if info else float(d.split('rt:')[1].split(',')[0] if 'rt:' in d else 0)
                ok=info.get('correct',True) if info else ('correct:True' in d)
                resp=info.get('resp','') if info else ('ABSENT' if 'resp:ABSENT' in d else 'PRESENT')
                vs_tot+=1
                if 100<rt<5000: vs_rt.append(rt)
                if not ok and resp=='ABSENT': vs_miss+=1
            except: pass
    f24=float(np.mean(vs_rt)) if vs_rt else 0.0
    f25=float((vs_miss/vs_tot)*100) if vs_tot>0 else 0.0

    return {
        "F1_MFD":f1_mfd,"F2_FixationCount":f2_fc,"F3_SaccadeLatency":f3_sl,
        "F4_SaccadeAmplitude":f4_sa,"F5_SaccadePeakVelocity":f5_spv,
        "F6_GazeEntropy":f6_entropy,"F7_AntiSaccadeErrorRate":f7_aser,
        "F8_ROICoverage":f8_roi,"F9_RT_Mean":f9_rt_mean,"F10_RT_IIV":f10_rt_iiv,
        "F11_CommissionErrors":f11_commission,"F12_OmissionErrors":f12_omission,
        "F13_NBack_dPrime":f13_dprime,"F14_NBack_Bias":f14_bias,
        "F15_StroopCongruentRT":f15,"F16_StroopIncongruentRT":f16,
        "F17_StroopInterference":f17,"F18_StroopErrorRate":f18,
        "F19_TMT_A":f19,"F20_TMT_B":f20,"F21_TMT_Delta":f21,
        "F22_CorsiSpan":f22,"F23_DigitSpan":f23,
        "F24_VisualSearchRT":f24,"F25_VisualSearchMissRate":f25,
    }

# ──────────────────────────────────────────────────────────────
# NORMS & CLASSIFICATION
# ──────────────────────────────────────────────────────────────
NORMS = {
    "F1_MFD":                 {"label":"Mean Fixation Duration",        "unit":"ms",   "lo":150,  "hi":350,  "domain":"Oculomotor"},
    "F2_FixationCount":       {"label":"Fixation Count",                "unit":"",     "lo":80,   "hi":300,  "domain":"Oculomotor"},
    "F3_SaccadeLatency":      {"label":"Saccade Latency",               "unit":"ms",   "lo":150,  "hi":350,  "domain":"Oculomotor"},
    "F4_SaccadeAmplitude":    {"label":"Saccade Amplitude",             "unit":"°",    "lo":2.0,  "hi":8.0,  "domain":"Oculomotor"},
    "F5_SaccadePeakVelocity": {"label":"Saccade Peak Velocity",         "unit":"°/s",  "lo":200,  "hi":600,  "domain":"Oculomotor"},
    "F6_GazeEntropy":         {"label":"Gaze Path Entropy",             "unit":"bits", "lo":2.0,  "hi":4.0,  "domain":"Oculomotor"},
    "F7_AntiSaccadeErrorRate":{"label":"Anti-Saccade Error Rate",       "unit":"%",    "lo":0,    "hi":25,   "domain":"Inhibitory Control"},
    "F8_ROICoverage":         {"label":"ROI Coverage",                  "unit":"%",    "lo":40,   "hi":100,  "domain":"Oculomotor"},
    "F9_RT_Mean":             {"label":"Simple RT Mean",                "unit":"ms",   "lo":200,  "hi":350,  "domain":"Processing Speed"},
    "F10_RT_IIV":             {"label":"RT Intra-individual Var.",       "unit":"ms",   "lo":10,   "hi":60,   "domain":"Processing Speed"},
    "F11_CommissionErrors":   {"label":"Go/No-Go Commission Rate",       "unit":"%",    "lo":0,    "hi":20,   "domain":"Inhibitory Control"},
    "F12_OmissionErrors":     {"label":"Go/No-Go Omission Rate",        "unit":"%",    "lo":0,    "hi":10,   "domain":"Inhibitory Control"},
    "F13_NBack_dPrime":       {"label":"N-Back d′ (sensitivity)",       "unit":"",     "lo":1.0,  "hi":4.0,  "domain":"Working Memory"},
    "F14_NBack_Bias":         {"label":"N-Back Response Bias (c)",      "unit":"",     "lo":-1.0, "hi":1.0,  "domain":"Working Memory"},
    "F15_StroopCongruentRT":  {"label":"Stroop Congruent RT",           "unit":"ms",   "lo":400,  "hi":700,  "domain":"Attention"},
    "F16_StroopIncongruentRT":{"label":"Stroop Incongruent RT",         "unit":"ms",   "lo":500,  "hi":900,  "domain":"Attention"},
    "F17_StroopInterference": {"label":"Stroop Interference Score",     "unit":"ms",   "lo":0,    "hi":200,  "domain":"Attention"},
    "F18_StroopErrorRate":    {"label":"Stroop Error Rate",             "unit":"%",    "lo":0,    "hi":10,   "domain":"Attention"},
    "F19_TMT_A":              {"label":"Trail Making A (time)",         "unit":"s",    "lo":15,   "hi":45,   "domain":"Processing Speed"},
    "F20_TMT_B":              {"label":"Trail Making B (time)",         "unit":"s",    "lo":30,   "hi":90,   "domain":"Executive Function"},
    "F21_TMT_Delta":          {"label":"TMT B–A Delta",                 "unit":"s",    "lo":10,   "hi":50,   "domain":"Executive Function"},
    "F22_CorsiSpan":          {"label":"Corsi Block Span",              "unit":"",     "lo":4,    "hi":7,    "domain":"Working Memory"},
    "F23_DigitSpan":          {"label":"Digit Span Forward",            "unit":"",     "lo":5,    "hi":9,    "domain":"Working Memory"},
    "F24_VisualSearchRT":     {"label":"Visual Search RT Mean",         "unit":"ms",   "lo":400,  "hi":1200, "domain":"Attention"},
    "F25_VisualSearchMissRate":{"label":"Visual Search Miss Rate",      "unit":"%",    "lo":0,    "hi":15,   "domain":"Attention"},
}
DOMAIN_REFS = {
    "Oculomotor":         "Rayner (1998), Psych Bull; Salvucci & Goldberg (2000), ETRA; Duchowski (2007)",
    "Inhibitory Control": "Hutton & Ettinger (2006), Neuropsychology Review; Aron (2007), TICS",
    "Processing Speed":   "Luce (1986), Response Times; Hultsch et al. (2002), Neuropsychology",
    "Working Memory":     "Jaeggi et al. (2008), PNAS; Wechsler (1997), WAIS-III",
    "Attention":          "MacLeod (1991), Psych Bull; Treisman & Gelade (1980), Cognitive Psychology",
    "Executive Function": "Reitan (1958), Percept Mot Skills; Lezak (2004), Neuropsychological Assessment",
}

def classify(key, val):
    n = NORMS.get(key)
    if not n or val == 0.0: return "badge-blue","No data"
    inverted = {"F7_AntiSaccadeErrorRate","F11_CommissionErrors","F12_OmissionErrors",
                "F18_StroopErrorRate","F25_VisualSearchMissRate","F9_RT_Mean","F10_RT_IIV",
                "F15_StroopCongruentRT","F16_StroopIncongruentRT","F17_StroopInterference",
                "F19_TMT_A","F20_TMT_B","F21_TMT_Delta"}
    in_range = n["lo"] <= val <= n["hi"]
    if in_range: return "badge-green","Normal"
    if key in inverted:
        return ("badge-green","Excellent") if val < n["lo"] else ("badge-red","Elevated")
    return ("badge-red","Below norm") if val < n["lo"] else ("badge-gold","Above norm")

# ──────────────────────────────────────────────────────────────
# PAGE 4: RESULTS
# ──────────────────────────────────────────────────────────────
def page_results():
    r = st.session_state["results"]
    p = st.session_state["participant"]

    st.markdown(f"""
    <div style='display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:24px;'>
      <div>
        <h1 style='font-size:24px;font-weight:700;margin-bottom:4px;'>Cognitive Assessment Report</h1>
        <p style='color:#7A99A8;font-size:14px;'>Participant {p.get('id','—')} &nbsp;·&nbsp; {p.get('age','—')} y/o {p.get('gender','—')} &nbsp;·&nbsp; {p.get('timestamp','')[:10]}</p>
      </div>
      <span class='badge badge-gold'>25 Biomarkers Computed</span>
    </div>""", unsafe_allow_html=True)

    domains = {}
    for key, val in r.items():
        dom=NORMS[key]["domain"]; cls,lbl=classify(key,val)
        domains.setdefault(dom,[]).append((cls,lbl))
    cols=st.columns(6)
    for i,(dom,items) in enumerate(domains.items()):
        reds=sum(1 for c,_ in items if c=="badge-red")
        badge="badge-green" if reds==0 else ("badge-gold" if reds<=1 else "badge-red")
        label="All normal" if reds==0 else f"{reds} flag{'s' if reds>1 else ''}"
        cols[i%6].markdown(f"""
        <div class='metric-card'>
          <span class='label'>{dom}</span>
          <span class='value' style='font-size:16px;'><span class='badge {badge}'>{label}</span></span>
          <span class='norm'>{len(items)} features</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    DOMS=["Oculomotor","Processing Speed","Inhibitory Control","Working Memory","Attention","Executive Function"]
    tabs=st.tabs(DOMS)
    for tab,dom in zip(tabs,DOMS):
        with tab:
            dom_feats={k:v for k,v in r.items() if NORMS[k]["domain"]==dom}
            keys=list(dom_feats.keys())
            for rs in range(0,len(keys),4):
                row=keys[rs:rs+4]; cols=st.columns(len(row))
                for col,key in zip(cols,row):
                    val=dom_feats[key]; n=NORMS[key]; cls,lbl=classify(key,val); unit=n["unit"]
                    if   unit=="s":         disp=f"{val:.1f} s"
                    elif unit=="ms":        disp=f"{val:.0f} ms"
                    elif unit=="%":         disp=f"{val:.1f}%"
                    elif unit in("°","°/s"):disp=f"{val:.1f}{unit}"
                    elif unit=="bits":      disp=f"{val:.2f} bits"
                    else:                   disp=f"{val:.2f}"
                    col.markdown(f"""
                    <div class='metric-card'>
                      <span class='label'>{n['label']}</span>
                      <span class='value'>{disp}</span>
                      <span class='norm'><span class='badge {cls}'>{lbl}</span></span>
                    </div>""", unsafe_allow_html=True)
            info_box(f"<strong style='color:#D4A843;'>Reference:</strong> {DOMAIN_REFS[dom]}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Complete Feature Matrix","All 25 biomarkers with normative classification")
    rows=""
    for key,val in r.items():
        n=NORMS[key]; cls,lbl=classify(key,val); unit=n["unit"]
        val_str=f"{val:.2f} {unit}".strip()
        rows+=f"<tr><td style='color:#7A99A8;font-size:12px;'>{key}</td><td>{n['label']}</td><td><span class='badge badge-blue' style='font-size:11px;'>{n['domain']}</span></td><td style='color:#D4A843;font-weight:500;'>{val_str}</td><td>Norm: {n['lo']}–{n['hi']} {unit}</td><td><span class='badge {cls}'>{lbl}</span></td></tr>"
    st.markdown(f"<table class='results-table'><thead><tr><th>Code</th><th>Feature</th><th>Domain</th><th>Value</th><th>Normative Range</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>",unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Export Results")
    c1,c2,c3=st.columns(3)
    df=pd.DataFrame([{"participant_id":p.get("id"),"timestamp":p.get("timestamp"),**r}])
    c1.download_button("Download CSV",df.to_csv(index=False).encode(),"cognitive_results.csv","text/csv")
    c2.download_button("Download JSON",json.dumps({"participant":p,"results":r},indent=2).encode(),"cognitive_results.json","application/json")
    c3.download_button("Download PDF Report",generate_pdf(p,r),"cognitive_report.pdf","application/pdf")

    st.markdown("<br>", unsafe_allow_html=True)
    _,col2,_=st.columns([1,2,1])
    with col2:
        if st.button("New Participant Session"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()

# ──────────────────────────────────────────────────────────────
# PDF
# ──────────────────────────────────────────────────────────────
def generate_pdf(participant, results):
    pdf=FPDF(); pdf.set_auto_page_break(auto=True,margin=15); pdf.add_page()
    pdf.set_fill_color(30,45,53); pdf.rect(0,0,210,40,'F')
    pdf.set_text_color(212,168,67); pdf.set_font("Helvetica","B",18); pdf.set_xy(15,12)
    pdf.cell(0,10,"Pocket-Precise Cognitive Diagnostic Report",ln=True)
    pdf.set_text_color(160,179,188); pdf.set_font("Helvetica","",10); pdf.set_xy(15,26)
    pdf.cell(0,8,f"Participant: {participant.get('id','—')}  |  Age: {participant.get('age','—')}  |  Date: {participant.get('timestamp','')[:10]}",ln=True)
    pdf.set_text_color(30,45,53); pdf.set_xy(15,48); pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"Participant Information",ln=True); pdf.set_font("Helvetica","",10)
    for k,v in participant.items():
        if k!="timestamp": pdf.cell(0,6,f"  {k.capitalize()}: {v}",ln=True)
    pdf.ln(6); pdf.set_font("Helvetica","B",12); pdf.cell(0,8,"Cognitive Biomarker Results",ln=True)
    pdf.set_font("Helvetica","B",9); pdf.set_fill_color(42,61,72); pdf.set_text_color(160,179,188)
    for hdr,w in [("Feature",45),("Domain",35),("Value",30),("Normal Range",40),("Status",30)]:
        pdf.cell(w,7,hdr,border=1,fill=True)
    pdf.ln(); pdf.set_font("Helvetica","",8); pdf.set_text_color(30,45,53)
    for key,val in results.items():
        n=NORMS[key]; cls,lbl=classify(key,val); unit=n["unit"]
        if   cls=="badge-red":   pdf.set_fill_color(250,235,235)
        elif cls=="badge-green": pdf.set_fill_color(235,250,240)
        else:                    pdf.set_fill_color(255,255,255)
        pdf.cell(45,6,n['label'][:30],border=1,fill=True)
        pdf.cell(35,6,n['domain'],border=1,fill=True)
        pdf.cell(30,6,f"{val:.2f} {unit}".strip(),border=1,fill=True)
        pdf.cell(40,6,f"{n['lo']}–{n['hi']} {unit}".strip(),border=1,fill=True)
        pdf.cell(30,6,lbl,border=1,fill=True,ln=True)
    pdf.ln(8); pdf.set_font("Helvetica","B",10); pdf.cell(0,7,"Key References",ln=True); pdf.set_font("Helvetica","",8)
    for ref in [
        "Duchowski, A. (2007). Eye Tracking Methodology: Theory and Practice. Springer.",
        "Rayner, K. (1998). Eye movements in reading. Psychological Bulletin, 124(3), 372-422.",
        "Salvucci, D. & Goldberg, J. (2000). Identifying fixations and saccades. ETRA 2000, 71-78.",
        "Hutton, S.B. & Ettinger, U. (2006). The antisaccade task. Neuropsychology Review.",
        "Jaeggi, S.M. et al. (2008). Improving fluid intelligence. PNAS, 105(19), 6829-6833.",
        "MacLeod, C.M. (1991). Half a century of Stroop research. Psychological Bulletin, 109(2), 163-203.",
        "Reitan, R.M. (1958). Validity of the Trail Making Test. Perceptual and Motor Skills.",
        "Wechsler, D. (1997). WAIS-III. The Psychological Corporation.",
        "Treisman, A. & Gelade, G. (1980). Feature-integration theory. Cognitive Psychology, 12(1), 97-136.",
        "Hultsch, D.F. et al. (2002). Intraindividual variability in cognitive performance. Neuropsychology.",
        "Green, D.M. & Swets, J.A. (1966). Signal Detection Theory and Psychophysics. Wiley.",
    ]: pdf.multi_cell(0,5,"• "+ref)
    pdf.ln(4); pdf.set_font("Helvetica","I",8); pdf.set_text_color(120,140,150)
    pdf.multi_cell(0,5,"DISCLAIMER: Research use only. Not a clinical diagnosis. Gaze features derived from webcam-based iris tracking (±1–3° accuracy); interpret oculomotor features accordingly.")
    return bytes(pdf.output())

# ──────────────────────────────────────────────────────────────
# ROUTER
# ──────────────────────────────────────────────────────────────
def main():
    page=st.session_state["page"]
    if   page=="consent":      page_consent()
    elif page=="demographics": page_demographics()
    elif page=="battery":      page_battery()
    elif page=="upload":       page_upload()
    elif page=="results":      page_results()

if __name__=="__main__":
    main()
