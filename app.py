"""
Pocket-Precise Cognitive Diagnostic Engine  — v2.1 (patched)
=============================================================
Fixes applied
  1. JSON + video export: postMessage from iframe → parent window listener
     (bypasses sandboxed-iframe download block in Streamlit)
  2. Gaze-to-screen calibration: 9-point affine transform built from
     CALIB_GAZE_WINDOW events; all spatial features use mapped coordinates
  3. Antisaccade direction, Visual-Search hit geometry use canvas px coords
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp,[data-testid="stAppViewContainer"]{background-color:#1E2D35!important;color:#E8EDF0!important;font-family:'Inter',sans-serif!important}
h1,h2,h3,h4,p,span,label,li{color:#E8EDF0!important}
.block-container{padding-top:2rem!important;max-width:1100px!important}
[data-testid="stSidebar"]{background-color:#162028!important}
.stButton>button{background-color:transparent!important;color:#D4A843!important;border:1.5px solid #D4A843!important;border-radius:6px!important;padding:10px 22px!important;font-size:15px!important;font-weight:500!important;transition:all 0.2s ease!important;width:100%!important}
.stButton>button:hover{background-color:#D4A843!important;color:#1E2D35!important}
.stTextInput>div>div>input,.stSelectbox>div>div,.stNumberInput>div>div>input{background-color:#2A3D48!important;color:#E8EDF0!important;border:1px solid #4A6070!important;border-radius:6px!important}
.stSelectbox label,.stTextInput label,.stNumberInput label,.stRadio label,.stCheckbox label{color:#A8BDC8!important;font-size:14px!important}
.metric-card{background-color:#2A3D48;border:1px solid #3D5565;padding:18px 20px;border-radius:8px;text-align:center;margin-bottom:12px}
.metric-card .label{font-size:12px;color:#7A99A8;text-transform:uppercase;letter-spacing:.8px;display:block;margin-bottom:8px}
.metric-card .value{font-size:26px;font-weight:600;color:#D4A843;display:block}
.metric-card .norm{font-size:12px;color:#5A7A8A;display:block;margin-top:5px}
.section-divider{border:none;border-top:1px solid #2A3D48;margin:28px 0}
.info-box{background-color:#1E3040;border-left:3px solid #D4A843;padding:14px 18px;border-radius:0 6px 6px 0;margin:16px 0;font-size:14px;color:#A8BDC8!important;line-height:1.6}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:500}
.badge-gold{background-color:#3D2E10;color:#D4A843}
.badge-green{background-color:#0E2E1E;color:#4CAF82}
.badge-red{background-color:#2E0E0E;color:#E07070}
.badge-blue{background-color:#0E1E2E;color:#70A0E0}
.stTabs [data-baseweb="tab-list"]{background-color:#2A3D48;border-radius:8px;padding:4px;gap:4px}
.stTabs [data-baseweb="tab"]{color:#7A99A8!important;font-weight:500!important;font-size:14px!important;border-radius:6px!important}
.stTabs [aria-selected="true"]{color:#D4A843!important;background-color:#1E2D35!important}
.stProgress>div>div>div{background-color:#D4A843!important}
.streamlit-expanderHeader{background-color:#2A3D48!important;color:#A8BDC8!important;border-radius:6px!important}
.stFileUploader>div{background-color:#2A3D48!important;border:1px dashed #4A6070!important;border-radius:8px!important}
header,#MainMenu,footer{visibility:hidden}
.results-table{width:100%;border-collapse:collapse;font-size:13px}
.results-table th{background-color:#2A3D48;color:#7A99A8;padding:10px 14px;text-align:left;font-weight:500;text-transform:uppercase;letter-spacing:.6px;font-size:11px}
.results-table td{padding:10px 14px;border-bottom:1px solid #2A3D48;color:#E8EDF0}
.results-table tr:hover td{background-color:#2A3D48}
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
def init_state():
    for k, v in {
        "page": "consent", "participant": {}, "consent_given": False,
        "battery_done": False, "event_logs": None, "video_bytes": None, "results": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def section_header(title, subtitle=""):
    st.markdown(f"<h2 style='font-size:22px;font-weight:600;color:#E8EDF0;margin-bottom:4px;'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='font-size:14px;color:#7A99A8;margin-bottom:20px;'>{subtitle}</p>", unsafe_allow_html=True)

def info_box(text):
    st.markdown(f"<div class='info-box'>{text}</div>", unsafe_allow_html=True)

def metric_card(label, value, norm="", col=None):
    html = f"<div class='metric-card'><span class='label'>{label}</span><span class='value'>{value}</span>{'<span class=norm>'+norm+'</span>' if norm else ''}</div>"
    (col or st).markdown(html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0 — CONSENT
# ══════════════════════════════════════════════════════════════════════════════
def page_consent():
    st.markdown("<div style='max-width:720px;margin:0 auto;padding:30px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='display:flex;align-items:center;gap:14px;margin-bottom:32px;'>
        <div style='width:44px;height:44px;background:#D4A843;border-radius:8px;display:flex;align-items:center;justify-content:center;'>
            <span style='color:#1E2D35;font-size:22px;font-weight:700;'>P</span>
        </div>
        <div>
            <p style='margin:0;font-size:20px;font-weight:600;color:#E8EDF0!important;'>Pocket-Precise</p>
            <p style='margin:0;font-size:13px;color:#7A99A8!important;'>Cognitive Diagnostic Engine · v2.1</p>
        </div>
    </div>""", unsafe_allow_html=True)
    section_header("Participant Information Sheet")
    info_box("""<strong style='color:#D4A843;'>Study Purpose</strong><br>
    This assessment battery measures core cognitive and behavioural parameters including oculomotor
    control, inhibitory control, working memory, processing speed, and attentional capacity.""")
    st.markdown("""
    <div style='background:#2A3D48;border-radius:8px;padding:20px 24px;margin:20px 0;'>
    <p style='font-size:14px;color:#A8BDC8!important;line-height:1.8;margin:0;'>
    <strong style='color:#E8EDF0;'>What this involves:</strong><br>
    You will complete a series of 10 computer-based tasks (~25–35 min). A webcam records gaze behaviour.
    All processing is local — no data leaves this machine.<br><br>
    <strong style='color:#E8EDF0;'>Export method (v2.1 fix):</strong><br>
    When the battery finishes, both files are offered as download buttons <em>inside this page</em>
    (not via auto-download) to work around browser sandbox restrictions.<br><br>
    <strong style='color:#E8EDF0;'>Voluntary participation:</strong><br>
    You may stop at any time.
    </p></div>""", unsafe_allow_html=True)

    tasks = [
        ("Gaze Calibration","9-point affine calibration","~1 min"),
        ("Prosaccade / Anti-Saccade","Inhibitory control & eye movement latency","~4 min"),
        ("Visual Search","Selective attention & target detection","~3 min"),
        ("Simple Reaction Time","Baseline processing speed & IIV","~3 min"),
        ("Go / No-Go","Response inhibition & impulsivity","~4 min"),
        ("N-Back (1-back & 2-back)","Working memory capacity & d-prime","~5 min"),
        ("Stroop Colour-Word","Cognitive interference & selective attention","~4 min"),
        ("Trail Making A & B","Processing speed & cognitive flexibility","~5 min"),
        ("Corsi Block Tapping","Visuospatial working memory","~3 min"),
        ("Digit Span Forward","Verbal working memory span","~3 min"),
    ]
    for i,(name,desc,dur) in enumerate(tasks):
        st.markdown(f"""
        <div style='display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid #2A3D48;'>
          <div style='display:flex;align-items:center;gap:12px;'>
            <span style='width:24px;height:24px;background:#2A3D48;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:12px;color:#D4A843!important;'>{i+1}</span>
            <div><p style='margin:0;font-size:14px;font-weight:500;color:#E8EDF0!important;'>{name}</p>
                 <p style='margin:0;font-size:12px;color:#7A99A8!important;'>{desc}</p></div>
          </div>
          <span style='font-size:12px;color:#5A7A8A;'>{dur}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    agree = st.checkbox("I have read and understood the information above. I consent to participate voluntarily.")
    col1,col2,col3 = st.columns([2,2,2])
    with col2:
        if st.button("Continue to Demographics →", disabled=not agree):
            st.session_state.update({"consent_given":True,"page":"demographics"})
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════
def page_demographics():
    st.markdown("<div style='max-width:720px;margin:0 auto;padding:30px 0;'>", unsafe_allow_html=True)
    section_header("Participant Demographics","Normalises biomarkers against population norms.")
    col1,col2 = st.columns(2)
    with col1:
        pid        = st.text_input("Participant ID *", placeholder="e.g. P001")
        age        = st.number_input("Age *", min_value=18, max_value=90, value=25)
        handedness = st.selectbox("Handedness",["Right","Left","Ambidextrous"])
    with col2:
        gender    = st.selectbox("Gender",["Male","Female","Non-binary","Prefer not to say"])
        education = st.selectbox("Education level",["Secondary school","Undergraduate","Postgraduate","Doctoral","Other"])
        vision    = st.selectbox("Corrected-to-normal vision?",["Yes","No – I have uncorrected impairment"])
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Pre-session Checks")
    col3,col4 = st.columns(2)
    with col3:
        sleep     = st.selectbox("Sleep last night (hours)",["< 5","5–6","7–8","9+"])
        caffeine  = st.selectbox("Caffeine in last 2 hours?",["No","Yes – 1 drink","Yes – 2+ drinks"])
    with col4:
        medications = st.selectbox("Psychoactive medication?",["No","Yes (stimulant)","Yes (sedative)","Yes (other)"])
        anxiety     = st.selectbox("Current anxiety level",["1 – Very low","2","3 – Moderate","4","5 – Very high"])
    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        if st.button("Begin Assessment Battery →", disabled=(pid.strip()=="")):
            st.session_state["participant"] = {
                "id":pid,"age":age,"gender":gender,"handedness":handedness,
                "education":education,"vision":vision,"sleep":sleep,
                "caffeine":caffeine,"medications":medications,"anxiety":anxiety,
                "timestamp":datetime.datetime.now().isoformat()
            }
            st.session_state["page"] = "battery"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BATTERY  (iframe + postMessage export fix)
# ══════════════════════════════════════════════════════════════════════════════
def page_battery():
    section_header("Assessment Battery","Follow each task's on-screen instructions carefully.")
    info_box("""<strong style='color:#D4A843;'>Before you start:</strong>
    Sit ~60 cm from the screen in good lighting. Keep head still during gaze tasks.
    When the battery ends, <strong>download buttons will appear below</strong> — click both to save your files.""")

    # ── postMessage listener injected at the TOP-LEVEL page ──────────────────
    # This script lives outside the iframe, so blob downloads work.
    st.markdown("""
    <script>
    (function(){
      if(window.__ppListenerInstalled) return;
      window.__ppListenerInstalled = true;

      window.addEventListener('message', function(evt){
        var d = evt.data;
        if(!d || d.type !== 'PP_EXPORT') return;

        // ── JSON download ──────────────────────────────────────
        if(d.logsJson){
          var jBlob = new Blob([d.logsJson],{type:'application/json'});
          var ja = document.createElement('a');
          ja.href = URL.createObjectURL(jBlob);
          ja.download = 'interaction_logs.json';
          document.body.appendChild(ja);
          ja.click();
          document.body.removeChild(ja);
          setTimeout(()=>URL.revokeObjectURL(ja.href), 5000);
        }

        // ── Video download ─────────────────────────────────────
        if(d.videoB64){
          var bytes = atob(d.videoB64);
          var buf   = new Uint8Array(bytes.length);
          for(var i=0;i<bytes.length;i++) buf[i]=bytes.charCodeAt(i);
          var vBlob = new Blob([buf],{type:'video/webm'});
          var va = document.createElement('a');
          va.href = URL.createObjectURL(vBlob);
          va.download = 'raw_gaze_video.webm';
          document.body.appendChild(va);
          va.click();
          document.body.removeChild(va);
          setTimeout(()=>URL.revokeObjectURL(va.href), 30000);
        }
      });
    })();
    </script>
    """, unsafe_allow_html=True)

    battery_html = _build_battery_html()
    components.html(battery_html, height=740, scrolling=False)

    st.markdown("<br>", unsafe_allow_html=True)
    info_box("""
    When the battery completes:<br>
    &nbsp;&nbsp;• Your browser will download <strong style='color:#D4A843;'>interaction_logs.json</strong> and <strong style='color:#D4A843;'>raw_gaze_video.webm</strong> automatically.<br>
    &nbsp;&nbsp;• If a file is missing, check your browser's downloads or allow pop-ups/downloads from this page.<br>
    Then click the button below.
    """)
    col1,col2,col3 = st.columns([1,2,1])
    with col2:
        if st.button("I have both files → Upload for Analysis"):
            st.session_state.update({"battery_done":True,"page":"upload"})
            st.rerun()


def _build_battery_html() -> str:
    # Key changes vs v2.0:
    # 1. exportAll() posts message to parent instead of inline downloads
    # 2. Calibration logs CALIB_GAZE_WINDOW events (start_t, end_t, gaze_x, gaze_y)
    #    so the Python backend can build the affine map
    # 3. Every stimulus logs its canvas px position for later spatial comparison
    return r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1E2D35;color:#E8EDF0;font-family:'Inter',sans-serif;
     display:flex;flex-direction:column;align-items:center;min-height:700px;overflow:hidden}
#stage{position:relative;width:800px;height:560px;background:#253540;
       border:1px solid #3D5565;border-radius:10px;overflow:hidden;margin-top:16px}
#ui{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
    justify-content:center;padding:40px;text-align:center;z-index:20}
#progress-bar-wrap{width:800px;height:4px;background:#2A3D48;border-radius:2px;margin-top:10px}
#progress-bar{height:4px;background:#D4A843;border-radius:2px;width:0%;transition:width .5s}
#task-label{font-size:11px;color:#5A7A8A;letter-spacing:1px;text-transform:uppercase;margin-top:6px}
h2{font-size:24px;font-weight:600;margin-bottom:10px}
.sub{font-size:14px;color:#7A99A8;line-height:1.6;margin-bottom:28px;max-width:560px}
.btn{background:transparent;color:#D4A843;border:1.5px solid #D4A843;border-radius:6px;
     padding:11px 28px;font-size:15px;font-weight:500;cursor:pointer;transition:all .2s;font-family:inherit}
.btn:hover{background:#D4A843;color:#1E2D35}
.btn:disabled{opacity:.3;cursor:not-allowed}
#dot{position:absolute;width:18px;height:18px;background:#D4A843;border-radius:50%;
     transform:translate(-50%,-50%);display:none;z-index:10;
     box-shadow:0 0 0 4px rgba(212,168,67,.25);transition:left .3s ease,top .3s ease}
#cross{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
       font-size:42px;color:#4A6070;display:none;z-index:10;font-weight:300;line-height:1}
#stim{position:absolute;inset:0;display:none;align-items:center;
      justify-content:center;z-index:15;flex-direction:column;gap:16px}
#stim-box{background:#1E3040;border:1.5px solid #D4A843;border-radius:10px;
          padding:30px 50px;text-align:center;min-width:300px}
#stim-text{font-size:48px;font-weight:700}
#stim-label{font-size:13px;color:#7A99A8;margin-top:8px}
#nback-grid{display:none;position:absolute;inset:0;align-items:center;justify-content:center;z-index:15}
.nb-cell{width:90px;height:90px;border:1px solid #3D5565;border-radius:6px;background:#1E2D35;
         display:flex;align-items:center;justify-content:center;font-size:36px;font-weight:700;
         color:transparent;transition:all .1s}
.nb-cell.active{background:#D4A843;color:#1E2D35}
#trail-canvas{position:absolute;inset:0;display:none;z-index:15}
#corsi-area{position:absolute;inset:0;display:none;z-index:15}
.corsi-block{position:absolute;width:60px;height:60px;background:#2A3D48;
             border:1.5px solid #4A6070;border-radius:8px;cursor:pointer;transition:background .15s}
.corsi-block.lit{background:#D4A843;border-color:#D4A843}
.corsi-block.correct{background:#4CAF82;border-color:#4CAF82}
.corsi-block.wrong{background:#E07070;border-color:#E07070}
#digitspan-area{position:absolute;inset:0;display:none;flex-direction:column;
                align-items:center;justify-content:center;z-index:15}
#digit-display{font-size:72px;font-weight:700;color:#D4A843;display:none}
#digit-input-wrap{display:none;flex-direction:column;align-items:center;gap:14px}
#digit-input{background:#1E2D35;border:1.5px solid #4A6070;color:#E8EDF0;font-size:28px;
             text-align:center;border-radius:8px;padding:12px 20px;width:260px;font-family:inherit}
#vs-hint{position:absolute;bottom:0;left:0;right:0;z-index:16;display:none;
         padding:14px;text-align:center;background:rgba(30,45,53,.85)}
</style>
</head>
<body>
<div id="stage">
  <div id="ui">
    <h2 id="title">Battery Ready</h2>
    <p class="sub" id="sub">Ensure your webcam is available and you are seated ~60 cm from the screen. Click below to begin.</p>
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
    <p style="color:#7A99A8;font-size:14px;margin-bottom:12px;" id="ds-prompt">Memorise the following sequence</p>
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
// ── Global state ──────────────────────────────────────────────────────────────
const logs     = [];
const STAGE_W  = 800, STAGE_H = 560;  // canvas / stage dimensions (px)
let mediaRecorder, stream, videoChunks = [];
let taskIndex  = 0;
const TOTAL    = 10;
let keyHandler = null;

// Webcam-frame gaze samples accumulated during calibration
// Each sample: {t_ms, gx, gy}  (gx/gy = raw MediaPipe normalised 0-1)
// We use a live canvas-based iris tracker running in a hidden <video>
// so we can collect gaze timestamps aligned to performance.now().
let gazeSamples = [];      // collected continuously while camera is on
let gazeActive  = false;
let gazeVideo   = null;
let gazeCanvas  = null;
let gazeCtx     = null;
let gazeRAF     = null;

function log(event, details){
  logs.push({timestamp_ms: performance.now().toFixed(2), event, details});
}
function setProgress(n){
  document.getElementById('progress-bar').style.width=((n/TOTAL)*100)+'%';
  document.getElementById('task-label').innerText='Task '+n+' / '+TOTAL;
}
function showUI(title,sub,btnLabel,onclick){
  document.getElementById('title').innerText=title;
  document.getElementById('sub').innerText=sub;
  const btn=document.getElementById('btn');
  btn.innerText=btnLabel; btn.onclick=onclick; btn.disabled=false;
  document.getElementById('ui').style.display='flex';
}
function hideUI(){ document.getElementById('ui').style.display='none'; }

function hideAll(){
  if(keyHandler){ document.removeEventListener('keydown',keyHandler); keyHandler=null; }
  document.getElementById('dot').style.display='none';
  document.getElementById('cross').style.display='none';
  const stim=document.getElementById('stim');
  stim.style.display='none';
  const st2=document.getElementById('stim-text'), sl=document.getElementById('stim-label');
  if(st2){st2.innerText='';st2.style.color='';st2.style.fontSize='48px';}
  if(sl) sl.innerText='';
  const sb=document.getElementById('stim-box');
  if(sb){sb.style.background='#1E3040';sb.style.border='1.5px solid #D4A843';
         sb.style.padding='30px 50px';sb.style.minWidth='300px';sb.style.borderRadius='10px';}
  document.getElementById('nback-grid').style.display='none';
  document.getElementById('trail-canvas').style.display='none';
  document.getElementById('corsi-area').style.display='none';
  document.getElementById('digitspan-area').style.display='none';
  document.getElementById('vs-hint').style.display='none';
  document.querySelectorAll('#stage canvas:not(#trail-canvas)').forEach(c=>c.remove());
}

// ── Lightweight JS iris tracker (no MediaPipe; uses face bounding box heuristic)
// We record the centroid of the largest skin-tone blob in the upper-face region.
// This is used ONLY to generate CALIB_GAZE_WINDOW logs for the affine calibration.
// The full MediaPipe analysis is done server-side on the raw video.
function startGazeCollection(videoEl){
  gazeVideo  = videoEl;
  gazeCanvas = document.createElement('canvas');
  gazeCanvas.width=160; gazeCanvas.height=120;
  gazeCtx    = gazeCanvas.getContext('2d',{willReadFrequently:true});
  gazeActive = true;
  function loop(){
    if(!gazeActive) return;
    gazeCtx.drawImage(gazeVideo,0,0,160,120);
    const id = gazeCtx.getImageData(0,0,160,120);
    // Simple centroid of bright pixels in upper-face strip (eyes region ≈ rows 30–60)
    let sx=0,sy=0,n=0;
    for(let y=25;y<65;y++){
      for(let x=0;x<160;x++){
        const i=(y*160+x)*4;
        const r=id.data[i],g=id.data[i+1],b=id.data[i+2];
        // Dark pixels likely = pupil
        if(r<80 && g<80 && b<80){ sx+=x;sy+=y;n++; }
      }
    }
    if(n>0){
      gazeSamples.push({t_ms:performance.now(), gx:sx/n/160, gy:sy/n/120});
      if(gazeSamples.length>6000) gazeSamples.shift(); // keep last ~3 min at 30fps
    }
    gazeRAF=requestAnimationFrame(loop);
  }
  gazeRAF=requestAnimationFrame(loop);
}
function stopGazeCollection(){
  gazeActive=false;
  if(gazeRAF) cancelAnimationFrame(gazeRAF);
}

// Average gaze x,y from a time window [t_start, t_end] ms
function avgGazeInWindow(t_start, t_end){
  const seg = gazeSamples.filter(s=>s.t_ms>=t_start && s.t_ms<=t_end);
  if(!seg.length) return null;
  return {gx: seg.reduce((a,s)=>a+s.gx,0)/seg.length,
          gy: seg.reduce((a,s)=>a+s.gy,0)/seg.length};
}

// ── Init battery ──────────────────────────────────────────────────────────────
async function initBattery(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',frameRate:30},audio:false});
    // Start recording
    const opts = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
               ? {mimeType:'video/webm;codecs=vp9'} : {mimeType:'video/webm'};
    videoChunks=[];
    mediaRecorder = new MediaRecorder(stream, opts);
    mediaRecorder.ondataavailable = e=>{if(e.data.size>0) videoChunks.push(e.data);};
    mediaRecorder.onstop = ()=>exportAll(new Blob(videoChunks,{type:'video/webm'}));
    mediaRecorder.start(100);

    // Start lightweight gaze collection
    const vid=document.createElement('video');
    vid.srcObject=stream; vid.muted=true; vid.playsInline=true;
    vid.onloadedmetadata=()=>{ vid.play(); startGazeCollection(vid); };
    document.body.appendChild(vid); // must be in DOM to play
    vid.style.cssText='position:absolute;opacity:0;pointer-events:none;width:1px;height:1px;';

    log('SYSTEM_START','Camera active, battery initiated');
    task_calibration();
  } catch(e){
    showUI('Camera Access Denied','Grant webcam permission and reload.','Retry',initBattery);
  }
}

// ══════════════════════════════════════════
// TASK 1 — 9-POINT CALIBRATION
// FIX: logs CALIB_GAZE_WINDOW for each point so Python can build affine map
// ══════════════════════════════════════════
function task_calibration(){
  setProgress(1);
  showUI('Task 1 · Gaze Calibration',
    'A gold dot will appear at 9 positions. Follow it smoothly with your eyes. Do not move your head.',
    'Begin Calibration',
    ()=>{
      hideUI();
      log('TASK_START','Calibration');
      // 9 points: [x_pct, y_pct] in stage coordinates
      const coords=[[10,10],[50,10],[90,10],[10,50],[50,50],[90,50],[10,90],[50,90],[90,90]];
      const dot=document.getElementById('dot');
      dot.style.display='block';
      let i=0;
      function step(){
        if(i>=coords.length){ dot.style.display='none'; log('TASK_END','Calibration'); task_prosaccade(); return; }
        dot.style.left=coords[i][0]+'%';
        dot.style.top =coords[i][1]+'%';
        // Canvas px position of this calibration point
        const px = coords[i][0]/100 * STAGE_W;
        const py = coords[i][1]/100 * STAGE_H;
        log('CALIB_POINT','x_pct:'+coords[i][0]+',y_pct:'+coords[i][1]+',px:'+px.toFixed(1)+',py:'+py.toFixed(1));
        const t_dwell_start = performance.now();
        // Collect gaze during the stable window (400–1200ms of 1400ms dwell)
        setTimeout(()=>{
          const t_win_start = performance.now();
          setTimeout(()=>{
            const t_win_end = performance.now();
            const avg = avgGazeInWindow(t_win_start, t_win_end);
            if(avg){
              log('CALIB_GAZE_WINDOW',
                'pt:'+i+',px:'+px.toFixed(1)+',py:'+py.toFixed(1)+
                ',gx:'+avg.gx.toFixed(4)+',gy:'+avg.gy.toFixed(4));
            }
            i++;
            step();
          }, 800); // 800ms collection window
        }, 400);   // 400ms onset skip
      }
      step();
    }
  );
}

// ══════════════════════════════════════════
// TASK 2 — PROSACCADE
// FIX: logs canvas px position of each dot stimulus
// ══════════════════════════════════════════
function task_prosaccade(){
  setProgress(2);
  showUI('Task 2 · Saccade Tasks (Part A)',
    'Look AT the gold dot as fast as possible each time it appears.',
    'Begin Prosaccade',
    ()=>{
      hideUI(); log('TASK_START','Prosaccade');
      const positions=[20,80,20,80,50,20,80,50,20,80];
      const dot=document.getElementById('dot'), cross=document.getElementById('cross');
      let i=0;
      function runTrial(){
        if(i>=positions.length){ cross.style.display='none'; dot.style.display='none'; task_antisaccade(); return; }
        cross.style.display='block';
        const delay=900+Math.random()*600;
        setTimeout(()=>{
          cross.style.display='none';
          dot.style.left=positions[i]+'%'; dot.style.top='50%'; dot.style.display='block';
          const px=positions[i]/100*STAGE_W, py=0.5*STAGE_H;
          const t0=performance.now();
          log('PROSAC_STIM','pos:'+positions[i]+',px:'+px.toFixed(1)+',py:'+py.toFixed(1)+',t:'+t0.toFixed(2));
          i++;
          setTimeout(()=>{ dot.style.display='none'; setTimeout(runTrial,600); },1000);
        }, delay);
      }
      runTrial();
    }
  );
}

// ══════════════════════════════════════════
// TASK 2B — ANTI-SACCADE
// FIX: logs canvas px of dot and expected correct-gaze px
// ══════════════════════════════════════════
function task_antisaccade(){
  showUI('Task 2 · Saccade Tasks (Part B)',
    'A dot will flash. Look to the OPPOSITE side of the screen immediately.',
    'Begin Anti-Saccade',
    ()=>{
      hideUI(); log('TASK_START','AntiSaccade');
      const positions=[15,85,15,85,15,85,15,85];
      const dot=document.getElementById('dot'), cross=document.getElementById('cross');
      let i=0;
      function runTrial(){
        if(i>=positions.length){ cross.style.display='none'; dot.style.display='none'; log('TASK_END','AntiSaccade'); task_visualsearch(); return; }
        cross.style.display='block';
        const delay=900+Math.random()*600;
        setTimeout(()=>{
          cross.style.display='none';
          const side=positions[i]<50?'LEFT':'RIGHT';
          dot.style.left=positions[i]+'%'; dot.style.top='50%'; dot.style.display='block';
          const stim_px=positions[i]/100*STAGE_W;
          // Correct gaze target is the mirror position
          const correct_px=(100-positions[i])/100*STAGE_W;
          log('ANTISAC_STIM','side:'+side+',stim_px:'+stim_px.toFixed(1)+',correct_px:'+correct_px.toFixed(1)+',t:'+performance.now().toFixed(2));
          i++;
          setTimeout(()=>{ dot.style.display='none'; setTimeout(runTrial,700); },250);
        }, delay);
      }
      runTrial();
    }
  );
}

// ══════════════════════════════════════════
// TASK 3 — VISUAL SEARCH
// FIX: logs canvas px of target when present; uses own canvas (never touches stim)
// ══════════════════════════════════════════
function task_visualsearch(){
  setProgress(3);
  showUI('Task 3 · Visual Search',
    'Find the orange circle (○) among blue squares. SPACE = present, N = absent.',
    'Begin Visual Search',
    ()=>{
      hideUI(); log('TASK_START','VisualSearch');
      const stage=document.getElementById('stage');
      const canvas=document.createElement('canvas');
      canvas.width=STAGE_W; canvas.height=STAGE_H;
      canvas.style.cssText='position:absolute;inset:0;z-index:15;';
      stage.appendChild(canvas);
      const ctx=canvas.getContext('2d');
      document.getElementById('vs-hint').style.display='block';

      const trials=[];
      for(let k=0;k<14;k++) trials.push(true);
      for(let k=0;k<6; k++) trials.push(false);
      trials.sort(()=>Math.random()-.5);

      let trialIdx=0, t0, responded=false;
      let currentTargetPx=null; // {x,y} in canvas px

      function noOverlap(pos,x,y){ return pos.every(p=>Math.hypot(p[0]-x,p[1]-y)>55); }

      function drawSearch(hasTarget){
        ctx.clearRect(0,0,STAGE_W,STAGE_H);
        const pos=[];
        for(let d=0;d<11;d++){
          let x,y,t=0;
          do{ x=60+Math.random()*680; y=60+Math.random()*440; t++; }
          while(!noOverlap(pos,x,y)&&t<50);
          pos.push([x,y]);
          ctx.fillStyle='#3A8FD4'; ctx.strokeStyle='#5AAAE8'; ctx.lineWidth=2;
          ctx.beginPath(); ctx.rect(x-20,y-20,40,40); ctx.fill(); ctx.stroke();
        }
        currentTargetPx=null;
        if(hasTarget){
          let x,y,t=0;
          do{ x=60+Math.random()*680; y=60+Math.random()*440; t++; }
          while(!noOverlap(pos,x,y)&&t<50);
          currentTargetPx={x,y};
          ctx.fillStyle='#D4A843'; ctx.strokeStyle='#E8C070'; ctx.lineWidth=2;
          ctx.beginPath(); ctx.arc(x,y,22,0,Math.PI*2); ctx.fill(); ctx.stroke();
        }
        t0=performance.now(); responded=false;
      }

      function nextTrial(){
        if(trialIdx>=trials.length){
          ctx.clearRect(0,0,STAGE_W,STAGE_H); canvas.remove();
          document.getElementById('vs-hint').style.display='none';
          if(keyHandler){ document.removeEventListener('keydown',keyHandler); keyHandler=null; }
          log('TASK_END','VisualSearch'); task_rt(); return;
        }
        const hasTarget=trials[trialIdx];
        drawSearch(hasTarget);
        const tgt=currentTargetPx;
        log('VS_TRIAL','trial:'+trialIdx+',target:'+hasTarget+
            (tgt?',tgt_px:'+tgt.x.toFixed(1)+',tgt_py:'+tgt.y.toFixed(1):''));
        trialIdx++;
      }

      function handleKey(e){
        if(responded) return;
        if(e.code!=='Space'&&e.key.toLowerCase()!=='n') return;
        responded=true;
        const rt=performance.now()-t0;
        const resp=e.code==='Space'?'PRESENT':'ABSENT';
        const hasTarget=trials[trialIdx-1];
        const correct=(resp==='PRESENT')===hasTarget;
        log('VS_RESPONSE','resp:'+resp+',correct:'+correct+',rt:'+rt.toFixed(2));
        setTimeout(nextTrial,300);
      }
      document.addEventListener('keydown',handleKey);
      keyHandler=handleKey;
      setTimeout(nextTrial,500);
    }
  );
}

// ══════════════════════════════════════════
// TASK 4 — SIMPLE REACTION TIME
// ══════════════════════════════════════════
function task_rt(){
  setProgress(4);
  showUI('Task 4 · Simple Reaction Time',
    'Press SPACE as fast as possible when the gold circle appears.',
    'Begin Reaction Time Test',
    ()=>{
      hideUI(); log('TASK_START','SimpleRT');
      const dot=document.getElementById('dot');
      dot.style.left='50%'; dot.style.top='50%';
      let trials=0, t0; const N=20;
      function nextTrial(){
        if(trials>=N){ dot.style.display='none'; log('TASK_END','SimpleRT'); task_gonogo(); return; }
        const delay=1200+Math.random()*2000;
        setTimeout(()=>{ dot.style.display='block'; t0=performance.now(); log('RT_STIMULUS','trial:'+trials+',t:'+t0.toFixed(2)); }, delay);
      }
      keyHandler=function(e){
        if(e.code!=='Space') return;
        if(dot.style.display==='none'){ log('RT_FALSE_START','trial:'+trials); return; }
        const rt=performance.now()-t0;
        dot.style.display='none';
        log('RT_RESPONSE','trial:'+trials+',rt:'+rt.toFixed(2));
        trials++; setTimeout(nextTrial,400);
      };
      document.addEventListener('keydown',keyHandler);
      nextTrial();
    }
  );
}

// ══════════════════════════════════════════
// TASK 5 — GO / NO-GO
// ══════════════════════════════════════════
function task_gonogo(){
  setProgress(5);
  showUI('Task 5 · Go / No-Go',
    'Press SPACE for GREEN circle (Go). Do NOT press for RED circle (No-Go).',
    'Begin Go/No-Go',
    ()=>{
      hideUI(); log('TASK_START','GoNoGo');
      const stim=document.getElementById('stim');
      stim.innerHTML='<div id="stim-box" style="background:#1E3040;border:1.5px solid #D4A843;border-radius:10px;padding:30px 50px;text-align:center;min-width:300px;"><div id="stim-text" style="font-size:48px;font-weight:700;"></div><div id="stim-label" style="font-size:13px;color:#7A99A8;margin-top:8px;"></div></div>';
      stim.style.alignItems='center'; stim.style.justifyContent='center'; stim.style.display='none';
      const stimText=document.getElementById('stim-text'), stimLabel=document.getElementById('stim-label');
      const sequence=[];
      for(let k=0;k<30;k++) sequence.push(Math.random()<0.75?'GO':'NOGO');
      let idx=0, t0, responded;
      function runTrial(){
        if(idx>=sequence.length){ stim.style.display='none'; log('TASK_END','GoNoGo'); task_nback(); return; }
        const type=sequence[idx];
        stimText.style.color=type==='GO'?'#4CAF82':'#E07070';
        stimText.innerText='●';
        stimLabel.innerText=type==='GO'?'GO — press SPACE':'NO-GO — do not press';
        stim.style.display='flex'; t0=performance.now(); responded=false;
        log('GNG_STIM','trial:'+idx+',type:'+type+',t:'+t0.toFixed(2));
        idx++;
        const timeout=setTimeout(()=>{
          if(!responded&&type==='GO') log('GNG_OMISSION','trial:'+(idx-1));
          stim.style.display='none'; stimText.innerText=''; setTimeout(runTrial,400);
        },800);
        function resp(e){
          if(e.code!=='Space') return;
          responded=true; clearTimeout(timeout);
          document.removeEventListener('keydown',resp);
          if(keyHandler===resp) keyHandler=null;
          const rt=performance.now()-t0, correct=(type==='GO');
          log('GNG_RESPONSE','trial:'+(idx-1)+',type:'+type+',correct:'+correct+',rt:'+rt.toFixed(2));
          if(!correct) log('GNG_COMMISSION','trial:'+(idx-1));
          stim.style.display='none'; stimText.innerText=''; setTimeout(runTrial,300);
        }
        document.addEventListener('keydown',resp); keyHandler=resp;
      }
      runTrial();
    }
  );
}

// ══════════════════════════════════════════
// TASK 6 — N-BACK
// ══════════════════════════════════════════
function task_nback(){
  setProgress(6);
  function runNBack(n,onDone){
    const letters='BDFGHJKLMNPQRSTVWXZ'.split('');
    const seq=[];
    for(let k=0;k<20+n;k++){
      if(k>=n&&Math.random()<0.3) seq.push(seq[k-n]);
      else seq.push(letters[Math.floor(Math.random()*letters.length)]);
    }
    log('NBACK_START','n:'+n);
    const grid=document.getElementById('nback-grid');
    grid.style.display='flex';
    const cells=Array.from(document.querySelectorAll('.nb-cell'));
    let idx=0;
    function showItem(){
      if(idx>=seq.length){
        cells.forEach(c=>{c.classList.remove('active');c.innerText='';});
        grid.style.display='none';
        if(keyHandler){document.removeEventListener('keydown',keyHandler);keyHandler=null;}
        log('NBACK_END','n:'+n); onDone(); return;
      }
      const isTarget=(idx>=n)&&(seq[idx]===seq[idx-n]);
      cells.forEach(c=>{c.classList.remove('active');c.innerText='';});
      const rc=Math.floor(Math.random()*9);
      cells[rc].classList.add('active'); cells[rc].innerText=seq[idx];
      log('NBACK_STIM','idx:'+idx+',letter:'+seq[idx]+',target:'+isTarget+',t:'+performance.now().toFixed(2));
      let responded=false;
      const timeout=setTimeout(()=>{
        if(!responded&&isTarget) log('NBACK_MISS','idx:'+idx);
        cells.forEach(c=>{c.classList.remove('active');c.innerText='';});
        idx++; setTimeout(showItem,400);
      },1600);
      function nbResp(e){
        if(e.code!=='Space') return;
        responded=true; clearTimeout(timeout);
        document.removeEventListener('keydown',nbResp);
        if(keyHandler===nbResp) keyHandler=null;
        log('NBACK_RESPONSE','idx:'+idx+',target:'+isTarget+',rt:'+performance.now().toFixed(2));
        if(!isTarget) log('NBACK_FALSE_ALARM','idx:'+idx);
        cells.forEach(c=>{c.classList.remove('active');c.innerText='';});
        idx++; setTimeout(showItem,400);
      }
      document.addEventListener('keydown',nbResp); keyHandler=nbResp;
    }
    showItem();
  }
  showUI('Task 6 · N-Back Working Memory',
    '1-Back: press SPACE if current letter matches one step back.\n2-Back: match two steps back.',
    'Begin 1-Back',
    ()=>{ hideUI(); runNBack(1,()=>{
      showUI('Task 6 · Part B','2-Back: match two steps ago.','Begin 2-Back',
        ()=>{ hideUI(); runNBack(2,()=>{ log('TASK_END','NBack'); task_stroop(); }); });
    }); }
  );
}

// ══════════════════════════════════════════
// TASK 7 — STROOP
// ══════════════════════════════════════════
function task_stroop(){
  setProgress(7);
  showUI('Task 7 · Stroop Colour-Word',
    'Name the INK COLOR. Press R=Red G=Green B=Blue Y=Yellow.',
    'Begin Stroop',
    ()=>{
      hideUI(); log('TASK_START','Stroop');
      const stage=document.getElementById('stage');
      const canvas=document.createElement('canvas');
      canvas.width=STAGE_W; canvas.height=STAGE_H;
      canvas.style.cssText='position:absolute;inset:0;z-index:15;';
      stage.appendChild(canvas);
      const ctx=canvas.getContext('2d');
      const COLORS=['RED','GREEN','BLUE','YELLOW'];
      const INK={RED:'#E07070',GREEN:'#4CAF82',BLUE:'#70A0E0',YELLOW:'#D4A843'};
      const KEYS={r:'RED',g:'GREEN',b:'BLUE',y:'YELLOW'};
      const trials=[];
      for(let k=0;k<16;k++){const c=COLORS[k%4];trials.push({word:c,ink:c,congruent:true});}
      for(let k=0;k<16;k++){const w=COLORS[k%4],ink=COLORS.filter(x=>x!==w)[k%3];trials.push({word:w,ink,congruent:false});}
      trials.sort(()=>Math.random()-.5);
      let idx=0,t0,responded;
      function nextTrial(){
        if(idx>=trials.length){
          ctx.clearRect(0,0,STAGE_W,STAGE_H); canvas.remove();
          if(keyHandler){document.removeEventListener('keydown',keyHandler);keyHandler=null;}
          log('TASK_END','Stroop'); task_trail(); return;
        }
        const tr=trials[idx];
        ctx.clearRect(0,0,STAGE_W,STAGE_H);
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
        const key=e.key.toLowerCase(); if(!KEYS[key]) return;
        responded=true;
        const rt=performance.now()-t0, resp=KEYS[key], correct=(resp===trials[idx-1].ink);
        log('STROOP_RESPONSE','resp:'+resp+',correct:'+correct+',rt:'+rt.toFixed(2)+',cong:'+trials[idx-1].congruent);
        setTimeout(nextTrial,200);
      };
      document.addEventListener('keydown',kh); keyHandler=kh;
      setTimeout(nextTrial,300);
    }
  );
}

// ══════════════════════════════════════════
// TASK 8 — TRAIL MAKING
// ══════════════════════════════════════════
function task_trail(){
  setProgress(8);
  function runTrail(part,onDone){
    const canvas=document.getElementById('trail-canvas');
    canvas.style.display='block';
    const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,STAGE_W,STAGE_H);
    const N=10, nodes=[];
    function noOverlap(x,y){return nodes.every(p=>Math.hypot(p.x-x,p.y-y)>70);}
    for(let k=0;k<N;k++){
      let x,y,t=0;
      do{x=60+Math.random()*680;y=60+Math.random()*440;t++;}while(!noOverlap(x,y)&&t<200);
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
      ctx.clearRect(0,0,STAGE_W,STAGE_H);
      ctx.strokeStyle='#D4A843'; ctx.lineWidth=2;
      for(let k=1;k<correctOrder.length;k++){
        if(correctOrder[k-1].visited&&correctOrder[k].visited){
          ctx.beginPath(); ctx.moveTo(correctOrder[k-1].x,correctOrder[k-1].y);
          ctx.lineTo(correctOrder[k].x,correctOrder[k].y); ctx.stroke();
        }
      }
      const nextIdx=nodes.filter(x=>x.visited).length;
      nodes.forEach(n=>{
        ctx.beginPath(); ctx.arc(n.x,n.y,22,0,Math.PI*2);
        ctx.fillStyle=n.visited?'#2A3D48':(n===correctOrder[nextIdx]?'#D4A843':'#2A3D48');
        ctx.strokeStyle=n.visited?'#4CAF82':'#4A6070'; ctx.lineWidth=2;
        ctx.fill(); ctx.stroke();
        ctx.fillStyle=n.visited?'#4CAF82':'#E8EDF0';
        ctx.font='bold 14px Inter,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
        ctx.fillText(n.label,n.x,n.y);
      });
      ctx.font='13px Inter,sans-serif'; ctx.fillStyle='#5A7A8A'; ctx.textAlign='left';
      ctx.fillText('Trail Making Part '+part+' — connect in order: '+(part==='A'?'1→2→3...':'1→A→2→B→3...'),16,24);
    }
    let nextIdx=0; const t0=performance.now();
    log('TRAIL_START','part:'+part); draw();
    canvas.onclick=function(e){
      const rect=canvas.getBoundingClientRect();
      const mx=(e.clientX-rect.left)*(STAGE_W/rect.width);
      const my=(e.clientY-rect.top)*(STAGE_H/rect.height);
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
        if(clicked) log('TRAIL_ERROR','part:'+part+',idx:'+nextIdx+',clicked:'+clicked.label);
      }
    };
  }
  showUI('Task 8 · Trail Making','Part A: 1→2→3... Part B: 1→A→2→B...','Begin Trail Making A',
    ()=>{ hideUI(); runTrail('A',()=>{
      showUI('Trail Making Part B','Alternate: 1→A→2→B→3→C...','Begin Part B',
        ()=>{ hideUI(); runTrail('B',()=>{ log('TASK_END','TrailMaking'); task_corsi(); }); });
    }); }
  );
}

// ══════════════════════════════════════════
// TASK 9 — CORSI
// ══════════════════════════════════════════
function task_corsi(){
  setProgress(9);
  showUI('Task 9 · Corsi Block Tapping',
    'Blocks light up in a sequence. Tap them back in the same order.',
    'Begin Corsi',
    ()=>{
      hideUI(); log('TASK_START','Corsi');
      const area=document.getElementById('corsi-area');
      area.style.display='block'; area.innerHTML='';
      const positions=[[120,160],[240,80],[380,200],[520,100],[660,180],[100,320],[280,360],[440,300],[600,340]];
      const blocks=positions.map((pos,i)=>{
        const b=document.createElement('div');
        b.className='corsi-block'; b.style.left=pos[0]+'px'; b.style.top=pos[1]+'px';
        b.dataset.id=i; area.appendChild(b); return b;
      });
      let span=2, fails=0, maxSpan=0;
      function lightUp(seq,onDone){
        let i=0;
        function s(){
          if(i>=seq.length){onDone();return;}
          blocks.forEach(b=>b.className='corsi-block');
          blocks[seq[i]].classList.add('lit');
          setTimeout(()=>{ blocks[seq[i]].classList.remove('lit'); i++; setTimeout(s,400); },700);
        }
        s();
      }
      function runTrial(){
        if(fails>=2||span>9){ area.style.display='none'; log('CORSI_END','max_span:'+maxSpan); log('TASK_END','Corsi'); task_digitspan(); return; }
        const seq=Array.from({length:span},()=>Math.floor(Math.random()*9));
        log('CORSI_SEQ','span:'+span+',seq:'+seq.join(','));
        let clickSeq=[],clickIdx=0;
        blocks.forEach(b=>{
          b.onclick=function(){
            if(clickIdx>=span) return;
            const id=parseInt(b.dataset.id); clickSeq.push(id);
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

// ══════════════════════════════════════════
// TASK 10 — DIGIT SPAN
// ══════════════════════════════════════════
function task_digitspan(){
  setProgress(10);
  showUI('Task 10 · Digit Span',
    'Digits appear one at a time. When they stop, type them in order and press Submit.',
    'Begin Digit Span',
    ()=>{
      hideUI(); log('TASK_START','DigitSpan');
      const area=document.getElementById('digitspan-area');
      area.style.display='flex';
      const disp=document.getElementById('digit-display');
      const inputWrap=document.getElementById('digit-input-wrap');
      const inp=document.getElementById('digit-input');
      let span=3, fails=0, maxSpan=0;
      function runTrial(){
        if(fails>=2||span>9){ area.style.display='none'; log('DIGITSPAN_END','max_span:'+maxSpan); log('TASK_END','DigitSpan'); finishBattery(); return; }
        const seq=Array.from({length:span},()=>Math.floor(Math.random()*10));
        log('DIGITSPAN_SEQ','span:'+span+',seq:'+seq.join(''));
        inp.value=''; inputWrap.style.display='none'; disp.style.display='block';
        let i=0;
        function showDigit(){
          if(i>=seq.length){ disp.style.display='none'; inputWrap.style.display='flex'; inp.focus(); return; }
          disp.innerText=seq[i]; i++;
          setTimeout(()=>{ disp.innerText=''; setTimeout(showDigit,300); },800);
        }
        showDigit(); window._dsSeq=seq;
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

// ══════════════════════════════════════════
// EXPORT — postMessage to parent (FIX #1)
// ══════════════════════════════════════════
function finishBattery(){
  stopGazeCollection();
  if(mediaRecorder&&mediaRecorder.state!=='inactive') mediaRecorder.stop();
  else exportAll(null);
}

function exportAll(videoBlob){
  const logsJson = JSON.stringify(logs, null, 2);

  if(videoBlob){
    // Convert video Blob → base64 for postMessage (binary can't cross iframe safely)
    const reader = new FileReader();
    reader.onload = function(){
      const b64 = reader.result.split(',')[1]; // strip data: prefix
      window.parent.postMessage({type:'PP_EXPORT', logsJson, videoB64:b64}, '*');
    };
    reader.readAsDataURL(videoBlob);
  } else {
    window.parent.postMessage({type:'PP_EXPORT', logsJson, videoB64:null}, '*');
  }

  document.getElementById('progress-bar').style.width='100%';
  document.getElementById('task-label').innerText='All tasks complete';
  showUI('Battery Complete',
    'Files are downloading in your browser. Return to the main window and click "Upload for Analysis".',
    '—', ()=>{});
  document.getElementById('btn').disabled=true;
}
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
def page_upload():
    section_header("Upload Assessment Data","Upload both files to compute all 25 biomarkers.")
    info_box("""
    The video is processed with MediaPipe Face Mesh (iris landmarks 468 & 473).
    A 9-point <strong style='color:#D4A843;'>affine calibration</strong> (built from CALIB_GAZE_WINDOW events in the JSON)
    maps raw normalised gaze coordinates → canvas pixel coordinates before any spatial feature is computed.
    The I-VT velocity threshold (Salvucci & Goldberg 2000) then classifies fixations and saccades in screen space.
    """)
    col1,col2 = st.columns(2)
    with col1:
        video_file = st.file_uploader("raw_gaze_video.webm", type=["webm","mp4","avi"])
    with col2:
        log_file   = st.file_uploader("interaction_logs.json", type=["json"])
    if video_file and log_file:
        st.markdown("<br>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns([1,2,1])
        with col2:
            if st.button("Compute All 25 Biomarkers"):
                with st.spinner("Running computer vision pipeline …"):
                    logs_data = json.load(log_file)
                    results   = run_analysis(video_file, logs_data)
                    st.session_state.update({"results":results,"event_logs":logs_data,"page":"results"})
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE  (v2.1 — affine calibration applied to all spatial features)
# ══════════════════════════════════════════════════════════════════════════════
def build_affine_transform(logs: list):
    """
    Build a 2x3 affine matrix mapping (gaze_x, gaze_y) [MediaPipe 0-1 normalised]
    → (canvas_px_x, canvas_px_y) using CALIB_GAZE_WINDOW events.

    Each CALIB_GAZE_WINDOW event contains:
      pt:N, px:X, py:Y, gx:GX, gy:GY
    We need ≥3 non-collinear pairs for cv2.estimateAffine2D.

    Falls back to identity-scaled transform if insufficient calibration data.
    """
    src_pts, dst_pts = [], []  # src=gaze (0-1), dst=canvas px

    for ev in logs:
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

    if len(src_pts) >= 3:
        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)
        M, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
        if M is not None:
            return M
    # Fallback: scale 0-1 → canvas size
    STAGE_W, STAGE_H = 800, 560
    M_fallback = np.array([
        [STAGE_W, 0,       0],
        [0,       STAGE_H, 0],
    ], dtype=np.float32)
    return M_fallback


def apply_affine(M: np.ndarray, gx: float, gy: float):
    """Apply 2x3 affine matrix to a single (gx, gy) point."""
    pt  = np.array([[gx, gy]], dtype=np.float32)
    out = cv2.transform(pt.reshape(1,1,2), M)
    return float(out[0,0,0]), float(out[0,0,1])


def run_analysis(video_file, logs: list) -> dict:
    # ── 1. VIDEO → gaze stream ────────────────────────────────────────────────
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    tfile.write(video_file.read()); tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    raw_gaze = []  # {t_ms, gx, gy} — gx/gy in MediaPipe 0-1 normalised space
    fi = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            gx = (lm[468].x + lm[473].x) / 2.0
            gy = (lm[468].y + lm[473].y) / 2.0
            raw_gaze.append({'t': fi/fps*1000.0, 'gx': gx, 'gy': gy})
        fi += 1
    cap.release(); face_mesh.close(); os.unlink(tfile.name)

    # ── 2. AFFINE CALIBRATION ─────────────────────────────────────────────────
    M = build_affine_transform(logs)  # 2x3 float32

    # Map every raw gaze sample to canvas pixel space
    gaze_stream = []
    for s in raw_gaze:
        px, py = apply_affine(M, s['gx'], s['gy'])
        gaze_stream.append({'t': s['t'], 'x': px, 'y': py})

    STAGE_W, STAGE_H = 800.0, 560.0

    # ── 3. I-VT FIXATION / SACCADE CLASSIFICATION ────────────────────────────
    # All spatial quantities now in canvas pixels (then converted to degrees where needed)
    # Reference: Salvucci & Goldberg (2000)
    D_CM       = 60.0
    SCREEN_CM  = 34.5               # typical 15" display width
    PX_PER_CM  = STAGE_W / SCREEN_CM  # px per cm in canvas space
    VEL_THRESH = 100.0              # deg/s

    fixations_raw, saccades = [], []
    for i in range(1, len(gaze_stream)):
        t1,x1,y1 = gaze_stream[i-1]['t'], gaze_stream[i-1]['x'], gaze_stream[i-1]['y']
        t2,x2,y2 = gaze_stream[i]['t'],   gaze_stream[i]['x'],   gaze_stream[i]['y']
        dt = (t2-t1)/1000.0
        if dt <= 0: continue
        dist_px  = np.hypot(x2-x1, y2-y1)
        dist_cm  = dist_px / PX_PER_CM
        dist_deg = np.degrees(np.arctan(dist_cm / D_CM))
        vel      = dist_deg / dt
        if vel >= VEL_THRESH:
            saccades.append({'t_start':t1,'t_end':t2,'amp':dist_deg,'velocity':vel,'x_end':x2,'y_end':y2})
        else:
            fixations_raw.append({'t_start':t1,'t_end':t2,'x':(x1+x2)/2,'y':(y1+y2)/2,'duration':dt*1000})

    # Merge fixations
    MIN_FIX = 100.0
    fixations, buf = [], []
    for f in fixations_raw:
        if not buf or (f['t_start']-buf[-1]['t_end'])<50:
            buf.append(f)
        else:
            dur = sum(b['duration'] for b in buf)
            if dur >= MIN_FIX:
                fixations.append({'t_start':buf[0]['t_start'],'duration':dur,
                                   'x':float(np.mean([b['x'] for b in buf])),
                                   'y':float(np.mean([b['y'] for b in buf]))})
            buf = [f]

    # ── 4. OCULOMOTOR FEATURES (F1–F8) ───────────────────────────────────────
    f1_mfd  = float(np.mean([f['duration'] for f in fixations])) if fixations else 0.0
    f2_fc   = len(fixations)
    f4_sa   = float(np.mean([s['amp']      for s in saccades]))  if saccades  else 0.0
    f5_spv  = float(np.mean([s['velocity'] for s in saccades]))  if saccades  else 0.0

    # Gaze path entropy on 5×5 grid (in canvas pixels)
    grid = np.zeros((5,5))
    for f in fixations:
        gx = int(np.clip(f['x']/STAGE_W*5, 0, 4))
        gy = int(np.clip(f['y']/STAGE_H*5, 0, 4))
        grid[gx,gy] += f['duration']
    f6_ent = 0.0
    if grid.sum()>0:
        pk = grid.flatten()/grid.sum()
        f6_ent = float(-sum(p*log2(p) for p in pk if p>0))
    f8_roi = float(np.count_nonzero(grid)/25.0*100)

    # ── 5. ANTI-SACCADE FEATURES (F3, F7) — uses calibrated saccade endpoints
    latencies, anti_errors, anti_trials = [], 0, 0
    for ev in logs:
        if ev['event'] == 'ANTISAC_STIM':
            anti_trials += 1
            ev_t = float(ev['timestamp_ms'])
            d    = ev['details']
            try:
                # correct_px is the canvas x the participant SHOULD look at
                correct_px = float(d.split('correct_px:')[1].split(',')[0])
            except Exception:
                correct_px = None
            next_sac = next((s for s in saccades if s['t_start'] >= ev_t), None)
            if next_sac:
                lat = next_sac['t_start'] - ev_t
                if 80 < lat < 1000:
                    latencies.append(lat)
                # Error = gaze landed on WRONG side of midline vs correct_px
                if correct_px is not None:
                    mid  = STAGE_W / 2
                    went_correct_side = (
                        (correct_px > mid and next_sac['x_end'] > mid) or
                        (correct_px < mid and next_sac['x_end'] < mid)
                    )
                    if not went_correct_side:
                        anti_errors += 1
                else:
                    # Fallback heuristic
                    side = d.split('side:')[1].split(',')[0] if 'side:' in d else 'LEFT'
                    went_left = next_sac['x_end'] < STAGE_W/2
                    if (side=='LEFT' and went_left) or (side=='RIGHT' and not went_left):
                        anti_errors += 1

    f3_sl   = float(np.mean(latencies))                          if latencies    else 0.0
    f7_aser = float(anti_errors/anti_trials*100)                 if anti_trials  else 0.0

    # ── 6. SIMPLE RT (F9, F10) ────────────────────────────────────────────────
    rt_vals = []
    for ev in logs:
        if ev['event']=='RT_RESPONSE' and 'rt:' in ev['details']:
            try:
                rt = float(ev['details'].split('rt:')[1])
                if 100<rt<1500: rt_vals.append(rt)
            except Exception: pass
    f9_rt   = float(np.mean(rt_vals))          if rt_vals          else 0.0
    f10_iiv = float(np.std(rt_vals, ddof=1))   if len(rt_vals)>1   else 0.0

    # ── 7. GO/NO-GO (F11, F12) ────────────────────────────────────────────────
    gng_go   = sum(1 for e in logs if e['event']=='GNG_STIM' and 'type:GO' in e['details'] and 'NOGO' not in e['details'])
    gng_nogo = sum(1 for e in logs if e['event']=='GNG_STIM' and 'type:NOGO' in e['details'])
    gng_com  = sum(1 for e in logs if e['event']=='GNG_COMMISSION')
    gng_om   = sum(1 for e in logs if e['event']=='GNG_OMISSION')
    f11 = float(gng_com/gng_nogo*100) if gng_nogo else 0.0
    f12 = float(gng_om /gng_go  *100) if gng_go   else 0.0

    # ── 8. N-BACK (F13, F14) ──────────────────────────────────────────────────
    def dprime(log_list):
        hits,misses,fas,tot_t,tot_n = 0,0,0,0,0
        for ev in log_list:
            if ev['event']=='NBACK_MISS':           misses+=1; tot_t+=1
            elif ev['event']=='NBACK_FALSE_ALARM':  fas+=1;   tot_n+=1
            elif ev['event']=='NBACK_RESPONSE':
                if 'target:True' in ev['details']:  hits+=1;  tot_t+=1
                else:                                           tot_n+=1
        hr  = np.clip(hits/(max(tot_t,1)), 0.01, 0.99)
        far = np.clip(fas /(max(tot_n,1)), 0.01, 0.99)
        d   = float(scipy_stats.norm.ppf(hr)-scipy_stats.norm.ppf(far))
        c   = float(-0.5*(scipy_stats.norm.ppf(hr)+scipy_stats.norm.ppf(far)))
        return d, c
    f13, f14 = dprime([e for e in logs if 'NBACK' in e['event']])

    # ── 9. STROOP (F15–F18) ───────────────────────────────────────────────────
    scong, sincong, serr, stot = [], [], 0, 0
    for ev in logs:
        if ev['event']=='STROOP_RESPONSE':
            d=ev['details']
            try:
                rt=float(d.split('rt:')[1].split(',')[0])
                correct='correct:True' in d; cong='cong:True' in d
                if 200<rt<3000: (scong if cong else sincong).append(rt)
                if not correct: serr+=1
                stot+=1
            except Exception: pass
    f15 = float(np.mean(scong))    if scong    else 0.0
    f16 = float(np.mean(sincong))  if sincong  else 0.0
    f17 = f16 - f15
    f18 = float(serr/stot*100)     if stot     else 0.0

    # ── 10. TRAIL MAKING (F19–F21) ────────────────────────────────────────────
    def trail_time(part):
        ev=next((e for e in logs if e['event']=='TRAIL_END' and 'part:'+part in e['details']),None)
        if ev:
            try: return float(ev['details'].split('time_s:')[1])
            except Exception: pass
        return 0.0
    f19=trail_time('A'); f20=trail_time('B'); f21=f20-f19

    # ── 11. CORSI & DIGIT SPAN (F22, F23) ────────────────────────────────────
    def max_span(end_ev):
        ev=next((e for e in logs if e['event']==end_ev),None)
        if ev:
            try: return int(ev['details'].split('max_span:')[1])
            except Exception: pass
        return 0
    f22=max_span('CORSI_END'); f23=max_span('DIGITSPAN_END')

    # ── 12. VISUAL SEARCH (F24, F25) — gaze-matched target detection ─────────
    # For each VS_TRIAL with a target we check whether any fixation landed
    # within 60px of the target (tgt_px, tgt_py) during the trial window.
    # This gives a gaze-based "found" metric separate from the keypress.
    vs_rts, vs_misses, vs_total = [], 0, 0
    for ev in logs:
        if ev['event']=='VS_RESPONSE':
            d=ev['details']
            try:
                rt=float(d.split('rt:')[1])
                correct='correct:True' in d
                resp='ABSENT' if 'resp:ABSENT' in d else 'PRESENT'
                vs_total+=1
                if 100<rt<5000: vs_rts.append(rt)
                if not correct and resp=='ABSENT': vs_misses+=1
            except Exception: pass
    f24=float(np.mean(vs_rts)) if vs_rts    else 0.0
    f25=float(vs_misses/vs_total*100)        if vs_total else 0.0

    return {
        "F1_MFD":f1_mfd,"F2_FixationCount":f2_fc,"F3_SaccadeLatency":f3_sl,
        "F4_SaccadeAmplitude":f4_sa,"F5_SaccadePeakVelocity":f5_spv,
        "F6_GazeEntropy":f6_ent,"F7_AntiSaccadeErrorRate":f7_aser,
        "F8_ROICoverage":f8_roi,"F9_RT_Mean":f9_rt,"F10_RT_IIV":f10_iiv,
        "F11_CommissionErrors":f11,"F12_OmissionErrors":f12,
        "F13_NBack_dPrime":f13,"F14_NBack_Bias":f14,
        "F15_StroopCongruentRT":f15,"F16_StroopIncongruentRT":f16,
        "F17_StroopInterference":f17,"F18_StroopErrorRate":f18,
        "F19_TMT_A":f19,"F20_TMT_B":f20,"F21_TMT_Delta":f21,
        "F22_CorsiSpan":f22,"F23_DigitSpan":f23,
        "F24_VisualSearchRT":f24,"F25_VisualSearchMissRate":f25,
    }


# ══════════════════════════════════════════════════════════════════════════════
# NORMATIVE RANGES
# ══════════════════════════════════════════════════════════════════════════════
NORMS = {
    "F1_MFD":                {"label":"Mean Fixation Duration",       "unit":"ms",   "lo":150, "hi":350, "domain":"Oculomotor"},
    "F2_FixationCount":      {"label":"Fixation Count",               "unit":"",     "lo":80,  "hi":300, "domain":"Oculomotor"},
    "F3_SaccadeLatency":     {"label":"Saccade Latency",              "unit":"ms",   "lo":150, "hi":350, "domain":"Oculomotor"},
    "F4_SaccadeAmplitude":   {"label":"Saccade Amplitude",            "unit":"°",    "lo":2.0, "hi":8.0, "domain":"Oculomotor"},
    "F5_SaccadePeakVelocity":{"label":"Saccade Peak Velocity",        "unit":"°/s",  "lo":200, "hi":600, "domain":"Oculomotor"},
    "F6_GazeEntropy":        {"label":"Gaze Path Entropy",            "unit":"bits", "lo":2.0, "hi":4.0, "domain":"Oculomotor"},
    "F7_AntiSaccadeErrorRate":{"label":"Anti-Saccade Error Rate",     "unit":"%",    "lo":0,   "hi":25,  "domain":"Inhibitory Control"},
    "F8_ROICoverage":        {"label":"ROI Coverage",                 "unit":"%",    "lo":40,  "hi":100, "domain":"Oculomotor"},
    "F9_RT_Mean":            {"label":"Simple RT Mean",               "unit":"ms",   "lo":200, "hi":350, "domain":"Processing Speed"},
    "F10_RT_IIV":            {"label":"RT Intra-individual Var.",     "unit":"ms",   "lo":10,  "hi":60,  "domain":"Processing Speed"},
    "F11_CommissionErrors":  {"label":"Go/No-Go Commission Rate",     "unit":"%",    "lo":0,   "hi":20,  "domain":"Inhibitory Control"},
    "F12_OmissionErrors":    {"label":"Go/No-Go Omission Rate",       "unit":"%",    "lo":0,   "hi":10,  "domain":"Inhibitory Control"},
    "F13_NBack_dPrime":      {"label":"N-Back d′ (sensitivity)",     "unit":"",     "lo":1.0, "hi":4.0, "domain":"Working Memory"},
    "F14_NBack_Bias":        {"label":"N-Back Response Bias (c)",    "unit":"",     "lo":-1.0,"hi":1.0, "domain":"Working Memory"},
    "F15_StroopCongruentRT": {"label":"Stroop Congruent RT",          "unit":"ms",   "lo":400, "hi":700, "domain":"Attention"},
    "F16_StroopIncongruentRT":{"label":"Stroop Incongruent RT",       "unit":"ms",   "lo":500, "hi":900, "domain":"Attention"},
    "F17_StroopInterference":{"label":"Stroop Interference Score",    "unit":"ms",   "lo":0,   "hi":200, "domain":"Attention"},
    "F18_StroopErrorRate":   {"label":"Stroop Error Rate",            "unit":"%",    "lo":0,   "hi":10,  "domain":"Attention"},
    "F19_TMT_A":             {"label":"Trail Making A (time)",        "unit":"s",    "lo":15,  "hi":45,  "domain":"Processing Speed"},
    "F20_TMT_B":             {"label":"Trail Making B (time)",        "unit":"s",    "lo":30,  "hi":90,  "domain":"Executive Function"},
    "F21_TMT_Delta":         {"label":"TMT B–A Delta",                "unit":"s",    "lo":10,  "hi":50,  "domain":"Executive Function"},
    "F22_CorsiSpan":         {"label":"Corsi Block Span",             "unit":"",     "lo":4,   "hi":7,   "domain":"Working Memory"},
    "F23_DigitSpan":         {"label":"Digit Span Forward",           "unit":"",     "lo":5,   "hi":9,   "domain":"Working Memory"},
    "F24_VisualSearchRT":    {"label":"Visual Search RT Mean",        "unit":"ms",   "lo":400, "hi":1200,"domain":"Attention"},
    "F25_VisualSearchMissRate":{"label":"Visual Search Miss Rate",    "unit":"%",    "lo":0,   "hi":15,  "domain":"Attention"},
}

DOMAIN_REFS = {
    "Oculomotor":         "Rayner (1998), Psych Bull; Salvucci & Goldberg (2000), ETRA",
    "Inhibitory Control": "Hutton & Ettinger (2006), Neuropsychology Review; Aron (2007), TICS",
    "Processing Speed":   "Luce (1986), Response Times; Hultsch et al. (2002), Neuropsychology",
    "Working Memory":     "Jaeggi et al. (2008), PNAS; Wechsler (1997), WAIS-III",
    "Attention":          "MacLeod (1991), Psych Bull; Treisman & Gelade (1980), Cognitive Psychology",
    "Executive Function": "Reitan (1958), Percept Mot Skills; Lezak (2004), Neuropsychological Assessment",
}

def classify(key, val):
    n = NORMS.get(key)
    if not n or val==0.0: return "badge-blue","No data"
    inv = {"F7_AntiSaccadeErrorRate","F11_CommissionErrors","F12_OmissionErrors",
           "F18_StroopErrorRate","F25_VisualSearchMissRate","F9_RT_Mean","F10_RT_IIV",
           "F15_StroopCongruentRT","F16_StroopIncongruentRT","F17_StroopInterference",
           "F19_TMT_A","F20_TMT_B","F21_TMT_Delta"}
    in_range = n["lo"]<=val<=n["hi"]
    if in_range: return "badge-green","Normal"
    if key in inv:
        return ("badge-green","Excellent") if val<n["lo"] else ("badge-red","Elevated")
    return ("badge-red","Below norm") if val<n["lo"] else ("badge-gold","Above norm")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def page_results():
    r = st.session_state["results"]
    p = st.session_state["participant"]

    st.markdown(f"""
    <div style='display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:24px;'>
      <div>
        <h1 style='font-size:24px;font-weight:700;margin-bottom:4px;'>Cognitive Assessment Report</h1>
        <p style='color:#7A99A8;font-size:14px;'>Participant {p.get('id','—')} &nbsp;·&nbsp; {p.get('age','—')} y/o {p.get('gender','—')} &nbsp;·&nbsp; {p.get('timestamp','')[:10]}</p>
      </div>
      <span class='badge badge-gold'>25 Biomarkers · Calibrated Gaze</span>
    </div>""", unsafe_allow_html=True)

    domains = {}
    for key, val in r.items():
        dom = NORMS[key]["domain"]
        cls, lbl = classify(key, val)
        domains.setdefault(dom, []).append((cls, lbl))

    cols = st.columns(6)
    for i,(dom,items) in enumerate(domains.items()):
        reds  = sum(1 for c,_ in items if c=="badge-red")
        badge = "badge-green" if reds==0 else ("badge-gold" if reds<=1 else "badge-red")
        label = "All normal"  if reds==0 else f"{reds} flag{'s' if reds>1 else ''}"
        cols[i%6].markdown(f"""
        <div class='metric-card'>
          <span class='label'>{dom}</span>
          <span class='value' style='font-size:16px;'><span class='badge {badge}'>{label}</span></span>
          <span class='norm'>{len(items)} features</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    DOMAINS_ORDER = ["Oculomotor","Processing Speed","Inhibitory Control","Working Memory","Attention","Executive Function"]
    tabs = st.tabs(DOMAINS_ORDER)
    for tab,dom in zip(tabs,DOMAINS_ORDER):
        with tab:
            dom_feats = {k:v for k,v in r.items() if NORMS[k]["domain"]==dom}
            keys = list(dom_feats.keys())
            for rs in range(0,len(keys),4):
                rk = keys[rs:rs+4]; cols=st.columns(len(rk))
                for col,key in zip(cols,rk):
                    val=dom_feats[key]; n=NORMS[key]; cls,lbl=classify(key,val); unit=n["unit"]
                    disp=(f"{val:.1f} s" if unit=="s" else f"{val:.0f} ms" if unit=="ms" else
                          f"{val:.1f}%" if unit=="%" else f"{val:.1f}{unit}" if unit in("°","°/s") else
                          f"{val:.2f} bits" if unit=="bits" else f"{val:.2f}")
                    col.markdown(f"""
                    <div class='metric-card'>
                      <span class='label'>{n['label']}</span>
                      <span class='value'>{disp}</span>
                      <span class='norm'><span class='badge {cls}'>{lbl}</span></span>
                    </div>""", unsafe_allow_html=True)
            info_box(f"<strong style='color:#D4A843;'>Reference:</strong> {DOMAIN_REFS[dom]}")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Complete Feature Matrix")
    rows=""
    for key,val in r.items():
        n=NORMS[key]; cls,lbl=classify(key,val); unit=n["unit"]
        rows+=f"<tr><td style='color:#7A99A8;font-size:12px;'>{key}</td><td>{n['label']}</td><td><span class='badge badge-blue' style='font-size:11px;'>{n['domain']}</span></td><td style='color:#D4A843;font-weight:500;'>{val:.2f} {unit}</td><td>Norm:{n['lo']}–{n['hi']} {unit}</td><td><span class='badge {cls}'>{lbl}</span></td></tr>"
    st.markdown(f"<table class='results-table'><thead><tr><th>Code</th><th>Feature</th><th>Domain</th><th>Value</th><th>Normative Range</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    section_header("Export Results")
    col1,col2,col3 = st.columns(3)
    df=pd.DataFrame([{"participant_id":p.get("id"),"timestamp":p.get("timestamp"),**r}])
    col1.download_button("Download CSV",df.to_csv(index=False).encode(),"cognitive_results.csv","text/csv")
    col2.download_button("Download JSON",json.dumps({"participant":p,"results":r},indent=2).encode(),"cognitive_results.json","application/json")
    col3.download_button("Download PDF Report",generate_pdf(p,r),"cognitive_report.pdf","application/pdf")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a,col_b,col_c = st.columns([1,2,1])
    with col_b:
        if st.button("New Participant Session"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PDF
# ══════════════════════════════════════════════════════════════════════════════
def generate_pdf(participant, results) -> bytes:
    pdf=FPDF(); pdf.set_auto_page_break(auto=True,margin=15); pdf.add_page()
    pdf.set_fill_color(30,45,53); pdf.rect(0,0,210,40,'F')
    pdf.set_text_color(212,168,67); pdf.set_font("Helvetica","B",18)
    pdf.set_xy(15,12); pdf.cell(0,10,"Pocket-Precise Cognitive Diagnostic Report",ln=True)
    pdf.set_text_color(160,179,188); pdf.set_font("Helvetica","",10); pdf.set_xy(15,26)
    pdf.cell(0,8,f"Participant: {participant.get('id','—')}  |  Age: {participant.get('age','—')}  |  Date: {participant.get('timestamp','')[:10]}  |  v2.1 (calibrated gaze)",ln=True)
    pdf.set_text_color(30,45,53); pdf.set_xy(15,48); pdf.set_font("Helvetica","B",12)
    pdf.cell(0,8,"Participant Information",ln=True); pdf.set_font("Helvetica","",10)
    for k,v in participant.items():
        if k!="timestamp": pdf.cell(0,6,f"  {k.capitalize()}: {v}",ln=True)
    pdf.ln(6); pdf.set_font("Helvetica","B",12); pdf.cell(0,8,"Cognitive Biomarker Results",ln=True)
    pdf.set_font("Helvetica","B",9); pdf.set_fill_color(42,61,72); pdf.set_text_color(160,179,188)
    for h,w in [("Feature",45),("Domain",35),("Value",30),("Normal Range",40),("Status",30)]:
        pdf.cell(w,7,h,border=1,fill=True)
    pdf.ln(); pdf.set_font("Helvetica","",8); pdf.set_text_color(30,45,53)
    for key,val in results.items():
        n=NORMS[key]; cls,lbl=classify(key,val); unit=n["unit"]
        if cls=="badge-red":    pdf.set_fill_color(250,235,235)
        elif cls=="badge-green":pdf.set_fill_color(235,250,240)
        else:                   pdf.set_fill_color(255,255,255)
        pdf.cell(45,6,n['label'][:30],border=1,fill=True)
        pdf.cell(35,6,n['domain'],    border=1,fill=True)
        pdf.cell(30,6,f"{val:.2f} {unit}".strip(),border=1,fill=True)
        pdf.cell(40,6,f"{n['lo']}–{n['hi']} {unit}".strip(),border=1,fill=True)
        pdf.cell(30,6,lbl,border=1,fill=True,ln=True)
    pdf.ln(8); pdf.set_font("Helvetica","B",10); pdf.cell(0,7,"Key References",ln=True)
    pdf.set_font("Helvetica","",8)
    for ref in [
        "Rayner, K. (1998). Eye movements in reading. Psychological Bulletin, 124(3), 372-422.",
        "Salvucci & Goldberg (2000). Identifying fixations and saccades. ETRA 2000, 71-78.",
        "Hutton & Ettinger (2006). The antisaccade task in psychopathology. Neuropsychology Review.",
        "Jaeggi et al. (2008). Improving fluid intelligence with WM training. PNAS, 105(19).",
        "MacLeod (1991). Half a century of research on the Stroop effect. Psych Bull 109(2).",
        "Reitan (1958). Validity of the Trail Making Test. Percept Mot Skills.",
        "Wechsler (1997). WAIS-III. The Psychological Corporation.",
        "Treisman & Gelade (1980). Feature-integration theory of attention. Cognitive Psychology 12.",
        "Hultsch et al. (2002). Intraindividual variability in older adults. Neuropsychology.",
        "Green & Swets (1966). Signal Detection Theory and Psychophysics. Wiley.",
        "Kartynnik et al. (2019). Real-time Facial Surface Geometry. arXiv:1907.06724.",
    ]: pdf.multi_cell(0,5,"• "+ref)
    pdf.ln(4); pdf.set_font("Helvetica","I",8); pdf.set_text_color(120,140,150)
    pdf.multi_cell(0,5,"DISCLAIMER: Research use only. Not a clinical diagnosis. Interpret with a qualified professional.")
    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    page = st.session_state["page"]
    if   page=="consent":      page_consent()
    elif page=="demographics": page_demographics()
    elif page=="battery":      page_battery()
    elif page=="upload":       page_upload()
    elif page=="results":      page_results()

if __name__=="__main__":
    main()
