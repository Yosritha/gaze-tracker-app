import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import time
from datetime import datetime

# --- 1. CORE UI CONFIGURATION ---
st.set_page_config(page_title="Cognitive Diagnostic Suite", layout="centered", initial_sidebar_state="collapsed")

# Aggressive CSS to force TreadWill aesthetics and PREVENT invisible text bugs
st.markdown("""
    <style>
    /* Force Light Mode Background & Text */
    .stApp { background-color: #F8FAFC !important; }
    h1, h2, h3, p, span, div { color: #0F172A !important; font-family: 'Inter', sans-serif; }
    
    /* Fix Checkbox Visibility */
    div[data-testid="stCheckbox"] > label > div > p {
        color: #0F172A !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    
    /* Input Fields */
    .stTextInput>div>div>input { 
        background-color: #FFFFFF !important; 
        color: #0F172A !important; 
        border: 2px solid #CBD5E1 !important; 
        border-radius: 8px !important; 
        padding: 12px !important; 
    }
    
    /* Standardized Buttons */
    .stButton>button {
        background-color: #2563EB !important; 
        color: #FFFFFF !important; 
        border-radius: 8px !important;
        padding: 14px 24px !important; 
        font-size: 16px !important; 
        font-weight: 600 !important; 
        border: none !important; 
        width: 100% !important; 
        transition: 0.2s ease !important;
    }
    .stButton>button:hover { background-color: #1D4ED8 !important; }
    
    /* Info Box (replaces standard consent box) */
    div[data-testid="stWebsocket"] { display: none; }
    .consent-card {
        background-color: #EFF6FF; border: 1px solid #BFDBFE; border-left: 6px solid #3B82F6;
        padding: 20px; border-radius: 8px; margin-bottom: 20px;
    }
    
    /* Gaze Stimulus Canvas */
    .gaze-canvas {
        position: relative; width: 100%; height: 400px; 
        background-color: #E2E8F0; border: 2px solid #94A3B8; border-radius: 12px; overflow: hidden; margin: 20px 0;
    }
    .auto-dot {
        position: absolute; width: 24px; height: 24px; 
        background-color: #EF4444; border-radius: 50%; transform: translate(-50%, -50%);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.6);
    }
    .center-cross {
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        color: #475569 !important; font-size: 40px; font-weight: 300;
    }
    
    /* Grid for Memory Task */
    .grid-container {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; max-width: 300px; margin: 20px auto;
    }
    .grid-tile { height: 80px; border-radius: 8px; transition: background-color 0.1s; border: 1px solid #94A3B8;}
    </style>
""", unsafe_allow_html=True)

# --- 2. STATE MANAGEMENT ---
states = [
    ('phase', "LOGIN"), ('subject_id', ""), ('session_logs', []),
    ('step_calib', 0), ('step_anti', 0), ('step_pursuit', 0),
    ('step_nback', 0), ('score_nback', 0)
]
for key, val in states:
    if key not in st.session_state:
        st.session_state[key] = val

def log_event(event_id, metadata=""):
    st.session_state.session_logs.append({
        "timestamp_ms": int(time.time() * 1000),
        "event": event_id,
        "details": metadata
    })

# --- 3. SILENT VIDEO ENGINE ---
def load_camera(filename):
    html = f"""
    <script>
    (async function() {{
        try {{
            const stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "user", frameRate: {{ ideal: 30 }} }} }});
            const recorder = new MediaRecorder(stream, {{ mimeType: 'video/webm' }});
            let chunks = [];
            recorder.ondataavailable = (e) => {{ if (e.data.size > 0) chunks.push(e.data); }};
            recorder.onstop = () => {{
                const blob = new Blob(chunks, {{ type: 'video/webm' }});
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url; a.download = "{st.session_state.subject_id}_{filename}.webm";
                document.body.appendChild(a); a.click(); document.body.removeChild(a);
            }};
            recorder.start();
            window.parent.killRecorder = () => {{ recorder.stop(); stream.getTracks().forEach(t => t.stop()); }};
        }} catch (e) {{ console.log("Camera failed."); }}
    }})();
    </script>
    """
    components.html(html, height=0)

# --- 4. JS AUTOMATION TRIGGER ---
def auto_advance(delay_ms):
    """Clicks a hidden Streamlit button to advance tasks without user input."""
    st.markdown("""<style>div.stButton > button[title="AUTO"] { display: none !important; }</style>""", unsafe_allow_html=True)
    components.html(f"""
        <script>
        setTimeout(function() {{
            const btns = window.parent.document.querySelectorAll('button');
            btns.forEach(b => {{ if(b.innerText === 'AUTO') b.click(); }});
        }}, {delay_ms});
        </script>
    """, height=0)

# =====================================================================
# UI ROUTING (TREADWILL STYLE)
# =====================================================================

# --- PHASE 1: LOGIN ---
if st.session_state.phase == "LOGIN":
    st.markdown("<h1>Cognitive Diagnostic Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Clinical Data Acquisition Environment</p>", unsafe_allow_html=True)
    st.write("---")
    
    token = st.text_input("Enter Participant Token:", placeholder="e.g. P001")
    if st.button("Unlock Assessment Suite"):
        if token.strip():
            st.session_state.subject_id = token.strip()
            st.session_state.phase = "CONSENT"
            st.rerun()
        else:
            st.error("Please enter a valid token.")

# --- PHASE 2: CONSENT ---
elif st.session_state.phase == "CONSENT":
    st.markdown("<h1>Informed Consent</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="consent-card">
        <h4 style="margin-top: 0; color: #1E3A8A !important;">Study Parameters:</h4>
        <ol style="color: #1E40AF; margin-bottom: 0;">
            <li>Your webcam will record your eye movements silently at 30 frames per second.</li>
            <li>No recording preview is shown to prevent visual bias and self-correction loops.</li>
            <li>All exported interaction arrays are permanently anonymized.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    c1 = st.checkbox("I authorize silent camera access for biometric parsing.")
    c2 = st.checkbox("I authorize anonymized data collection for research use.")
    
    st.write("")
    if st.button("Accept & Continue"):
        if c1 and c2:
            log_event("CONSENT_GIVEN")
            st.session_state.phase = "CALIB_INTRO"
            st.rerun()
        else:
            st.error("You must select both checkboxes to proceed.")

# --- PHASE 3: CALIBRATION ---
elif st.session_state.phase == "CALIB_INTRO":
    st.markdown("<h1>Task 1: Optical Calibration</h1>", unsafe_allow_html=True)
    st.write("Place your device on a stable surface. Sit **30–35 cm** from the screen.")
    st.markdown("👉 **Keep your head perfectly still. Follow the red dot purely with your eyes.**")
    st.info("This task advances automatically. Do not click anything once it starts.")
    
    if st.button("Start Calibration"):
        log_event("CALIB_START")
        st.session_state.phase = "PLAY_CALIB"
        st.rerun()

elif st.session_state.phase == "PLAY_CALIB":
    load_camera("1_calibration")
    coords = [("10%", "10%"), ("50%", "10%"), ("90%", "10%"), ("10%", "50%"), ("50%", "50%"), ("90%", "50%"), ("10%", "90%"), ("50%", "90%"), ("90%", "90%")]
    idx = st.session_state.step_calib
    x, y = coords[idx]
    
    st.markdown(f'<div class="gaze-canvas"><div class="auto-dot" style="left: {x}; top: {y};"></div></div>', unsafe_allow_html=True)
    
    auto_advance(1500)
    if st.button("AUTO", help="AUTO"):
        log_event(f"CALIB_POINT_{idx}", f"X:{x}, Y:{y}")
        if idx < 8:
            st.session_state.step_calib += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.phase = "ANTI_INTRO"
        st.rerun()

# --- PHASE 4: ANTI-SACCADE ---
elif st.session_state.phase == "ANTI_INTRO":
    st.markdown("<h1>Task 2: Inhibitory Challenge</h1>", unsafe_allow_html=True)
    st.markdown("""
    A central cross (**+**) will appear, followed by a red dot flashing on the **Left** or **Right**.
    
    👉 **CRITICAL:** Instantly look in the **EXACT OPPOSITE DIRECTION** of the dot.
    """)
    st.info("This task advances automatically.")
    
    if st.button("Start Inhibitory Task"):
        log_event("ANTI_START")
        st.session_state.phase = "PLAY_ANTI"
        st.rerun()

elif st.session_state.phase == "PLAY_ANTI":
    load_camera("2_antisaccade")
    trials = [("20%", "50%"), ("80%", "50%"), ("80%", "50%"), ("20%", "50%")]
    idx = st.session_state.step_anti
    x, y = trials[idx]
    
    st.markdown(f'<div class="gaze-canvas"><div class="center-cross">+</div><div class="auto-dot" style="left: {x}; top: {y};"></div></div>', unsafe_allow_html=True)
    
    auto_advance(1800)
    if st.button("AUTO", help="AUTO"):
        log_event(f"ANTI_TRIAL_{idx}", f"Flashed: {x}")
        if idx < 3:
            st.session_state.step_anti += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.phase = "PURSUIT_INTRO"
        st.rerun()

# --- PHASE 5: SMOOTH PURSUIT ---
elif st.session_state.phase == "PURSUIT_INTRO":
    st.markdown("<h1>Task 3: Smooth Pursuit</h1>", unsafe_allow_html=True)
    st.write("Keep your head perfectly steady. Track the moving dot continuously with your eyes as it sweeps horizontally.")
    st.info("This task advances automatically.")
    
    if st.button("Start Pursuit Task"):
        log_event("PURSUIT_START")
        st.session_state.phase = "PLAY_PURSUIT"
        st.rerun()

elif st.session_state.phase == "PLAY_PURSUIT":
    load_camera("3_smooth_pursuit")
    pursuit_coords = ["10%", "25%", "40%", "55%", "70%", "85%", "70%", "55%", "40%", "25%", "10%"]
    idx = st.session_state.step_pursuit
    x = pursuit_coords[idx]
    
    st.markdown(f'<div class="gaze-canvas"><div class="auto-dot" style="left: {x}; top: 50%;"></div></div>', unsafe_allow_html=True)
    
    auto_advance(800)
    if st.button("AUTO", help="AUTO"):
        log_event(f"PURSUIT_STEP_{idx}", f"X: {x}")
        if idx < 10:
            st.session_state.step_pursuit += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.phase = "NBACK_INTRO"
        st.rerun()

# --- PHASE 6: N-BACK (MANUAL INTERACTION) ---
elif st.session_state.phase == "NBACK_INTRO":
    st.markdown("<h1>Task 4: Spatial Memory (2-Back)</h1>", unsafe_allow_html=True)
    st.markdown("""
    A blue square will jump across a grid.
    
    👉 Click **MATCH DETECTED** if the square lands in the **exact same spot** it was in **2 steps ago**. Otherwise, click **Next**.
    """)
    
    if st.button("Start Memory Task"):
        log_event("NBACK_START")
        st.session_state.phase = "PLAY_NBACK"
        st.rerun()

elif st.session_state.phase == "PLAY_NBACK":
    load_camera("4_spatial_memory")
    seq = [3, 6, 9, 6, 2, 9, 2, 4, 7]
    ans = [0, 0, 0, 1, 0, 0, 1, 0, 0]
    idx = st.session_state.step_nback
    
    html = "<div class='grid-container'>"
    for i in range(1, 10):
        color = "#3B82F6" if i == seq[idx] else "#F1F5F9"
        html += f"<div class='grid-tile' style='background-color: {color} !important;'></div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💥 MATCH DETECTED"):
            log_event(f"NBACK_MATCH_{idx}")
            if ans[idx] == 1: st.session_state.score_nback += 1
            st.session_state.step_nback += 1
            if st.session_state.step_nback >= len(seq):
                components.html("<script>window.parent.killRecorder();</script>", height=0)
                st.session_state.phase = "EXPORT"
            st.rerun()
    with c2:
        if st.button("➡️ NEXT (NO MATCH)"):
            log_event(f"NBACK_NEXT_{idx}")
            if ans[idx] == 0: st.session_state.score_nback += 1
            st.session_state.step_nback += 1
            if st.session_state.step_nback >= len(seq):
                components.html("<script>window.parent.killRecorder();</script>", height=0)
                st.session_state.phase = "EXPORT"
            st.rerun()

# --- PHASE 7: EXPORT ---
elif st.session_state.phase == "EXPORT":
    st.markdown("<h1>Assessment Complete</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="consent-card" style="border-left-color: #10B981; background-color: #ECFDF5; border-color: #A7F3D0;">
        <h4 style="margin-top: 0; color: #065F46 !important;">Session data successfully recorded.</h4>
        <p style="color: #047857; margin-bottom: 0;">Your behavioral metrics and asynchronous camera logs are complete.</p>
    </div>
    """, unsafe_allow_html=True)
    
    payload = {
        "subject_id": st.session_state.subject_id,
        "date_utc": str(datetime.utcnow()),
        "memory_accuracy": f"{(st.session_state.score_nback / 9) * 100:.1f}%",
        "logs": st.session_state.session_logs
    }
    
    b64 = base64.b64encode(json.dumps(payload, indent=4).encode()).decode()
    
    st.write("Click below to download your JSON log file. Return this file along with the downloaded video tracks.")
    href = f'<a href="data:file/json;base64,{b64}" download="{st.session_state.subject_id}_metrics.json"><button style="width:100%; padding:14px; background-color:#10B981 !important; color:white !important; border-radius:8px; border:none; font-weight:bold; cursor:pointer;">Download Session Logs (.JSON)</button></a>'
    st.markdown(href, unsafe_allow_html=True)
