import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import time
from datetime import datetime

# --- 1. CLEAN CLINICAL UI CONFIGURATION ---
st.set_page_config(page_title="Cognitive Diagnostic Suite", layout="centered", initial_sidebar_state="collapsed")

# Force Light Theme & TreadWill Minimalist Aesthetics
st.markdown("""
    <style>
    /* Base Theme Override */
    .stApp { background-color: #F8FAFC; color: #1E293B; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #0F172A; text-align: center; font-weight: 700; margin-bottom: 20px; }
    p { font-size: 16px; line-height: 1.6; text-align: center; }
    
    /* Input & Button Styling */
    .stTextInput>div>div>input { background-color: #FFFFFF; color: #1E293B; border: 1px solid #CBD5E1; border-radius: 6px; padding: 10px; }
    .stButton>button {
        background-color: #2563EB; color: #FFFFFF; border-radius: 8px;
        padding: 12px 24px; font-size: 16px; font-weight: 600; border: none; width: 100%; transition: 0.2s ease;
    }
    .stButton>button:hover { background-color: #1D4ED8; color: #FFFFFF; }
    
    /* Gaze Stimulus Canvas */
    .gaze-canvas {
        position: relative; width: 100%; height: 400px; 
        background-color: #E2E8F0; border: 2px solid #CBD5E1; border-radius: 12px; overflow: hidden; margin: 20px 0;
    }
    .auto-dot {
        position: absolute; width: 24px; height: 24px; 
        background-color: #EF4444; border-radius: 50%; transform: translate(-50%, -50%);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }
    .center-cross {
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        color: #64748B; font-size: 40px; font-weight: 300;
    }
    
    /* Grid for Memory Task */
    .grid-container {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; max-width: 300px; margin: 20px auto;
    }
    .grid-tile { height: 80px; border-radius: 8px; transition: background-color 0.1s; }
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
    """Clicks a hidden Streamlit button after a set time to advance tasks without user input."""
    st.markdown("""
        <style>div.stButton > button[title="AUTO"] { display: none; }</style>
    """, unsafe_allow_html=True)
    components.html(f"""
        <script>
        setTimeout(function() {{
            const btns = window.parent.document.querySelectorAll('button');
            btns.forEach(b => {{ if(b.innerText === 'AUTO') b.click(); }});
        }}, {delay_ms});
        </script>
    """, height=0)

# =====================================================================
# UI ROUTING (CLEAN & MINIMAL)
# =====================================================================

if st.session_state.phase == "LOGIN":
    st.title("Cognitive Diagnostic Portal")
    st.write("Clinical Data Acquisition Environment")
    
    # Using native containers to prevent CSS clipping errors
    with st.container():
        st.write("---")
        token = st.text_input("Enter Participant Token:")
        if st.button("Unlock Assessment Suite"):
            if token.strip():
                st.session_state.subject_id = token.strip()
                st.session_state.phase = "CONSENT"
                st.rerun()
            else:
                st.error("Please enter a valid token.")

elif st.session_state.phase == "CONSENT":
    st.title("Informed Consent")
    with st.container():
        st.info("""
        **Study Parameters:**
        1. Your webcam will record your eye movements silently.
        2. No recording preview is shown to prevent visual bias.
        3. All data is anonymized.
        """)
        c1 = st.checkbox("I authorize silent camera access.")
        c2 = st.checkbox("I authorize anonymized data collection.")
        if st.button("Accept & Continue"):
            if c1 and c2:
                log_event("CONSENT_GIVEN")
                st.session_state.phase = "CALIB_INTRO"
                st.rerun()
            else:
                st.warning("Please check both boxes.")

elif st.session_state.phase == "CALIB_INTRO":
    st.title("Task 1: Optical Calibration")
    st.write("Place your device on a stable surface. Sit 30–35 cm from the screen.")
    st.write("👉 **Keep your head perfectly still. Follow the red dot purely with your eyes.**")
    st.write("*(This task will advance automatically. Do not click anything.)*")
    
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
    
    # Auto-advances every 1500ms
    auto_advance(1500)
    if st.button("AUTO", help="AUTO"):
        log_event(f"CALIB_POINT_{idx}", f"X:{x}, Y:{y}")
        if idx < 8:
            st.session_state.step_calib += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.phase = "ANTI_INTRO"
        st.rerun()

elif st.session_state.phase == "ANTI_INTRO":
    st.title("Task 2: Inhibitory Challenge")
    st.markdown("""
    A central cross (**+**) will appear, followed by a red dot on the **Left** or **Right**.
    👉 **CRITICAL:** Instantly look in the **EXACT OPPOSITE DIRECTION** of the dot.
    """)
    st.write("*(This task will advance automatically.)*")
    
    if st.button("Start Task 2"):
        log_event("ANTI_START")
        st.session_state.phase = "PLAY_ANTI"
        st.rerun()

elif st.session_state.phase == "PLAY_ANTI":
    load_camera("2_antisaccade")
    trials = [("20%", "50%"), ("80%", "50%"), ("80%", "50%"), ("20%", "50%")]
    idx = st.session_state.step_anti
    x, y = trials[idx]
    
    st.markdown(f'<div class="gaze-canvas"><div class="center-cross">+</div><div class="auto-dot" style="left: {x}; top: {y};"></div></div>', unsafe_allow_html=True)
    
    # Auto-advances every 1800ms
    auto_advance(1800)
    if st.button("AUTO", help="AUTO"):
        log_event(f"ANTI_TRIAL_{idx}", f"Flashed: {x}")
        if idx < 3:
            st.session_state.step_anti += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.phase = "NBACK_INTRO"
        st.rerun()

elif st.session_state.phase == "NBACK_INTRO":
    st.title("Task 3: Spatial Memory (2-Back)")
    st.markdown("""
    A blue square will jump across a grid.
    👉 Click **MATCH DETECTED** if the square lands in the **exact same spot** it was in **2 steps ago**.
    """)
    if st.button("Start Memory Task"):
        log_event("NBACK_START")
        st.session_state.phase = "PLAY_NBACK"
        st.rerun()

elif st.session_state.phase == "PLAY_NBACK":
    load_camera("3_spatial_memory")
    seq = [3, 6, 9, 6, 2, 9, 2, 4, 7]
    ans = [0, 0, 0, 1, 0, 0, 1, 0, 0]
    idx = st.session_state.step_nback
    
    html = "<div class='grid-container'>"
    for i in range(1, 10):
        color = "#3B82F6" if i == seq[idx] else "#CBD5E1"
        html += f"<div class='grid-tile' style='background-color: {color};'></div>"
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

elif st.session_state.phase == "EXPORT":
    st.title("Assessment Complete")
    st.success("Session data successfully recorded.")
    
    payload = {
        "subject_id": st.session_state.subject_id,
        "date_utc": str(datetime.utcnow()),
        "memory_accuracy": f"{(st.session_state.score_nback / 9) * 100:.1f}%",
        "logs": st.session_state.session_logs
    }
    
    b64 = base64.b64encode(json.dumps(payload, indent=4).encode()).decode()
    st.write("Click below to download your log file. Return this file along with the downloaded video tracks.")
    
    href = f'<a href="data:file/json;base64,{b64}" download="{st.session_state.subject_id}_metrics.json"><button style="width:100%; padding:14px; background-color:#10B981; color:white; border-radius:6px; border:none; font-weight:bold; cursor:pointer;">Download Session Logs (.JSON)</button></a>'
    st.markdown(href, unsafe_allow_html=True)
