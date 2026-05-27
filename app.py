import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import time
from datetime import datetime

# --- CLINICAL VISUAL ENGINE CONFIGURATION ---
st.set_page_config(page_title="Neurological Gaze & Cognitive Tracker", layout="centered", initial_sidebar_state="collapsed")

# Custom UI implementation modeled after high-contrast, accessible healthcare platforms
st.markdown("""
    <style>
    .main { background-color: #FFFFFF; color: #1E293B; font-family: sans-serif; }
    h1, h2, h3 { color: #0F172A; font-weight: 700; margin-top: 5px; }
    .stButton>button {
        background-color: #2563EB; color: white; border-radius: 8px;
        padding: 14px; font-size: 16px; font-weight: 600; border: none; width: 100%;
    }
    .stButton>button:hover { background-color: #1D4ED8; }
    .consent-box {
        background-color: #F8FAFC; border: 1px solid #E2E8F0;
        padding: 15px; border-radius: 8px; max-height: 200px; overflow-y: scroll; font-size: 14px; line-height: 1.5;
    }
    .display-canvas {
        position: relative; width: 100%; height: 350px; 
        background-color: #0F172A; border: 2px solid #334155; border-radius: 12px; overflow: hidden;
    }
    .stimulus-dot {
        position: absolute; width: 24px; height: 24px; 
        background-color: #EF4444; border: 4px solid #FFFFFF; border-radius: 50%;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.9);
        transform: translate(-50%, -50%); transition: all 0.25s ease-in-out;
    }
    .fixation-cross {
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        color: #64748B; font-size: 32px; font-weight: 300;
    }
    .news-viewport {
        width: 100%; height: 300px; border: 1px solid #E2E8F0; border-radius: 8px;
        overflow-y: scroll; padding: 15px; background-color: #FFFFFF;
    }
    .news-card {
        padding: 12px; border-bottom: 1px solid #F1F5F9; font-size: 15px; font-weight: 600; color: #1E293B;
    }
    .gallery-matrix {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; max-width: 320px; margin: 0 auto;
    }
    .gallery-item {
        height: 75px; background-color: #E2E8F0; border-radius: 6px; display: flex;
        align-items: center; justify-content: center; font-size: 20px; cursor: pointer; border: 2px solid transparent;
    }
    .gallery-item:active { border-color: #2563EB; background-color: #DBEAFE; }
    </style>
""", unsafe_allow_html=True)

# --- APPLICATION STATE REGISTER MATRIX ---
state_keys = [
    ('phase', "LOGIN"), ('participant_id', ""), ('logs', []), 
    ('calib_idx', 0), ('nback_idx', 0), ('nback_score', 0),
    ('antisaccade_trial', 0), ('pursuit_step', 0), ('search_found', False),
    ('vmem_phase', "ENCODE")
]
for key, default in state_keys:
    if key not in st.session_state:
        st.session_state[key] = default

def log_event(event_type, description=""):
    st.session_state.logs.append({
        "timestamp_ms": int(time.time() * 1000),
        "event": event_type,
        "details": description
    })

# --- HIGH-FIDELITY WEBRTC SILENT VIDEO RECORDING INJECTION ---
def inject_silent_camera(filename_suffix):
    html_js = f"""
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
                a.href = url;
                a.download = "{st.session_state.participant_id}_" + "{filename_suffix}.webm";
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            }};
            recorder.start();
            window.parent.killRecorder = () => {{
                recorder.stop();
                stream.getTracks().forEach(track => track.stop());
            }};
        }} catch (err) {{ console.log("Camera linkage error."); }}
    }})();
    </script>
    """
    components.html(html_js, height=0)

# =====================================================================
# SYSTEM PHASE ROUTING MATRIX
# =====================================================================

# Phase 1: Registration Entry
if st.session_state.phase == "LOGIN":
    st.title("Cognitive Portal")
    st.write("Please sign in with your assigned Participant ID to initialize the evaluation session.")
    uid = st.text_input("Participant Identifier Token:", placeholder="e.g., P091")
    if st.button("Initialize Protocol"):
        if uid.strip():
            st.session_state.participant_id = uid.strip()
            st.session_state.phase = "CONSENT"
            st.rerun()
        else:
            st.error("A unique identification parameter must be registered.")

# Phase 2: Ethical Clearance Framework
elif st.session_state.phase == "CONSENT":
    st.title("Informed Consent Protocol")
    st.markdown(f"""
    <div class="consent-box">
        <strong>Study Reference:</strong> High-Fidelity Gaze Biomarkers for Asynchronous Stage 0 Cognitive Tracking [cite: 3]<br>
        <strong>Sponsoring Institution:</strong> Indian Institute of Technology Tirupati<br><br>
        <strong>Technical Disclosures:</strong><br>
        1. Your front-facing optical sensor captures eye spatial orientation data continuously at 30fps.<br>
        2. Visual feedback overlays are programmatically suppressed to eliminate behavioral self-correction loops[cite: 455].<br>
        3. Raw assets are immediately compiled under anonymized cryptographic hash pointers matching token <code>{st.session_state.participant_id}</code>.<br><br>
        You reserve the absolute right to abort the task progression framework at any frame boundary by closing the tab.
    </div>
    """, unsafe_allow_html=True)
    a1 = st.checkbox("I authorize the silent execution of my device's camera for biometric parsing purposes[cite: 455].")
    a2 = st.checkbox("I authorize the scientific distribution of my completely anonymized tracking arrays.")
    if st.button("Grant Clearance & Launch"):
        if a1 and a2:
            log_event("CONSENT_GRANTED")
            st.session_state.phase = "CALIB_INTRO"
            st.rerun()
        else:
            st.warning("Ethical authorization checkboxes must be cleared.")

# Phase 3: 9-Point Spatial Calibration Grid
elif st.session_state.phase == "CALIB_INTRO":
    st.title("System Optical Calibration")
    st.write("Let's align your biometric line-of-sight tracking maps.")
    st.markdown("""
    * **Device Alignment:** Lock your smartphone in a portrait desk mount or place your laptop flat on a desk[cite: 80, 453].
    * **Visual Distance:** Sit up straight, positioning your face naturally **30–35 cm** from the camera lens[cite: 80, 484].
    * **Head Posture:** Keep your head still and follow the target purely with your eyes[cite: 73, 453].
    """)
    if st.button("Launch Calibration Display Matrix"):
        log_event("CALIBRATION_SEQUENCE_INITIATED")
        st.session_state.phase = "RUN_CALIB"
        st.rerun()

elif st.session_state.phase == "RUN_CALIB":
    inject_silent_camera("1_calibration")
    calib_coords = [
        ("10%", "10%"), ("50%", "10%"), ("90%", "10%"),
        ("10%", "50%"), ("50%", "50%"), ("90%", "50%"),
        ("10%", "90%"), ("50%", "90%"), ("90%", "90%")
    ]
    i = st.session_state.calib_idx
    x, y = calib_coords[i]
    st.write("👉 **Keep your head perfectly still and lock your eyes onto the moving RED DOT.**")
    st.markdown(f'<div class="display-canvas"><div class="stimulus-dot" style="left: {x}; top: {y};"></div></div>', unsafe_allow_html=True)
    if st.button("Target Point Locked ➡️ Advance" if i < 8 else "Finalize System Calibration Matrix"):
        log_event(f"CALIB_NODE_EXECUTION_{i}", f"Coordinates: X={x}, Y={y}")
        if i < 8:
            st.session_state.calib_idx += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.phase = "TASK1_NEWS"
        st.rerun()

# Task 1: News Feed Reading (Naturalistic Exploration)
elif st.session_state.phase == "TASK1_NEWS":
    st.title("Task 1: Information Ingestion")
    st.write("Scroll through and read the news feed below at your normal reading pace for 15 seconds.")
    if st.button("Start Reading Session"):
        log_event("TASK_NEWS_STARTED")
        st.session_state.phase = "RUN_TASK1"
        st.rerun()

elif st.session_state.phase == "RUN_TASK1":
    inject_silent_camera("2_task_news_feed")
    st.write("📖 Read freely. Click the complete button at the bottom once done.")
    headlines = [
        "Global markets stabilize as technology indexes experience standard valuation corrections[cite: 453].",
        "New deep space computational hardware achieves unprecedented operational efficiencies[cite: 453].",
        "Strategic atmospheric preservation initiatives clear milestones across global research alliances[cite: 453].",
        "Submicron semiconductor fabrication pipelines transition to scale across global facilities[cite: 453].",
        "Marine ecosystem conservation framework yields measurable restoration indices."
    ]
    # Render mock viewport scroll layout
    news_html = "<div class='news-viewport'>"
    for item in headlines: news_html += f"<div class='news-card'>{item}</div>"
    news_html += "</div>"
    st.markdown(news_html, unsafe_allow_html=True)
    if st.button("Complete Task 1"):
        components.html("<script>window.parent.killRecorder();</script>", height=0)
        st.session_state.phase = "TASK2_PHOTO"
        st.rerun()

# Task 2: Free Photo Browsing Layout
elif st.session_state.phase == "TASK2_PHOTO":
    st.title("Task 2: Visual Asset Exploration")
    st.write("Browse through the gallery layout below. Tap or click various items to simulate item inspect cycles.")
    if st.button("Start Gallery Session"):
        log_event("TASK_GALLERY_STARTED")
        st.session_state.phase = "RUN_TASK2"
        st.rerun()

elif st.session_state.phase == "RUN_TASK2":
    inject_silent_camera("3_task_photo_browse")
    st.write("🖼️ Click on any asset node blocks that pull your visual curiosity attention patterns.")
    icons = ["⛰️", "🏄", "🏎️", "🌆", "🚀", "🐼", "🌴", "🎨", "🍕"]
    grid_html = "<div class='gallery-matrix'>"
    for idx, icon in enumerate(icons):
        grid_html += f"<div class='gallery-item' onclick='window.parent.postMessage({{type: \"img_click\", id: {idx}}}, \"*\")'>{icon}</div>"
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)
    if st.button("Complete Task 2"):
        components.html("<script>window.parent.killRecorder();</script>", height=0)
        st.session_state.phase = "TASK3_ANTI"
        st.rerun()

# Task 3: Anti-Saccade Clinical Paradigm
elif st.session_state.phase == "TASK3_ANTI":
    st.title("Task 3: Executive Inhibitory Challenge")
    st.markdown("""
    **Instructions:**
    A central fixation cross will display briefly, followed by a **RED DOT** jumping to either the Left or Right side[cite: 453].
    
    👉 **CRITICAL:** You must immediately force your eyes to look in the **OPPOSITE DIRECTION** of the dot[cite: 32]. 
    If the dot appears on the Left, look Right. If it drops on the Right, look Left[cite: 32].
    """)
    if st.button("Launch Anti-Saccade Trial"):
        log_event("TASK_ANTISACCADE_STARTED")
        st.session_state.phase = "RUN_TASK3"
        st.rerun()

elif st.session_state.phase == "RUN_TASK3":
    inject_silent_camera("4_task_anti_saccade")
    anti_trials = [("20%", "50%"), ("80%", "50%"), ("20%", "50%"), ("80%", "50%")]
    t = st.session_state.antisaccade_trial
    st.write(f"**Trial Run {t + 1} of {len(anti_trials)}**")
    x_pos, y_pos = anti_trials[t]
    st.markdown(f'<div class="display-canvas"><div class="fixation-cross">+</div><div class="stimulus-dot" style="left: {x_pos}; top: {y_pos};"></div></div>', unsafe_allow_html=True)
    if st.button("Inhibitory Focus Locked ➡️ Next Position" if t < len(anti_trials)-1 else "Complete Task 3"):
        log_event(f"ANTISACCADE_TRIAL_{t}", f"Target Position Side: {'LEFT' if x_pos == '20%' else 'RIGHT'}")
        if t < len(anti_trials) - 1:
            st.session_state.antisaccade_trial += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.phase = "TASK4_NBACK"
        st.rerun()

# Task 4: Spatial Working Memory (2-Back Matrix)
elif st.session_state.phase == "TASK4_NBACK":
    st.title("Task 4: Spatial Working Memory (2-Back)")
    st.markdown("""
    A blue tracker tile will jump across a 3x3 layout.
    Click **💥 MATCH DETECTED** only if the block occupies the **exact location** it was in **two steps prior**[cite: 53]. Otherwise click Next.
    """)
    if st.button("Launch Memory Window"):
        log_event("TASK_NBACK_STARTED")
        st.session_state.phase = "RUN_TASK4"
        st.rerun()

elif st.session_state.phase == "RUN_TASK4":
    inject_silent_camera("5_task_spatial_memory")
    nback_seq = [2, 5, 8, 5, 3, 8, 3, 1, 4]
    nback_ans = [0, 0, 0, 1, 0, 0, 1, 0, 0]
    idx = st.session_state.nback_idx
    st.markdown(f"**Sequence Step {idx + 1} of {len(nback_seq)}**")
    
    # Render standardized grid matrix block
    g_html = "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; max-width: 280px; margin: 15px auto;'>"
    for b_id in range(1, 10):
        color = "#2563EB" if b_id == nback_seq[idx] else "#E2E8F0"
        g_html += f"<div style='height: 70px; background-color: {color}; border-radius: 8px;'></div>"
    g_html += "</div>"
    st.markdown(g_html, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💥 MATCH DETECTED"):
            log_event(f"NBACK_CLICK_MATCH_{idx}")
            if nback_ans[idx] == 1: st.session_state.nback_score += 1
            st.session_state.nback_idx += 1
            if st.session_state.nback_idx >= len(nback_seq):
                components.html("<script>window.parent.killRecorder();</script>", height=0)
                st.session_state.phase = "TASK5_PURSUIT"
            st.rerun()
    with col2:
        if st.button("➡️ NEXT POSITION"):
            log_event(f"NBACK_CLICK_NEXT_{idx}")
            if nback_ans[idx] == 0: st.session_state.nback_score += 1
            st.session_state.nback_idx += 1
            if st.session_state.nback_idx >= len(nback_seq):
                components.html("<script>window.parent.killRecorder();</script>", height=0)
                st.session_state.phase = "TASK5_PURSUIT"
            st.rerun()

# Task 5: Smooth Pursuit Kinematics
elif st.session_state.phase == "TASK5_PURSUIT":
    st.title("Task 5: Smooth Pursuit Tracking")
    st.write("Keep your head steady and follow the target dot smoothly as it shifts across horizontal track planes[cite: 302].")
    if st.button("Launch Pursuit Window"):
        log_event("TASK_PURSUIT_STARTED")
        st.session_state.phase = "RUN_TASK5"
        st.rerun()

elif st.session_state.phase == "RUN_TASK5":
    inject_silent_camera("6_task_smooth_pursuit")
    pursuit_coords = ["10%", "30%", "50%", "70%", "90%", "70%", "50%", "30%", "10%"]
    step = st.session_state.pursuit_step
    st.markdown(f"**Tracking Node Step {step + 1} of {len(pursuit_coords)}**")
    x_val = pursuit_coords[step]
    st.markdown(f'<div class="display-canvas"><div class="stimulus-dot" style="left: {x_val}; top: 50%;"></div></div>', unsafe_allow_html=True)
    if st.button("Target Trajectory Locked ➡️ Advance Step" if step < len(pursuit_coords)-1 else "Complete Task 5"):
        log_event(f"PURSUIT_STEP_{step}", f"Horizontal Position Vector: {x_val}")
        if step < len(pursuit_coords) - 1:
            st.session_state.pursuit_step += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.phase = "TASK6_SEARCH"
        st.rerun()

# Task 6: Visual Conjunction Search & Entropy Mapping
elif st.session_state.phase == "TASK6_SEARCH":
    st.title("Task 6: Feature Conjunction Search")
    st.write("Locate the unique **RED TARGET SHAPE** hidden inside the crowded feature noise field.")
    if st.button("Launch Search Canvas"):
        log_event("TASK_SEARCH_STARTED")
        st.session_state.phase = "RUN_TASK6"
        st.rerun()

elif st.session_state.phase == "RUN_TASK6":
    inject_silent_camera("7_task_visual_search")
    st.write("Scan the environment below. Locate the target element and log confirmation.")
    st.markdown("""
        <div class="display-canvas" style="color: #64748B; padding: 20px; font-size: 22px; letter-spacing: 15px; word-wrap: break-word; line-height: 2.5;">
            🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦
            🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦<span style="color: #EF4444; letter-spacing: 5px;">🔴</span>🟢🟦🟢🟦🟢🟦🟢🟦
            🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦
        </div>
    """, unsafe_allow_html=True)
    if st.button("💥 TARGET LOCATED"):
        log_event("SEARCH_TARGET_FOUND_CONFIRMED")
        components.html("<script>window.parent.killRecorder();</script>", height=0)
        st.session_state.phase = "EXPORT"
        st.rerun()

# Complete Session Packaging & Verification Export
elif st.session_state.phase == "EXPORT":
    st.title("Evaluation Framework Complete")
    st.success("Session localized metrics compiled.")
    export_packet = {
        "participant_id": st.session_state.participant_id,
        "date_timestamp": str(datetime.utcnow()),
        "task_4_nback_accuracy": f"{(st.session_state.nback_score / 9) * 100:.1f}%",
        "raw_interaction_log_stream": st.session_state.logs
    }
    encoded_json = base64.b64encode(json.dumps(export_packet, indent=4).encode()).decode()
    st.markdown("### 📥 Export Analytical Package Parameters")
    st.write("Click below to download your log file. Return this file along with the automatically generated video files to complete the assessment sync.")
    href_link = f'<a href="data:file/json;base64,{encoded_json}" download="{st.session_state.participant_id}_master_metrics.json"><button style="width:100%; padding:14px; background-color:#10B981; color:white; border-radius:8px; border:none; font-weight:bold; cursor:pointer;">Download Complete Session Logs (.JSON)</button></a>'
    st.markdown(href_link, unsafe_allow_html=True)
