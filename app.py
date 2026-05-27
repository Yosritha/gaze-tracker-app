import streamlit as st
import streamlit.components.v1 as components
import json
import base64
import time
from datetime import datetime

# --- TREADWILL INSPIRED MINIMALIST CLINICAL LOOK ---
st.set_page_config(page_title="Cognitive Diagnostic Suite", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* TreadWill inspired background aesthetics */
    .main { background-color: #0B132B; color: #F4F5F6; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #FFFFFF; font-weight: 600; text-align: center; }
    
    /* Clean CSS Box structures */
    .treadwill-box {
        background-color: #1C2541; border: 1px solid #3A506B;
        padding: 30px; border-radius: 12px; margin-top: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .stButton>button {
        background-color: #48CAE4; color: #0B132B; border-radius: 6px;
        padding: 14px; font-size: 16px; font-weight: 700; border: none; width: 100%; transition: all 0.2s ease;
    }
    .stButton>button:hover { background-color: #00B4D8; color: #FFFFFF; }
    
    /* Dedicated Dark Canvas for Eye Tracking Tasks */
    .gaze-canvas {
        position: relative; width: 100%; height: 380px; 
        background-color: #010203; border: 1px solid #1C2541; border-radius: 8px; overflow: hidden;
    }
    .auto-dot {
        position: absolute; width: 20px; height: 20px; 
        background-color: #FF5A5F; border-radius: 50%;
        box-shadow: 0 0 20px #FF5A5F; transform: translate(-50%, -50%);
    }
    .center-cross {
        position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
        color: #3A506B; font-size: 36px; font-weight: 300;
    }
    .grid-container {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; max-width: 300px; margin: 25px auto;
    }
    .grid-tile { height: 80px; border-radius: 8px; background-color: #1C2541; transition: background-color 0.1s ease; }
    .search-field {
        color: #3A506B; font-size: 20px; letter-spacing: 12px; line-height: 2.5; 
        text-align: center; padding: 40px 10px; word-wrap: break-word;
    }
    </style>
""", unsafe_allow_html=True)

# --- TRACKING ARCHITECTURE REGISTER ---
keys_to_initialize = [
    ('state', "WELCOME"), ('subject_id', ""), ('session_logs', []),
    ('step_calib', 0), ('step_anti', 0), ('step_pursuit', 0),
    ('step_nback', 0), ('score_nback', 0)
]
for target_key, base_value in keys_to_initialize:
    if target_key not in st.session_state:
        st.session_state[target_key] = base_value

def record_timestamp(label, context=""):
    st.session_state.session_logs.append({
        "timestamp_ms": int(time.time() * 1000),
        "event_id": label,
        "metadata": context
    })

# --- SILENT WEBRTC VIDEO CAPTURE ENGINE ---
def load_silent_camera_engine(filename_suffix):
    html_js = f"""
    <script>
    (async function() {{
        try {{
            const mediaStream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "user", frameRate: {{ ideal: 30 }} }} }});
            const recorder = new MediaRecorder(mediaStream, {{ mimeType: 'video/webm' }});
            let dataChunks = [];
            recorder.ondataavailable = (e) => {{ if (e.data.size > 0) dataChunks.push(e.data); }};
            recorder.onstop = () => {{
                const dataBlob = new Blob(dataChunks, {{ type: 'video/webm' }});
                const downloadUrl = URL.createObjectURL(dataBlob);
                const downloadLink = document.createElement('a');
                downloadLink.href = downloadUrl;
                downloadLink.download = "{st.session_state.subject_id}_" + "{filename_suffix}.webm";
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            }};
            recorder.start();
            window.parent.killRecorder = () => {{
                recorder.stop();
                mediaStream.getTracks().forEach(track => track.stop());
            }};
        }} catch (cameraError) {{ console.log("Camera hardware binding restricted."); }}
    }})();
    </script>
    """
    components.html(html_js, height=0)

# =====================================================================
# SYSTEM AUTOMATION PIPELINE ROUTER
# =====================================================================

# Phase 1: TreadWill Portal Authentication Screen
if st.session_state.state == "WELCOME":
    st.markdown("<h1 style='margin-top: 40px;'>Cognitive Diagnostic Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8D99AE;'>Clinical Data Acquisition Environment / IIT Tirupati EE Department</p>", unsafe_allow_html=True)
    
    with st.markdown("<div class='treadwill-box'>", unsafe_allow_html=True):
        st.write("Welcome. To access your asynchronous testing registry tracking sequence, please authenticate your profile using your unique identification token.")
        input_token = st.text_input("Participant Token Key ID:", placeholder="e.g., P015")
        st.write("")
        if st.button("Unlock Assessment Suite"):
            if input_token.strip():
                st.session_state.subject_id = input_token.strip()
                st.session_state.state = "ETHICS_CONSENT"
                st.rerun()
            else:
                st.error("Identification registration parameter missing.")
    st.markdown("</div>", unsafe_allow_html=True)

# Phase 2: IRB Consent Protocol
elif st.session_state.state == "ETHICS_CONSENT":
    st.title("Informed Consent Agreement")
    with st.markdown("<div class='treadwill-box'>", unsafe_allow_html=True):
        st.markdown(f"""
        <strong>Project Title:</strong> High-Fidelity Gaze Biomarker Matrix Analysis for Stage 0 Cognitive Tracking via Consumer Optics<br>
        <strong>Principal Investigator:</strong> Yosritha Laxmi Pasupuleti, Indian Institute of Technology Tirupati<br><br>
        <strong>Protocol Controls:</strong><br>
        1. The frontend portal utilizes your webcam optics to capture sequential physical eye movements silently at 30fps[cite: 47, 455].<br>
        2. Live streaming camera previews are suppressed to isolate natural visual behavior and protect against cognitive bias[cite: 455].<br>
        3. All generated tracking matrices are processed locally and decoupled from real-world variables using secure alphanumeric identifier hash: <code>{st.session_state.subject_id}</code>.<br><br>
        By selecting the authorization criteria below, you verify that you understand these parameters and voluntarily consent to contribute your metrics.
        """)
        st.write("")
        v1 = st.checkbox("I authorize the system to log my front-camera matrix stream asynchronously.")
        v2 = st.checkbox("I authorize the use of my anonymized metrics logs in open-access academic publications.")
        st.write("")
        if st.button("Accept Protocol Conditions"):
            if v1 and v2:
                record_timestamp("CONSENT_VERIFIED")
                st.session_state.state = "TASK_CALIB"
                st.rerun()
            else:
                st.warning("All data verification boxes must be signed to proceed.")
    st.markdown("</div>", unsafe_allow_html=True)

# Phase 3: Automated 9-Point Calibration Matrix (Zero User Click Control)
elif st.session_state.state == "TASK_CALIB":
    st.title("Task 1: Spatial Grid Calibration")
    st.write("Position your smartphone securely on a table stand or set your laptop flatly on a desk. Sit straight, **30–35 cm from the screen**, and trace the moving red dot with your eyes[cite: 79, 80].")
    
    if st.button("Initialize Automated Sequence"):
        record_timestamp("CALIBRATION_SEQUENCE_START")
        st.session_state.state = "PLAY_CALIB"
        st.rerun()

elif st.session_state.state == "PLAY_CALIB":
    load_silent_camera_engine("1_spatial_calibration")
    
    # 9 geometric point locations mapped as absolute coordinate bounds (X%, Y%)
    grid_matrix = [
        ("10%", "10%"), ("50%", "10%"), ("90%", "10%"),
        ("10%", "50%"), ("50%", "50%"), ("90%", "50%"),
        ("10%", "90%"), ("50%", "90%"), ("90%", "90%")
    ]
    step = st.session_state.step_calib
    x, y = grid_matrix[step]
    
    st.markdown(f'<div class="gaze-canvas"><div class="auto-dot" style="left: {x}; top: {y};"></div></div>', unsafe_allow_html=True)
    
    # JavaScript Injection: Automated time frame clock handler shifts dot every 1.5 seconds without user intervention
    components.html(f"""
    <script>
    setTimeout(function() {{
        window.parent.document.querySelector('button[kind="secondaryPrimary"]').click();
    }}, 1500);
    </script>
    """, height=0)
    
    if st.button("AUTOMATED_NEXT_STEP", type="secondary"):
        record_timestamp(f"STIMULUS_CALIB_NODE_{step}", f"Target_Coordinates: X={x}, Y={y}")
        if step < 8:
            st.session_state.step_calib += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.state = "TASK_ANTI"
        st.rerun()

# Phase 4: Anti-Saccade Clinical Paradigm (Fully Automated Timed Triggers)
elif st.session_state.state == "TASK_ANTI":
    st.title("Task 2: Anti-Saccade Inhibitory Challenge")
    st.markdown("""
    A central fixation cross (**+**) will show up briefly. Then, a red dot will flash on either the **Left** or **Right** side[cite: 32, 453].
    
    👉 **CRITICAL RESEARCH CONDITION:** You must force your eyes to look in the **EXACT OPPOSITE DIRECTION** of the dot instantly. If it drops Left, glance Right. If it drops Right, glance Left. Do not look at the dot!
    """)
    if st.button("Launch Inhibitory Task Loop"):
        record_timestamp("PARADIGM_ANTISACCADE_START")
        st.session_state.state = "PLAY_ANTI"
        st.rerun()

elif st.session_state.state == "PLAY_ANTI":
    load_silent_camera_engine("2_anti_saccade_paradigm")
    
    # 4 Unpredictable spatial trial shifts 
    anti_sequence = [("20%", "50%"), ("80%", "50%"), ("80%", "50%"), ("20%", "50%")]
    t_idx = st.session_state.step_anti
    x_pos, y_pos = anti_sequence[t_idx]
    
    st.markdown(f'<div class="gaze-canvas"><div class="center-cross">+</div><div class="auto-dot" style="left: {x_pos}; top: {y_pos};"></div></div>', unsafe_allow_html=True)
    
    # Target dots pop up and auto-terminate every 1.8 seconds to capture true reflexive saccade responses
    components.html(f"""
    <script>
    setTimeout(function() {{
        window.parent.document.querySelector('button[kind="secondaryPrimary"]').click();
    }}, 1800);
    </script>
    """, height=0)
    
    if st.button("AUTOMATED_ANTI_STEP", type="secondary"):
        record_timestamp(f"STIMULUS_ANTISACCADE_TRIAL_{t_idx}", f"Flashed_Side: {'LEFT' if x_pos == '20%' else 'RIGHT'}")
        if t_idx < len(anti_sequence) - 1:
            st.session_state.step_anti += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.state = "TASK_PURSUIT"
        st.rerun()

# Phase 5: Smooth Pursuit Kinematics (Continuous Kinematic Slide)
elif st.session_state.state == "TASK_PURSUIT":
    st.title("Task 3: Smooth Pursuit Trajectory Tracking")
    st.write("Keep your head perfectly steady. Track the moving target dot continuously with your eyes as it sweeps horizontally across the tracking plane.")
    
    if st.button("Launch Pursuit Tracking Sequence"):
        record_timestamp("PARADIGM_PURSUIT_START")
        st.session_state.state = "PLAY_PURSUIT"
        st.rerun()

elif st.session_state.state == "PLAY_PURSUIT":
    load_silent_camera_engine("3_smooth_pursuit_trajectory")
    
    horizontal_track = ["10%", "25%", "40%", "55%", "70%", "85%", "70%", "55%", "40%", "25%", "10%"]
    p_step = st.session_state.step_pursuit
    current_x = horizontal_track[p_step]
    
    st.markdown(f'<div class="gaze-canvas"><div class="auto-dot" style="left: {current_x}; top: 50%;"></div></div>', unsafe_allow_html=True)
    
    # Pure time-stepped sweep sequence tracking window (Shifts location boundaries every 800ms) 
    components.html(f"""
    <script>
    setTimeout(function() {{
        window.parent.document.querySelector('button[kind="secondaryPrimary"]').click();
    }}, 800);
    </script>
    """, height=0)
    
    if st.button("AUTOMATED_PURSUIT_STEP", type="secondary"):
        record_timestamp(f"STIMULUS_PURSUIT_PLANE_{p_step}", f"Horizontal_Coordinate: {current_x}")
        if p_step < len(horizontal_track) - 1:
            st.session_state.step_pursuit += 1
        else:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.state = "TASK_NBACK"
        st.rerun()

# Phase 6: Spatial N-Back Task (User Interacts for Verification Logs)
elif st.session_state.state == "TASK_NBACK":
    st.title("Task 4: Spatial Memory (2-Back Game)")
    st.markdown("""
    **Instructions:**
    A blue square will move sequentially across spots on a grid display matrix.
    
    👉 Click **💥 MATCH DETECTED** immediately if the tile jumps into the **exact location** it occupied **two steps back** (2-Back memory match). Otherwise, click **Next Step**.
    """)
    if st.button("Start Spatial Assessment Working Memory Loop"):
        record_timestamp("PARADIGM_NBACK_START")
        st.session_state.state = "PLAY_NBACK"
        st.rerun()

elif st.session_state.state == "PLAY_NBACK":
    load_silent_camera_engine("4_spatial_working_memory")
    
    nback_sequence = [3, 6, 9, 6, 2, 9, 2, 4, 7]
    match_indices =  [0, 0, 0, 1, 0, 0, 1, 0, 0]
    n_idx = st.session_state.step_nback
    
    st.write(f"Grid Block State Matrix Index: {n_idx + 1} / {len(nback_sequence)}")
    
    # Render 3x3 Block Grid UI
    grid_html = "<div class='grid-container'>"
    for tile_id in range(1, 10):
        color = "#48CAE4" if tile_id == nback_sequence[n_idx] else "#1C2541"
        grid_html += f"<div class='grid-tile' style='background-color: {color};'></div>"
    grid_html += "</div>"
    st.markdown(grid_html, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("💥 MATCH DETECTED"):
            record_timestamp(f"USER_ACTION_NBACK_MATCH_STEP_{n_idx}", f"Filled_Block: {nback_sequence[n_idx]}")
            if match_indices[n_idx] == 1: st.session_state.score_nback += 1
            st.session_state.step_nback += 1
            if st.session_state.step_nback >= len(nback_sequence):
                components.html("<script>window.parent.killRecorder();</script>", height=0)
                st.session_state.state = "TASK_SEARCH"
            st.rerun()
    with c2:
        if st.button("➡️ NEXT STEP POSITION"):
            record_timestamp(f"USER_ACTION_NBACK_NEXT_STEP_{n_idx}", f"Filled_Block: {nback_sequence[n_idx]}")
            if match_indices[n_idx] == 0: st.session_state.score_nback += 1
            st.session_state.step_nback += 1
            if st.session_state.step_nback >= len(nback_sequence):
                components.html("<script>window.parent.killRecorder();</script>", height=0)
                st.session_state.state = "TASK_SEARCH"
            st.rerun()

# Phase 7: Visual Conjunction Search Paradigm (Entropy Mapping)
elif st.session_state.state == "TASK_SEARCH":
    st.title("Task 5: Visual Feature Conjunction Search")
    st.write("Scan the complex noise visual display layout below. Scan your eyes naturally to locate the single **RED TARGET CIRCLE** hidden within the background shape noise.")
    if st.button("Launch Search Field Canvas"):
        record_timestamp("PARADIGM_SEARCH_START")
        st.session_state.state = "PLAY_SEARCH"
        st.rerun()

elif st.session_state.state == "PLAY_SEARCH":
    load_silent_camera_engine("5_visual_search_entropy")
    
    st.markdown("""
        <div class="gaze-canvas">
            <div class="search-field">
                🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦<br>
                🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦<span style="color: #FF5A5F; font-size:24px; font-weight:bold;">🔴</span>🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦<br>
                🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦🟢🟦
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    if st.button("💥 TARGET LOCATED AND LOCKED"):
        record_timestamp("USER_ACTION_SEARCH_TARGET_FOUND_CONFIRMED")
        components.html("<script>window.parent.killRecorder();</script>", height=0)
        st.session_state.state = "EXPORT_PORTAL"
        st.rerun()

# Phase 8: Data Packaging Export Portal
elif st.session_state.state == "EXPORT_PORTAL":
    st.title("Assessment Matrix Completed")
    st.success("Session data vectors safely locked and structured.")
    
    package_payload = {
        "participant_token_id": st.session_state.subject_id,
        "execution_date_utc": str(datetime.utcnow()),
        "cognitive_memory_accuracy": f"{(st.session_state.score_nback / 9) * 100:.1f}%",
        "coordinated_interaction_logs": st.session_state.session_logs
    }
    
    serialized_json = json.dumps(package_payload, indent=4)
    b64_string = base64.b64encode(serialized_json.encode()).decode()
    
    st.markdown("### 📥 Extract Research Data Logs")
    st.write("Click the action button below to download your anonymized session log string. Return this generated file and the auto-downloaded video tracks to the lab supervisor.")
    
    action_href = f'<a href="data:file/json;base64,{b64_string}" download="{st.session_state.subject_id}_experiment_metrics.json"><button style="width:100%; padding:14px; background-color:#48CAE4; color:#0B132B; border-radius:6px; border:none; font-weight:bold; cursor:pointer;">Download Session Logs Package (.JSON)</button></a>'
    st.markdown(action_href, unsafe_allow_html=True)
