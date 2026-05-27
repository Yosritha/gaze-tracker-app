import streamlit as st
import streamlit.components.v1 as components
import json
import base64
from datetime import datetime

# Enforce clean, minimalist web layout matching premium clinical applications
st.set_page_config(page_title="Cognitive Evaluation Portal", layout="centered", initial_sidebar_state="collapsed")

# Visual interface styles (High-contrast blue and slate)
st.markdown("""
    <style>
    .main { background-color: #FFFFFF; color: #1E293B; font-family: sans-serif; }
    h1, h2, h3 { color: #0F172A; }
    .stButton>button {
        background-color: #2563EB; color: white; border-radius: 8px;
        padding: 12px; font-size: 16px; font-weight: 600; border: none; width: 100%;
    }
    .stButton>button:hover { background-color: #1D4ED8; }
    .consent-box {
        background-color: #F8FAFC; border: 1px solid #E2E8F0;
        padding: 15px; border-radius: 8px; max-height: 250px; overflow-y: scroll; font-size: 14px;
    }
    .grid-box {
        background-color: #F1F5F9; border-radius: 12px; padding: 20px;
        text-align: center; margin: 10px 0; border: 2px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize application tracking variables
if 'phase' not in st.session_state: st.session_state.phase = "LOGIN"
if 'participant_id' not in st.session_state: st.session_state.participant_id = ""
if 'logs' not in st.session_state: st.session_state.logs = []
if 'trial_index' not in st.session_state: st.session_state.trial_index = 0
if 'correct_clicks' not in st.session_state: st.session_state.correct_clicks = 0

def add_log_entry(event_type, description=""):
    st.session_state.logs.append({
        "timestamp_ms": int(datetime.utcnow().timestamp() * 1000),
        "event": event_type,
        "details": description
    })

# JavaScript Injection to silently stream camera frames directly to user's browser download cache
def inject_silent_camera(filename_suffix):
    html_js = f"""
    <div style="font-size:12px; color:#64748B; margin-bottom:8px;">🎥 Biometric Sensor Connected (Camera preview hidden to prevent visual distraction)</div>
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
                downloadLink.download = "{st.session_state.participant_id}_" + "{filename_suffix}.webm";
                document.body.appendChild(downloadLink);
                downloadLink.click();
                document.body.removeChild(downloadLink);
            }};
            
            recorder.start();
            window.parent.killRecorder = () => {{
                recorder.stop();
                mediaStream.getTracks().forEach(track => track.stop());
            }};
        }} catch (cameraError) {{
            console.log("Camera access denied.");
        }}
    }})();
    </script>
    """
    components.html(html_js, height=35)

# --- LAYOUT STATE 1: LOGIN ENTRY ---
if st.session_state.phase == "LOGIN":
    st.title("Cognitive Assessment Portal")
    st.write("Please sign in with the unique Participant ID provided by your researcher.")
    user_id_input = st.text_input("Enter Participant ID Token:", placeholder="e.g., P012")
    
    if st.button("Open Assessment Portal"):
        if user_id_input.strip():
            st.session_state.participant_id = user_id_input.strip()
            st.session_state.phase = "CONSENT_FORM"
            st.rerun()
        else:
            st.error("A valid identifier token is required to start the session.")

# --- LAYOUT STATE 2: INFORMED CONSENT ---
elif st.session_state.phase == "CONSENT_FORM":
    st.title("Informed Consent Agreement")
    st.write("Please read through the ethical framework before confirming authorization checkboxes.")
    
    st.markdown(f"""
    <div class="consent-box">
        <strong>Study Title:</strong> Eye Tracking Metrics for Early-Stage Cognitive Variational Analysis<br>
        <strong>Lead Investigator:</strong> Yosritha Laxmi Pasupuleti, Indian Institute of Technology Tirupati<br><br>
        <strong>Data Privacy Standards:</strong><br>
        1. <strong>Biometric Access:</strong> The app programmatically uses your front-facing selfie camera or webcam to record your eyes at 30 frames per second.<br>
        2. <strong>Anonymization:</strong> Video sequences are saved under your random ID code: <code>{st.session_state.participant_id}</code>. No personal identification metrics are shared.<br>
        3. <strong>Right to Exit:</strong> You can close this browser tab at any time to immediately delete your active session and drop out of the project.
    </div>
    """, unsafe_allow_html=True)
    
    check_video = st.checkbox("I authorize the app to access my camera silently to record eye movements.")
    check_data = st.checkbox("I authorize my completely anonymized data logs to be used in academic publications.")
    
    if st.button("Sign Consent & Continue"):
        if check_video and check_data:
            add_log_entry("CONSENT_GIVEN")
            st.session_state.phase = "CALIBRATION_PROMPT"
            st.rerun()
        else:
            st.warning("You must click both checkboxes to proceed with the assessment.")

# --- LAYOUT STATE 3: CALIBRATION SETUP ---
elif st.session_state.phase == "CALIBRATION_PROMPT":
    st.title("Step 1: Eye Alignment Calibration")
    st.markdown("""
    To calibrate the tracking eye models correctly:
    * **If on a Phone:** Put the phone in a stable stand on your desk. Don't hold it in your hand.
    * **If on a Laptop:** Keep the laptop completely flat on a table.
    * **Distance:** Sit naturally at a normal viewing distance (**30–35 cm** away from the screen).
    """)
    if st.button("Start 9-Point Calibration Grid"):
        add_log_entry("CALIBRATION_STARTED")
        st.session_state.phase = "EXECUTE_CALIBRATION"
        st.rerun()

elif st.session_state.phase == "EXECUTE_CALIBRATION":
    inject_silent_camera("calibration_video")
    st.write("👉 **Look directly at the center of the blue target square as it moves. Do not tilt your head.**")
    
    grid_points = ["Top-Left", "Top-Center", "Top-Right", "Mid-Left", "Center-Focus", "Mid-Right", "Bottom-Left", "Bottom-Center", "Bottom-Right"]
    active_point = st.slider("Calibration Progression Step", 0, 8, 0)
    
    st.markdown(f"""
        <div class="grid-box">
            <span style='font-size: 24px; font-weight: bold; color: #2563EB;'>🎯 LOOK HERE: {grid_points[active_point]}</span>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("Record Point Coordinates" if active_point < 8 else "Lock Calibration Grid"):
        add_log_entry(f"CALIBRATION_NODE_SAVED_{active_point}", f"Grid Location: {grid_points[active_point]}")
        if active_point == 8:
            components.html("<script>window.parent.killRecorder();</script>", height=0)
            st.session_state.phase = "TASK_MEM_INTRO"
        st.rerun()

# --- LAYOUT STATE 4: COGNITIVE N-BACK TASK ---
elif st.session_state.phase == "TASK_MEM_INTRO":
    st.title("Step 2: Spatial Memory Assessment")
    st.markdown("""
    **How to play this task:**
    A blue square will jump sequentially around a 3x3 grid layout.
    
    Click the **💥 MATCH DETECTED** button only if the blue square lands in the **exact same spot** it was in **two steps ago** (2-Back memory match). Otherwise, click Next.
    """)
    if st.button("Launch Memory Window"):
        add_log_entry("TASK_2BACK_STARTED")
        st.session_state.phase = "EXECUTE_TASK_MEM"
        st.rerun()

elif st.session_state.phase == "EXECUTE_TASK_MEM":
    inject_silent_camera("task_memory_video")
    
    grid_sequence = [2, 5, 8, 5, 3, 8, 3, 1, 4]
    match_answers = [0, 0, 0, 1, 0, 0, 1, 0, 0] 
    idx = st.session_state.trial_index
    
    st.markdown(f"**Grid Sequence Item {idx + 1} of {len(grid_sequence)}**")
    
    # Force a max-width layout container so the grid looks uniform on both laptops and phones
    left_margin, core_ui, right_margin = st.columns([1, 2, 1])
    with core_ui:
        # Build 3x3 Grid
        html_grid = "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; max-width: 280px; margin: 15px auto;'> "
        for box_id in range(1, 10):
            color = "#2563EB" if box_id == grid_sequence[idx] else "#E2E8F0"
            html_grid += f"<div style='height: 75px; background-color: {color}; border-radius: 8px;'></div>"
        html_grid += "</div>"
        st.markdown(html_grid, unsafe_allow_html=True)
        
        btn_left, btn_right = st.columns(2)
        with btn_left:
            if st.button("💥 MATCH DETECTED"):
                add_log_entry(f"USER_CLICKED_MATCH_TRIAL_{idx}")
                if match_answers[idx] == 1: st.session_state.correct_clicks += 1
                st.session_state.trial_index += 1
                if st.session_state.trial_index >= len(grid_sequence):
                    components.html("<script>window.parent.killRecorder();</script>", height=0)
                    st.session_state.phase = "COMPLETE_DOWNLOAD"
                st.rerun()
        with btn_right:
            if st.button("➡️ NO MATCH / NEXT"):
                add_log_entry(f"USER_CLICKED_NEXT_TRIAL_{idx}")
                if match_answers[idx] == 0: st.session_state.correct_clicks += 1
                st.session_state.trial_index += 1
                if st.session_state.trial_index >= len(grid_sequence):
                    components.html("<script>window.parent.killRecorder();</script>", height=0)
                    st.session_state.phase = "COMPLETE_DOWNLOAD"
                st.rerun()

# --- LAYOUT STATE 5: SESSION PACKAGING & EXPORT ---
elif st.session_state.phase == "COMPLETE_DOWNLOAD":
    st.title("Assessment Complete")
    st.success("Great job! Your testing files have been generated.")
    
    structured_json_output = {
        "participant_id": st.session_state.participant_id,
        "session_date": str(datetime.utcnow()),
        "behavioral_accuracy": f"{(st.session_state.correct_clicks / 9) * 100:.1f}%",
        "raw_interaction_timestamps": st.session_state.logs
    }
    
    compiled_json = json.dumps(structured_json_output, indent=4)
    b64_data = base64.b64encode(compiled_json.encode()).decode()
    
    st.markdown("### 📥 Download Your Session Results")
    st.write("Click the green button below to save your experiment log file. Send this file and the two video files in your downloads folder to your research coordinator.")
    
    download_button_html = f'<a href="data:file/json;base64,{b64_data}" download="{st.session_state.participant_id}_interaction_logs.json"><button style="width:100%; padding:14px; background-color:#10B981; color:white; border-radius:8px; border:none; font-weight:bold; cursor:pointer;">Download Session Logs (.JSON)</button></a>'
    st.markdown(download_button_html, unsafe_allow_html=True)