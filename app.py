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

# =====================================================================
# 1. CORE UI CONFIGURATION (TREADWILL AESTHETIC)
# =====================================================================
st.set_page_config(page_title="Cognitive Diagnostic Engine", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Dark Slate Background & White Text */
    .stApp, .main, [data-testid="stAppViewContainer"] { background-color: #28373E !important; color: #FFFFFF !important; font-family: 'Inter', sans-serif; }
    h1, h2, h3, p, span, label { color: #FFFFFF !important; }
    
    /* Gold Accent Buttons */
    .stButton>button { 
        background-color: transparent !important; 
        color: #E3B658 !important; 
        border: 2px solid #E3B658 !important; 
        border-radius: 6px !important; 
        padding: 12px 24px !important; 
        font-size: 18px !important; 
        font-weight: 600 !important; 
        width: 100% !important; 
        transition: 0.3s ease !important;
    }
    .stButton>button:hover { background-color: #E3B658 !important; color: #28373E !important; }
    
    /* Input Fields & Cards */
    .stTextInput>div>div>input { background-color: #384A54 !important; color: #FFFFFF !important; border: 1px solid #5A7280 !important; border-radius: 6px !important; }
    .metric-card { background-color: #384A54; border: 1px solid #5A7280; padding: 20px; border-radius: 8px; text-align: center; color: #FFFFFF; }
    .metric-val { font-size: 28px; font-weight: 700; color: #E3B658; display: block; margin-top: 10px; }
    
    /* Hide Default Clutter */
    header, #MainMenu, footer {visibility: hidden;}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { background-color: #384A54; border-radius: 8px; padding: 5px; }
    .stTabs [data-baseweb="tab"] { color: #A0B2BC !important; font-weight: bold; }
    .stTabs [aria-selected="true"] { color: #E3B658 !important; background-color: #28373E !important; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

st.title("Cognitive & Gaze Diagnostic Suite")
tab1, tab2 = st.tabs(["Stage 1: Asynchronous Data Acquisition", "Stage 2: Clinical Parameter Extraction"])

# =====================================================================
# TAB 1: ZERO-LAG JAVASCRIPT ENGINE
# =====================================================================
with tab1:
    st.markdown("<p style='font-size: 16px; color: #A0B2BC !important;'>Complete the automated assessment battery. Video recording is suppressed to maintain ecological validity.</p>", unsafe_allow_html=True)
    
    html_engine = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { margin: 0; background-color: #28373E; color: #FFFFFF; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 500px; overflow: hidden; }
            #canvas-container { position: relative; width: 600px; height: 400px; background-color: #384A54; border: 1px solid #5A7280; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); }
            #ui-layer { position: absolute; width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; z-index: 10; text-align: center; padding: 20px; box-sizing: border-box;}
            h2 { font-size: 26px; margin-bottom: 10px;}
            p { color: #A0B2BC; font-size: 16px; margin-bottom: 30px;}
            .target-dot { position: absolute; width: 20px; height: 20px; background-color: #E3B658; border-radius: 50%; transform: translate(-50%, -50%); display: none; box-shadow: 0 0 12px rgba(227, 182, 88, 0.8); }
            .crosshair { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #A0B2BC; font-size: 36px; font-weight: 300; display: none; }
            button { background-color: transparent; color: #E3B658; border: 2px solid #E3B658; border-radius: 6px; padding: 12px 30px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.3s; }
            button:hover { background-color: #E3B658; color: #28373E; }
        </style>
    </head>
    <body>
        <div id="canvas-container">
            <div id="ui-layer">
                <h2 id="msg-title">System Ready</h2>
                <p id="msg-desc">Click below to initialize the camera and begin the automated task battery.</p>
                <button id="btn-start" onclick="initCamera()">Initialize Protocol</button>
            </div>
            <div id="dot" class="target-dot"></div>
            <div id="cross" class="crosshair">+</div>
        </div>

        <script>
            let mediaRecorder;
            let recordedChunks = [];
            let eventLogs = [];
            
            const uiLayer = document.getElementById('ui-layer');
            const dot = document.getElementById('dot');
            const cross = document.getElementById('cross');
            const msgTitle = document.getElementById('msg-title');
            const msgDesc = document.getElementById('msg-desc');

            function logEvent(name, details) {
                eventLogs.push({ timestamp_ms: performance.now().toFixed(2), event: name, details: details });
            }

            async function initCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", frameRate: 30 } });
                    mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
                    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
                    mediaRecorder.onstop = exportData;
                    mediaRecorder.start();
                    logEvent("SYSTEM_START", "Camera Active");
                    startCalibration();
                } catch(e) {
                    msgTitle.innerText = "Camera Access Denied";
                    msgDesc.innerText = "Please allow permissions in your browser to proceed.";
                }
            }

            // --- TASK 1: 9-POINT CALIBRATION ---
            function startCalibration() {
                uiLayer.style.display = 'none';
                dot.style.display = 'block';
                logEvent("TASK_START", "Calibration");
                const coords = [ [10,10], [50,10], [90,10], [10,50], [50,50], [90,50], [10,90], [50,90], [90,90] ];
                let step = 0;
                
                function moveDot() {
                    if (step >= coords.length) { dot.style.display = 'none'; startAntiSaccade(); return; }
                    dot.style.left = coords[step][0] + '%'; dot.style.top = coords[step][1] + '%';
                    logEvent("CALIB_POINT", `X:${coords[step][0]}, Y:${coords[step][1]}`);
                    step++;
                    setTimeout(moveDot, 1500); // 1.5s interval
                }
                moveDot();
            }

            // --- TASK 2: ANTI-SACCADE ---
            function startAntiSaccade() {
                logEvent("TASK_START", "Anti-Saccade");
                const trials = [20, 80, 80, 20]; // 20% = Left, 80% = Right
                let step = 0;
                
                function runTrial() {
                    if (step >= trials.length) { dot.style.display = 'none'; mediaRecorder.stop(); return; }
                    cross.style.display = 'block';
                    
                    setTimeout(() => { // Show cross for 1000ms
                        cross.style.display = 'none';
                        dot.style.left = trials[step] + '%'; dot.style.top = '50%'; dot.style.display = 'block';
                        logEvent("ANTI_STIMULUS", trials[step] === 20 ? "LEFT" : "RIGHT");
                        
                        setTimeout(() => { // Show dot for 1000ms
                            dot.style.display = 'none'; 
                            step++; 
                            setTimeout(runTrial, 500); // 500ms gap before next trial
                        }, 1000);
                    }, 1000);
                }
                runTrial();
            }

            // --- DATA EXPORT HANDLER ---
            function exportData() {
                // Export Video
                const vidBlob = new Blob(recordedChunks, { type: 'video/webm' });
                const aVid = document.createElement('a');
                aVid.href = URL.createObjectURL(vidBlob); aVid.download = "raw_gaze_video.webm"; aVid.click();
                
                // Export Timestamps
                const jsonBlob = new Blob([JSON.stringify(eventLogs, null, 2)], { type: 'application/json' });
                const aJson = document.createElement('a');
                aJson.href = URL.createObjectURL(jsonBlob); aJson.download = "interaction_logs.json"; aJson.click();
                
                uiLayer.style.display = 'flex';
                msgTitle.innerText = "Assessment Complete";
                msgDesc.innerText = "Files downloaded securely. Please proceed to Stage 2.";
                document.getElementById('btn-start').style.display = 'none';
            }
        </script>
    </body>
    </html>
    """
    components.html(html_engine, height=550)

# =====================================================================
# TAB 2: CLINICAL PARAMETER EXTRACTION (PYTHON BACKEND)
# =====================================================================
with tab2:
    st.markdown("<p style='font-size: 16px; color: #A0B2BC !important;'>Upload the exported raw video and JSON log matrix to compute clinical features via I-VT.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        video_file = st.file_uploader("Upload raw_gaze_video.webm", type=['webm', 'mp4'])
    with col2:
        log_file = st.file_uploader("Upload interaction_logs.json", type=['json'])

    if video_file and log_file:
        if st.button("Execute Clinical Parameter Calculation"):
            with st.spinner("Processing computer vision matrices and extracting gaze vectors..."):
                
                # Load JSON logs
                logs = json.load(log_file)
                
                # Save video bytes to a temporary file for OpenCV to read
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
                tfile.write(video_file.read())
                tfile.close()
                
                cap = cv2.VideoCapture(tfile.name)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                
                # Initialize MediaPipe Face Mesh
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5)
                
                gaze_stream = []
                frame_idx = 0
                
                # Process video frames
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        landmarks = results.multi_face_landmarks[0].landmark
                        # Extract left (468) and right (473) iris center coordinates [cite: 108]
                        lx, ly = landmarks[468].x, landmarks[468].y
                        rx, ry = landmarks[473].x, landmarks[473].y
                        
                        # Average position for stabilization
                        avg_x, avg_y = (lx + rx) / 2.0, (ly + ry) / 2.0
                        timestamp_ms = (frame_idx / fps) * 1000.0
                        gaze_stream.append({'t': timestamp_ms, 'x': avg_x, 'y': avg_y})
                        
                    frame_idx += 1
                cap.release()
                face_mesh.close()
                os.unlink(tfile.name)

                # -------------------------------------------------------------
                # THE I-VT VELOCITY ALGORITHM & FEATURE EXTRACTION
                # -------------------------------------------------------------
                VELOCITY_THRESH = 100.0 # deg/s [cite: 184]
                D_CM = 33.0 # Viewing distance [cite: 154]
                PX_PER_CM = 1920 / 6.8 # Assuming standard 1080p horizontal layout math
                
                fixations = []
                saccades = []
                
                # Parse Fixations and Saccades based on angular velocity [cite: 171-185]
                for i in range(1, len(gaze_stream)):
                    t1, x1, y1 = gaze_stream[i-1]['t'], gaze_stream[i-1]['x'], gaze_stream[i-1]['y']
                    t2, x2, y2 = gaze_stream[i]['t'], gaze_stream[i]['x'], gaze_stream[i]['y']
                    
                    dt = (t2 - t1) / 1000.0 # Seconds
                    if dt > 0:
                        # Convert normalized relative coordinates to visual degrees
                        dist_px = np.sqrt(((x2 - x1)*1920)**2 + ((y2 - y1)*1080)**2)
                        dist_cm = dist_px / PX_PER_CM
                        dist_deg = np.degrees(np.arctan(dist_cm / D_CM))
                        velocity = dist_deg / dt
                        
                        if velocity >= VELOCITY_THRESH:
                            saccades.append({'t_start': t1, 't_end': t2, 'amp': dist_deg, 'x_end': x2})
                        else:
                            fixations.append({'t_start': t1, 't_end': t2, 'x': x2, 'y': y2, 'duration': dt * 1000})

                # Aggregate consecutive fixations to filter noise
                cleaned_fixations = []
                current_fix = []
                for f in fixations:
                    if not current_fix or (f['t_start'] - current_fix[-1]['t_end']) < 50:
                        current_fix.append(f)
                    else:
                        dur = sum([cf['duration'] for cf in current_fix])
                        if dur >= 100.0: # Minimum 100ms threshold [cite: 199]
                            cleaned_fixations.append({
                                't_start': current_fix[0]['t_start'],
                                'duration': dur,
                                'x': np.mean([cf['x'] for cf in current_fix]),
                                'y': np.mean([cf['y'] for cf in current_fix])
                            })
                        current_fix = [f]

                # --- CALCULATE SPECIFIC 10 FEATURES [cite: 485, 487] ---
                
                # F1 & F2: Mean Fixation Duration & Count
                f1_mfd = np.mean([f['duration'] for f in cleaned_fixations]) if cleaned_fixations else 0
                f2_fc = len(cleaned_fixations)
                
                # F4: Saccade Amplitude
                f4_sa = np.mean([s['amp'] for s in saccades]) if saccades else 0

                # F6 & F8: Gaze Path Entropy & ROI Coverage (5x5 Grid)
                grid = np.zeros((5, 5))
                for f in cleaned_fixations:
                    gx, gy = int(np.clip(f['x'] * 5, 0, 4)), int(np.clip(f['y'] * 5, 0, 4))
                    grid[gx, gy] += f['duration']
                    
                f8_roi = (np.count_nonzero(grid) / 25.0) * 100
                f6_entropy = 0
                if np.sum(grid) > 0:
                    pk = grid.flatten() / np.sum(grid)
                    f6_entropy = -sum([p * log2(p) for p in pk if p > 0])
                
                # F3 & F7: Anti-Saccade Latency & Error Rate
                anti_errors = 0
                latencies = []
                anti_trials = 0
                
                for event in logs:
                    if event['event'] == "ANTI_STIMULUS":
                        anti_trials += 1
                        is_left = event['details'] == "LEFT"
                        event_time = float(event['timestamp_ms'])
                        
                        # Find the first saccade executed after the stimulus onset
                        first_sac = next((s for s in saccades if s['t_start'] >= event_time), None)
                        if first_sac:
                            latencies.append(first_sac['t_start'] - event_time)
                            # Error if they looked towards the stimulus (Left is < 0.5 normalized X)
                            sac_went_left = first_sac['x_end'] < 0.5
                            if (is_left and sac_went_left) or (not is_left and not sac_went_left):
                                anti_errors += 1
                                
                f3_sl = np.mean(latencies) if latencies else 0
                f7_aser = (anti_errors / anti_trials * 100) if anti_trials > 0 else 0.0

                # --- RENDER RESULTS MATRIX ---
                st.markdown("<h3 style='margin-top: 30px; color: #FFFFFF;'>Calculated Biomarkers (Stage 0 Assessment)</h3>", unsafe_allow_html=True)
                
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"<div class='metric-card'>Mean Fixation (F1)<span class='metric-val'>{f1_mfd:.1f} ms</span></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='metric-card'>Fixation Count (F2)<span class='metric-val'>{f2_fc}</span></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='metric-card'>Saccade Latency (F3)<span class='metric-val'>{f3_sl:.1f} ms</span></div>", unsafe_allow_html=True)
                c4.markdown(f"<div class='metric-card'>Saccade Amp (F4)<span class='metric-val'>{f4_sa:.1f} deg</span></div>", unsafe_allow_html=True)
                
                st.write("")
                
                c5, c6, c7, c8 = st.columns(4)
                c5.markdown(f"<div class='metric-card'>Gaze Entropy (F6)<span class='metric-val'>{f6_entropy:.2f} bits</span></div>", unsafe_allow_html=True)
                c6.markdown(f"<div class='metric-card'>Anti-Sac Error (F7)<span class='metric-val'>{f7_aser:.1f} %</span></div>", unsafe_allow_html=True)
                c7.markdown(f"<div class='metric-card'>ROI Coverage (F8)<span class='metric-val'>{f8_roi:.1f} %</span></div>", unsafe_allow_html=True)
                
                # Output Packaging
                results_df = pd.DataFrame([{
                    "F1_MeanFixationDuration_ms": f1_mfd,
                    "F2_FixationCount": f2_fc,
                    "F3_SaccadeLatency_ms": f3_sl,
                    "F4_SaccadeAmplitude_deg": f4_sa,
                    "F6_GazeEntropy_bits": f6_entropy,
                    "F7_AntiSaccadeErrorRate_pct": f7_aser,
                    "F8_ROICoverage_pct": f8_roi
                }])
                
                csv_data = results_df.to_csv(index=False).encode('utf-8')
                
                st.write("")
                st.download_button(
                    label="Download Clinical Parameters (.CSV)",
                    data=csv_data,
                    file_name="clinical_parameters_output.csv",
                    mime="text/csv",
                )
                
