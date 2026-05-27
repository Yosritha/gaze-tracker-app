import streamlit as st
import streamlit.components.v1 as components

# --- 1. CORE UI SHELL ---
st.set_page_config(page_title="Cognitive Diagnostic Engine", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC !important; color: #0F172A !important; font-family: 'Inter', sans-serif; }
    h1 { text-align: center; font-weight: 700; color: #1E293B; }
    .stTextInput>div>div>input { border: 2px solid #CBD5E1; border-radius: 8px; padding: 12px; }
    .stButton>button { background-color: #2563EB; color: white; border-radius: 8px; padding: 14px; font-weight: bold; width: 100%; }
    .stButton>button:hover { background-color: #1D4ED8; color: white; }
    </style>
""", unsafe_allow_html=True)

if 'subject_id' not in st.session_state:
    st.session_state.subject_id = ""
if 'phase' not in st.session_state:
    st.session_state.phase = "LOGIN"

# --- 2. PHASE ROUTING ---
if st.session_state.phase == "LOGIN":
    st.markdown("<h1>Cognitive Diagnostic Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Clinical Data Acquisition Environment</p><hr>", unsafe_allow_html=True)
    
    token = st.text_input("Enter Participant Token:")
    if st.button("Unlock Assessment Suite"):
        if token.strip():
            st.session_state.subject_id = token.strip()
            st.session_state.phase = "CONSENT"
            st.rerun()
        else:
            st.error("Token required.")

elif st.session_state.phase == "CONSENT":
    st.markdown("<h1>Informed Consent</h1>", unsafe_allow_html=True)
    st.info("""
    **Study Parameters:**
    1. Your webcam will record your eye movements silently.
    2. No recording preview is shown to prevent visual bias.
    3. The assessment will advance automatically.
    """)
    c1 = st.checkbox("I authorize silent camera access.")
    c2 = st.checkbox("I authorize anonymized data collection.")
    if st.button("Accept & Initialize Engine"):
        if c1 and c2:
            st.session_state.phase = "ENGINE"
            st.rerun()
        else:
            st.warning("Please check both boxes.")

elif st.session_state.phase == "ENGINE":
    st.markdown("<h1>Clinical Diagnostic Engine Running</h1>", unsafe_allow_html=True)
    st.write("Please keep your head still and follow the on-screen instructions. The tasks will transition automatically.")
    
    # --- 3. THE ISOLATED JAVASCRIPT ENGINE ---
    # This block runs entirely in the browser. Zero Python latency. Millisecond precision.
    html_engine = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 0; background-color: #0F172A; color: white; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 500px; overflow: hidden; }}
            #canvas-container {{ position: relative; width: 600px; height: 400px; background-color: #1E293B; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            #ui-layer {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; z-index: 10; padding: 20px; box-sizing: border-box; }}
            #stimulus-layer {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 5; pointer-events: none; }}
            .target-dot {{ position: absolute; width: 24px; height: 24px; background-color: #EF4444; border-radius: 50%; transform: translate(-50%, -50%); display: none; box-shadow: 0 0 10px rgba(239, 68, 68, 0.8); }}
            .crosshair {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #94A3B8; font-size: 32px; display: none; }}
            button {{ margin-top: 20px; padding: 12px 24px; font-size: 16px; font-weight: bold; cursor: pointer; background-color: #3B82F6; color: white; border: none; border-radius: 6px; }}
            button:hover {{ background-color: #2563EB; }}
        </style>
    </head>
    <body>
        <div id="canvas-container">
            <div id="ui-layer">
                <h2 id="msg-title">System Ready</h2>
                <p id="msg-desc">Camera access required to begin.</p>
                <button id="btn-action" style="display:none;">Start</button>
            </div>
            <div id="stimulus-layer">
                <div id="dot" class="target-dot"></div>
                <div id="cross" class="crosshair">+</div>
            </div>
        </div>

        <script>
            const subject_id = "{st.session_state.subject_id}";
            let mediaRecorder;
            let recordedChunks = [];
            let eventLogs = [];
            
            const uiLayer = document.getElementById('ui-layer');
            const title = document.getElementById('msg-title');
            const desc = document.getElementById('msg-desc');
            const dot = document.getElementById('dot');
            const cross = document.getElementById('cross');

            function logEvent(name, details) {{
                eventLogs.push({{ timestamp: performance.now().toFixed(2), event: name, data: details }});
            }}

            async function initCamera() {{
                try {{
                    const stream = await navigator.mediaDevices.getUserMedia({{ video: {{ facingMode: "user", frameRate: 30 }} }});
                    mediaRecorder = new MediaRecorder(stream, {{ mimeType: 'video/webm' }});
                    mediaRecorder.ondataavailable = (e) => {{ if (e.data.size > 0) recordedChunks.push(e.data); }};
                    mediaRecorder.onstop = exportData;
                    mediaRecorder.start();
                    logEvent("SYSTEM", "Camera Started");
                    startCalibration();
                }} catch (e) {{
                    title.innerText = "Camera Error"; desc.innerText = "Please allow camera access.";
                }}
            }}

            // --- TASK 1: CALIBRATION ---
            function startCalibration() {{
                uiLayer.style.display = 'none';
                dot.style.display = 'block';
                const coords = [ [10,10], [50,10], [90,10], [10,50], [50,50], [90,50], [10,90], [50,90], [90,90] ];
                let step = 0;

                function moveDot() {{
                    if (step >= coords.length) {{
                        dot.style.display = 'none';
                        startAntiSaccade();
                        return;
                    }}
                    dot.style.left = coords[step][0] + '%';
                    dot.style.top = coords[step][1] + '%';
                    logEvent("CALIB_POINT", `X:${{coords[step][0]}}, Y:${{coords[step][1]}}`);
                    step++;
                    setTimeout(moveDot, 1500); // Strict 1.5s interval
                }}
                moveDot();
            }}

            // --- TASK 2: ANTI-SACCADE ---
            function startAntiSaccade() {{
                const trials = [20, 80, 80, 20]; // X percentages
                let step = 0;

                function runTrial() {{
                    if (step >= trials.length) {{
                        dot.style.display = 'none';
                        endSession();
                        return;
                    }}
                    
                    // Show crosshair
                    cross.style.display = 'block';
                    setTimeout(() => {{
                        // Hide crosshair, show dot
                        cross.style.display = 'none';
                        dot.style.left = trials[step] + '%';
                        dot.style.top = '50%';
                        dot.style.display = 'block';
                        logEvent("ANTI_STIMULUS", `Side:${{trials[step] === 20 ? 'LEFT' : 'RIGHT'}}`);
                        
                        // Wait, then next trial
                        setTimeout(() => {{
                            dot.style.display = 'none';
                            step++;
                            setTimeout(runTrial, 500); // Inter-trial interval
                        }}, 1000);
                    }}, 1000); // Crosshair duration
                }}
                runTrial();
            }}

            // --- DATA EXPORT ---
            function endSession() {{
                uiLayer.style.display = 'flex';
                title.innerText = "Assessment Complete";
                desc.innerText = "Saving data files...";
                mediaRecorder.stop(); // Triggers exportData()
            }}

            function exportData() {{
                // Save Video
                const videoBlob = new Blob(recordedChunks, {{ type: 'video/webm' }});
                const videoUrl = URL.createObjectURL(videoBlob);
                const aVid = document.createElement('a');
                aVid.href = videoUrl;
                aVid.download = subject_id + "_video.webm";
                aVid.click();

                // Save Logs
                const jsonBlob = new Blob([JSON.stringify(eventLogs, null, 2)], {{ type: 'application/json' }});
                const jsonUrl = URL.createObjectURL(jsonBlob);
                const aJson = document.createElement('a');
                aJson.href = jsonUrl;
                aJson.download = subject_id + "_metrics.json";
                aJson.click();
                
                desc.innerText = "Files downloaded. You may close this window and return the files.";
            }}

            // Bootstrap
            setTimeout(initCamera, 1000); // Give user a second to read before requesting camera
        </script>
    </body>
    </html>
    """
    components.html(html_engine, height=550)
