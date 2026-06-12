import streamlit as st

def get_tts_html(text, lang="en"):
    """
    Generates HTML/JS containing browser-native Web Speech API SpeechSynthesis controls.
    Supports English (en-US), Hindi (hi-IN), and Telugu (te-IN).
    """
    lang_map = {
        "en": "en-US",
        "hi": "hi-IN",
        "te": "te-IN"
    }
    lang_code = lang_map.get(lang.lower()[:2], "en-US")
    
    # We create standard Play, Pause, and Replay controls
    js_code = f"""
    <div style="display: flex; gap: 10px; align-items: center; background: #1E293B; padding: 12px; border-radius: 8px;">
        <button id="playBtn" style="background:#2563EB; color:#fff; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-weight:bold;">▶ Play</button>
        <button id="pauseBtn" style="background:#475569; color:#fff; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-weight:bold;">⏸ Pause</button>
        <button id="replayBtn" style="background:#059669; color:#fff; border:none; padding:8px 16px; border-radius:6px; cursor:pointer; font-weight:bold;">🔄 Replay</button>
    </div>
    
    <script>
        const synth = window.speechSynthesis;
        let utterance = null;
        
        function speak() {{
            synth.cancel();
            utterance = new SpeechSynthesisUtterance("{text}");
            utterance.lang = "{lang_code}";
            synth.speak(utterance);
        }}
        
        document.getElementById('playBtn').onclick = () => {{
            if (synth.paused) {{
                synth.resume();
            }} else if (!synth.speaking) {{
                speak();
            }}
        }};
        
        document.getElementById('pauseBtn').onclick = () => {{
            if (synth.speaking && !synth.paused) {{
                synth.pause();
            }}
        }};
        
        document.getElementById('replayBtn').onclick = () => {{
            speak();
        }};
    </script>
    """
    return js_code

def render_stt_listener(lang="en"):
    """
    Renders an HTML5 Speech Recognition button inside the browser.
    Transcribes spoken voice into text and displays it directly.
    """
    lang_map = {
        "en": "en-US",
        "hi": "hi-IN",
        "te": "te-IN"
    }
    lang_code = lang_map.get(lang.lower()[:2], "en-US")
    
    js_code = f"""
    <div style="background: #1E293B; padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); color:#fff; font-family: sans-serif;">
        <button id="startSttBtn" style="background:#DC2626; color:#fff; border:none; padding:10px 20px; border-radius:8px; cursor:pointer; font-size:16px; font-weight:bold; display:flex; align-items:center; gap:8px;">
            🎙️ Start Listening
        </button>
        <p id="statusMsg" style="color:#94A3B8; font-size:14px; margin-top:8px;">Click button to start speaking...</p>
        <div style="margin-top:15px; background:#0F172A; padding:12px; border-radius:6px; min-height:60px; border:1px solid #334155;">
            <span id="transcriptionOutput" style="color:#F1F5F9; font-size:15px; font-style:italic;">No speech detected yet.</span>
        </div>
    </div>
    
    <script>
        const btn = document.getElementById('startSttBtn');
        const status = document.getElementById('statusMsg');
        const output = document.getElementById('transcriptionOutput');
        
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {{
            status.innerText = "Speech Recognition API not supported in this browser. Please use Chrome.";
            btn.disabled = true;
        }} else {{
            const recognition = new SpeechRecognition();
            recognition.lang = "{lang_code}";
            recognition.continuous = false;
            recognition.interimResults = false;
            
            btn.onclick = () => {{
                recognition.start();
                status.innerText = "Listening... speak now.";
                btn.style.background = "#991B1B";
            }};
            
            recognition.onresult = (event) => {{
                const text = event.results[0][0].transcript;
                output.innerText = text;
                status.innerText = "Speech captured successfully.";
                btn.style.background = "#DC2626";
                try {{
                    const url = new URL(window.parent.location.href);
                    url.searchParams.set("stt_text", text);
                    window.parent.location.href = url.toString();
                }} catch (e) {{
                    console.error("STT sync error:", e);
                }}
            }};
            
            recognition.onerror = (event) => {{
                status.innerText = "Error occurred: " + event.error;
                btn.style.background = "#DC2626";
            }};
            
            recognition.onend = () => {{
                if (status.innerText === "Listening... speak now.") {{
                    status.innerText = "Listening stopped.";
                }}
                btn.style.background = "#DC2626";
            }};
        }}
    </script>
    """
    st.html(js_code)
