"""
Smart Vision AI — Main Application (Live-Only Edition)
World-class assistive navigation interface with real-time WebRTC streaming.
Optimized for blind navigation with hands-free operation.
"""

import streamlit as st
import os, sys, time
import numpy as np
from PIL import Image

# Add the project root to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from services.vision_processor import VisionProcessor
from components.webrtc_handler import SmartVisionProcessor

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "best.pt")
LOGO_PATH  = os.path.join(ROOT, "assets",  "logo.svg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Smart Vision AI",
    page_icon   = "👁",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  WORLD-CLASS CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', sans-serif !important;
  background: #030B18 !important;
  color: #E8EFF8 !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display: none !important;
}
[data-testid="collapsedControl"] { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #030B18; }
::-webkit-scrollbar-thumb { background: #0A4F8A; border-radius: 4px; }

.stApp::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 80% 60% at 10% 0%, rgba(0,100,255,0.09) 0%, transparent 60%),
    radial-gradient(ellipse 50% 40% at 90% 100%, rgba(0,212,255,0.06) 0%, transparent 55%);
}

.main .block-container {
  max-width: 1400px !important;
  padding: 0 2rem 4rem !important;
  position: relative; z-index: 1;
}

.sv-header {
  display: flex; align-items: center; gap: 20px;
  padding: 2.2rem 0 0.5rem;
  border-bottom: 1px solid rgba(0,180,255,0.12);
  margin-bottom: 2rem;
}
.sv-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(1.6rem, 3vw, 2.4rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #FFFFFF 30%, #00C8FF 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text; line-height: 1.1;
}
.sv-subtitle {
  font-size: 0.85rem; font-weight: 400;
  color: #5A8AAA; letter-spacing: 0.06em;
  text-transform: uppercase; margin-top: 4px;
}
.sv-version {
  margin-left: auto;
  font-family: 'Syne', sans-serif;
  font-size: 0.7rem; font-weight: 700;
  color: #0099CC; letter-spacing: 0.1em;
  background: rgba(0,153,204,0.1);
  border: 1px solid rgba(0,153,204,0.25);
  padding: 4px 12px; border-radius: 20px;
}

.sv-video-panel {
  background: rgba(5,15,35,0.7);
  border: 1px solid rgba(0,150,220,0.18);
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 0 60px rgba(0,100,220,0.08), 0 20px 40px rgba(0,0,0,0.4);
}
.sv-video-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 20px;
  background: rgba(0,20,50,0.6);
  border-bottom: 1px solid rgba(0,150,220,0.12);
}
.sv-video-label {
  font-family: 'Syne', sans-serif;
  font-size: 0.72rem; font-weight: 700;
  letter-spacing: 0.12em; text-transform: uppercase; color: #5A8AAA;
}
.sv-live-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: #FF3B3B; margin-right: 8px; display: inline-block;
  animation: pulse-red 1.4s ease-in-out infinite;
}
.sv-live-dot.active { background: #00DD88; animation: pulse-green 1.4s ease-in-out infinite; }
@keyframes pulse-red   { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(255,59,59,.5)}  50%{opacity:.6;box-shadow:0 0 0 6px rgba(255,59,59,0)} }
@keyframes pulse-green { 0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,221,136,.5)}  50%{opacity:.8;box-shadow:0 0 0 8px rgba(0,221,136,0)} }

.sv-card {
  background: rgba(5,15,35,0.75);
  border: 1px solid rgba(0,150,220,0.15);
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.25);
  margin-bottom: 1rem;
}
.sv-card-title {
  font-family: 'Syne', sans-serif;
  font-size: 0.65rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase;
  color: #3A6A8A; margin-bottom: 14px;
}

.sv-status-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.sv-status-row:last-child { border-bottom: none; }
.sv-status-key { font-size: 0.78rem; color: #5A8AAA; font-weight: 400; }
.sv-pill {
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.04em;
  padding: 3px 10px; border-radius: 20px;
}
.sv-pill-green  { background: rgba(0,200,100,0.15);  color: #00DD88; border: 1px solid rgba(0,200,100,0.3); }
.sv-pill-red    { background: rgba(255,60,60,0.12);  color: #FF6B6B; border: 1px solid rgba(255,60,60,0.25); }
.sv-pill-blue   { background: rgba(0,160,255,0.12);  color: #40C0FF; border: 1px solid rgba(0,160,255,0.25); }
.sv-pill-gray   { background: rgba(100,120,150,0.12);color: #7A9AB8; border: 1px solid rgba(100,120,150,0.2); }

.sv-alert {
  border-radius: 10px; padding: 14px 16px;
  font-size: 1.1rem; font-weight: 700; line-height: 1.45;
  transition: all 0.35s ease;
}
.sv-alert-critical {
  background: rgba(220,30,30,0.12);
  border: 1px solid rgba(220,30,30,0.35);
  color: #FF7070;
  box-shadow: 0 0 20px rgba(220,30,30,0.08);
}
.sv-alert-warning {
  background: rgba(255,160,0,0.10);
  border: 1px solid rgba(255,160,0,0.3);
  color: #FFBB44;
}
.sv-alert-clear {
  background: rgba(0,200,100,0.10);
  border: 1px solid rgba(0,200,100,0.28);
  color: #44DD99;
}
.sv-alert-idle {
  background: rgba(0,100,180,0.08);
  border: 1px solid rgba(0,100,180,0.2);
  color: #4488BB;
}
.sv-alert-icon { font-size: 1.4rem; margin-right: 10px; }

.stButton > button {
  border-radius: 12px !important;
  border: none !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 800 !important;
  font-size: 1rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  padding: 20px !important;
  width: 100% !important;
  transition: all 0.2s ease !important;
}
div[data-testid="column"]:nth-child(1) .stButton > button {
  background: linear-gradient(135deg, #006633, #00AA55) !important;
  color: #fff !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button {
  background: linear-gradient(135deg, #7A0000, #CC2200) !important;
  color: #fff !important;
}

.stWebRtcStreamer video {
  border-radius: 0 !important;
  width: 100% !important;
  max-height: 70vh;
  object-fit: cover;
}

.sv-footer {
  text-align: center; color: #213A55;
  font-size: 0.75rem; padding: 2rem 0;
  border-top: 1px solid rgba(0,100,180,0.08);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "voice_enabled":  True,
    "processor":      None,
    "last_alert":     "",
    "alert_priority": "idle",
    "object_count":   0,
    "fps":            0.0,
    "last_spoken":    "",
    "active":         False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_processor(path: str) -> VisionProcessor:
    return VisionProcessor(path)

if st.session_state.processor is None:
    with st.spinner("Initialising Smart Vision AI…"):
        try:
            st.session_state.processor = load_processor(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Model failed to load: {e}")
            st.stop()

vp: VisionProcessor = st.session_state.processor

# ══════════════════════════════════════════════════════════════════════════════
#  BROWSER TTS (Web Speech API)
# ══════════════════════════════════════════════════════════════════════════════
tts_slot = st.empty()

def speak(text: str):
    """Inject a Web Speech API utterance into the browser."""
    if not text or not st.session_state.voice_enabled:
        return
    safe = text.replace("'", "\\'").replace('"', '\\"')
    tts_slot.markdown(f"""
    <script>
    (function(){{
      if(!window.speechSynthesis) return;
      window.speechSynthesis.cancel();
      var u = new SpeechSynthesisUtterance('{safe}');
      u.lang='en-US'; u.rate=0.92; u.pitch=1.05; u.volume=1.0;
      window.speechSynthesis.speak(u);
    }})();
    </script>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sv-header">
  <div class="sv-title-block">
    <div class="sv-title">Smart Vision AI</div>
    <div class="sv-subtitle">Live Navigation for the Visually Impaired</div>
  </div>
  <div class="sv-version">LIVE · v2.2</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
c1, c2 = st.columns(2)
with c1:
    if st.button("▶  START LIVE NAVIGATION", use_container_width=True):
        st.session_state.active = True
        vp.reset()
        st.rerun()
with c2:
    if st.button("⏹  STOP", use_container_width=True):
        st.session_state.active = False
        vp.reset()
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

is_active = st.session_state.active

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([2.5, 1], gap="large")

# ── Right panel: alerts + status ──────────────────────────────────────────────
with right_col:
    al  = st.session_state.last_alert
    pri = st.session_state.alert_priority

    if not is_active:
        aclass, aicon, atxt = "sv-alert-idle", "💤", "Press START to begin navigation."
    elif pri == "critical":
        aclass, aicon, atxt = "sv-alert-critical", "🚨", al
    elif pri == "warning":
        aclass, aicon, atxt = "sv-alert-warning", "⚠️", al
    elif pri == "clear":
        aclass, aicon, atxt = "sv-alert-clear", "✅", al
    else:
        aclass, aicon, atxt = "sv-alert-idle", "👁", al or "Scanning…"

    alert_slot  = st.empty()
    status_slot = st.empty()

    def _render_panels():
        _al  = st.session_state.last_alert
        _pri = st.session_state.alert_priority
        _act = st.session_state.active

        if not _act:
            _ac, _ai, _at = "sv-alert-idle", "💤", "Press START to begin navigation."
        elif _pri == "critical":
            _ac, _ai, _at = "sv-alert-critical", "🚨", _al
        elif _pri == "warning":
            _ac, _ai, _at = "sv-alert-warning", "⚠️", _al
        elif _pri == "clear":
            _ac, _ai, _at = "sv-alert-clear", "✅", _al
        else:
            _ac, _ai, _at = "sv-alert-idle", "👁", _al or "Scanning…"

        alert_slot.markdown(f"""
        <div class="sv-card">
          <div class="sv-card-title">Navigation Guidance</div>
          <div class="sv-alert {_ac}">
            <span class="sv-alert-icon">{_ai}</span>{_at}
          </div>
        </div>
        """, unsafe_allow_html=True)

        cam_pill = '<span class="sv-pill sv-pill-green">● LIVE</span>' if _act else '<span class="sv-pill sv-pill-red">● OFF</span>'
        status_slot.markdown(f"""
        <div class="sv-card">
          <div class="sv-card-title">System Status</div>
          <div class="sv-status-row"><span class="sv-status-key">Camera</span>{cam_pill}</div>
          <div class="sv-status-row"><span class="sv-status-key">Voice</span><span class="sv-pill sv-pill-blue">● ACTIVE</span></div>
          <div class="sv-status-row"><span class="sv-status-key">Objects</span><span class="sv-pill sv-pill-blue">{st.session_state.object_count}</span></div>
          <div class="sv-status-row"><span class="sv-status-key">FPS</span><span class="sv-pill sv-pill-gray">{st.session_state.fps}</span></div>
        </div>
        """, unsafe_allow_html=True)

    _render_panels()

# ── Left panel: WebRTC video ───────────────────────────────────────────────────
with left_col:
    st.markdown(f"""
    <div class="sv-video-panel">
      <div class="sv-video-header">
        <span class="sv-video-label">{'<span class="sv-live-dot active"></span> LIVE' if is_active else '<span class="sv-live-dot"></span> STANDBY'}</span>
        <span class="sv-video-label">AI Navigation Feed</span>
      </div>
    """, unsafe_allow_html=True)

    if is_active:
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

            RTC_CONFIG = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                ]
            })

            ctx = webrtc_streamer(
                key="smart-vision-live",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIG,
                video_processor_factory=lambda: SmartVisionProcessor(vp),
                media_stream_constraints={
                    "video": {
                        "width":     {"ideal": 640},
                        "height":    {"ideal": 480},
                        "frameRate": {"ideal": 15},
                        "facingMode": "environment",   # prefer rear/world camera on mobile
                    },
                    "audio": False,
                },
                async_processing=True,
            )

            # ── Continuous result polling ──────────────────────────────────────
            # WebRTC processing runs in a background thread; we poll every second
            # so the UI stays fresh without hammering the server.
            if ctx.state.playing and ctx.video_processor:
                result = ctx.video_processor.get_result()
                if result["fps"] > 0:
                    st.session_state.fps          = result["fps"]
                    st.session_state.object_count = len(result["detections"])
                    st.session_state.last_alert   = result["alert"]
                    st.session_state.alert_priority = result["priority"]

                    new_speech = result["speech"]
                    if new_speech and new_speech != st.session_state.last_spoken:
                        st.session_state.last_spoken = new_speech
                        speak(new_speech)

                    # Re-render panels with fresh data (no full page rerun)
                    _render_panels()

                # Auto-refresh every 1 s while streaming
                time.sleep(1)
                st.rerun()

        except Exception as e:
            st.error(f"Camera Connection Error: {e}")
    else:
        st.markdown("""
        <div style="background:rgba(2,8,22,0.8);height:400px;display:flex;align-items:center;justify-content:center;color:#1A3A55;">
          <span style="font-family:Syne,sans-serif;font-size:1rem;letter-spacing:.1em;text-transform:uppercase;">
            Press START LIVE NAVIGATION to begin
          </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="sv-footer">
  Smart Vision AI — Assistive Navigation System<br>
  Powered by YOLOv8 · Browser Voice Guidance · WebRTC Live Feed
</div>
""", unsafe_allow_html=True)
