# 👁 Smart Vision AI — v2.0

**Real-time AI navigation assistance for visually impaired users.**

Powered by a custom YOLOv8 model, WebRTC live streaming, and browser-native voice guidance.

---

## ✨ Features

- 🎥 **Real-time live video** via WebRTC (browser → server → processed feed back)
- 🧠 **Smart navigation alerts** — not raw object names, but contextual guidance
- 📍 **Position awareness** — Left / Centre / Right zone detection
- ↔️ **Movement tracking** — detects if an object is moving towards you
- 🚦 **Priority system** — Critical (people/vehicles) → Warning (obstacles) → Clear
- 🔊 **Zero-latency voice** — browser Web Speech API, no TTS server needed
- 📁 **Photo fallback** — analyse any uploaded image
- 🌑 **World-class dark UI** — professional assistive-tech aesthetic

---

## 🚀 Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
streamlit run app/app.py
```

Open http://localhost:8501 — allow camera and microphone when prompted.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this entire folder to a **GitHub repository** (make it public or connect your account).

2. Go to → https://share.streamlit.io

3. Click **New app** → select your repo.

4. Set:
   - **Main file path:** `app/app.py`
   - **Python version:** 3.11

5. Click **Deploy**.

Streamlit Cloud reads:
- `requirements.txt` → Python packages
- `packages.txt` → system-level apt packages (libGL, WebRTC codecs)

> ⚠️ **Note on WebRTC & Cloud:** `streamlit-webrtc` requires a TURN server
> for reliable connections through firewalls on production. For Streamlit Cloud,
> the built-in STUN (Google) works for most networks. If video doesn't connect,
> add a free TURN server via [Metered](https://www.metered.ca/tools/openrelay/)
> and update `RTC_CONFIG` in `app/app.py`.

---

## 📁 Project Structure

```
smart_vision_ai/
├── app/
│   └── app.py                  # Main Streamlit application
├── models/
│   └── best.pt                 # Trained YOLOv8 model
├── services/
│   ├── __init__.py
│   └── vision_processor.py     # Detection + smart alert engine
├── components/
│   ├── __init__.py
│   └── webrtc_handler.py       # WebRTC video frame processor
├── assets/
│   └── logo.svg                # App logo
├── .streamlit/
│   └── config.toml             # Dark theme + server config
├── requirements.txt            # Python dependencies
├── packages.txt                # System dependencies (apt)
└── README.md
```

---

## 🚨 Alert Priority System

| Level    | Trigger Objects                        | Example Voice Alert                          |
|----------|----------------------------------------|----------------------------------------------|
| 🚨 Critical | Person, Car, Bus, Truck, Motorcycle | *"Person directly ahead. Move slightly left."* |
| ⚠️ Warning  | Chair, Bench, Traffic light, Door   | *"Chair on your right."*                       |
| ✅ Clear    | No navigation hazards detected      | *"Path ahead is clear."*                       |

### Cooldown rules
- Same alert won't repeat within **3.5 seconds**
- "Path clear" announcement waits **7 seconds** between repeats
- Movement alerts fire immediately when object changes zone

---

## 🔊 Voice System

Voice uses the **browser's built-in Web Speech API**:
- ✅ Zero server load
- ✅ No internet required for audio
- ✅ Works on Chrome, Edge, Safari, Firefox
- ✅ Prefers local/offline voices for lowest latency
- Toggle with the **Voice On/Off** button

---

## 🛠️ Customisation

### Change detection confidence
In `services/vision_processor.py`:
```python
results = self.model(frame, verbose=False, conf=0.32)  # lower = more detections
```

### Add new object types
In `services/vision_processor.py`, add to `PRIORITY` and `NOUN` dicts:
```python
PRIORITY["escalator"] = 2
NOUN["escalator"]     = "Escalator"
```

### Adjust alert cooldowns
```python
self._cooldown       = 3.5   # seconds between same alert
self._clear_cooldown = 7.0   # seconds between "path clear" alerts
```
