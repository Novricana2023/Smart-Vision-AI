"""
Smart Vision AI — Vision Processor
Converts raw YOLO detections into intelligent navigation alerts.
"""

import cv2
import numpy as np
import time
from ultralytics import YOLO
from av import VideoFrame


# ── Priority map ─────────────────────────────────────────────────────────────
# 3 = critical  (always alert)
# 2 = warning   (alert if no critical)
# 1 = info      (background only)
# 0 = ignore
PRIORITY = {
    "person": 3, "pedestrian": 3, "people": 3,
    "car": 3, "truck": 3, "bus": 3, "motorcycle": 3,
    "bicycle": 3, "motorbike": 3, "train": 3, "van": 3,
    "stop sign": 2, "traffic light": 2,
    "chair": 2, "bench": 2, "table": 2, "couch": 2, "sofa": 2,
    "fire hydrant": 2, "parking meter": 2, "suitcase": 2,
    "potted plant": 2, "door": 2, "stairs": 2, "step": 2,
    "backpack": 1, "handbag": 1, "umbrella": 1,
    "bottle": 0, "cup": 0, "book": 0, "cell phone": 0, "keyboard": 0,
    "mouse": 0, "remote": 0, "scissors": 0, "toothbrush": 0,
}

NOUN = {
    "person": "Person", "pedestrian": "Person", "people": "People",
    "car": "Car", "truck": "Truck", "bus": "Bus",
    "motorcycle": "Motorcycle", "motorbike": "Motorcycle",
    "bicycle": "Bicycle", "van": "Vehicle", "train": "Train",
    "stop sign": "Stop sign", "traffic light": "Traffic light",
    "chair": "Chair", "bench": "Bench", "table": "Table",
    "couch": "Furniture", "sofa": "Furniture",
    "fire hydrant": "Fire hydrant", "potted plant": "Obstacle",
    "suitcase": "Luggage", "door": "Door",
    "stairs": "Stairs", "step": "Step",
    "parking meter": "Parking meter",
}

# Box colours per priority (BGR)
BOX_COLOR = {3: (20, 30, 230), 2: (10, 150, 255), 1: (40, 200, 60), 0: (130, 130, 130)}


def _zone(xc: float, fw: int) -> str:
    r = xc / fw
    if r < 0.33: return "left"
    if r > 0.66: return "right"
    return "ahead"

def _zone_label(xc: float, fw: int) -> str:
    z = _zone(xc, fw)
    return {"left": "on your left", "right": "on your right", "ahead": "directly ahead"}[z]


class VisionProcessor:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self._prev_x: dict = {}
        self._last_alert: str = ""
        self._last_alert_time: float = 0.0
        self._cooldown: float = 3.5
        self._last_clear_time: float = 0.0
        self._clear_cooldown: float = 7.0
        self._pending_speech: str = ""
        self.stats = {"frames": 0, "alerts": 0, "objects_total": 0}

    def reset(self):
        self._prev_x = {}
        self._last_alert = ""
        self._last_alert_time = 0.0
        self._last_clear_time = 0.0
        self._pending_speech = ""

    def pop_speech(self) -> str:
        """Return pending speech text and clear it."""
        s = self._pending_speech
        self._pending_speech = ""
        return s

    def process_frame(self, frame: np.ndarray):
        """
        Run YOLO and produce annotated frame + alert.
        Returns: (annotated_frame, detections, alert_text, priority_label)
        """
        self.stats["frames"] += 1
        results = self.model(frame, verbose=False, conf=0.32, iou=0.45)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                name = self.model.names[cls].lower()
                xc   = (x1 + x2) / 2
                detections.append({
                    "name":       name,
                    "confidence": conf,
                    "bbox":       [x1, y1, x2, y2],
                    "xc":         xc,
                    "area":       (x2-x1)*(y2-y1),
                    "zone":       _zone(xc, frame.shape[1]),
                    "zone_label": _zone_label(xc, frame.shape[1]),
                })

        self.stats["objects_total"] += len(detections)
        annotated = self._annotate(frame.copy(), detections)
        alert, pri = self._build_alert(detections, frame.shape[1])

        if alert:
            self.stats["alerts"] += 1

        return annotated, detections, alert, pri

    # ── Annotation ───────────────────────────────────────────────────────────

    def _annotate(self, frame: np.ndarray, detections: list) -> np.ndarray:
        h, w = frame.shape[:2]

        # Dark overlay on thirds
        overlay = frame.copy()
        cv2.line(overlay, (w//3, 0),   (w//3, h),   (255,255,255), 1)
        cv2.line(overlay, (2*w//3, 0), (2*w//3, h), (255,255,255), 1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Zone labels
        for txt, x in [("LEFT", 8), ("CENTER", w//2-28), ("RIGHT", 2*w//3+8)]:
            cv2.putText(frame, txt, (x, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

        for det in detections:
            pri = PRIORITY.get(det["name"], 0)
            if pri == 0:
                continue
            x1, y1, x2, y2 = det["bbox"]
            color = BOX_COLOR.get(pri, (150,150,150))
            thick = 3 if pri == 3 else 2

            # Box with rounded-corner effect (double rect)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
            cv2.rectangle(frame, (x1+2, y1+2), (x2-2, y2-2), color, 1)

            # Label pill
            label = f"{NOUN.get(det['name'], det['name'].title())}  {det['confidence']:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            pill_y = max(y1 - th - 12, 0)
            cv2.rectangle(frame, (x1, pill_y), (x1+tw+10, pill_y+th+10), color, -1)
            cv2.putText(frame, label, (x1+5, pill_y+th+4),
                        cv2.FONT_HERSHEY_DUPLEX, 0.52, (255,255,255), 1, cv2.LINE_AA)

        return frame

    # ── Alert logic ───────────────────────────────────────────────────────────

    def _build_alert(self, detections: list, fw: int):
        now = time.time()
        nav = [d for d in detections if PRIORITY.get(d["name"], 0) >= 2]

        if not nav:
            if now - self._last_clear_time > self._clear_cooldown:
                self._last_clear_time = now
                alert = "Path ahead is clear."
                self._maybe_speak(alert, now)
                return alert, "clear"
            return self._last_alert if self._last_alert else "", "idle"

        nav.sort(key=lambda d: (PRIORITY.get(d["name"], 0), d["area"]), reverse=True)
        top  = nav[0]
        name = top["name"]
        pri  = PRIORITY.get(name, 2)
        noun = NOUN.get(name, name.title())
        zone = top["zone"]
        zlbl = top["zone_label"]

        # Movement
        prev_xc = self._prev_x.get(name)
        curr_xc = top["xc"]
        self._prev_x[name] = curr_xc

        if prev_xc is not None and abs(curr_xc - prev_xc) > fw * 0.04:
            mv    = "right" if curr_xc > prev_xc else "left"
            alert = f"{noun} moving {mv}."
        else:
            alert = f"{noun} {zlbl}."

        # Navigation suggestion when blocking centre
        if zone == "ahead" and pri >= 2:
            ln = sum(1 for d in nav if d["zone"] == "left")
            rn = sum(1 for d in nav if d["zone"] == "right")
            alert += " Move slightly left." if ln <= rn else " Move slightly right."

        self._last_alert = alert
        self._maybe_speak(alert, now)
        return alert, ("critical" if pri == 3 else "warning")

    def _maybe_speak(self, alert: str, now: float):
        same    = alert == self._pending_speech or alert == self._last_alert
        elapsed = now - self._last_alert_time
        if not same or elapsed >= self._cooldown:
            self._pending_speech   = alert
            self._last_alert_time  = now
