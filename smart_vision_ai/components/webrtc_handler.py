"""
WebRTC callback — processes each video frame from the browser stream.
Shares state with the Streamlit app via a thread-safe container.
"""

import threading
import numpy as np
import av
from streamlit_webrtc import VideoProcessorBase


class SmartVisionProcessor(VideoProcessorBase):
    """
    Plugs into streamlit-webrtc.
    Runs YOLO on every incoming video frame and exposes
    the latest detection results for the UI to read.
    """

    def __init__(self, vision_proc):
        self._vp = vision_proc
        self._lock = threading.Lock()
        self.result = {
            "detections":  [],
            "alert":       "",
            "priority":    "idle",
            "speech":      "",
            "fps":         0.0,
        }
        self._last_ts = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        import time
        img = frame.to_ndarray(format="bgr24")

        now = time.time()
        fps = 1.0 / max(now - self._last_ts, 1e-6) if self._last_ts else 0.0
        self._last_ts = now

        annotated, detections, alert, priority = self._vp.process_frame(img)
        speech = self._vp.pop_speech()

        with self._lock:
            self.result = {
                "detections": detections,
                "alert":      alert,
                "priority":   priority,
                "speech":     speech,
                "fps":        round(fps, 1),
            }

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    def get_result(self) -> dict:
        with self._lock:
            return dict(self.result)
