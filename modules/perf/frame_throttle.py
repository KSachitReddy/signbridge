"""
frame_throttle.py — Performance utilities for real-time frame pipeline.

Provides:
  - PERF_MODES: pre-configured settings for Performance / Balanced / Accuracy
  - FrameThrottle: fires True every N frames (gate for expensive inference)
  - CentroidTracker: lightweight distance-based face identity persistence
  - AsyncWorker: thread-pool wrapper for non-blocking background tasks (Ollama etc.)
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Performance Mode Configurations
# ─────────────────────────────────────────────────────────────────────────────
PERF_MODES: Dict[str, Dict[str, Any]] = {
    "⚡ Performance": {
        # Fastest mode: skip face mesh entirely, lowest resolution
        "face_recog_interval": 25,   # run InsightFace every 25 frames
        "sign_interval":        3,   # run sign classifier every 3 frames
        "use_face_mesh":        False,  # mesh disabled for max speed
        "infer_w":              320,
        "infer_h":              240,
        "det_size":             (320, 320),
        "pose_conf":            0.4,
        "hand_conf":            0.4,
    },
    "⚖️ Balanced": {
        # Default: full 468-pt face mesh ON, moderate resolution
        "face_recog_interval": 15,
        "sign_interval":        2,
        "use_face_mesh":        True,   # restored: full face mesh in balanced mode
        "infer_w":              480,
        "infer_h":              360,
        "det_size":             (480, 480),
        "pose_conf":            0.5,
        "hand_conf":            0.5,
    },
    "🎯 Accuracy": {
        # Highest quality: full mesh, full pose, both hands, highest resolution
        "face_recog_interval":  8,
        "sign_interval":        1,
        "use_face_mesh":        True,
        "infer_w":              640,
        "infer_h":              480,
        "det_size":             (640, 640),
        "pose_conf":            0.5,
        "hand_conf":            0.5,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FrameThrottle
# ─────────────────────────────────────────────────────────────────────────────
class FrameThrottle:
    """
    Returns True every `interval` calls; False otherwise.
    Allows expensive inference to be skipped between active frames.

    Usage:
        face_throttle = FrameThrottle(interval=20)
        if face_throttle.should_run():
            faces = recognize_multiple_faces(frame)
            cache = faces
        else:
            faces = cache
    """

    def __init__(self, interval: int = 20):
        self.interval = max(1, interval)
        self._counter = 0

    def should_run(self) -> bool:
        self._counter += 1
        if self._counter >= self.interval:
            self._counter = 0
            return True
        return False

    def reset(self):
        self._counter = 0

    def set_interval(self, interval: int):
        self.interval = max(1, interval)


class TimeThrottle:
    """
    Returns True when `interval_seconds` have elapsed since the last execution.
    Allows for time-based throttling of heavy inference functions, independent of loop frame rates.
    """

    def __init__(self, interval_seconds: float = 1.0):
        self.interval = float(interval_seconds)
        self.last_run = 0.0

    def should_run(self) -> bool:
        now = time.time()
        if now - self.last_run >= self.interval:
            self.last_run = now
            return True
        return False

    def reset(self):
        self.last_run = 0.0

    def set_interval(self, interval_seconds: float):
        self.interval = float(interval_seconds)


# ─────────────────────────────────────────────────────────────────────────────
# CentroidTracker
# ─────────────────────────────────────────────────────────────────────────────
class CentroidTracker:
    """
    Lightweight centroid-distance tracker for face bounding boxes.

    Between recognition frames, existing tracks are updated with new
    bounding boxes from InsightFace detection (which is fast) and the
    last known identity is reused.

    Tracks expire after `max_disappeared` missed frames.
    """

    def __init__(self, max_disappeared: int = 30, max_distance: float = 120.0):
        self.next_id = 0
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _centroid(self, box: List[int]) -> Tuple[float, float]:
        x, y, w, h = box
        return x + w / 2.0, y + h / 2.0

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        detections: list of dicts with at least {"box": [x,y,w,h], "name": ..., ...}
        Returns the same list with "track_id" added; identities persisted from cache.
        """
        if not detections:
            for tid in list(self.disappeared.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
            return detections

        if not self.tracks:
            for det in detections:
                self._register(det)
            return detections

        track_ids = list(self.tracks.keys())
        track_centroids = [self._centroid(self.tracks[tid]["box"]) for tid in track_ids]
        det_centroids = [self._centroid(d["box"]) for d in detections]

        # Distance matrix
        used_track = set()
        used_det = set()

        for di, dc in enumerate(det_centroids):
            best_tid = None
            best_dist = self.max_distance
            for ti, tc in enumerate(track_centroids):
                if ti in used_track:
                    continue
                dist = np.sqrt((dc[0] - tc[0]) ** 2 + (dc[1] - tc[1]) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_tid = ti

            if best_tid is not None:
                tid = track_ids[best_tid]
                # Carry identity from track, update box
                old_name = self.tracks[tid].get("name", detections[di]["name"])
                new_name = detections[di]["name"]
                # Only update identity if recognition ran this frame (name != cached)
                self.tracks[tid]["box"] = detections[di]["box"]
                self.tracks[tid]["name"] = new_name if new_name != "Unknown Person" else old_name
                detections[di]["name"] = self.tracks[tid]["name"]
                detections[di]["track_id"] = tid
                self.disappeared[tid] = 0
                used_track.add(best_tid)
                used_det.add(di)

        for di, det in enumerate(detections):
            if di not in used_det:
                self._register(det)

        for ti, tid in enumerate(track_ids):
            if ti not in used_track:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]

        return detections

    def _register(self, det: Dict[str, Any]):
        tid = self.next_id
        self.tracks[tid] = {"box": det["box"], "name": det.get("name", "Unknown Person")}
        self.disappeared[tid] = 0
        det["track_id"] = tid
        self.next_id += 1

    def clear(self):
        self.tracks.clear()
        self.disappeared.clear()
        self.next_id = 0


# ─────────────────────────────────────────────────────────────────────────────
# AsyncWorker
# ─────────────────────────────────────────────────────────────────────────────
class AsyncWorker:
    """
    Runs callables in a background thread pool without blocking the camera loop.

    Usage:
        worker = AsyncWorker(max_workers=1)
        worker.submit(generate_response, sign, text, person, lang)
        # later...
        result = worker.get_result()  # None if not ready yet
    """

    def __init__(self, max_workers: int = 1):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="SignBridgeWorker")
        self._future = None
        self._lock = threading.Lock()

    def submit(self, fn: Callable, *args, **kwargs):
        """Submit a new task. Any previous pending task is abandoned."""
        with self._lock:
            self._future = self._pool.submit(fn, *args, **kwargs)

    def get_result(self) -> Optional[Any]:
        """Returns result if ready, None otherwise. Non-blocking."""
        with self._lock:
            if self._future is None:
                return None
            if self._future.done():
                try:
                    result = self._future.result()
                except Exception as e:
                    print(f"[AsyncWorker] Task error: {e}")
                    result = None
                self._future = None
                return result
        return None

    def is_running(self) -> bool:
        with self._lock:
            return self._future is not None and not self._future.done()

    def shutdown(self):
        self._pool.shutdown(wait=False)
