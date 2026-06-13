"""
holistic.py — Pose tracking using MediaPipe Tasks API PoseLandmarker.

Uses pose_landmarker_lite.task model (auto-downloaded on first run).
Draws skeleton overlay using cv2 for pixel-perfect accuracy.
"""
import cv2
import numpy as np
import os

# ── MediaPipe Tasks API ───────────────────────────────────────────────────────
_pose_landmarker = None
_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "pose_landmarker_lite.task")
)

# Pose landmark indices we care about (MediaPipe Pose 33-point model)
JOINT_INDICES = {
    "nose":            0,
    "left_shoulder":   11,
    "right_shoulder":  12,
    "left_elbow":      13,
    "right_elbow":     14,
    "left_wrist":      15,
    "right_wrist":     16,
    "left_hip":        23,
    "right_hip":       24,
    "left_knee":       25,
    "right_knee":      26,
    "left_ankle":      27,
    "right_ankle":     28,
}

# Skeleton connections (index pairs)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (17, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (18, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26),
    (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32),
]


def _ensure_model():
    """Downloads the model file if not present."""
    if not os.path.exists(_MODEL_PATH):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/latest/"
                   "pose_landmarker_lite.task")
            print(f"[PoseLandmarker] Downloading model...")
            urllib.request.urlretrieve(url, _MODEL_PATH)
            print(f"[PoseLandmarker] Model downloaded.")
        except Exception as e:
            print(f"[PoseLandmarker] Download failed: {e}")
            return False
    return True


def _get_pose_landmarker():
    global _pose_landmarker
    if _pose_landmarker is not None:
        return _pose_landmarker

    if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("SPACE_ID"):
        print("[PoseLandmarker] Skip PoseLandmarker in Streamlit Cloud/Spaces environment.")
        _pose_landmarker = "FAILED"
        return _pose_landmarker

    if not _ensure_model():
        _pose_landmarker = "FAILED"
        return _pose_landmarker

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        print("[PoseLandmarker] Initialized.")
    except Exception as e:
        print(f"[PoseLandmarker] Init error: {e}")
        _pose_landmarker = "FAILED"

    return _pose_landmarker


def track_and_draw_pose(frame, use_mock=False, rgb_frame=None):
    """
    Detects pose using MediaPipe PoseLandmarker and draws skeleton on frame.

    Args:
        frame:     BGR image (numpy array) — drawn on in-place.
        use_mock:  If True, use synthetic joint positions.
        rgb_frame: Pre-converted RGB array. When provided, skips an extra cvtColor call.
    Returns: (joints dict with normalized coords, annotated frame)
    """
    h, w = frame.shape[:2]

    landmarker = _get_pose_landmarker()

    if landmarker is not None and landmarker != "FAILED":
        try:
            import mediapipe as mp

            if rgb_frame is not None:
                rgb = rgb_frame
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            # Store results in session state for hands module to reuse
            try:
                import streamlit as st
                st.session_state.current_pose_result = result
            except Exception:
                pass

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]  # first person

                # ── Extract joint dict ────────────────────────────────────────
                joints = {}
                for name, idx in JOINT_INDICES.items():
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        joints[name] = {"x": lm.x, "y": lm.y, "z": lm.z}

                # Compute neck midpoint between shoulders
                if "left_shoulder" in joints and "right_shoulder" in joints:
                    ls, rs = joints["left_shoulder"], joints["right_shoulder"]
                    joints["neck"] = {
                        "x": (ls["x"] + rs["x"]) / 2,
                        "y": (ls["y"] + rs["y"]) / 2,
                        "z": (ls["z"] + rs["z"]) / 2,
                    }

                # ── Draw skeleton with cv2 ────────────────────────────────────
                # Convert all 33 landmarks to pixel coords
                px = []
                for lm in landmarks:
                    px.append((int(lm.x * w), int(lm.y * h)))

                # Draw connections
                for a, b in POSE_CONNECTIONS:
                    if a < len(px) and b < len(px):
                        cv2.line(frame, px[a], px[b], (180, 0, 220), 2)

                # Draw joint dots
                for point in px:
                    cv2.circle(frame, point, 3, (0, 230, 0), -1)

                # Larger circles for key joints
                for name, idx in JOINT_INDICES.items():
                    if idx < len(px):
                        cv2.circle(frame, px[idx], 5, (0, 255, 80), -1)

                return joints, frame

            elif not use_mock:
                return {}, frame

        except Exception as e:
            print(f"[PoseLandmarker] Processing error: {e}")
            if not use_mock:
                return {}, frame

    if not use_mock:
        return {}, frame

    # ── Mock skeleton fallback ────────────────────────────────────────────────
    joints = {
        "nose":           {"x": 0.50, "y": 0.22, "z": 0.0},
        "neck":           {"x": 0.50, "y": 0.38, "z": 0.0},
        "left_shoulder":  {"x": 0.38, "y": 0.44, "z": 0.0},
        "right_shoulder": {"x": 0.62, "y": 0.44, "z": 0.0},
        "left_elbow":     {"x": 0.30, "y": 0.62, "z": 0.0},
        "right_elbow":    {"x": 0.70, "y": 0.62, "z": 0.0},
        "left_wrist":     {"x": 0.26, "y": 0.76, "z": 0.0},
        "right_wrist":    {"x": 0.74, "y": 0.76, "z": 0.0},
        "left_hip":       {"x": 0.42, "y": 0.72, "z": 0.0},
        "right_hip":      {"x": 0.58, "y": 0.72, "z": 0.0},
        "left_knee":      {"x": 0.40, "y": 0.86, "z": 0.0},
        "right_knee":     {"x": 0.60, "y": 0.86, "z": 0.0},
        "left_ankle":     {"x": 0.40, "y": 0.97, "z": 0.0},
        "right_ankle":    {"x": 0.60, "y": 0.97, "z": 0.0},
    }

    px_joints = {name: (int(pt["x"] * w), int(pt["y"] * h)) for name, pt in joints.items()}

    mock_connections = [
        ("nose", "neck"), ("neck", "left_shoulder"), ("neck", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"), ("right_knee", "right_ankle"),
    ]

    for a, b in mock_connections:
        if a in px_joints and b in px_joints:
            cv2.line(frame, px_joints[a], px_joints[b], (180, 0, 220), 2)

    for name, coord in px_joints.items():
        cv2.circle(frame, coord, 5, (0, 230, 0), -1)

    return joints, frame
