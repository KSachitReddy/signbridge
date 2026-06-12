"""
landmarks.py — Hand tracking using MediaPipe Tasks API HandLandmarker.

Uses hand_landmarker.task model (auto-downloaded on first run).
Returns 21 normalized landmarks per hand (left/right separated by handedness).
Draws hand skeleton overlay using cv2.
"""
import cv2
import numpy as np
import os

_hand_landmarker = None
_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "hand_landmarker.task")
)

# Standard MediaPipe Hand connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (0, 9), (9, 10), (10, 11), (11, 12),    # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (5, 9), (9, 13), (13, 17),              # palm arch
]


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            print("[HandLandmarker] Downloading model...")
            urllib.request.urlretrieve(url, _MODEL_PATH)
            print("[HandLandmarker] Model downloaded.")
        except Exception as e:
            print(f"[HandLandmarker] Download failed: {e}")
            return False
    return True


def _get_hand_landmarker():
    global _hand_landmarker
    if _hand_landmarker is not None:
        return _hand_landmarker

    if not _ensure_model():
        _hand_landmarker = "FAILED"
        return _hand_landmarker

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _hand_landmarker = mp_vision.HandLandmarker.create_from_options(options)
        print("[HandLandmarker] Initialized.")
    except Exception as e:
        print(f"[HandLandmarker] Init error: {e}")
        _hand_landmarker = "FAILED"

    return _hand_landmarker


def _draw_hand(frame, landmarks_px, color_joint, color_conn):
    """Draws a single hand skeleton on the frame using pixel coordinates."""
    for a, b in HAND_CONNECTIONS:
        if a < len(landmarks_px) and b < len(landmarks_px):
            cv2.line(frame, landmarks_px[a], landmarks_px[b], color_conn, 2)
    for px in landmarks_px:
        cv2.circle(frame, px, 4, color_joint, -1)
    # Wrist dot slightly larger
    if landmarks_px:
        cv2.circle(frame, landmarks_px[0], 6, color_joint, -1)


def track_and_draw_hands(frame, use_mock=False):
    """
    Detects hands using MediaPipe HandLandmarker and draws skeleton on frame.
    Returns: ({"left": [...], "right": [...]}, annotated frame)
    Each hand is a list of 21 dicts with x, y, z keys (normalized 0-1).
    """
    h, w = frame.shape[:2]
    hands_data = {"left": [], "right": []}

    landmarker = _get_hand_landmarker()

    if landmarker is not None and landmarker != "FAILED":
        try:
            import mediapipe as mp

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                for hand_idx, landmarks in enumerate(result.hand_landmarks):
                    # Determine handedness
                    side = "right"  # default
                    if result.handedness and hand_idx < len(result.handedness):
                        cat = result.handedness[hand_idx]
                        # MediaPipe handedness is from the camera's perspective (mirrored)
                        # "Right" in MediaPipe = user's left hand in mirror
                        side = "left" if cat[0].category_name == "Right" else "right"

                    lm_list = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks]
                    hands_data[side] = lm_list

                    # Draw
                    px = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                    if side == "left":
                        _draw_hand(frame, px, (0, 140, 255), (255, 220, 0))
                    else:
                        _draw_hand(frame, px, (0, 200, 255), (0, 255, 200))

                return hands_data, frame

            elif not use_mock:
                return hands_data, frame

        except Exception as e:
            print(f"[HandLandmarker] Processing error: {e}")
            if not use_mock:
                return hands_data, frame

    if not use_mock:
        return hands_data, frame

    # ── Mock fallback ─────────────────────────────────────────────────────────
    for side, wrist_x, wrist_y in [("left", 0.28, 0.72), ("right", 0.72, 0.72)]:
        lm_list = []
        # Generate 21 hand landmarks relative to wrist
        finger_angles = [0.8, 0.5, 0.25, 0.0, -0.25]
        lm_list.append({"x": wrist_x, "y": wrist_y, "z": 0.0})
        for fi, angle in enumerate(finger_angles):
            for ji in range(1, 5):
                lm_list.append({
                    "x": wrist_x + np.cos(angle) * 0.03 * ji * (1 if side == "right" else -1),
                    "y": wrist_y - np.sin(angle + 0.3) * 0.04 * ji,
                    "z": -0.01 * ji,
                })
        if len(lm_list) != 21:
            lm_list = [{"x": wrist_x, "y": wrist_y, "z": 0.0}] * 21

        hands_data[side] = lm_list
        px = [(int(lm["x"] * w), int(lm["y"] * h)) for lm in lm_list]
        color_j = (0, 140, 255) if side == "left" else (0, 200, 255)
        color_c = (255, 220, 0) if side == "left" else (0, 255, 200)
        _draw_hand(frame, px, color_j, color_c)

    return hands_data, frame
