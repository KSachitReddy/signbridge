import numpy as np
import os
import pickle
from modules.database import get_setting

# Vocabulary List (30 ISL signs — Phase 1)
VOCABULARY = [
    "Hello", "Thank You", "Yes", "No", "Help",
    "Water", "Food", "Mother", "Father", "Brother",
    "Sister", "Friend", "School", "Teacher", "Hospital",
    "Doctor", "Emergency", "Pain", "Medicine", "Bathroom",
    "Home", "Eat", "Drink", "Sleep", "Stop",
    "Come", "Go", "Good", "Bad", "Please"
]


class SignSequenceBuffer:
    def __init__(self, size=20):
        self.size = size
        self.buffer = []

    def add(self, left_hand, right_hand, pose):
        """Adds landmarks to temporal frame buffer."""
        frame_data = {
            "left_hand": left_hand,
            "right_hand": right_hand,
            "pose": pose
        }
        self.buffer.append(frame_data)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def clear(self):
        self.buffer.clear()

    def is_full(self):
        return len(self.buffer) >= self.size


def extract_frame_features(frame_data):
    """
    Extracts a 162-dimensional translation-invariant feature vector from frame data.
      - Left hand:  21 pts × 3 dims = 63 (wrist-relative)
      - Right hand: 21 pts × 3 dims = 63 (wrist-relative)
      - Pose:        6 joints × 3 dims = 18 (midpoint-relative)
      - Hand presence flags: 2 binary features (left/right active)
      - Right wrist global position: 2 features (x, y)
    Total = 63 + 63 + 18 + 2 + 2 = 148  →  padded to 148
    """
    features = []

    # 1. Left Hand — wrist-relative (63 features)
    left = frame_data.get("left_hand", [])
    if left and len(left) == 21:
        wrist = left[0]
        wx, wy, wz = wrist.get("x", 0.0), wrist.get("y", 0.0), wrist.get("z", 0.0)
        for lm in left:
            features.extend([
                lm.get("x", 0.0) - wx,
                lm.get("y", 0.0) - wy,
                lm.get("z", 0.0) - wz
            ])
        features.append(1.0)  # left hand present flag
    else:
        features.extend([0.0] * 63)
        features.append(0.0)  # left hand absent

    # 2. Right Hand — wrist-relative (63 features)
    right = frame_data.get("right_hand", [])
    if right and len(right) == 21:
        wrist = right[0]
        wx, wy, wz = wrist.get("x", 0.0), wrist.get("y", 0.0), wrist.get("z", 0.0)
        for lm in right:
            features.extend([
                lm.get("x", 0.0) - wx,
                lm.get("y", 0.0) - wy,
                lm.get("z", 0.0) - wz
            ])
        features.append(1.0)  # right hand present flag
        # Global wrist position (where in frame the hand is)
        features.extend([wrist.get("x", 0.5), wrist.get("y", 0.5)])
    else:
        features.extend([0.0] * 63)
        features.append(0.0)  # right hand absent
        features.extend([0.5, 0.5])  # default center position

    # 3. Pose — midpoint-relative (18 features)
    pose = frame_data.get("pose", {})
    joints_of_interest = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist"
    ]

    # Shoulder midpoint as anchor
    mid_x, mid_y, mid_z = 0.5, 0.5, 0.0
    if "left_shoulder" in pose and "right_shoulder" in pose:
        ls = pose["left_shoulder"]
        rs = pose["right_shoulder"]
        mid_x = (ls.get("x", 0.5) + rs.get("x", 0.5)) / 2
        mid_y = (ls.get("y", 0.5) + rs.get("y", 0.5)) / 2
        mid_z = (ls.get("z", 0.0) + rs.get("z", 0.0)) / 2

    for j in joints_of_interest:
        pt = pose.get(j, None)
        if pt:
            features.extend([
                pt.get("x", mid_x) - mid_x,
                pt.get("y", mid_y) - mid_y,
                pt.get("z", mid_z) - mid_z
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def extract_sequence_features(sequence, target_length=20):
    """
    Resamples a sequence to target_length frames, computes per-frame features,
    then appends velocity (frame-to-frame delta) features for temporal dynamics.

    Output size: target_length * n_feat + (target_length-1) * n_feat
    With n_feat=148, target_length=20:
      = 20*148 + 19*148 = 2960 + 2812 = 5772 features
    """
    T = len(sequence)
    if T == 0:
        n_feat = len(extract_frame_features({}))
        total = target_length * n_feat + (target_length - 1) * n_feat
        return np.zeros(total, dtype=np.float32)

    # Resample to target_length using linear interpolation
    indices = np.linspace(0, T - 1, target_length)
    resampled = []
    for idx in indices:
        low = int(np.floor(idx))
        high = min(int(np.ceil(idx)), T - 1)
        weight = idx - low
        f_low = extract_frame_features(sequence[low])
        f_high = extract_frame_features(sequence[high])
        resampled.append((1.0 - weight) * f_low + weight * f_high)

    resampled = np.array(resampled)  # shape: (target_length, n_feat)

    # Velocity: consecutive frame differences
    velocity = np.diff(resampled, axis=0)  # shape: (target_length-1, n_feat)

    # Flatten and concatenate
    return np.concatenate([resampled.flatten(), velocity.flatten()]).astype(np.float32)


class TemporalSignClassifier:
    def __init__(self):
        self.model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "sign_classifier.pkl")
        )
        self.clf = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.clf = pickle.load(f)
                print(f"[Classifier] Loaded model from {self.model_path}")
            except Exception as e:
                print(f"[Classifier] Error loading model: {e}")
                self.clf = None

    def predict(self, sequence_buffer: SignSequenceBuffer):
        """
        Predicts top 3 ISL signs from sequence of frame landmarks.
        Returns: list of (label, confidence) tuples, length 3.
        """
        buffer = sequence_buffer.buffer
        if not buffer:
            return [("None", 1.0), ("None", 0.0), ("None", 0.0)]

        # Lazy reload if missing
        if self.clf is None:
            self.load_model()

        if self.clf is not None:
            try:
                features = extract_sequence_features(buffer, target_length=20).reshape(1, -1)
                probs = self.clf.predict_proba(features)[0]
                classes = self.clf.classes_
                preds = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
                while len(preds) < 3:
                    preds.append(("None", 0.0))
                return preds[:3]
            except Exception as e:
                print(f"[Classifier] Prediction error: {e}")

        # ── Heuristic fallback (only if RF unavailable) ──────────────────────
        left_ys, right_ys = [], []
        right_xs = []
        for frame in buffer:
            if frame["right_hand"] and len(frame["right_hand"]) > 0:
                right_ys.append(frame["right_hand"][0]["y"])
                right_xs.append(frame["right_hand"][0]["x"])
            if frame["left_hand"] and len(frame["left_hand"]) > 0:
                left_ys.append(frame["left_hand"][0]["y"])

        has_right = len(right_ys) > 3
        has_left = len(left_ys) > 3
        right_up = has_right and (right_ys[-1] < right_ys[0] - 0.05)
        left_up = has_left and (left_ys[-1] < left_ys[0] - 0.05)
        right_high = has_right and np.mean(right_ys) < 0.40
        right_forward = has_right and len(right_xs) > 3 and (right_xs[-1] > right_xs[0] + 0.05)

        scores = {v: 0.01 for v in VOCABULARY}
        if right_up and left_up:
            scores["Help"] = 0.85; scores["Emergency"] = 0.70
        elif right_high:
            scores["Hello"] = 0.80; scores["Father"] = 0.65
        elif right_up:
            scores["Thank You"] = 0.75; scores["Good"] = 0.60
        elif right_forward:
            scores["Stop"] = 0.78; scores["Go"] = 0.65
        else:
            scores["Yes"] = 0.60; scores["Please"] = 0.50

        total = sum(scores.values())
        for k in scores:
            scores[k] /= total
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]


# Global singleton
sign_classifier = TemporalSignClassifier()
