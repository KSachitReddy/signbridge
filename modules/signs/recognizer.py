"""
recognizer.py — ISL sign sequence classifier.

Feature pipeline (per frame, 148 dims):
  Left hand:  21 pts × 3 = 63  (wrist-relative)           presence flag: 1
  Right hand: 21 pts × 3 = 63  (wrist-relative)           presence flag: 1
  Right wrist global: 2
  Pose: 6 joints × 3 = 18  (shoulder-midpoint-relative, shoulder-width normalised)
  Total per frame = 148

Sequence features (5776 dims per sequence):
  Positions:         20 × 148 = 2960
  Velocity (Δ1):     19 × 148 = 2812
  Acceleration (Δ2): 18 × 148 = 2664  ← new
  Hand-presence:      2 scalars        ← new
  Grand total = 8438

Confidence gate:
  If the RF top-1 probability < CONFIDENCE_THRESHOLD → "No Sign Detected"
  If < HANDS_PRESENT_MIN_RATIO of buffer frames have any hand → "No Sign Detected"
"""

import numpy as np
import os
import pickle
from modules.database import get_setting

# ── Public constants ──────────────────────────────────────────────────────────
NO_SIGN_LABEL = "No Sign Detected"
CONFIDENCE_THRESHOLD = 0.55          # below this the model is not confident enough
HANDS_PRESENT_MIN_RATIO = 0.20       # need ≥20 % of buffer frames to have a hand

# Vocabulary List (30 ISL signs — Phase 1)
VOCABULARY = [
    "Hello", "Thank You", "Yes", "No", "Help",
    "Water", "Food", "Mother", "Father", "Brother",
    "Sister", "Friend", "School", "Teacher", "Hospital",
    "Doctor", "Emergency", "Pain", "Medicine", "Bathroom",
    "Home", "Eat", "Drink", "Sleep", "Stop",
    "Come", "Go", "Good", "Bad", "Please"
]


# ─────────────────────────────────────────────────────────────────────────────
class SignSequenceBuffer:
    def __init__(self, size=20):
        self.size = size
        self.buffer = []

    def add(self, left_hand, right_hand, pose):
        self.buffer.append({"left_hand": left_hand, "right_hand": right_hand, "pose": pose})
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def clear(self):
        self.buffer.clear()

    def is_full(self):
        return len(self.buffer) >= self.size


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_frame_features(frame_data: dict) -> np.ndarray:
    """
    Returns a 148-dim feature vector from one frame.

    Improvements over v2:
    - Pose features are divided by shoulder_width (scale/distance normalisation).
      Signs at 0.5 m and 1.5 m from camera now produce similar feature values.
    """
    features = []

    # 1. Left hand — wrist-relative (63 + 1 presence = 64)
    left = frame_data.get("left_hand", [])
    if left and len(left) == 21:
        wrist = left[0]
        wx, wy, wz = wrist.get("x", 0.0), wrist.get("y", 0.0), wrist.get("z", 0.0)
        for lm in left:
            features.extend([lm.get("x", 0.0) - wx,
                              lm.get("y", 0.0) - wy,
                              lm.get("z", 0.0) - wz])
        features.append(1.0)
    else:
        features.extend([0.0] * 64)   # 63 coords + 1 flag

    # 2. Right hand — wrist-relative (63 + 1 presence + 2 global = 66)
    right = frame_data.get("right_hand", [])
    if right and len(right) == 21:
        wrist = right[0]
        wx, wy, wz = wrist.get("x", 0.0), wrist.get("y", 0.0), wrist.get("z", 0.0)
        for lm in right:
            features.extend([lm.get("x", 0.0) - wx,
                              lm.get("y", 0.0) - wy,
                              lm.get("z", 0.0) - wz])
        features.append(1.0)
        features.extend([wrist.get("x", 0.5), wrist.get("y", 0.5)])
    else:
        features.extend([0.0] * 63)
        features.append(0.0)
        features.extend([0.5, 0.5])

    # 3. Pose — midpoint-relative, shoulder-width normalised (6 × 3 = 18)
    pose = frame_data.get("pose", {})
    joints_of_interest = [
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    ]

    mid_x, mid_y, mid_z = 0.5, 0.5, 0.0
    shoulder_width = 1.0          # normalisation denominator — default = no scaling

    if "left_shoulder" in pose and "right_shoulder" in pose:
        ls = pose["left_shoulder"]
        rs = pose["right_shoulder"]
        mid_x = (ls.get("x", 0.5) + rs.get("x", 0.5)) / 2
        mid_y = (ls.get("y", 0.5) + rs.get("y", 0.5)) / 2
        mid_z = (ls.get("z", 0.0) + rs.get("z", 0.0)) / 2
        # Euclidean distance between shoulders in normalised image coords
        dx = ls.get("x", 0.5) - rs.get("x", 0.5)
        dy = ls.get("y", 0.5) - rs.get("y", 0.5)
        w = float(np.sqrt(dx * dx + dy * dy))
        if w > 1e-4:
            shoulder_width = w

    for j in joints_of_interest:
        pt = pose.get(j)
        if pt:
            features.extend([
                (pt.get("x", mid_x) - mid_x) / shoulder_width,
                (pt.get("y", mid_y) - mid_y) / shoulder_width,
                (pt.get("z", mid_z) - mid_z) / shoulder_width,
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

    # Total: 64 + 66 + 18 = 148
    return np.array(features, dtype=np.float32)


def extract_sequence_features(sequence: list, target_length: int = 20) -> np.ndarray:
    """
    Resamples to target_length frames, then stacks:
      • Positions   (target_length × n_feat)
      • Velocity Δ1 (target_length-1 × n_feat)
      • Acceleration Δ2 (target_length-2 × n_feat)  ← new
      • Hand-presence ratio left + right (2 scalars)  ← new

    With n_feat=148, target_length=20:
      2960 + 2812 + 2664 + 2 = 8438 features
    """
    T = len(sequence)
    if T == 0:
        n_feat = len(extract_frame_features({}))
        pos_dim = target_length * n_feat
        vel_dim = (target_length - 1) * n_feat
        acc_dim = (target_length - 2) * n_feat
        return np.zeros(pos_dim + vel_dim + acc_dim + 2, dtype=np.float32)

    # ── Resample ──────────────────────────────────────────────────────────────
    indices = np.linspace(0, T - 1, target_length)
    resampled = []
    for idx in indices:
        low = int(np.floor(idx))
        high = min(int(np.ceil(idx)), T - 1)
        w = idx - low
        f_low = extract_frame_features(sequence[low])
        f_high = extract_frame_features(sequence[high])
        resampled.append((1.0 - w) * f_low + w * f_high)

    resampled = np.array(resampled)         # (target_length, n_feat)

    # ── Temporal derivatives ──────────────────────────────────────────────────
    velocity = np.diff(resampled, axis=0)   # (target_length-1, n_feat)
    acceleration = np.diff(velocity, axis=0)  # (target_length-2, n_feat)

    # ── Hand-presence ratio (sequence-level summary) ──────────────────────────
    n_frames = len(sequence)
    right_frames = sum(1 for f in sequence if len(f.get("right_hand", [])) == 21)
    left_frames  = sum(1 for f in sequence if len(f.get("left_hand",  [])) == 21)
    right_ratio = np.float32(right_frames / max(n_frames, 1))
    left_ratio  = np.float32(left_frames  / max(n_frames, 1))

    return np.concatenate([
        resampled.flatten(),
        velocity.flatten(),
        acceleration.flatten(),
        [right_ratio, left_ratio],
    ]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────

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
                print(f"[Classifier] Loaded model from {self.model_path} "
                      f"(expects {self.clf.n_features_in_} features)")
            except Exception as e:
                print(f"[Classifier] Load error: {e}")
                self.clf = None

    # ── public ────────────────────────────────────────────────────────────────

    def predict(self, sequence_buffer: SignSequenceBuffer) -> list:
        """
        Returns top-3 (label, confidence) list.

        Gates (in order, early-exit):
          1. Empty buffer → NO_SIGN_LABEL
          2. Too few frames with a hand detected → NO_SIGN_LABEL
          3. RF probability below CONFIDENCE_THRESHOLD → NO_SIGN_LABEL
        """
        buffer = sequence_buffer.buffer
        if not buffer:
            return [(NO_SIGN_LABEL, 1.0), (NO_SIGN_LABEL, 0.0), (NO_SIGN_LABEL, 0.0)]

        # Gate 1: hand presence
        n = len(buffer)
        right_count = sum(1 for f in buffer if len(f.get("right_hand", [])) == 21)
        left_count  = sum(1 for f in buffer if len(f.get("left_hand",  [])) == 21)
        if (right_count + left_count) / (2 * max(n, 1)) < HANDS_PRESENT_MIN_RATIO:
            return [(NO_SIGN_LABEL, 1.0), (NO_SIGN_LABEL, 0.0), (NO_SIGN_LABEL, 0.0)]

        # Lazy reload
        if self.clf is None:
            self.load_model()

        if self.clf is not None:
            try:
                features = extract_sequence_features(buffer, target_length=20).reshape(1, -1)

                # Gate 2: feature-size mismatch means stale model — force heuristic
                if features.shape[1] != self.clf.n_features_in_:
                    print(f"[Classifier] Feature mismatch "
                          f"({features.shape[1]} vs {self.clf.n_features_in_}). "
                          f"Model needs retraining.")
                    self.clf = None
                    return self._heuristic(buffer)

                probs  = self.clf.predict_proba(features)[0]
                classes = self.clf.classes_
                preds  = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)

                while len(preds) < 3:
                    preds.append((NO_SIGN_LABEL, 0.0))
                return preds[:3]

            except Exception as e:
                print(f"[Classifier] Prediction error: {e}")

        return self._heuristic(buffer)

    # ── private ───────────────────────────────────────────────────────────────

    def _heuristic(self, buffer: list) -> list:
        """
        Velocity-based fallback when RF is unavailable.
        Returns NO_SIGN_LABEL when motion is too low to identify.
        """
        left_ys, right_ys, right_xs = [], [], []
        for frame in buffer:
            rh = frame.get("right_hand", [])
            lh = frame.get("left_hand", [])
            if rh and len(rh) == 21:
                right_ys.append(rh[0]["y"])
                right_xs.append(rh[0]["x"])
            if lh and len(lh) == 21:
                left_ys.append(lh[0]["y"])

        has_right = len(right_ys) >= 4
        has_left  = len(left_ys)  >= 4

        # If hands are present but barely moving → no sign
        if has_right:
            travel = abs(right_ys[-1] - right_ys[0]) + abs(right_xs[-1] - right_xs[0])
            if travel < 0.04:
                return [(NO_SIGN_LABEL, 0.95), (NO_SIGN_LABEL, 0.0), (NO_SIGN_LABEL, 0.0)]

        right_up  = has_right and (right_ys[-1] < right_ys[0] - 0.05)
        left_up   = has_left  and (left_ys[-1]  < left_ys[0]  - 0.05)
        right_high = has_right and np.mean(right_ys) < 0.40
        right_fwd  = has_right and len(right_xs) >= 4 and (right_xs[-1] > right_xs[0] + 0.05)

        scores = {v: 0.01 for v in VOCABULARY}
        if right_up and left_up:
            scores["Help"] = 0.85; scores["Emergency"] = 0.70
        elif right_high:
            scores["Hello"] = 0.80; scores["Father"] = 0.65
        elif right_up:
            scores["Thank You"] = 0.75; scores["Good"] = 0.60
        elif right_fwd:
            scores["Stop"] = 0.78; scores["Go"] = 0.65
        else:
            # Not enough motion to identify — return no-sign
            return [(NO_SIGN_LABEL, 0.90), (NO_SIGN_LABEL, 0.0), (NO_SIGN_LABEL, 0.0)]

        total = sum(scores.values())
        for k in scores:
            scores[k] /= total
        preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        return preds


# Global singleton — loaded once at module import
sign_classifier = TemporalSignClassifier()
