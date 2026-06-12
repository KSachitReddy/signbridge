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
NO_SIGN_LABEL = "No Gesture Detected"
CONFIDENCE_THRESHOLD = 0.55          # below this the model is not confident enough
HANDS_PRESENT_MIN_RATIO = 0.20       # need ≥20 % of buffer frames to have a hand

# Vocabulary List (10 gestures)
VOCABULARY = [
    "Thumbs Up", "Thumbs Down", "Point Left", "Point Right", "Point Up",
    "Point Down", "Open Palm", "Closed Fist", "Hello", "Bye"
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
        Uses a robust rule-based model utilizing active hand 21 landmarks.
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

        # 1. Classify each frame's active hand gesture
        open_palm_count = 0
        detected_static_list = []
        
        for f in buffer:
            hand = f.get("right_hand", [])
            if not (hand and len(hand) == 21):
                hand = f.get("left_hand", [])
            if hand and len(hand) == 21:
                static_g = self._detect_static_gesture(hand)
                detected_static_list.append(static_g)
                if static_g == "Open Palm":
                    open_palm_count += 1
            else:
                detected_static_list.append("None")

        # Get active hand coordinates
        x_coords = []
        y_coords = []
        for f in buffer:
            hand = f.get("right_hand", [])
            if not (hand and len(hand) == 21):
                hand = f.get("left_hand", [])
            if hand and len(hand) == 21:
                x_coords.append(hand[0]["x"])
                y_coords.append(hand[0]["y"])

        pred_label = NO_SIGN_LABEL
        pred_conf = 0.0

        # Check temporal Hello/Bye first if enough frames have an Open Palm
        valid_frames_count = len(x_coords)
        if valid_frames_count >= 6 and (open_palm_count / valid_frames_count) >= 0.30:
            diffs = [x_coords[i] - x_coords[i-1] for i in range(1, valid_frames_count)]
            sig_diffs = [d for d in diffs if abs(d) > 0.005]
            
            dir_changes = 0
            last_sign = 0
            for d in sig_diffs:
                curr_sign = 1 if d > 0 else -1
                if last_sign != 0 and curr_sign != last_sign:
                    dir_changes += 1
                last_sign = curr_sign
                
            total_path = sum(abs(d) for d in diffs)
            displacement = x_coords[-1] - x_coords[0]
            abs_disp = abs(displacement)
            
            if dir_changes >= 2 and total_path > 0.12:
                pred_label = "Bye"
                pred_conf = 0.90
            elif dir_changes <= 1 and abs_disp > 0.08 and total_path < 1.6 * abs_disp:
                pred_label = "Hello"
                pred_conf = 0.85

        # If no temporal gesture detected, find the most common static gesture
        if pred_label == NO_SIGN_LABEL:
            valid_static = [g for g in detected_static_list if g != "None"]
            if valid_static:
                from collections import Counter
                counter = Counter(valid_static)
                best_static, count = counter.most_common(1)[0]
                ratio = count / len(detected_static_list)
                if ratio >= 0.40:
                    pred_label = best_static
                    pred_conf = float(0.55 + 0.40 * ratio)  # 0.71 to 0.95

        # Fill top 3 predictions
        scores = {g: 0.0 for g in VOCABULARY}
        if pred_label in scores:
            scores[pred_label] = pred_conf
            remaining = 1.0 - pred_conf
            # Distribute remaining scores among next top static gestures
            other_candidates = [g for g in VOCABULARY if g != pred_label]
            for i, cand in enumerate(other_candidates[:2]):
                scores[cand] = remaining * (0.6 if i == 0 else 0.4)
        else:
            scores[NO_SIGN_LABEL] = 1.0

        preds = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # If top label is NO_SIGN_LABEL, return it
        if pred_label == NO_SIGN_LABEL:
            return [(NO_SIGN_LABEL, 1.0), (NO_SIGN_LABEL, 0.0), (NO_SIGN_LABEL, 0.0)]
        
        return preds[:3]

    def _detect_static_gesture(self, hand_lms) -> str:
        if not hand_lms or len(hand_lms) < 21:
            return "None"
        
        # Distance helper
        def d(i1, i2):
            p1 = hand_lms[i1]
            p2 = hand_lms[i2]
            return float(np.sqrt((p1["x"] - p2["x"])**2 + (p1["y"] - p2["y"])**2))

        # Check four fingers: Index, Middle, Ring, Pinky
        index_ext = d(8, 0) > d(6, 0)
        middle_ext = d(12, 0) > d(10, 0)
        ring_ext = d(16, 0) > d(14, 0)
        pinky_ext = d(20, 0) > d(18, 0)

        # Thumb extension: check distance from thumb tip (4) to index MCP (5)
        # and thumb tip (4) to middle MCP (9).
        thumb_ext = d(4, 9) > d(2, 9)

        extensions = [thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext]
        num_extended = sum(extensions)

        # 1. Closed Fist: all fingers folded
        if num_extended <= 1 and not thumb_ext and not index_ext:
            return "Closed Fist"
        
        # 2. Open Palm: all fingers extended
        if num_extended >= 4:
            return "Open Palm"

        # 3. Thumbs Up / Down: only thumb extended
        if thumb_ext and num_extended == 1:
            # If tip y (4) is significantly higher than MCP y (2) (Y coordinate is smaller in image space)
            if hand_lms[4]["y"] < hand_lms[2]["y"]:
                return "Thumbs Up"
            else:
                return "Thumbs Down"

        # 4. Point Up/Down/Left/Right: only index extended
        if index_ext and not middle_ext and not ring_ext and not pinky_ext:
            dx = hand_lms[8]["x"] - hand_lms[5]["x"]
            dy = hand_lms[8]["y"] - hand_lms[5]["y"]
            if abs(dx) > abs(dy):
                return "Point Right" if dx > 0 else "Point Left"
            else:
                return "Point Down" if dy > 0 else "Point Up"

        return "None"


# Global singleton — loaded once at module import
sign_classifier = TemporalSignClassifier()
