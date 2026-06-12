import json
import os
import pickle
import numpy as np
from modules.database import add_sign_history, get_sign_history, delete_sign_sample, save_setting, get_setting, add_model_version


def record_sign_sample(person_id, sign_label, sequence_buffer, model_version="1.0"):
    """Saves a recorded sequence buffer of hand/pose landmarks to the SQLite database."""
    if not sequence_buffer or len(sequence_buffer.buffer) == 0:
        return False, "Cannot record empty sequence buffer."

    landmarks_data = sequence_buffer.buffer
    add_sign_history(sign_label, landmarks_data, person_id, model_version)
    return True, f"Recorded sign sample for '{sign_label}' successfully."


def get_recorded_samples(label=None):
    """Returns list of recorded sign samples from database."""
    return get_sign_history(label)


def delete_recorded_sample(sample_id):
    """Deletes sign sample from database by ID."""
    delete_sign_sample(sample_id)
    return True


def retrain_sign_model():
    """
    Trains a scikit-learn RandomForestClassifier on the recorded samples in DatasetSamples.
    Saves the trained model to models/sign_classifier.pkl.
    """
    samples = get_recorded_samples()
    if not samples:
        return False, "Training failed: No recorded sign samples in database."

    X = []
    y = []

    from modules.signs.recognizer import extract_sequence_features

    for s in samples:
        features = extract_sequence_features(s["landmarks"], target_length=20)
        X.append(features)
        y.append(s["sign_label"])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        return False, "Training failed: Feature extraction returned empty arrays."

    unique_labels = list(set(y))
    if len(unique_labels) < 2:
        return False, f"Training failed: Need at least 2 classes, got {len(unique_labels)}."

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X, y)

        # Ensure models dir exists
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "sign_classifier.pkl")

        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        # Lazy reload model in recognizer instance
        from modules.signs.recognizer import sign_classifier
        sign_classifier.load_model()

        # Save model version metadata
        active_arch = get_setting("model_architecture", "Phase 1: Random Forest")
        current_version = float(get_setting("model_version", "1.0"))
        new_version = round(current_version + 0.1, 2)
        save_setting("model_version", str(new_version))

        # Estimate accuracy from training score
        train_score = float(clf.score(X, y))
        add_model_version(str(new_version), train_score, len(samples), active_arch, "Active")

        report = {
            "status": "success",
            "model_architecture": active_arch,
            "new_version": str(new_version),
            "total_samples": len(samples),
            "labels_trained": sorted(unique_labels),
            "training_accuracy": round(train_score, 4),
        }

        save_setting("training_report", json.dumps(report))
        return True, report
    except Exception as e:
        return False, f"Training failed: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Biomechanically-distinct synthetic ISL sign templates
# Each sign has a unique combination of:
#   - which hands are active
#   - trajectory (upward, forward, circular, static, etc.)
#   - starting position (forehead, chin, chest, neutral)
#   - finger configuration (represented by spread vs. closed landmarks)
# ─────────────────────────────────────────────────────────────────────────────

def _make_hand_open(cx, cy, spread=0.05):
    """Open hand: fingers spread out from center."""
    lm = [{"x": cx, "y": cy, "z": 0.0}]  # wrist
    for finger in range(5):
        angle = np.pi * (0.1 + finger * 0.2)
        for joint in range(1, 5):
            lm.append({
                "x": cx + spread * joint * np.cos(angle),
                "y": cy - spread * joint * np.sin(angle) * 0.5,
                "z": -0.01 * joint
            })
    return lm


def _make_hand_closed(cx, cy):
    """Closed fist: all landmarks clustered near wrist."""
    lm = [{"x": cx, "y": cy, "z": 0.0}]
    for i in range(20):
        lm.append({
            "x": cx + np.random.uniform(-0.01, 0.01),
            "y": cy + np.random.uniform(-0.01, 0.01),
            "z": -0.005
        })
    return lm


def _make_hand_pointing(cx, cy, direction_x=0.0, direction_y=-1.0):
    """Pointing hand: index finger extended."""
    lm = [{"x": cx, "y": cy, "z": 0.0}]
    # Thumb (curled)
    for i in range(1, 5):
        lm.append({"x": cx - 0.015 * i, "y": cy + 0.005 * i, "z": -0.005})
    # Index (extended in direction)
    for i in range(1, 5):
        lm.append({
            "x": cx + direction_x * 0.025 * i,
            "y": cy + direction_y * 0.025 * i,
            "z": -0.01 * i
        })
    # Middle/Ring/Pinky (curled)
    for finger in range(3):
        for j in range(1, 5):
            lm.append({"x": cx + 0.01 * finger, "y": cy + 0.008 * j, "z": -0.005})
    return lm


def _make_empty_hand():
    """No hand detected."""
    return []


def _base_pose(right_wrist_x=0.72, right_wrist_y=0.83,
               left_wrist_x=0.28, left_wrist_y=0.83):
    return {
        "left_shoulder":  {"x": 0.38, "y": 0.52, "z": 0.0},
        "right_shoulder": {"x": 0.62, "y": 0.52, "z": 0.0},
        "left_elbow":     {"x": 0.31, "y": 0.68, "z": 0.0},
        "right_elbow":    {"x": 0.69, "y": 0.68, "z": 0.0},
        "left_wrist":     {"x": left_wrist_x,  "y": left_wrist_y,  "z": 0.0},
        "right_wrist":    {"x": right_wrist_x, "y": right_wrist_y, "z": 0.0},
    }


def _sequence(frames):
    """Wrap list of frame dicts as a buffer-compatible list."""
    return frames


def _add_noise(val, sigma=0.008):
    return val + np.random.normal(0, sigma)


def _gen_sign_frames(label, n_frames=20, sample_idx=0):
    """
    Generate n_frames of landmark data for a given ISL sign.
    Each sign has a distinct kinematic pattern grounded in real ISL motion primitives.
    """
    np.random.seed(hash(label + str(sample_idx)) % 10000)
    t = np.linspace(0, 1, n_frames)
    frames = []

    # ── HELLO: right hand raised to forehead level, wave side-to-side ──────
    if label == "Hello":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62 + 0.06 * np.sin(ti * 2 * np.pi))
            ry = _add_noise(0.30 + 0.02 * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.045),
                           "pose": pose})

    # ── THANK YOU: right open hand from chin, arcs forward/down ─────────────
    elif label == "Thank You":
        for i, ti in enumerate(t):
            rx = _add_noise(0.60 + 0.05 * ti)
            ry = _add_noise(0.40 - 0.05 * ti + 0.08 * ti * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.04),
                           "pose": pose})

    # ── YES: right closed fist, nod down ────────────────────────────────────
    elif label == "Yes":
        for i, ti in enumerate(t):
            rx = _add_noise(0.65)
            ry = _add_noise(0.55 + 0.04 * np.sin(ti * 3 * np.pi))
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── NO: right index finger wags side to side ────────────────────────────
    elif label == "No":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62 + 0.07 * np.sin(ti * 4 * np.pi))
            ry = _add_noise(0.45)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, 0.0, -1.0),
                           "pose": pose})

    # ── HELP: both hands raise upward ───────────────────────────────────────
    elif label == "Help":
        for i, ti in enumerate(t):
            rx = _add_noise(0.65)
            ry = _add_noise(0.75 - 0.30 * ti)
            lx = _add_noise(0.35)
            ly = _add_noise(0.75 - 0.30 * ti)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_open(lx, ly, 0.04),
                           "right_hand": _make_hand_open(rx, ry, 0.04),
                           "pose": pose})

    # ── WATER: W-handshape (three fingers) at chin, twisted twice ───────────
    elif label == "Water":
        for i, ti in enumerate(t):
            rx = _add_noise(0.60 + 0.01 * np.sin(ti * 4 * np.pi))
            ry = _add_noise(0.42 + 0.02 * np.sin(ti * 4 * np.pi))
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, 0.3, -0.5),
                           "pose": pose})

    # ── FOOD: fingers bunched to mouth, repeated ────────────────────────────
    elif label == "Food":
        for i, ti in enumerate(t):
            rx = _add_noise(0.60)
            ry = _add_noise(0.38 + 0.03 * np.sin(ti * 3 * np.pi))
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── MOTHER: right open hand at chin level ───────────────────────────────
    elif label == "Mother":
        for i, ti in enumerate(t):
            rx = _add_noise(0.60)
            ry = _add_noise(0.42 + 0.01 * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.055),
                           "pose": pose})

    # ── FATHER: right open hand at forehead ─────────────────────────────────
    elif label == "Father":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62)
            ry = _add_noise(0.28 - 0.01 * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.055),
                           "pose": pose})

    # ── BROTHER: right fist at forehead, arc to neutral ─────────────────────
    elif label == "Brother":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62)
            ry = _add_noise(0.28 + 0.30 * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── SISTER: right fist at chin, arc to neutral ──────────────────────────
    elif label == "Sister":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62)
            ry = _add_noise(0.42 + 0.20 * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── FRIEND: both index fingers link/hook ────────────────────────────────
    elif label == "Friend":
        for i, ti in enumerate(t):
            rx = _add_noise(0.55 + 0.02 * np.sin(ti * np.pi))
            ry = _add_noise(0.55)
            lx = _add_noise(0.45 - 0.02 * np.sin(ti * np.pi))
            ly = _add_noise(0.55)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_pointing(lx, ly, 1.0, 0.0),
                           "right_hand": _make_hand_pointing(rx, ry, -1.0, 0.0),
                           "pose": pose})

    # ── SCHOOL: clap both open hands twice ──────────────────────────────────
    elif label == "School":
        for i, ti in enumerate(t):
            offset = 0.06 * abs(np.sin(ti * 2 * np.pi))
            rx = _add_noise(0.53 + offset)
            lx = _add_noise(0.47 - offset)
            ry = _add_noise(0.55)
            ly = _add_noise(0.55)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_open(lx, ly, 0.04),
                           "right_hand": _make_hand_open(rx, ry, 0.04),
                           "pose": pose})

    # ── TEACHER: right fingers spread at forehead, push forward ─────────────
    elif label == "Teacher":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62 + 0.08 * ti)
            ry = _add_noise(0.28)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.045),
                           "pose": pose})

    # ── HOSPITAL: H-shape (two fingers) drawn on upper arm ──────────────────
    elif label == "Hospital":
        for i, ti in enumerate(t):
            # Right hand draws H on left upper arm region
            rx = _add_noise(0.38 + 0.04 * np.sin(ti * np.pi * 2))
            ry = _add_noise(0.52 + 0.04 * np.cos(ti * np.pi * 2))
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, 0.0, -1.0),
                           "pose": pose})

    # ── DOCTOR: D-handshape (index+thumb) wrist tap ─────────────────────────
    elif label == "Doctor":
        for i, ti in enumerate(t):
            rx = _add_noise(0.45 + 0.02 * np.sin(ti * 3 * np.pi))
            ry = _add_noise(0.65 + 0.02 * np.sin(ti * 3 * np.pi))
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, -0.3, -0.3),
                           "pose": pose})

    # ── EMERGENCY: both fists raised high and shaken ─────────────────────────
    elif label == "Emergency":
        for i, ti in enumerate(t):
            shake = 0.04 * np.sin(ti * 6 * np.pi)
            rx = _add_noise(0.65 + shake)
            ry = _add_noise(0.28)
            lx = _add_noise(0.35 - shake)
            ly = _add_noise(0.28)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_closed(lx, ly),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── PAIN: both index fingers jab toward each other at chest ─────────────
    elif label == "Pain":
        for i, ti in enumerate(t):
            rx = _add_noise(0.55 - 0.03 * np.sin(ti * 2 * np.pi))
            ry = _add_noise(0.60)
            lx = _add_noise(0.45 + 0.03 * np.sin(ti * 2 * np.pi))
            ly = _add_noise(0.60)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_pointing(lx, ly, 1.0, 0.0),
                           "right_hand": _make_hand_pointing(rx, ry, -1.0, 0.0),
                           "pose": pose})

    # ── MEDICINE: right M-shape rotated at palm ──────────────────────────────
    elif label == "Medicine":
        for i, ti in enumerate(t):
            rx = _add_noise(0.50 + 0.02 * np.sin(ti * 4 * np.pi))
            ry = _add_noise(0.60)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_hand_open(0.45, 0.60, 0.03),
                           "right_hand": _make_hand_pointing(rx, ry, 0.0, 1.0),
                           "pose": pose})

    # ── BATHROOM: B-hand twist at wrist ─────────────────────────────────────
    elif label == "Bathroom":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62 + 0.02 * np.sin(ti * 6 * np.pi))
            ry = _add_noise(0.52)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.02),
                           "pose": pose})

    # ── HOME: fingertips touch cheek then chin ───────────────────────────────
    elif label == "Home":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62)
            ry = _add_noise(0.35 + 0.12 * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── EAT: fingers to mouth, repeated ─────────────────────────────────────
    elif label == "Eat":
        for i, ti in enumerate(t):
            rx = _add_noise(0.60)
            ry = _add_noise(0.37 + 0.02 * np.sin(ti * 4 * np.pi))
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── DRINK: C-handshape tilted toward mouth ───────────────────────────────
    elif label == "Drink":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62)
            ry = _add_noise(0.38 - 0.05 * ti + 0.05 * ti * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.025),
                           "pose": pose})

    # ── SLEEP: right hand, palm on cheek, head tilts ────────────────────────
    elif label == "Sleep":
        for i, ti in enumerate(t):
            rx = _add_noise(0.65 - 0.02 * ti)
            ry = _add_noise(0.35 + 0.02 * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.05),
                           "pose": pose})

    # ── STOP: right palm pushes forward/outward ──────────────────────────────
    elif label == "Stop":
        for i, ti in enumerate(t):
            rx = _add_noise(0.62 + 0.12 * ti)
            ry = _add_noise(0.55)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.055),
                           "pose": pose})

    # ── COME: right index beckons (curls inward) ─────────────────────────────
    elif label == "Come":
        for i, ti in enumerate(t):
            rx = _add_noise(0.70 - 0.08 * ti)
            ry = _add_noise(0.55 + 0.02 * np.sin(ti * 2 * np.pi))
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, -1.0, 0.0),
                           "pose": pose})

    # ── GO: right index points away and moves outward ───────────────────────
    elif label == "Go":
        for i, ti in enumerate(t):
            rx = _add_noise(0.60 + 0.10 * ti)
            ry = _add_noise(0.52)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, 1.0, 0.0),
                           "pose": pose})

    # ── GOOD: right open hand at chin, moves forward ────────────────────────
    elif label == "Good":
        for i, ti in enumerate(t):
            rx = _add_noise(0.60 + 0.06 * ti)
            ry = _add_noise(0.40 + 0.02 * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.05),
                           "pose": pose})

    # ── BAD: right open hand at chin, flips downward ────────────────────────
    elif label == "Bad":
        for i, ti in enumerate(t):
            rx = _add_noise(0.60 + 0.03 * ti)
            ry = _add_noise(0.40 + 0.15 * ti)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.05),
                           "pose": pose})

    # ── PLEASE: right open hand circles on chest ─────────────────────────────
    elif label == "Please":
        for i, ti in enumerate(t):
            rx = _add_noise(0.60 + 0.05 * np.cos(ti * 2 * np.pi))
            ry = _add_noise(0.55 + 0.05 * np.sin(ti * 2 * np.pi))
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.05),
                           "pose": pose})

    else:
        # Generic fallback: random directional movement unique per label
        angle = (hash(label) % 360) * np.pi / 180.0
        for i, ti in enumerate(t):
            rx = _add_noise(0.62 + 0.10 * ti * np.cos(angle))
            ry = _add_noise(0.55 + 0.10 * ti * np.sin(angle))
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.04),
                           "pose": pose})

    return frames


def initialize_default_dataset_if_empty():
    """
    Checks if DatasetSamples table is empty, and if so:
    Generates biomechanically-distinct synthetic sign sequences per ISL label,
    then trains the RandomForest classifier.
    """
    samples = get_recorded_samples()
    if len(samples) == 0:
        from modules.signs.recognizer import VOCABULARY

        print("[Dataset] Generating discriminative synthetic ISL dataset...")

        class MockSeqBuf:
            def __init__(self, buf):
                self.buffer = buf

        SAMPLES_PER_SIGN = 10  # Increased from 3 for better model generalization

        for label in VOCABULARY:
            for sample_idx in range(SAMPLES_PER_SIGN):
                frames = _gen_sign_frames(label, n_frames=20, sample_idx=sample_idx)
                record_sign_sample("Synthetic", label, MockSeqBuf(frames))

        print(f"[Dataset] Generated {len(VOCABULARY) * SAMPLES_PER_SIGN} samples. Training model...")
        retrain_sign_model()
        print("[Dataset] Model training complete.")
