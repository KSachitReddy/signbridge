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
    Trains a RandomForestClassifier on all recorded samples (real + synthetic).
    Computes a before/after confusion analysis using a stratified 80/20 split.
    Saves model to models/sign_classifier.pkl.
    """
    from modules.signs.recognizer import extract_sequence_features, NO_SIGN_LABEL

    samples = get_recorded_samples()
    if not samples:
        return False, "Training failed: No recorded sign samples in database."

    X, y = [], []
    for s in samples:
        feat = extract_sequence_features(s["landmarks"], target_length=20)
        X.append(feat)
        y.append(s["sign_label"])

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        return False, "Training failed: Feature extraction returned empty arrays."

    unique_labels = sorted(set(y))
    if len(unique_labels) < 2:
        return False, f"Training failed: Need ≥2 classes, got {len(unique_labels)}."

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import confusion_matrix, classification_report

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )

        # ── Before/After analysis on a held-out 20 % ─────────────────────────
        confusion_analysis = {}
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, test_idx = next(splitter.split(X, y))
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            clf_eval = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            clf_eval.fit(X_tr, y_tr)
            y_pred = clf_eval.predict(X_te)

            # Confusion on test set
            labels_present = sorted(set(y_te) | set(y_pred))
            cm = confusion_matrix(y_te, y_pred, labels=labels_present)

            # Top confused pairs (off-diagonal with count > 0)
            confused_pairs = []
            for i, la in enumerate(labels_present):
                for j, lb in enumerate(labels_present):
                    if i != j and cm[i][j] > 0:
                        confused_pairs.append({"true": la, "pred": lb, "count": int(cm[i][j])})
            confused_pairs.sort(key=lambda d: d["count"], reverse=True)

            # Per-class accuracy on test set
            cr = classification_report(y_te, y_pred, labels=labels_present,
                                        output_dict=True, zero_division=0)
            per_class = {
                lbl: {
                    "precision": round(cr[lbl]["precision"], 3),
                    "recall":    round(cr[lbl]["recall"],    3),
                    "f1":        round(cr[lbl]["f1-score"],  3),
                    "support":   int(cr[lbl]["support"]),
                }
                for lbl in labels_present if lbl in cr
            }

            # Idle (No Sign) precision/recall specifically
            no_sign_stats = per_class.get(NO_SIGN_LABEL, {})

            confusion_analysis = {
                "holdout_accuracy":   round(float(np.mean(y_pred == y_te)), 4),
                "top_confused_pairs": confused_pairs[:10],
                "no_sign_precision":  no_sign_stats.get("precision", "N/A"),
                "no_sign_recall":     no_sign_stats.get("recall",    "N/A"),
                "no_sign_f1":         no_sign_stats.get("f1",        "N/A"),
                "per_class_f1":       {k: v["f1"] for k, v in per_class.items()},
            }
        except Exception as ce:
            confusion_analysis = {"error": str(ce)}

        # ── Train final model on all data ─────────────────────────────────────
        clf.fit(X, y)

        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "sign_classifier.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        from modules.signs.recognizer import sign_classifier
        sign_classifier.load_model()

        active_arch = get_setting("model_architecture", "Phase 1: Random Forest")
        current_version = float(get_setting("model_version", "1.0"))
        new_version = round(current_version + 0.1, 2)
        save_setting("model_version", str(new_version))

        train_score = float(clf.score(X, y))
        add_model_version(str(new_version), train_score, len(samples), active_arch, "Active")

        report = {
            "status":             "success",
            "model_architecture": active_arch,
            "new_version":        str(new_version),
            "total_samples":      len(samples),
            "labels_trained":     unique_labels,
            "no_sign_class":      NO_SIGN_LABEL in unique_labels,
            "training_accuracy":  round(train_score, 4),
            "feature_dims":       int(X.shape[1]),
            "confusion_analysis": confusion_analysis,
        }

        save_setting("training_report", json.dumps(report))
        return True, report
    except Exception as e:
        return False, f"Training failed: {e}"


def generate_confusion_report() -> dict:
    """
    Standalone: loads existing model, generates idle test frames,
    and shows before (no threshold) vs after (with threshold) predictions.
    Returns a structured report dict.
    """
    from modules.signs.recognizer import (
        sign_classifier, SignSequenceBuffer,
        NO_SIGN_LABEL, CONFIDENCE_THRESHOLD,
        HANDS_PRESENT_MIN_RATIO,
    )

    report = {
        "before_threshold": {},
        "after_threshold":  {},
        "summary": "",
    }

    if sign_classifier.clf is None:
        report["summary"] = "Model not loaded — run retrain first."
        return report

    # Build 20 idle test sequences (no intentional gesture)
    idle_sequences = []
    for i in range(20):
        frames = _gen_no_sign_frames(n_frames=20, sample_idx=i)
        idle_sequences.append(frames)

    before_preds, after_preds = [], []

    for seq in idle_sequences:
        buf = SignSequenceBuffer(size=20)
        for f in seq:
            buf.add(f["left_hand"], f["right_hand"], f["pose"])

        # ── BEFORE: raw RF output (no gate) ──────────────────────────────────
        try:
            from modules.signs.recognizer import extract_sequence_features
            feat = extract_sequence_features(buf.buffer, target_length=20).reshape(1, -1)
            if feat.shape[1] == sign_classifier.clf.n_features_in_:
                probs  = sign_classifier.clf.predict_proba(feat)[0]
                top_i  = int(np.argmax(probs))
                top_lb = sign_classifier.clf.classes_[top_i]
                top_c  = float(probs[top_i])
                before_preds.append((top_lb, round(top_c, 3)))
            else:
                before_preds.append(("MISMATCH", 0.0))
        except Exception as ex:
            before_preds.append((f"ERR:{ex}", 0.0))

        # ── AFTER: gated predict() ────────────────────────────────────────────
        top = sign_classifier.predict(buf)
        after_preds.append((top[0][0], round(top[0][1], 3)))

    # Tally
    before_no_sign = sum(1 for lb, _ in before_preds if lb == NO_SIGN_LABEL)
    after_no_sign  = sum(1 for lb, _ in after_preds  if lb == NO_SIGN_LABEL)
    before_fp = len(before_preds) - before_no_sign   # false positive signs
    after_fp  = len(after_preds)  - after_no_sign

    report["before_threshold"] = {
        "idle_frames_tested": len(idle_sequences),
        "correctly_labelled_no_sign": before_no_sign,
        "false_positive_signs": before_fp,
        "sample_predictions": before_preds[:10],
    }
    report["after_threshold"] = {
        "idle_frames_tested": len(idle_sequences),
        "correctly_labelled_no_sign": after_no_sign,
        "false_positive_signs": after_fp,
        "sample_predictions": after_preds[:10],
    }
    report["summary"] = (
        f"BEFORE: {before_fp}/20 idle frames falsely labelled as signs. "
        f"AFTER: {after_fp}/20 idle frames falsely labelled as signs."
    )
    return report


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
    """Add Gaussian noise. Sigma scales with sample_idx for intra-class variation."""
    return val + np.random.normal(0, sigma)


def _add_noise_s(val, sigma=0.008, sample_idx=0):
    """Noise scaled by sample_idx to produce more diversity across samples."""
    # Base variation + extra spread for higher sample indices
    effective_sigma = sigma * (1.0 + 0.15 * (sample_idx % 5))
    return val + np.random.normal(0, effective_sigma)


def _gen_sign_frames(label, n_frames=30, sample_idx=0):
    """
    Generate n_frames of landmark data for a given ISL sign.
    Each sign has a distinct kinematic pattern grounded in real ISL motion primitives.
    Dataset v2: more frames, richer noise variation per sample_idx.
    """
    np.random.seed(hash(label + str(sample_idx)) % 10000)
    t = np.linspace(0, 1, n_frames)
    # Speed variation: some samples faster or slower
    speed = 1.0 + 0.2 * np.sin(sample_idx * 1.3)
    t_s = np.clip(t * speed, 0, 1)
    frames = []
    n = lambda v, s=0.010: _add_noise_s(v, s, sample_idx)  # shorthand with sample scaling

    # ── HELLO: right open hand raised to forehead, wave side-to-side ────────
    # Distinctive: forehead-level (y~0.28), lateral oscillation, right only
    if label == "Hello":
        for i, ti in enumerate(t_s):
            rx = n(0.62 + 0.07 * np.sin(ti * 2 * np.pi), 0.012)
            ry = n(0.28 + 0.015 * ti, 0.008)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.048),
                           "pose": pose})

    # ── THANK YOU: right open hand from chin arcs forward+down ───────────────
    # Distinctive: starts at chin (y~0.42), arcs outward, right only
    elif label == "Thank You":
        for i, ti in enumerate(t_s):
            rx = n(0.60 + 0.08 * ti, 0.010)
            ry = n(0.42 - 0.06 * ti + 0.10 * ti * ti, 0.008)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.042),
                           "pose": pose})

    # ── YES: right closed fist, nodding up-down repeatedly ───────────────────
    # Distinctive: closed fist at chest (y~0.55), vertical oscillation ×3
    elif label == "Yes":
        for i, ti in enumerate(t_s):
            rx = n(0.65, 0.007)
            ry = n(0.55 + 0.05 * np.sin(ti * 3 * np.pi), 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── NO: right index finger wags side-to-side ×4 ──────────────────────────
    # Distinctive: pointing hand at mid level, high-freq horizontal oscillation
    elif label == "No":
        for i, ti in enumerate(t_s):
            rx = n(0.62 + 0.09 * np.sin(ti * 4 * np.pi), 0.009)
            ry = n(0.45, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, 0.0, -1.0),
                           "pose": pose})

    # ── HELP: BOTH hands rise from low to high (bilateral upward sweep) ──────
    # Distinctive: two open hands, starts at y~0.78, ends at y~0.30
    elif label == "Help":
        for i, ti in enumerate(t_s):
            rx = n(0.65, 0.010)
            ry = n(0.78 - 0.48 * ti, 0.010)
            lx = n(0.35, 0.010)
            ly = n(0.78 - 0.48 * ti, 0.010)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_open(lx, ly, 0.045),
                           "right_hand": _make_hand_open(rx, ry, 0.045),
                           "pose": pose})

    # ── WATER: W-handshape at chin, small oscillation ───────────────────────
    # Distinctive: pointing hand near chin (y~0.42), small circular motion
    elif label == "Water":
        for i, ti in enumerate(t_s):
            rx = n(0.60 + 0.015 * np.sin(ti * 4 * np.pi), 0.009)
            ry = n(0.42 + 0.018 * np.sin(ti * 4 * np.pi), 0.008)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, 0.3, -0.5),
                           "pose": pose})

    # ── FOOD: bunched fingers to mouth, repeated taps ────────────────────────
    # Distinctive: closed fist near mouth (y~0.38), vertical oscillation
    elif label == "Food":
        for i, ti in enumerate(t_s):
            rx = n(0.60, 0.008)
            ry = n(0.38 + 0.04 * np.sin(ti * 3 * np.pi), 0.008)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── MOTHER: right open hand at chin level, static ───────────────────────
    # Distinctive: spread fingers at chin (y~0.42), very slow downward drift
    elif label == "Mother":
        for i, ti in enumerate(t_s):
            rx = n(0.60, 0.008)
            ry = n(0.42 + 0.015 * ti, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.055),
                           "pose": pose})

    # ── FATHER: right open hand at forehead (y~0.26) ────────────────────────
    # Distinctive: spread fingers at forehead, slightly higher than Hello
    elif label == "Father":
        for i, ti in enumerate(t_s):
            rx = n(0.62, 0.008)
            ry = n(0.26 - 0.012 * ti, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.055),
                           "pose": pose})

    # ── BROTHER: right fist from forehead arcs down to chest ─────────────────
    # Distinctive: closed fist, long vertical sweep downward
    elif label == "Brother":
        for i, ti in enumerate(t_s):
            rx = n(0.62, 0.008)
            ry = n(0.26 + 0.40 * ti, 0.009)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── SISTER: right fist from chin arcs to chest ───────────────────────────
    # Distinctive: closed fist, shorter sweep starting from chin (y~0.42)
    elif label == "Sister":
        for i, ti in enumerate(t_s):
            rx = n(0.62, 0.008)
            ry = n(0.42 + 0.25 * ti, 0.009)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── FRIEND: both index fingers hook together at center ───────────────────
    # Distinctive: two pointing hands approaching center, bilateral
    elif label == "Friend":
        for i, ti in enumerate(t_s):
            rx = n(0.55 + 0.025 * np.sin(ti * np.pi), 0.009)
            ry = n(0.55, 0.007)
            lx = n(0.45 - 0.025 * np.sin(ti * np.pi), 0.009)
            ly = n(0.55, 0.007)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_pointing(lx, ly, 1.0, 0.0),
                           "right_hand": _make_hand_pointing(rx, ry, -1.0, 0.0),
                           "pose": pose})

    # ── SCHOOL: both open hands clap twice ──────────────────────────────────
    # Distinctive: both open hands converge/diverge horizontally
    elif label == "School":
        for i, ti in enumerate(t_s):
            offset = 0.07 * abs(np.sin(ti * 2 * np.pi))
            rx = n(0.53 + offset, 0.009)
            lx = n(0.47 - offset, 0.009)
            ry = n(0.55, 0.007)
            ly = n(0.55, 0.007)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_open(lx, ly, 0.04),
                           "right_hand": _make_hand_open(rx, ry, 0.04),
                           "pose": pose})

    # ── TEACHER: right fingers at forehead, push forward ────────────────────
    # Distinctive: open hand at forehead, moves forward (x increases)
    elif label == "Teacher":
        for i, ti in enumerate(t_s):
            rx = n(0.62 + 0.10 * ti, 0.009)
            ry = n(0.28, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.045),
                           "pose": pose})

    # ── HOSPITAL: H-shape drawn on upper arm (small circle on left arm) ──────
    # Distinctive: pointing hand traces circle at left arm region (x~0.38)
    elif label == "Hospital":
        for i, ti in enumerate(t_s):
            rx = n(0.38 + 0.045 * np.sin(ti * np.pi * 2), 0.008)
            ry = n(0.52 + 0.045 * np.cos(ti * np.pi * 2), 0.008)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, 0.0, -1.0),
                           "pose": pose})

    # ── DOCTOR: D-handshape tap on wrist (right taps left wrist) ────────────
    # Distinctive: pointing hand at wrist region (y~0.65), small vertical taps
    elif label == "Doctor":
        for i, ti in enumerate(t_s):
            rx = n(0.45 + 0.025 * np.sin(ti * 3 * np.pi), 0.008)
            ry = n(0.65 + 0.025 * np.sin(ti * 3 * np.pi), 0.008)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, -0.3, -0.3),
                           "pose": pose})

    # ── EMERGENCY: both fists raised high, shaken rapidly ───────────────────
    # Distinctive: bilateral closed fists at forehead (y~0.28), high-freq shake
    elif label == "Emergency":
        for i, ti in enumerate(t_s):
            shake = 0.05 * np.sin(ti * 6 * np.pi)
            rx = n(0.65 + shake, 0.009)
            ry = n(0.28, 0.007)
            lx = n(0.35 - shake, 0.009)
            ly = n(0.28, 0.007)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_closed(lx, ly),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── PAIN: both index fingers jab toward each other at chest ─────────────
    # Distinctive: bilateral pointing hands, oscillate toward center (x~0.50)
    elif label == "Pain":
        for i, ti in enumerate(t_s):
            rx = n(0.55 - 0.04 * np.sin(ti * 2 * np.pi), 0.009)
            ry = n(0.60, 0.007)
            lx = n(0.45 + 0.04 * np.sin(ti * 2 * np.pi), 0.009)
            ly = n(0.60, 0.007)
            pose = _base_pose(rx, ry, lx, ly)
            frames.append({"left_hand": _make_hand_pointing(lx, ly, 1.0, 0.0),
                           "right_hand": _make_hand_pointing(rx, ry, -1.0, 0.0),
                           "pose": pose})

    # ── MEDICINE: right pointing hand rotates over open left palm ────────────
    elif label == "Medicine":
        for i, ti in enumerate(t_s):
            rx = n(0.50 + 0.025 * np.sin(ti * 4 * np.pi), 0.009)
            ry = n(0.60, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_hand_open(0.45, 0.60, 0.03),
                           "right_hand": _make_hand_pointing(rx, ry, 0.0, 1.0),
                           "pose": pose})

    # ── BATHROOM: B-hand twists at wrist, high-freq rotation ─────────────────
    # Distinctive: open hand at chest-right (x~0.62), rapid small oscillation
    elif label == "Bathroom":
        for i, ti in enumerate(t_s):
            rx = n(0.62 + 0.025 * np.sin(ti * 6 * np.pi), 0.008)
            ry = n(0.52, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.022),
                           "pose": pose})

    # ── HOME: fingertips from cheek (y~0.35) down to chin (y~0.47) ──────────
    # Distinctive: closed hand slow downward slide along face
    elif label == "Home":
        for i, ti in enumerate(t_s):
            rx = n(0.62, 0.007)
            ry = n(0.35 + 0.15 * ti, 0.008)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── EAT: bunched fingers repeated to mouth ───────────────────────────────
    # Distinctive: closed near mouth (y~0.37), fast vertical oscillation ×4
    elif label == "Eat":
        for i, ti in enumerate(t_s):
            rx = n(0.60, 0.008)
            ry = n(0.37 + 0.025 * np.sin(ti * 4 * np.pi), 0.008)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_closed(rx, ry),
                           "pose": pose})

    # ── DRINK: C-handshape tips toward mouth (upward tilt) ───────────────────
    # Distinctive: partially-open hand moves upward toward mouth
    elif label == "Drink":
        for i, ti in enumerate(t_s):
            rx = n(0.62, 0.007)
            ry = n(0.45 - 0.08 * ti, 0.008)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.027),
                           "pose": pose})

    # ── SLEEP: right palm rests on cheek, slow stationary ───────────────────
    # Distinctive: open hand near cheek (y~0.35, x~0.66), nearly static
    elif label == "Sleep":
        for i, ti in enumerate(t_s):
            rx = n(0.66 - 0.015 * ti, 0.007)
            ry = n(0.35 + 0.015 * ti, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.05),
                           "pose": pose})

    # ── STOP: right palm push forward (large x sweep) ───────────────────────
    # Distinctive: open hand, fast rightward push (large x displacement)
    elif label == "Stop":
        for i, ti in enumerate(t_s):
            rx = n(0.62 + 0.15 * ti, 0.009)
            ry = n(0.55, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.055),
                           "pose": pose})

    # ── COME: right index beckons (moves inward, left x direction) ───────────
    # Distinctive: pointing hand moves leftward toward signer
    elif label == "Come":
        for i, ti in enumerate(t_s):
            rx = n(0.72 - 0.10 * ti, 0.009)
            ry = n(0.55 + 0.025 * np.sin(ti * 2 * np.pi), 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, -1.0, 0.0),
                           "pose": pose})

    # ── GO: right index points and moves outward (rightward) ─────────────────
    # Distinctive: pointing hand moves rightward away from signer
    elif label == "Go":
        for i, ti in enumerate(t_s):
            rx = n(0.58 + 0.14 * ti, 0.009)
            ry = n(0.52, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_pointing(rx, ry, 1.0, 0.0),
                           "pose": pose})

    # ── GOOD: right open hand at chin, arcs forward ──────────────────────────
    # Distinctive: open hand at chin (y~0.40), forward+down arc
    elif label == "Good":
        for i, ti in enumerate(t_s):
            rx = n(0.60 + 0.07 * ti, 0.009)
            ry = n(0.40 + 0.03 * ti, 0.007)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.05),
                           "pose": pose})

    # ── BAD: right open hand at chin, flips downward steeply ─────────────────
    # Distinctive: open hand at chin, steep downward flip
    elif label == "Bad":
        for i, ti in enumerate(t_s):
            rx = n(0.60 + 0.03 * ti, 0.009)
            ry = n(0.40 + 0.22 * ti, 0.009)
            pose = _base_pose(rx, ry)
            frames.append({"left_hand": _make_empty_hand(),
                           "right_hand": _make_hand_open(rx, ry, 0.05),
                           "pose": pose})

    # ── PLEASE: right open hand circles on chest ─────────────────────────────
    # Distinctive: open hand traces circle at chest level
    elif label == "Please":
        for i, ti in enumerate(t_s):
            rx = n(0.60 + 0.055 * np.cos(ti * 2 * np.pi), 0.008)
            ry = n(0.55 + 0.055 * np.sin(ti * 2 * np.pi), 0.008)
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


# ─────────────────────────────────────────────────────────────────────────────
# "No Sign Detected" — idle / resting-state template generator
# Four idle archetypes, cycled by sample_idx:
#   0: no hands visible at all
#   1: right hand stationary at waist level (small jitter only)
#   2: both hands stationary at sides
#   3: right hand random-walk drift (non-intentional micro-movement)
# ─────────────────────────────────────────────────────────────────────────────

def _gen_no_sign_frames(n_frames: int = 30, sample_idx: int = 0) -> list:
    """Generates idle / no-gesture frames for the 'No Sign Detected' training class."""
    np.random.seed(50000 + sample_idx)
    archetype = sample_idx % 4
    frames = []

    if archetype == 0:
        # No hands in frame at all
        for _ in range(n_frames):
            frames.append({
                "left_hand":  _make_empty_hand(),
                "right_hand": _make_empty_hand(),
                "pose":       _base_pose(0.72, 0.88, 0.28, 0.88),
            })

    elif archetype == 1:
        # Right hand stationary at resting waist position — tiny jitter
        rx0 = 0.65 + np.random.uniform(-0.06, 0.06)
        ry0 = 0.83 + np.random.uniform(-0.04, 0.04)
        for _ in range(n_frames):
            rx = rx0 + np.random.normal(0, 0.004)
            ry = ry0 + np.random.normal(0, 0.003)
            frames.append({
                "left_hand":  _make_empty_hand(),
                "right_hand": _make_hand_open(rx, ry, 0.032),
                "pose":       _base_pose(rx, ry),
            })

    elif archetype == 2:
        # Both hands stationary at sides
        rx0 = 0.64 + np.random.uniform(-0.05, 0.05)
        lx0 = 0.36 + np.random.uniform(-0.05, 0.05)
        ry0 = 0.83 + np.random.uniform(-0.03, 0.03)
        ly0 = ry0 + np.random.uniform(-0.02, 0.02)
        for _ in range(n_frames):
            rx = rx0 + np.random.normal(0, 0.004)
            lx = lx0 + np.random.normal(0, 0.004)
            ry = ry0 + np.random.normal(0, 0.003)
            ly = ly0 + np.random.normal(0, 0.003)
            frames.append({
                "left_hand":  _make_hand_closed(lx, ly),
                "right_hand": _make_hand_closed(rx, ry),
                "pose":       _base_pose(rx, ry, lx, ly),
            })

    else:
        # Right hand random-walk micro-drift — no systematic gesture trajectory
        rx, ry = 0.60 + np.random.uniform(-0.05, 0.05), 0.72 + np.random.uniform(-0.05, 0.05)
        for _ in range(n_frames):
            rx = float(np.clip(rx + np.random.normal(0, 0.007), 0.30, 0.80))
            ry = float(np.clip(ry + np.random.normal(0, 0.006), 0.50, 0.92))
            frames.append({
                "left_hand":  _make_empty_hand(),
                "right_hand": _make_hand_open(rx, ry, 0.030),
                "pose":       _base_pose(rx, ry),
            })

    return frames


# ─────────────────────────────────────────────────────────────────────────────
# Dataset versioning — bump this when synthetic templates change significantly.
# On startup, if the stored version doesn't match, the dataset is regenerated.
# ─────────────────────────────────────────────────────────────────────────────
DATASET_VERSION = "3.0"


def initialize_default_dataset_if_empty():
    """
    Checks if DatasetSamples table is empty OR if the stored dataset version is
    older than DATASET_VERSION.  If so, clears stale synthetic samples and
    generates a fresh ISL dataset, then trains the RandomForest classifier.

    Dataset v3 improvements over v2:
      - 'No Sign Detected' class added (40 idle samples, 4 archetypes)
      - Scale-normalised pose features (shoulder-width normalisation)
      - Acceleration feature layer added to sequence representation
      - Hand-presence ratio appended to each feature vector
      - Same 20 samples × 30 frames per ISL sign
    """
    from modules.signs.recognizer import VOCABULARY, NO_SIGN_LABEL

    samples = get_recorded_samples()
    stored_version = get_setting("dataset_version", "0.0")
    needs_regen = (len(samples) == 0) or (stored_version != DATASET_VERSION)

    if not needs_regen:
        return

    if len(samples) > 0 and stored_version != DATASET_VERSION:
        print(f"[Dataset] Upgrading v{stored_version} -> v{DATASET_VERSION}. "
              f"Clearing old synthetic samples...")
        for s in get_recorded_samples():
            if s.get("person_id") == "Synthetic":
                delete_sign_sample(s["id"])

    print(f"[Dataset] Generating v{DATASET_VERSION} ISL dataset with '{NO_SIGN_LABEL}' class...")

    class MockSeqBuf:
        def __init__(self, buf):
            self.buffer = buf

    SAMPLES_PER_SIGN = 20
    N_FRAMES = 30
    NO_SIGN_SAMPLES = 40   # more idle samples — harder negative class

    # ── ISL signs ─────────────────────────────────────────────────────────────
    for label in VOCABULARY:
        for sample_idx in range(SAMPLES_PER_SIGN):
            frames = _gen_sign_frames(label, n_frames=N_FRAMES, sample_idx=sample_idx)
            record_sign_sample("Synthetic", label, MockSeqBuf(frames))

    # ── No Sign Detected class ────────────────────────────────────────────────
    for sample_idx in range(NO_SIGN_SAMPLES):
        frames = _gen_no_sign_frames(n_frames=N_FRAMES, sample_idx=sample_idx)
        record_sign_sample("Synthetic", NO_SIGN_LABEL, MockSeqBuf(frames))

    save_setting("dataset_version", DATASET_VERSION)
    total_isl  = len(VOCABULARY) * SAMPLES_PER_SIGN
    total_idle = NO_SIGN_SAMPLES
    print(f"[Dataset] Generated {total_isl} ISL + {total_idle} idle samples. Training…")
    retrain_sign_model()
    print("[Dataset] Model training complete.")
