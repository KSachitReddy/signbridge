"""
face_ai.py — Face recognition using MediaPipe Tasks API FaceLandmarker.

Uses face_landmarker.task model (auto-downloaded on first run).
Extracts 50-key-point normalized embedding for face identification.
Stores embeddings in SQLite via cosine similarity matching.
"""
import cv2
import numpy as np
import os
import secrets
from modules.database import add_face_vector, get_all_face_vectors, save_person, update_last_seen

_face_landmarker = None
_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")
)

# Key face landmark indices (from 478-point FaceLandmarker mesh)
# Captures eye corners, nose tip, mouth, chin, cheekbones, forehead
FACE_KEY_INDICES = [
    1, 2, 4, 5, 6,           # nose bridge + tip
    33, 133, 159, 145, 362, 263, 386, 374,  # eye corners + lids
    61, 291, 13, 14, 0, 17,  # mouth
    152, 234, 454, 10,        # chin + jaw corners + forehead top
    54, 103, 332, 284,        # cheekbones
    70, 63, 105, 66, 107,     # left brow
    300, 293, 334, 296, 336,  # right brow
    168, 6, 197, 195, 5,      # nose bridge detail
]
# Deduplicate while preserving order
_seen = set()
FACE_KEY_INDICES = [i for i in FACE_KEY_INDICES if not (i in _seen or _seen.add(i))][:50]

EXPRESSIONS = ["Neutral", "Happy", "Focused", "Surprised", "Tired", "Attentive"]


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
            print("[FaceLandmarker] Downloading model...")
            urllib.request.urlretrieve(url, _MODEL_PATH)
            print("[FaceLandmarker] Model downloaded.")
        except Exception as e:
            print(f"[FaceLandmarker] Download failed: {e}")
            return False
    return True


def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is not None:
        return _face_landmarker

    if not _ensure_model():
        _face_landmarker = "FAILED"
        return _face_landmarker

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=4,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        _face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print("[FaceLandmarker] Initialized.")
    except Exception as e:
        print(f"[FaceLandmarker] Init error: {e}")
        _face_landmarker = "FAILED"

    return _face_landmarker


def extract_face_embedding(face_landmarks, img_w, img_h):
    """
    Extracts a compact 100-dim normalized embedding from FaceLandmarker result.
    Uses 50 key landmark points (x, y only).
    Normalization: centroid subtraction + max-extent scaling → translation+scale invariant.
    """
    pts = []
    for idx in FACE_KEY_INDICES:
        if idx < len(face_landmarks):
            lm = face_landmarks[idx]
            pts.append([lm.x, lm.y])

    if len(pts) < 10:
        return None

    pts = np.array(pts, dtype=np.float32)
    centroid = pts.mean(axis=0)
    pts -= centroid
    extent = np.max(np.abs(pts))
    if extent > 0:
        pts /= extent

    return pts.flatten().tolist()


def cosine_similarity(v1, v2):
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na > 0 and nb > 0:
        return float(np.dot(a, b) / (na * nb))
    return 0.0


def check_lighting(gray_region):
    mean = float(np.mean(gray_region))
    std = float(np.std(gray_region))
    if mean < 35:
        return False, "Too dark — improve lighting"
    if mean > 235:
        return False, "Overexposed — reduce brightness"
    if std < 8:
        return False, "Low contrast — face may be blurry"
    return True, "Good"


def _face_bbox(face_landmarks, img_w, img_h):
    """Returns (x1, y1, x2, y2) bounding box in pixel coords."""
    xs = [lm.x * img_w for lm in face_landmarks]
    ys = [lm.y * img_h for lm in face_landmarks]
    x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
    x2, y2 = min(img_w, int(max(xs))), min(img_h, int(max(ys)))
    return x1, y1, x2, y2


def validate_and_enroll_face(frame, name, notes=""):
    """
    Validates and enrolls a face from the given frame.
    Returns dict with 'status' ('success' or 'error') and message.
    """
    landmarker = _get_face_landmarker()
    if landmarker is None or landmarker == "FAILED":
        return {"status": "error", "message": "Face Landmarker model unavailable."}

    import mediapipe as mp
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {"status": "error", "message": "No face detected in frame."}

    face_lm = result.face_landmarks[0]
    x1, y1, x2, y2 = _face_bbox(face_lm, w, h)
    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return {"status": "error", "message": "Invalid face region."}

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    light_ok, light_msg = check_lighting(gray)
    if not light_ok:
        return {"status": "error", "message": f"Quality check failed: {light_msg}"}

    embedding = extract_face_embedding(face_lm, w, h)
    if embedding is None:
        return {"status": "error", "message": "Could not extract face landmarks."}

    # Duplicate check
    ENROLL_THRESHOLD = 0.92
    existing = get_all_face_vectors()
    from modules.database import get_all_people
    people_map = {p["id"]: p["name"] for p in get_all_people()}
    for ev in existing:
        sim = cosine_similarity(embedding, ev["embedding"])
        if sim > ENROLL_THRESHOLD:
            existing_name = people_map.get(ev["person_id"], "Unknown")
            return {
                "status": "error",
                "message": f"This face is already enrolled as '{existing_name}'."
            }

    # Save
    person_id = f"P_{secrets.token_hex(6)}"
    faces_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "database", "faces")
    )
    os.makedirs(faces_dir, exist_ok=True)
    image_path = os.path.join(faces_dir, f"{person_id}.jpg")
    cv2.imwrite(image_path, face_crop)

    save_person(person_id, name, notes)
    add_face_vector(person_id, image_path, embedding)

    return {
        "status": "success",
        "person_id": person_id,
        "name": name,
        "message": f"Face profile for '{name}' enrolled successfully!"
    }


def recognize_multiple_faces(frame):
    """
    Detects and identifies all faces using MediaPipe FaceLandmarker.
    Draws bounding boxes + name labels on frame.
    Returns: (list of result dicts, annotated frame)
    """
    landmarker = _get_face_landmarker()
    results_list = []

    if landmarker is None or landmarker == "FAILED":
        return results_list, frame

    import mediapipe as mp
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    mp_result = landmarker.detect(mp_image)

    if not mp_result.face_landmarks:
        return results_list, frame

    existing_vectors = get_all_face_vectors()
    from modules.database import get_all_people
    people_map = {p["id"]: p["name"] for p in get_all_people()}

    MATCH_THRESHOLD = 0.88

    for i, face_lm in enumerate(mp_result.face_landmarks):
        x1, y1, x2, y2 = _face_bbox(face_lm, w, h)
        bw, bh = x2 - x1, y2 - y1

        if bw < 10 or bh < 10:
            continue

        embedding = extract_face_embedding(face_lm, w, h)
        if embedding is None:
            continue

        # Match against enrolled faces
        best_name = "Unknown"
        best_id = "Unknown"
        best_sim = 0.0

        for ev in existing_vectors:
            sim = cosine_similarity(embedding, ev["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = ev["person_id"]

        if best_sim >= MATCH_THRESHOLD:
            best_name = people_map.get(best_id, "Unknown")
            update_last_seen(best_id)
        else:
            best_id = "Unknown"
            best_sim = 0.0

        # Expression heuristic from mouth geometry
        try:
            upper_lip = face_lm[13]  # upper inner lip
            lower_lip = face_lm[14]  # lower inner lip
            mouth_open = abs(upper_lip.y - lower_lip.y) * h
            left_corner = face_lm[61]
            right_corner = face_lm[291]
            mouth_width = abs(right_corner.x - left_corner.x) * w

            if mouth_open > 12:
                expression = "Surprised"
            elif bw > 0 and (mouth_width / bw) > 0.70:
                expression = "Happy"
            else:
                expression = "Neutral"
        except Exception:
            expression = "Neutral"

        # Draw bounding box
        color = (0, 220, 160) if best_name != "Unknown" else (0, 80, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"{best_name} ({int(best_sim * 100)}%)" if best_name != "Unknown" \
                else f"Unknown {chr(65 + i)}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        results_list.append({
            "person_id": best_id,
            "name": best_name,
            "confidence": float(best_sim),
            "expression": expression,
            "box": [x1, y1, bw, bh],
        })

    return results_list, frame
