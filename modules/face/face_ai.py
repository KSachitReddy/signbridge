"""
face_ai.py — Face recognition and face mesh rendering.

Uses InsightFace buffalo_sc model (MobileFaceNet) for 512-dim face recognition.
Uses MediaPipe Tasks API FaceLandmarker for detailed 468-point face mesh rendering.
Supports multi-sample enrollment and head orientation estimation.
"""
import cv2
import numpy as np
import os
import secrets
import json
from modules.database import add_face_vector, get_all_face_vectors, save_person, update_last_seen

_face_landmarker = None
_insightface_app = None

# InsightFace buffalo_sc outputs 512-dim embeddings.
# Any DB vector with a different length is a legacy artifact and must be skipped.
_EXPECTED_EMBEDDING_DIM = 512

_MEDIAPIPE_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "face_landmarker.task")
)


def _ensure_mediapipe_model():
    if not os.path.exists(_MEDIAPIPE_MODEL_PATH):
        try:
            import urllib.request
            os.makedirs(os.path.dirname(_MEDIAPIPE_MODEL_PATH), exist_ok=True)
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
            print("[FaceLandmarker] Downloading model...")
            urllib.request.urlretrieve(url, _MEDIAPIPE_MODEL_PATH)
            print("[FaceLandmarker] Model downloaded.")
        except Exception as e:
            print(f"[FaceLandmarker] Download failed: {e}")
            return False
    return True


def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is not None:
        return _face_landmarker

    if not _ensure_mediapipe_model():
        _face_landmarker = "FAILED"
        return _face_landmarker

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=_MEDIAPIPE_MODEL_PATH)
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


# Current det_size used by InsightFace — can be changed via reinit_insightface()
_insightface_det_size = (640, 640)


def _get_insightface_app():
    global _insightface_app
    if _insightface_app is not None:
        return _insightface_app

    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name='buffalo_sc')
        app.prepare(ctx_id=-1, det_size=_insightface_det_size)
        _insightface_app = app
        print(f"[InsightFace] Initialized buffalo_sc @ det_size={_insightface_det_size}.")
    except Exception as e:
        print(f"[InsightFace] Init error: {e}")
        _insightface_app = "FAILED"

    return _insightface_app


def reinit_insightface(det_size: tuple = (640, 640)):
    """Re-prepare InsightFace with a new det_size (called on performance mode change)."""
    global _insightface_app, _insightface_det_size
    if det_size == _insightface_det_size and _insightface_app is not None and _insightface_app != "FAILED":
        return  # nothing to do
    _insightface_det_size = det_size
    _insightface_app = None  # force re-init
    _get_insightface_app()


def extract_face_embedding(face_landmarks, img_w, img_h):
    """
    Compatibility shim. Returns a mock vector if frame is not available.
    In production, embeddings should be extracted from frames using InsightFace.
    """
    return [0.0] * 512


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
        return False, "Too dark - improve lighting"
    if mean > 235:
        return False, "Too bright - reduce brightness"
    if std < 8:
        return False, "Low contrast - face may be blurry"
    return True, "Good lighting"


def estimate_face_orientation(landmarks):
    """
    Estimates 3D head orientation (Front, Left, Right, Up, Down) using FaceMesh landmarks.
    Landmarks correspond to:
      4: Nose tip
      234: Left cheek outer boundary
      454: Right cheek outer boundary
      10: Forehead top
      152: Chin bottom
    """
    if len(landmarks) < 455:
        return "Front"

    p_nose = np.array([landmarks[4].x, landmarks[4].y])
    p_left = np.array([landmarks[234].x, landmarks[234].y])
    p_right = np.array([landmarks[454].x, landmarks[454].y])
    p_top = np.array([landmarks[10].x, landmarks[10].y])
    p_bottom = np.array([landmarks[152].x, landmarks[152].y])

    d_left = np.linalg.norm(p_nose - p_left)
    d_right = np.linalg.norm(p_nose - p_right)
    d_top = np.linalg.norm(p_nose - p_top)
    d_bottom = np.linalg.norm(p_nose - p_bottom)

    if d_right == 0 or d_bottom == 0:
        return "Front"

    r_horiz = d_left / d_right
    r_vert = d_top / d_bottom

    if r_horiz > 1.35:
        return "Left"
    elif r_horiz < 0.73:
        return "Right"
    elif r_vert > 1.25:
        return "Down"
    elif r_vert < 0.62:
        return "Up"
    else:
        return "Front"


def _draw_detailed_face_mesh(frame, face_landmarks, color=(0, 255, 200)):
    """Draws 468 landmarks as small dots and lines connecting primary contours."""
    h, w = frame.shape[:2]
    
    # 1. Draw all 468 points as tiny dots
    for lm in face_landmarks[:468]:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 1, color, -1)

    # 2. Draw connections for contours to construct mesh visual
    contours = [
        [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33],  # Left Eye
        [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 263],  # Right Eye
        [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 95, 61],  # Outer Lips
        [70, 63, 105, 66, 107],  # Left Eyebrow
        [300, 293, 334, 296, 336],  # Right Eyebrow
        [168, 6, 197, 195, 5, 4],  # Nose details
        [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]  # Jawline
    ]

    for c in contours:
        pts = []
        for idx in c:
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                pts.append([int(lm.x * w), int(lm.y * h)])
        if len(pts) > 1:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], isClosed=False, color=color, thickness=1)


def validate_and_enroll_face(frame, name, notes=""):
    """
    Validates a face from the frame. If valid, extracts its 512-dim embedding.
    Does NOT write directly to DB if called inside the multi-sample flow.
    If called standalone (e.g., in test scripts), saves the profile immediately.
    """
    app = _get_insightface_app()
    if app is None or app == "FAILED":
        return {"status": "error", "message": "InsightFace FaceAnalysis unavailable."}

    h, w = frame.shape[:2]
    faces = app.get(frame)

    if not faces:
        return {"status": "error", "message": "No face detected in frame."}

    # Use first detected face
    face = faces[0]
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_crop = frame[y1:y2, x1:x2]

    if face_crop.size == 0:
        return {"status": "error", "message": "Invalid face region boundaries."}

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    light_ok, light_msg = check_lighting(gray)
    if not light_ok:
        return {"status": "error", "message": f"Quality check failed: {light_msg}"}

    embedding = face.embedding.tolist()

    # Standalone direct enrollment fallback (if called from tests)
    if name and not name.startswith("[VALIDATE_ONLY]"):
        ENROLL_THRESHOLD = 0.85
        existing = get_all_face_vectors()
        from modules.database import get_all_people
        people_map = {p["id"]: p["name"] for p in get_all_people()}
        for ev in existing:
            ev_emb = ev["embedding"]
            if len(ev_emb) != _EXPECTED_EMBEDDING_DIM:
                print(f"[FaceAI] Skipping legacy vector (dim={len(ev_emb)}) for person {ev['person_id']}")
                continue
            sim = cosine_similarity(embedding, ev_emb)
            if sim > ENROLL_THRESHOLD:
                existing_name = people_map.get(ev["person_id"], "Unknown")
                return {
                    "status": "error",
                    "message": f"This face is already enrolled as '{existing_name}'."
                }

        person_id = f"P_{secrets.token_hex(6)}"
        faces_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "database", "faces"))
        os.makedirs(faces_dir, exist_ok=True)
        image_path = os.path.join(faces_dir, f"{person_id}.jpg")
        cv2.imwrite(image_path, face_crop)

        save_person(person_id, name, notes)
        add_face_vector(person_id, image_path, embedding)

        try:
            from modules.perf.db_cache import invalidate_face_cache
            invalidate_face_cache()
        except Exception:
            pass

        return {
            "status": "success",
            "person_id": person_id,
            "name": name,
            "message": f"Face profile for '{name}' enrolled successfully!"
        }

    return {
        "status": "success",
        "embedding": embedding,
        "message": "Face quality validation passed."
    }


def recognize_multiple_faces(frame, run_mesh: bool = True):
    """
    Detects and identifies all faces using InsightFace.
    Optionally renders 468-point Face Mesh via MediaPipe FaceLandmarker.

    Args:
        frame:    BGR image (numpy array)
        run_mesh: If True, run MediaPipe FaceLandmarker for detailed mesh overlay.
                  Set False in Performance/Balanced modes for significant speedup.
    Returns: (list of result dicts, annotated frame)
    """
    h, w = frame.shape[:2]
    results_list = []
    mp_centroids = []

    # 1. Detailed Face Mesh Overlay (MediaPipe Landmarker) — optional for performance
    if run_mesh:
        landmarker = _get_face_landmarker()
        if landmarker is not None and landmarker != "FAILED":
            try:
                import mediapipe as mp
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                mp_result = landmarker.detect(mp_image)
                if mp_result.face_landmarks:
                    for face_lm in mp_result.face_landmarks:
                        _draw_detailed_face_mesh(frame, face_lm, color=(0, 255, 200))
                        xs = [lm.x * w for lm in face_lm]
                        ys = [lm.y * h for lm in face_lm]
                        centroid = (sum(xs)/len(xs), sum(ys)/len(ys))
                        mp_centroids.append((centroid, face_lm))
            except Exception as e:
                print(f"[FaceLandmarker] Mesh rendering error: {e}")

    # 2. Face Recognition (InsightFace w600k_mbf)
    app = _get_insightface_app()
    if app is None or app == "FAILED":
        return results_list, frame

    faces = app.get(frame)
    if not faces:
        return results_list, frame

    from modules.perf.db_cache import get_cached_face_vectors, get_cached_people
    existing_vectors = get_cached_face_vectors()
    people_map = {p["id"]: p["name"] for p in get_cached_people()}

    # Filter out any legacy vectors whose embedding dimension doesn't match InsightFace output
    compatible_vectors = []
    for ev in existing_vectors:
        ev_emb = ev["embedding"]
        if len(ev_emb) != _EXPECTED_EMBEDDING_DIM:
            print(f"[FaceAI] Skipping legacy vector (dim={len(ev_emb)}) for person {ev['person_id']} — run purge_legacy_face_vectors() to clean up.")
        else:
            compatible_vectors.append(ev)

    # Multi-threshold settings
    THRESHOLD_KNOWN = 0.68

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1

        if bw < 10 or bh < 10:
            continue

        embedding = face.embedding.tolist()

        # Spatial matching with MediaPipe centroids to get head orientation
        orientation = "Front"
        for (cx, cy), lm in mp_centroids:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                orientation = estimate_face_orientation(lm)
                break

        best_name = "Unknown Person"
        best_id = "Unknown"
        best_sim = 0.0

        # Pre-normalize query vector once; use dot product against pre-normalized DB vectors
        q_arr = np.array(embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr))
        if q_norm > 1e-8:
            q_arr /= q_norm

        for ev in compatible_vectors:
            # _norm_emb was pre-computed at cache-load time (unit vectors)
            ref = ev.get("_norm_emb") or ev["embedding"]
            sim = float(np.dot(q_arr, np.array(ref, dtype=np.float32)))
            if sim > best_sim:
                best_sim = sim
                best_id = ev["person_id"]

        expression = "Neutral"

        # Decision making logic: below threshold remains Unknown.
        if best_sim >= THRESHOLD_KNOWN:
            best_name = people_map.get(best_id, "Unknown Person")
            update_last_seen(best_id)
            label = f"{best_name} ({int(best_sim * 100)}%)"
            box_color = (0, 220, 160) # Green
        else:
            best_id = "Unknown"
            best_name = "Unknown Person"
            label = "Unknown Person"
            box_color = (0, 50, 255) # Red

        # Draw bounding boxes and text overlays
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        results_list.append({
            "person_id": best_id,
            "name": best_name,
            "confidence": float(best_sim),
            "expression": expression,
            "box": [x1, y1, bw, bh],
            "embedding": embedding,
            "orientation": orientation
        })

    return results_list, frame
