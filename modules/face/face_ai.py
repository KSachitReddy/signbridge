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

    if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("SPACE_ID"):
        print("[FaceLandmarker] Skip FaceLandmarker in Streamlit Cloud/Spaces environment.")
        _face_landmarker = "FAILED"
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


def reinit_insightface_async(det_size: tuple = (640, 640)):
    """Non-blocking variant — runs reinit_insightface in a daemon thread."""
    import threading
    threading.Thread(
        target=reinit_insightface, args=(det_size,),
        daemon=True, name="InsightFaceReinit"
    ).start()


def preload_models():
    """
    Trigger lazy-init for InsightFace, FaceLandmarker, PoseLandmarker, HandLandmarker.
    Intended to be called once at app startup in a background thread so that the
    first camera frame does not stall on a cold model load (typically 2–5 seconds).
    """
    import threading

    def _load():
        _get_insightface_app()
        _get_face_landmarker()
        try:
            from modules.pose.holistic import _get_pose_landmarker
            _get_pose_landmarker()
        except Exception as exc:
            print(f"[Preload] pose landmarker: {exc}")
        try:
            from modules.hands.landmarks import _get_hand_landmarker
            _get_hand_landmarker()
        except Exception as exc:
            print(f"[Preload] hand landmarker: {exc}")

    threading.Thread(target=_load, daemon=True, name="SignBridgePreload").start()


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


def _draw_detailed_face_mesh(frame, face_landmarks, color=(0, 255, 200), show_mesh: bool = True, show_ids: bool = False):
    """Draws 468 landmarks as small dots and lines connecting primary contours."""
    if not show_mesh and not show_ids:
        return

    h, w = frame.shape[:2]
    
    # 1. Draw points as tiny dots if show_mesh is enabled
    if show_mesh:
        for lm in face_landmarks[:468]:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, color, -1)

    # Draw numerical ID labels next to each landmark coordinate if show_ids is enabled
    if show_ids:
        for idx, lm in enumerate(face_landmarks[:468]):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.22, (0, 255, 200), 1, cv2.LINE_AA)

    # 2. Draw connections for the entire face mesh if show_mesh is enabled
    if show_mesh:
        from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

        # Draw face tesselation
        for conn in FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION:
            if conn.start < len(face_landmarks) and conn.end < len(face_landmarks):
                pt1 = (int(face_landmarks[conn.start].x * w), int(face_landmarks[conn.start].y * h))
                pt2 = (int(face_landmarks[conn.end].x * w), int(face_landmarks[conn.end].y * h))
                cv2.line(frame, pt1, pt2, color, 1)

        # Draw face contours (jawline, lips, nose, eyes, eyebrows)
        for conn in FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS:
            if conn.start < len(face_landmarks) and conn.end < len(face_landmarks):
                pt1 = (int(face_landmarks[conn.start].x * w), int(face_landmarks[conn.start].y * h))
                pt2 = (int(face_landmarks[conn.end].x * w), int(face_landmarks[conn.end].y * h))
                cv2.line(frame, pt1, pt2, color, 1)

        # Draw irises
        for conn in list(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS) + list(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS):
            if conn.start < len(face_landmarks) and conn.end < len(face_landmarks):
                pt1 = (int(face_landmarks[conn.start].x * w), int(face_landmarks[conn.start].y * h))
                pt2 = (int(face_landmarks[conn.end].x * w), int(face_landmarks[conn.end].y * h))
                cv2.line(frame, pt1, pt2, (255, 0, 0), 1)


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


def classify_expression(landmarks) -> tuple:
    """
    Heuristically classifies facial expressions using MediaPipe FaceMesh landmarks.
    Supported: Neutral, Happy, Sad, Angry, Fear, Pain, Surprised.
    Returns: (expression_label, confidence_score)
    """
    if len(landmarks) < 468:
        return "Neutral", 1.0

    # 3D Euclidean distance helper
    def dist(i1, i2):
        p1 = landmarks[i1]
        p2 = landmarks[i2]
        return float(np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2))

    def clip(val, min_val, max_val):
        return max(min_val, min(val, max_val))

    try:
        # Reference scale metrics
        face_h = dist(10, 152)  # forehead to chin
        face_w = dist(234, 454)  # outer cheekbones
        ref_scale = max(face_h, 1e-6)

        # Mouth features
        m_width = dist(78, 308)
        m_height = dist(13, 14)
        mar = m_height / max(m_width, 1e-6)  # Mouth Aspect Ratio

        # Smile/frown Y-coordinate comparison
        # (Y increases downwards in image space. Corners higher up means corners_y is smaller than center_y)
        corners_y = (landmarks[78].y + landmarks[308].y) / 2.0
        center_y = (landmarks[13].y + landmarks[14].y) / 2.0
        smile_score = (center_y - corners_y) / ref_scale

        # Eyebrow height relative to eyes
        left_brow_h = dist(105, 159)
        right_brow_h = dist(334, 386)
        brow_h = (left_brow_h + right_brow_h) / (2.0 * ref_scale)

        # Eyebrow furrow / squeeze
        brow_squeeze = dist(107, 336) / max(face_w, 1e-6)

        # Eye openness aspect ratio
        left_eye_h = dist(159, 145)
        left_eye_w = dist(33, 133)
        right_eye_h = dist(386, 374)
        right_eye_w = dist(263, 362)
        eye_ar = ((left_eye_h / max(left_eye_w, 1e-6)) + (right_eye_h / max(right_eye_w, 1e-6))) / 2.0

        # Continuous activation scores:
        # 1. Happy: smile_score ranges from 0.0 (neutral/flat) to 0.02+ (smile)
        happy_act = clip(smile_score / 0.015, 0.0, 1.0)

        # 2. Surprised: mar > 0.35 and raised eyebrows (brow_h > 0.075)
        surprised_act = clip((mar - 0.1) / 0.3, 0.0, 1.0) * clip((brow_h - 0.07) / 0.02, 0.0, 1.0)

        # Eyebrow furrow score: lower brow_squeeze means more furrowed.
        # Typically brow_squeeze is around 0.20-0.22, drops to 0.16-0.18 when furrowed.
        furrow_act = clip((0.21 - brow_squeeze) / 0.05, 0.0, 1.0)

        # 3. Angry: furrowed eyebrows + lowered brow_h
        angry_act = furrow_act * clip((0.08 - brow_h) / 0.02, 0.0, 1.0)

        # 4. Pain: furrowed eyebrows + squinted/narrowed eyes (eye_ar < 0.22)
        pain_act = furrow_act * clip((0.24 - eye_ar) / 0.08, 0.0, 1.0)

        # 5. Fear: furrowed eyebrows + raised/normal eyebrows (brow_h > 0.065)
        fear_act = furrow_act * clip((brow_h - 0.065) / 0.02, 0.0, 1.0)

        # 6. Sad: frowning mouth (negative smile score) + furrowed eyebrows
        sad_act = clip((-smile_score) / 0.01, 0.0, 1.0) * furrow_act

        activations = {
            "Happy": happy_act,
            "Surprised": surprised_act,
            "Angry": angry_act,
            "Pain": pain_act,
            "Fear": fear_act,
            "Sad": sad_act
        }

        # Find the maximum activation
        best_expr = max(activations, key=activations.get)
        best_val = activations[best_expr]

        if best_val >= 0.35:
            return best_expr, float(best_val)
        else:
            return "Neutral", float(1.0 - best_val)

    except Exception as e:
        print(f"[FaceAI] Expression classification error: {e}")
        return "Neutral", 1.0


def recognize_multiple_faces(frame, run_mesh: bool = True, show_mesh: bool = True, show_ids: bool = False, show_bbox: bool = True, run_recognition: bool = True):
    """
    Detects and identifies all faces using InsightFace.
    Optionally renders 468-point Face Mesh via MediaPipe FaceLandmarker.

    Args:
        frame:           BGR image (numpy array)
        run_mesh:        If True, run MediaPipe FaceLandmarker for detailed mesh overlay.
        show_mesh:       If True, draw connections.
        show_ids:        If True, draw landmark index text.
        show_bbox:       If True, draw bounding boxes on frame.
        run_recognition: If True, run InsightFace face recognition embedding comparison.
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
                        _draw_detailed_face_mesh(frame, face_lm, color=(0, 255, 200), show_mesh=show_mesh, show_ids=show_ids)
                        xs = [lm.x * w for lm in face_lm]
                        ys = [lm.y * h for lm in face_lm]
                        centroid = (sum(xs)/len(xs), sum(ys)/len(ys))
                        mp_centroids.append((centroid, face_lm))
            except Exception as e:
                print(f"[FaceLandmarker] Mesh rendering error: {e}")

    # 2. Skip recognition if run_recognition is False
    if not run_recognition:
        for (cx, cy), lm in mp_centroids:
            xs = [l.x * w for l in lm]
            ys = [l.y * h for l in lm]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            bw, bh = x2 - x1, y2 - y1
            
            orientation = estimate_face_orientation(lm)
            expression, expression_conf = classify_expression(lm)
            
            results_list.append({
                "person_id": "Unknown",
                "name": "Unknown Person",
                "confidence": 0.0,
                "match_status": "Unknown Person",
                "expression": expression,
                "expression_confidence": expression_conf,
                "box": [x1, y1, bw, bh],
                "embedding": None,
                "orientation": orientation
            })
        return results_list, frame

    # 3. Face Recognition (InsightFace w600k_mbf)
    app = _get_insightface_app()
    if app is None or app == "FAILED":
        return results_list, frame

    faces = app.get(frame)
    if not faces:
        return results_list, frame

    from modules.perf.db_cache import get_cached_face_centroids, get_cached_people
    centroids = get_cached_face_centroids()
    people_map = {p["id"]: p["name"] for p in get_cached_people()}

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw, bh = x2 - x1, y2 - y1

        if bw < 10 or bh < 10:
            continue

        embedding = face.embedding.tolist()

        # Spatial matching with MediaPipe centroids to get landmarks for orientation/expression
        orientation = "Front"
        expression = "Neutral"
        expression_conf = 1.0
        for (cx, cy), lm in mp_centroids:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                orientation = estimate_face_orientation(lm)
                expression, expression_conf = classify_expression(lm)
                break

        # Compute matches for margin verification
        matches = []
        q_arr = np.array(embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr))
        if q_norm > 1e-8:
            q_arr /= q_norm

        for pid, centroid in centroids.items():
            sim = float(np.dot(q_arr, centroid))
            matches.append((pid, sim))

        matches = sorted(matches, key=lambda x: x[1], reverse=True)

        best_name = "Unknown Person"
        best_id = "Unknown"
        best_sim = 0.0
        match_status = "Unknown Person"

        if matches:
            top_id, top_sim = matches[0]
            second_sim = matches[1][1] if len(matches) > 1 else 0.0
            margin = top_sim - second_sim

            # Margin Check Rule: if top minus second < 10% (0.10), match is considered ambiguous
            if len(matches) > 1 and margin < 0.10:
                best_id = "Unknown"
                best_name = "Unknown Person"
                best_sim = top_sim
                match_status = "Unknown Person"
            else:
                # Recognition thresholds
                if top_sim < 0.50:
                    best_id = "Unknown"
                    best_name = "Unknown Person"
                    best_sim = top_sim
                    match_status = "Unknown Person"
                elif 0.50 <= top_sim < 0.70:
                    best_id = top_id
                    best_name = people_map.get(top_id, "Unknown Person")
                    best_sim = top_sim
                    match_status = "Weak Match"
                elif 0.70 <= top_sim < 0.85:
                    best_id = top_id
                    best_name = people_map.get(top_id, "Unknown Person")
                    best_sim = top_sim
                    match_status = "Recognized"
                else:  # top_sim >= 0.85
                    best_id = top_id
                    best_name = people_map.get(top_id, "Unknown Person")
                    best_sim = top_sim
                    match_status = "Strong Match"

        if best_name == "Unknown Person":
            label = "Unknown Person"
            box_color = (0, 50, 255) # Red
        else:
            update_last_seen(best_id)
            label = f"{best_name} ({match_status} - {int(best_sim * 100)}%)"
            box_color = (0, 220, 160) # Green

        # Draw bounding boxes and text overlays if enabled
        if show_bbox:
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), box_color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        results_list.append({
            "person_id": best_id,
            "name": best_name,
            "confidence": float(best_sim),
            "match_status": match_status,
            "expression": expression,
            "expression_confidence": expression_conf,
            "box": [x1, y1, bw, bh],
            "embedding": embedding,
            "orientation": orientation
        })

    return results_list, frame
