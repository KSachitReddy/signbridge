"""
live.py — Real-time ISL sign translation page for SignBridge AI.

Pipeline (optimized):
  Camera → [resize to inference res] → RGB convert (once) →
    MediaPipe Pose + Hands (shared RGB) →
    InsightFace Face Recog (throttled, every N frames) →
    Sign Classifier (throttled, every M frames) →
    Translation → TTS + Ollama (async) →
    [upscale to display res] → Streamlit placeholder update

Performance Modes:
  ⚡ Performance  — 320×240 inference, face every 25 frames, no face mesh
  ⚖️ Balanced    — 480×360 inference, face every 15 frames, full face mesh ON
  🎯 Accuracy    — 640×480 inference, face every 8 frames,  full face mesh ON

Unknown Face Workflow (NON-BLOCKING):
  When an unknown face is detected, a sidebar notification appears.
  The video stream NEVER pauses. Recognition continues.
  User can [Save] (opens guided enrollment) or [Ignore] the notification.
"""

import streamlit as st
import cv2
import numpy as np
import time
import io
import os
import secrets

from modules.locales import t
from modules.camera import generate_mock_frame
from modules.pose.holistic import track_and_draw_pose
from modules.hands.landmarks import track_and_draw_hands
from modules.face.face_ai import validate_and_enroll_face, recognize_multiple_faces, reinit_insightface
from modules.signs.recognizer import SignSequenceBuffer, sign_classifier, NO_SIGN_LABEL
from modules.translation import translate_sign
from modules.speech import get_tts_html, render_stt_listener
from modules.database import (
    add_conversation,
    get_conversations,
    get_all_people,
    save_person,
    add_face_vector,
    get_all_face_vectors,
)
from modules.ollama import generate_response
from modules.perf import FrameThrottle, CentroidTracker, AsyncWorker, PERF_MODES
from modules.perf.db_cache import invalidate_face_cache


# ─── Singleton async worker (lives for the session) ───────────────────────────
_ollama_worker: AsyncWorker = None


def _get_ollama_worker() -> AsyncWorker:
    global _ollama_worker
    if _ollama_worker is None:
        _ollama_worker = AsyncWorker(max_workers=1)
    return _ollama_worker


def _init_state():
    defaults = {
        "sequence_buffer":    SignSequenceBuffer(size=20),
        "last_logged_sign":   "",
        "last_sign_time":     0.0,
        "current_results":    None,
        "ai_response":        "",
        # Enrollment state — NON-BLOCKING: video never pauses
        "enroll_pending":     False,   # True when sidebar notification is shown
        "enroll_name_input":  "",
        "enroll_frame_snap":  None,    # snapshot for the enrollment preview
        "fps_counter":        0,
        "fps_last_time":      time.time(),
        "fps_display":        0.0,
        "enroll_active":      False,   # True during guided multi-angle capture
        "enroll_name":        "",
        "enroll_samples":     {},
        "enroll_success_msg": "",
        "unknown_notif_shown": False,  # debounce: only notify once per unknown person
        "unknown_notif_ignored": False,
        # Performance engine
        "perf_mode":          "⚖️ Balanced",
        "face_throttle":      None,
        "sign_throttle":      None,
        "centroid_tracker":   None,
        "_face_cache":        [],
        "_sign_cache":        [("None", 1.0), ("None", 0.0), ("None", 0.0)],
        "_last_perf_mode":    "",
        # Pipeline diagnostics
        "_pipeline_hands_left":  False,
        "_pipeline_hands_right": False,
        "_pipeline_buffer_fill": 0,
        "_pipeline_model_src":   "unknown",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _ensure_throttles(mode_cfg: dict, mode_name: str):
    """Re-create throttle objects when mode changes."""
    if st.session_state._last_perf_mode == mode_name:
        return
    st.session_state._last_perf_mode = mode_name
    st.session_state.face_throttle = FrameThrottle(mode_cfg["face_recog_interval"])
    st.session_state.sign_throttle = FrameThrottle(mode_cfg["sign_interval"])
    if st.session_state.centroid_tracker is None:
        st.session_state.centroid_tracker = CentroidTracker(max_disappeared=30)
    else:
        st.session_state.centroid_tracker.clear()
    # Hot-swap InsightFace det_size
    reinit_insightface(det_size=mode_cfg["det_size"])


def _average_embeddings(embeddings_list):
    if not embeddings_list:
        return None
    arr = np.array(embeddings_list, dtype=np.float32)
    mean_vec = arr.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec /= norm
    return mean_vec.tolist()


def _complete_enrollment():
    name = st.session_state.enroll_name
    samples = [v for v in st.session_state.enroll_samples.values() if v is not None]
    if samples:
        avg_emb = _average_embeddings(samples)
        existing = get_all_face_vectors()
        people_map = {p["id"]: p["name"] for p in get_all_people()}
        for ev in existing:
            ev_emb = ev.get("embedding") or []
            if len(ev_emb) != len(avg_emb):
                continue
            sim = float(np.dot(np.array(avg_emb), np.array(ev_emb)) / (
                max(np.linalg.norm(avg_emb), 1e-8) * max(np.linalg.norm(ev_emb), 1e-8)
            ))
            if sim >= 0.85:
                existing_name = people_map.get(ev["person_id"], "an existing profile")
                st.session_state.enroll_success_msg = (
                    f"Face already enrolled as '{existing_name}' ({int(sim * 100)}% match)."
                )
                st.session_state.enroll_active = False
                st.session_state.enroll_name = ""
                st.session_state.enroll_samples = {}
                st.session_state.enroll_pending = False
                st.session_state.unknown_notif_shown = False
                return
        person_id = f"P_{secrets.token_hex(6)}"
        faces_dir = os.path.abspath(os.path.join("database", "faces"))
        os.makedirs(faces_dir, exist_ok=True)
        snap = st.session_state.get("enroll_frame_snap")
        image_path = os.path.join(faces_dir, f"{person_id}.jpg")
        if snap is not None:
            cv2.imwrite(image_path, snap)
        else:
            cv2.imwrite(image_path, np.zeros((100, 100, 3), dtype=np.uint8))
        save_person(person_id, name, "Multi-sample enrolled profile")
        add_face_vector(person_id, image_path, avg_emb)
        st.session_state.enroll_success_msg = f"✅ Face profile for '{name}' enrolled successfully!"
        # Invalidate DB cache so next recognition frame sees the new profile
        invalidate_face_cache()
        # Clear centroid tracker so re-identification fires immediately
        if st.session_state.centroid_tracker:
            st.session_state.centroid_tracker.clear()
    st.session_state.enroll_active = False
    st.session_state.enroll_name = ""
    st.session_state.enroll_samples = {}
    st.session_state.enroll_frame_snap = None
    # Reset notification state so new unknowns can be detected again
    st.session_state.enroll_pending = False
    st.session_state.unknown_notif_shown = False
    st.session_state.unknown_notif_ignored = False


def _scale_landmarks_to_display(frame_annotated, infer_w, infer_h, display_w=640, display_h=480):
    """
    When inference ran at lower resolution, upscale the annotated frame to display size.
    Landmark dots and lines were drawn at infer_w×infer_h pixel coordinates and
    will be correctly upscaled by cv2.resize with INTER_LINEAR.
    """
    if infer_w != display_w or infer_h != display_h:
        return cv2.resize(frame_annotated, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    return frame_annotated


def render_live_page(lang="en"):
    _init_state()

    st.title(f"📹 {t('nav.live', lang)}")
    st.markdown("---")

    if st.session_state.get("enroll_success_msg"):
        st.success(st.session_state.enroll_success_msg)
        st.session_state.enroll_success_msg = ""

    # ── STT query param sync ──────────────────────────────────────────────────
    query_params = st.query_params
    if "stt_text" in query_params:
        stt_text = query_params["stt_text"]
        st.query_params.clear()
        ai_resp = generate_response("[Speech]", stt_text, "You", lang)
        add_conversation("System", "[Speech]", stt_text, lang, 1.0)
        if ai_resp:
            add_conversation("System", "[AI Response]", ai_resp, lang, 1.0)
        st.rerun()

    # ── Performance Mode Selector ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚡ Performance Mode")
        selected_mode = st.radio(
            "Processing Mode",
            list(PERF_MODES.keys()),
            index=list(PERF_MODES.keys()).index(st.session_state.perf_mode),
            key="perf_mode_radio",
            help="⚡ Performance: no mesh, max FPS | ⚖️ Balanced: full mesh, good FPS | 🎯 Accuracy: full quality"
        )
        st.session_state.perf_mode = selected_mode
        mode_cfg = PERF_MODES[selected_mode]
        _ensure_throttles(mode_cfg, selected_mode)

        st.caption(
            f"Face recog every **{mode_cfg['face_recog_interval']}** frames  \n"
            f"Sign classify every **{mode_cfg['sign_interval']}** frames  \n"
            f"Inference res: **{mode_cfg['infer_w']}×{mode_cfg['infer_h']}**  \n"
            f"FaceMesh: **{'✅ On' if mode_cfg['use_face_mesh'] else '❌ Off'}**"
        )

        st.markdown("---")

        # ── NON-BLOCKING Unknown Face Notification ────────────────────────────
        if st.session_state.enroll_pending and not st.session_state.unknown_notif_ignored:
            st.markdown("""
            <div style="padding:12px;border-radius:10px;
                        background:rgba(245,158,11,0.15);
                        border:1px solid rgba(245,158,11,0.5);margin-bottom:8px;">
                <span style="color:#F59E0B;font-weight:700;font-size:15px;">👤 Unknown Person Detected</span><br>
                <span style="color:#CBD5E1;font-size:13px;">Video is still running. Save this person's profile?</span>
            </div>
            """, unsafe_allow_html=True)

            name_in = st.text_input(
                "Name for this person:",
                key="sidebar_enroll_name",
                placeholder="e.g. Rahul"
            )
            col_save, col_ignore = st.columns(2)
            if col_save.button("💾 Save", use_container_width=True, key="sidebar_save_btn"):
                if name_in.strip():
                    st.session_state.enroll_name = name_in.strip()
                    st.session_state.enroll_active = True
                    st.session_state.enroll_samples = {
                        "Front": None, "Left": None,
                        "Right": None, "Up": None, "Down": None
                    }
                    st.session_state.enroll_pending = False
                    st.rerun()
            if col_ignore.button("✕ Ignore", use_container_width=True, key="sidebar_ignore_btn"):
                st.session_state.unknown_notif_ignored = True
                st.session_state.enroll_pending = False
                st.rerun()

        # ── Guided Enrollment Progress (still non-blocking) ───────────────────
        if st.session_state.get("enroll_active", False):
            st.markdown(f"### 👤 Enrolling: **{st.session_state.enroll_name}**")
            st.info("Rotate your head slowly: Front → Left → Right → Up → Down")
            for o in ["Front", "Left", "Right", "Up", "Down"]:
                captured = "🟢" if st.session_state.enroll_samples.get(o) is not None else "🟡"
                st.markdown(f"{captured} **{o}**")
            col_save2, col_cancel = st.columns(2)
            if col_save2.button("💾 Save Profile", use_container_width=True, key="guided_save_btn"):
                _complete_enrollment()
                st.rerun()
            if col_cancel.button("❌ Cancel", use_container_width=True, key="guided_cancel_btn"):
                st.session_state.enroll_active = False
                st.session_state.enroll_name = ""
                st.session_state.enroll_samples = {}
                st.session_state.enroll_frame_snap = None
                st.session_state.enroll_pending = False
                st.session_state.unknown_notif_shown = False
                st.rerun()

    # ── Layout ────────────────────────────────────────────────────────────────
    col_cam, col_info = st.columns([6, 5])

    with col_cam:
        st.subheader(t("live.cameraTitle", lang))
        cam_source = st.radio(
            "Camera Source",
            ["🎥 Live Webcam", "🖥️ Simulated Feed"],
            horizontal=True
        )
        frame_placeholder = st.empty()

    with col_info:
        st.subheader(t("live.detailsTitle", lang))
        info_placeholder = st.empty()
        pipeline_placeholder = st.empty()
        ai_response_placeholder = st.empty()
        stt_placeholder = st.empty()
        history_placeholder = st.empty()

    # ═══════════════════════════════════════════════════════════════════════════
    # LIVE WEBCAM MODE
    # ═══════════════════════════════════════════════════════════════════════════
    if cam_source == "🎥 Live Webcam":
        run_cam = st.toggle("🟢 Start Live Stream", value=False, key="live_toggle")

        if run_cam:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            # Reduce internal buffer to minimize capture latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                st.error("❌ Could not open webcam. Check camera permissions.")
                return

            infer_w = mode_cfg["infer_w"]
            infer_h = mode_cfg["infer_h"]
            use_mesh = mode_cfg["use_face_mesh"]
            face_throttle: FrameThrottle = st.session_state.face_throttle
            sign_throttle: FrameThrottle = st.session_state.sign_throttle
            tracker: CentroidTracker = st.session_state.centroid_tracker
            ollama_worker = _get_ollama_worker()

            # FPS tracking locals
            fps_counter = 0
            fps_last_time = time.time()
            fps_display = 0.0

            # Render-throttle locals — reset every time the stream (re)starts
            render_tick = 0          # increments every frame, used for modular skipping
            _stt_rendered_lang = ""  # triggers STT re-render only on lang change
            _hist_last_ts = 0.0      # conversation history: update at most every 2s
            _last_info_sign = ""     # info panel: force update on new sign

            # Caches (local vars are faster than session_state lookups in tight loops)
            face_cache = st.session_state._face_cache
            sign_cache = st.session_state._sign_cache

            # Pipeline diagnostic locals
            hands_left_detected = False
            hands_right_detected = False

            try:
                while st.session_state.get("live_toggle", False):
                    t0 = time.perf_counter()

                    ret, raw_frame = cap.read()
                    if not ret:
                        st.error("Unable to retrieve webcam frame.")
                        break

                    # ── 1. Downscale for inference ────────────────────────────
                    if raw_frame.shape[1] != infer_w or raw_frame.shape[0] != infer_h:
                        infer_frame = cv2.resize(raw_frame, (infer_w, infer_h),
                                                 interpolation=cv2.INTER_LINEAR)
                    else:
                        infer_frame = raw_frame

                    # ── 2. Single RGB conversion shared by all AI modules ─────
                    rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)

                    # ── 3. Pose + Hands (share the single rgb_frame) ──────────
                    pose_joints, infer_frame = track_and_draw_pose(
                        infer_frame, rgb_frame=rgb_frame)
                    hands_data, infer_frame = track_and_draw_hands(
                        infer_frame, rgb_frame=rgb_frame)

                    # Track hand detection for pipeline diagnostics
                    hands_left_detected = len(hands_data.get("left", [])) == 21
                    hands_right_detected = len(hands_data.get("right", [])) == 21

                    # ── 4. Face Recognition (throttled) ──────────────────────
                    if face_throttle.should_run():
                        faces_results, infer_frame = recognize_multiple_faces(
                            infer_frame, run_mesh=use_mesh)
                        faces_results = tracker.update(faces_results)
                        face_cache = faces_results
                    else:
                        # Reuse cached identity; redraw cached boxes on frame
                        faces_results = face_cache
                        for f in faces_results:
                            x, y, w_b, h_b = f["box"]
                            name = f.get("name", "Unknown Person")
                            conf = f.get("confidence", 0.0)
                            sim_pct = int(conf * 100)
                            if name == "Unknown Person":
                                color = (0, 50, 255)
                                label = "Unknown Person"
                            elif "Possible" in name:
                                color = (0, 165, 255)
                                label = name
                            else:
                                color = (0, 220, 160)
                                label = f"{name} ({sim_pct}%)"
                            cv2.rectangle(infer_frame, (x, y), (x + w_b, y + h_b), color, 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(infer_frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
                            cv2.putText(infer_frame, label, (x + 2, y - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # ── 5. Guided Enrollment Sample Capturing ─────────────────
                    if st.session_state.get("enroll_active", False) and faces_results:
                        primary_face = faces_results[0]
                        orient = primary_face.get("orientation", "Front")
                        emb = primary_face.get("embedding", None)
                        if orient in ["Front", "Left", "Right", "Up", "Down"] and emb is not None:
                            if st.session_state.enroll_samples.get(orient) is None:
                                st.session_state.enroll_samples[orient] = emb
                                # Keep a snapshot for profile photo
                                if st.session_state.enroll_frame_snap is None:
                                    st.session_state.enroll_frame_snap = infer_frame.copy()

                        if all(st.session_state.enroll_samples.get(o) is not None
                               for o in ["Front", "Left", "Right", "Up", "Down"]):
                            _complete_enrollment()
                            st.rerun()

                    # ── 6. Sequence buffer ────────────────────────────────────
                    st.session_state.sequence_buffer.add(
                        left_hand=hands_data["left"],
                        right_hand=hands_data["right"],
                        pose=pose_joints
                    )

                    # ── 7. Sign prediction (throttled) ────────────────────────
                    if sign_throttle.should_run():
                        sign_cache = sign_classifier.predict(st.session_state.sequence_buffer)

                    top_preds = sign_cache
                    detected_sign = top_preds[0][0]
                    score = top_preds[0][1]
                    translation_text = translate_sign(detected_sign, lang)

                    # Determine prediction source for diagnostics
                    model_src = "RF Model" if sign_classifier.clf is not None else "Heuristic Fallback"

                    # ── 8. Log + AI response (throttled: 2s gap, 65% conf) ────
                    now_t = time.time()
                    should_log = (
                        detected_sign not in ("None", "", NO_SIGN_LABEL)
                        and score > 0.65
                        and detected_sign != st.session_state.last_logged_sign
                        and (now_t - st.session_state.last_sign_time) > 2.0
                    )
                    if should_log:
                        person_id = faces_results[0]["person_id"] if faces_results else "Unknown"
                        person_name = faces_results[0]["name"] if faces_results else "Unknown"
                        add_conversation(person_id, detected_sign, translation_text, lang, score)
                        ollama_worker.submit(generate_response, detected_sign,
                                             translation_text, person_name, lang)
                        st.session_state.last_logged_sign = detected_sign
                        st.session_state.last_sign_time = now_t

                    # Pick up any completed Ollama result
                    ollama_result = ollama_worker.get_result()
                    if ollama_result:
                        st.session_state.ai_response = ollama_result
                        add_conversation("System", "[AI Response]", ollama_result, lang, 1.0)

                    # ── 9. NON-BLOCKING Unknown face notification ─────────────
                    has_unknown = any(r["name"] == "Unknown Person" for r in faces_results)
                    # Only trigger notification if: unknown found, not already pending/ignored, not enrolling
                    if (has_unknown
                            and not st.session_state.enroll_pending
                            and not st.session_state.unknown_notif_ignored
                            and not st.session_state.get("enroll_active", False)
                            and not st.session_state.unknown_notif_shown):
                        st.session_state.enroll_pending = True
                        st.session_state.unknown_notif_shown = True
                        # Snapshot for enrollment reference (does NOT pause video)
                        st.session_state.enroll_frame_snap = infer_frame.copy()
                        st.session_state._face_cache = face_cache
                        st.session_state._sign_cache = sign_cache
                        # NOTE: NO st.rerun() here — video continues uninterrupted

                    # Reset notification when no unknowns are present (e.g. they left frame)
                    if not has_unknown and st.session_state.unknown_notif_shown:
                        if not st.session_state.get("enroll_active", False):
                            st.session_state.unknown_notif_shown = False
                            st.session_state.unknown_notif_ignored = False
                            st.session_state.enroll_pending = False

                    # ── 10. FPS tracking ──────────────────────────────────────
                    fps_counter += 1
                    elapsed = now_t - fps_last_time
                    if elapsed >= 1.0:
                        fps_display = round(fps_counter / elapsed, 1)
                        fps_counter = 0
                        fps_last_time = now_t
                        st.session_state.fps_display = fps_display

                    # ── 11. Frame display — JPEG bytes (~3× faster than Streamlit PNG) ──
                    display_frame = _scale_landmarks_to_display(infer_frame, infer_w, infer_h)

                    fps_color = (0, 220, 100) if fps_display >= 10 else (0, 80, 255)
                    cv2.putText(display_frame, f"FPS: {fps_display}  [{selected_mode}]",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2)

                    if st.session_state.enroll_pending:
                        cv2.putText(display_frame, "Unknown Person — See Sidebar to Save",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

                    if st.session_state.get("enroll_active", False):
                        cv2.putText(display_frame,
                                    f"Enrolling: {st.session_state.enroll_name} — Rotate Head",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

                    img_rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    ret_jpg, jpeg_buf = cv2.imencode(
                        '.jpg', img_rgb_display, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ret_jpg:
                        frame_placeholder.image(
                            io.BytesIO(jpeg_buf.tobytes()), use_container_width=True)
                    else:
                        frame_placeholder.image(img_rgb_display, use_container_width=True)

                    # ── 12. Throttled panel updates ───────────────────────────
                    # Each panel is updated at the rate it actually needs — this
                    # eliminates ~120-150ms of wasted Streamlit render work per frame.
                    render_tick += 1

                    # Info panel: refresh every 2nd frame OR immediately when sign changes
                    if render_tick % 2 == 0 or detected_sign != _last_info_sign:
                        _render_info_panel(
                            info_placeholder, ai_response_placeholder,
                            faces_results, detected_sign, score,
                            translation_text, top_preds, lang
                        )
                        _last_info_sign = detected_sign

                    # Pipeline diagnostics: refresh every 3rd frame
                    if render_tick % 3 == 0:
                        _render_pipeline_report(
                            pipeline_placeholder,
                            hands_left=hands_left_detected,
                            hands_right=hands_right_detected,
                            buffer_fill=len(st.session_state.sequence_buffer.buffer),
                            buffer_size=st.session_state.sequence_buffer.size,
                            model_src=model_src,
                            top_preds=top_preds,
                            fps=fps_display
                        )

                    # STT listener: re-render only when language changes (static JS widget)
                    if lang != _stt_rendered_lang:
                        with stt_placeholder.container():
                            st.markdown(f"### {t('live.twoWayHeader', lang)}")
                            render_stt_listener(lang)
                        _stt_rendered_lang = lang

                    # Conversation history: at most once every 2 seconds
                    _hist_now = now_t
                    if _hist_now - _hist_last_ts >= 2.0:
                        with history_placeholder.container():
                            _render_chat_history(lang)
                        _hist_last_ts = _hist_now

            finally:
                cap.release()
                st.session_state._face_cache = face_cache
                st.session_state._sign_cache = sign_cache

        else:
            frame_placeholder.info(
                "▶ Toggle **'Start Live Stream'** above to begin real-time ISL translation."
            )
            _render_static_widgets(info_placeholder, ai_response_placeholder,
                                   stt_placeholder, history_placeholder, lang)
            _render_pipeline_report(
                pipeline_placeholder,
                hands_left=False, hands_right=False,
                buffer_fill=0, buffer_size=20,
                model_src="RF Model" if sign_classifier.clf is not None else "Heuristic Fallback",
                top_preds=[("None", 0.0), ("None", 0.0), ("None", 0.0)],
                fps=0.0
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMULATED FEED MODE
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        frame = generate_mock_frame("SignBridge AI — Simulated Feed")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_joints, frame = track_and_draw_pose(frame, use_mock=True, rgb_frame=rgb_frame)
        hands_data, frame = track_and_draw_hands(frame, use_mock=True, rgb_frame=rgb_frame)
        faces_results, frame = recognize_multiple_faces(frame, run_mesh=mode_cfg["use_face_mesh"])

        st.session_state.sequence_buffer.add(
            left_hand=hands_data["left"],
            right_hand=hands_data["right"],
            pose=pose_joints
        )

        top_preds = sign_classifier.predict(st.session_state.sequence_buffer)
        detected_sign = top_preds[0][0]
        score = top_preds[0][1]
        translation_text = translate_sign(detected_sign, lang)
        model_src = "RF Model" if sign_classifier.clf is not None else "Heuristic Fallback"

        if detected_sign not in ("None", "", NO_SIGN_LABEL):
            if st.session_state.last_logged_sign != detected_sign:
                person_id = faces_results[0]["person_id"] if faces_results else "Unknown"
                add_conversation(person_id, detected_sign, translation_text, lang, score)
                st.session_state.last_logged_sign = detected_sign

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(img_rgb, use_container_width=True)

        _render_info_panel(
            info_placeholder, ai_response_placeholder,
            faces_results, detected_sign, score,
            translation_text, top_preds, lang
        )

        _render_pipeline_report(
            pipeline_placeholder,
            hands_left=len(hands_data.get("left", [])) == 21,
            hands_right=len(hands_data.get("right", [])) == 21,
            buffer_fill=len(st.session_state.sequence_buffer.buffer),
            buffer_size=st.session_state.sequence_buffer.size,
            model_src=model_src,
            top_preds=top_preds,
            fps=st.session_state.get("fps_display", 0.0)
        )

        with stt_placeholder.container():
            st.markdown(f"### {t('live.twoWayHeader', lang)}")
            render_stt_listener(lang)

        with history_placeholder.container():
            _render_chat_history(lang)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-components
# ─────────────────────────────────────────────────────────────────────────────

def _render_info_panel(info_pl, ai_pl, faces_results, detected_sign, score,
                       translation_text, top_preds, lang):
    person_name = faces_results[0]["name"] if faces_results else "Unknown"
    expression = faces_results[0]["expression"] if faces_results else "Neutral"

    with info_pl.container():
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin:0 0 10px 0;color:#3B82F6;">👤 {t('live.person', lang)}:
                <span style="color:#FFF;">{person_name}</span></h4>
            <h4 style="margin:0 0 10px 0;color:#10B981;">🤟 {t('live.sign', lang)}:
                <span style="color:#FFF;">{detected_sign} ({int(score * 100)}%)</span></h4>
            <h4 style="margin:0 0 10px 0;color:#EC4899;">🎭 {t('live.expressionContext', lang)}:
                <span style="color:#FFF;">{expression}</span></h4>
            <h4 style="margin:0 0 10px 0;color:#F59E0B;">🌍 {t('live.language', lang)}:
                <span style="color:#FFF;">{lang.upper()}</span></h4>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"### 💬 {t('live.output', lang)}")
        st.success(translation_text if translation_text else "Awaiting sign...")

        if translation_text and detected_sign != "None":
            st.markdown("#### 🔊 Voice Output")
            st.html(get_tts_html(translation_text, lang))

    with ai_pl.container():
        ai_resp = st.session_state.get("ai_response", "")
        if ai_resp:
            st.markdown("""
            <div style="padding:12px;border-radius:10px;
                        background:rgba(16,185,129,0.12);
                        border-left:4px solid #10B981;margin-top:8px;">
                <span style="color:#10B981;font-weight:600;">🤖 AI Response</span><br>
            """ + ai_resp + "</div>", unsafe_allow_html=True)


def _render_pipeline_report(pipeline_pl, hands_left: bool, hands_right: bool,
                             buffer_fill: int, buffer_size: int,
                             model_src: str, top_preds: list, fps: float):
    """
    Renders a live pipeline diagnostic card showing:
    - Model status (RF loaded vs heuristic fallback)
    - Hand detection status (left / right)
    - Sequence buffer fill level
    - Prediction source
    - Top 3 sign predictions with confidence bars
    """
    model_loaded = (model_src == "RF Model")
    model_icon = "✅" if model_loaded else "⚠️"
    model_color = "#10B981" if model_loaded else "#F59E0B"

    lh_icon = "🟢" if hands_left else "⚪"
    rh_icon = "🟢" if hands_right else "⚪"
    buf_pct = int((buffer_fill / max(buffer_size, 1)) * 100)
    buf_color = "#10B981" if buf_pct >= 80 else "#F59E0B" if buf_pct >= 40 else "#EF4444"

    with pipeline_pl.container():
        st.markdown(f"""
        <div style="padding:12px;border-radius:10px;
                    background:rgba(15,23,42,0.8);
                    border:1px solid rgba(255,255,255,0.1);
                    margin:8px 0 12px 0;">
            <span style="color:#94A3B8;font-size:12px;font-weight:600;letter-spacing:0.5px;">
                🔬 PIPELINE DIAGNOSTICS
            </span>
            <div style="margin-top:8px;display:grid;grid-template-columns:1fr 1fr;gap:6px;">
                <div>
                    <span style="color:#64748B;font-size:11px;">MODEL</span><br>
                    <span style="color:{model_color};font-size:13px;font-weight:600;">
                        {model_icon} {model_src}
                    </span>
                </div>
                <div>
                    <span style="color:#64748B;font-size:11px;">FPS</span><br>
                    <span style="color:{'#10B981' if fps >= 5 else '#EF4444'};font-size:13px;font-weight:600;">
                        {fps:.1f}
                    </span>
                </div>
                <div>
                    <span style="color:#64748B;font-size:11px;">LEFT HAND</span><br>
                    <span style="font-size:13px;">{lh_icon} {'Detected (21 pts)' if hands_left else 'Not detected'}</span>
                </div>
                <div>
                    <span style="color:#64748B;font-size:11px;">RIGHT HAND</span><br>
                    <span style="font-size:13px;">{rh_icon} {'Detected (21 pts)' if hands_right else 'Not detected'}</span>
                </div>
            </div>
            <div style="margin-top:8px;">
                <span style="color:#64748B;font-size:11px;">SEQUENCE BUFFER</span>
                <span style="color:{buf_color};font-size:11px;float:right;">{buffer_fill}/{buffer_size} frames</span>
                <div style="background:#1E293B;border-radius:4px;height:6px;margin-top:4px;">
                    <div style="background:{buf_color};border-radius:4px;height:6px;width:{buf_pct}%;transition:width 0.3s;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Top-3 predictions — single markdown call replaces 3×(columns+progress+write)
        valid_preds = [
            (lbl, conf) for lbl, conf in top_preds[:3]
            if lbl not in ("None", NO_SIGN_LABEL, "")
        ]
        if valid_preds:
            bars = "".join(
                f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;">'
                f'<span style="color:#E2E8F0;font-size:12px;min-width:72px;font-weight:600;">{lbl}</span>'
                f'<div style="flex:1;background:#1E293B;border-radius:3px;height:7px;">'
                f'<div style="background:#2563EB;border-radius:3px;height:7px;width:{int(max(0,min(100,conf*100)))}%;"></div></div>'
                f'<span style="color:#94A3B8;font-size:11px;min-width:30px;">{int(max(0,min(100,conf*100)))}%</span></div>'
                for lbl, conf in valid_preds
            )
            st.markdown(
                '<p style="margin:4px 0 3px;color:#94A3B8;font-size:11px;font-weight:600;">'
                'TOP PREDICTIONS</p>' + bars,
                unsafe_allow_html=True,
            )
        else:
            st.caption("Awaiting hand signs...")


def _render_chat_history(lang):
    """Renders two-way conversation history as a chat-style log."""
    from modules.perf.db_cache import get_cached_people
    st.markdown("#### 💬 Conversation History")
    logs = get_conversations()[:10]
    people = get_cached_people()
    people_map = {p["id"]: p["name"] for p in people}
    people_map.update({"Unknown": "Unknown", "System": "System", "Synthetic": "Synthetic"})

    for entry in reversed(logs):
        ts = entry["timestamp"]
        ts_short = ts.split(" ")[1][:5] if " " in ts else ts
        p_name = people_map.get(entry["person_id"], "Unknown")
        sign = entry["recognized_sign"]
        text = entry["translated_text"]
        conf = entry.get("confidence", 0.0)

        if sign == "[AI Response]":
            st.markdown(f"""
            <div style="margin:4px 0;padding:10px 14px;border-radius:10px 10px 10px 2px;
                        background:rgba(16,185,129,0.1);border-left:4px solid #10B981;
                        max-width:90%;">
                <span style="color:#6B7280;font-size:11px;">[{ts_short}] 🤖 AI</span><br>
                <span style="color:#E2E8F0;">{text}</span>
            </div>
            """, unsafe_allow_html=True)
        elif sign == "[Speech]":
            st.markdown(f"""
            <div style="margin:4px 0;padding:10px 14px;border-radius:10px 10px 2px 10px;
                        background:rgba(139,92,246,0.1);border-right:4px solid #8B5CF6;
                        max-width:90%;margin-left:auto;">
                <span style="color:#6B7280;font-size:11px;">[{ts_short}] 🎤 {p_name}</span><br>
                <span style="color:#E2E8F0;">{text}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin:4px 0;padding:10px 14px;border-radius:10px 10px 10px 2px;
                        background:rgba(37,99,235,0.1);border-left:4px solid #2563EB;
                        max-width:90%;">
                <span style="color:#6B7280;font-size:11px;">[{ts_short}] 🤟 {p_name}</span><br>
                <span style="color:#E2E8F0;"><strong>{sign}</strong> → <em>{text}</em>
                <span style="color:#64748B;font-size:11px;">({int(conf * 100)}%)</span></span>
            </div>
            """, unsafe_allow_html=True)


def _render_static_widgets(info_pl, ai_pl, stt_pl, hist_pl, lang):
    """Renders static (non-streaming) right-panel widgets."""
    with info_pl.container():
        st.markdown("""
        <div class="glass-card">
            <h4 style="margin:0 0 10px 0;color:#3B82F6;">👤 Person: <span style="color:#FFF;">—</span></h4>
            <h4 style="margin:0 0 10px 0;color:#10B981;">🤟 Sign: <span style="color:#FFF;">Awaiting stream...</span></h4>
            <h4 style="margin:0 0 10px 0;color:#EC4899;">🎭 Expression: <span style="color:#FFF;">—</span></h4>
        </div>
        """, unsafe_allow_html=True)
        st.info("Start the webcam stream to begin real-time sign recognition.")

    with stt_pl.container():
        st.markdown(f"### {t('live.twoWayHeader', lang)}")
        render_stt_listener(lang)

    with hist_pl.container():
        _render_chat_history(lang)
