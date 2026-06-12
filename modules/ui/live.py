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
from modules.pose.holistic import track_and_draw_pose
from modules.hands.landmarks import track_and_draw_hands
from modules.face.face_ai import validate_and_enroll_face, recognize_multiple_faces, reinit_insightface
from modules.signs.recognizer import SignSequenceBuffer, sign_classifier, NO_SIGN_LABEL, CONFIDENCE_THRESHOLD
from modules.translation import translate_sign
from modules.speech import get_tts_html
from modules.database import (
    add_conversation,
    get_all_people,
    save_person,
    add_face_vector,
    get_all_face_vectors,
)
from modules.perf import FrameThrottle, TimeThrottle, CentroidTracker, AsyncWorker, PERF_MODES
from modules.perf.db_cache import invalidate_face_cache


def _init_state():
    defaults = {
        "sequence_buffer":    SignSequenceBuffer(size=20),
        "last_logged_sign":   "",
        "last_sign_time":     0.0,
        "current_results":    None,
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
        "expr_throttle":      None,
        "sign_throttle":      None,
        "centroid_tracker":   None,
        "_face_cache":        [],
        "_sign_cache":        [("None", 1.0), ("None", 0.0), ("None", 0.0)],
        "_last_perf_mode":    "",
        "face_worker":        None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "face_worker" not in st.session_state or st.session_state.face_worker is None:
        st.session_state.face_worker = AsyncWorker(max_workers=1)


def _ensure_throttles(mode_cfg: dict, mode_name: str):
    """Re-create throttle objects when mode changes."""
    if st.session_state._last_perf_mode == mode_name:
        return
    st.session_state._last_perf_mode = mode_name
    
    # Time-based throttle for face recognition: Performance=1.5s, Balanced=1.0s, Accuracy=0.5s
    face_int = 1.5 if "Performance" in mode_name else 1.0 if "Balanced" in mode_name else 0.5
    st.session_state.face_throttle = TimeThrottle(face_int)
    
    # Time-based throttle for expressions: Performance=1.0s, Balanced/Accuracy=0.5s
    expr_int = 1.0 if "Performance" in mode_name else 0.5
    st.session_state.expr_throttle = TimeThrottle(expr_int)
    
    st.session_state.sign_throttle = FrameThrottle(mode_cfg["sign_interval"])
    if st.session_state.centroid_tracker is None:
        st.session_state.centroid_tracker = CentroidTracker(max_disappeared=30)
    else:
        st.session_state.centroid_tracker.clear()
    # Hot-swap InsightFace det_size
    reinit_insightface(det_size=mode_cfg["det_size"])
    
    st.session_state.show_mesh = mode_cfg["use_face_mesh"]


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

        st.markdown("### 🛠️ Display Overlays")
        show_mesh = st.checkbox("Show Face Mesh", value=st.session_state.get("show_mesh", mode_cfg["use_face_mesh"]), key="show_mesh")
        show_ids = st.checkbox("Show Landmark IDs", value=False, key="show_ids")
        show_bbox = st.checkbox("Show Face Bounding Box", value=True, key="show_bbox")
        show_hands = st.checkbox("Show Hand Skeleton", value=True, key="show_hands")

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
        
        col_cam_source, col_cam_btn = st.columns([3, 2])
        with col_cam_source:
            cam_source = st.radio(
                "Camera Source",
                ["🎥 Live Webcam"],
                label_visibility="collapsed"
            )
        with col_cam_btn:
            run_cam = st.toggle("🟢 Start Live Stream", value=False, key="live_toggle")

        frame_placeholder = st.empty()

    with col_info:
        st.subheader(t("live.detailsTitle", lang))
        info_placeholder = st.empty()

    # ═══════════════════════════════════════════════════════════════════════════
    # LIVE WEBCAM MODE
    # ═══════════════════════════════════════════════════════════════════════════
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

        # FPS tracking locals
        fps_counter = 0
        fps_last_time = time.time()
        fps_display = 0.0

        # Render-throttle locals
        render_tick = 0
        _last_info_sign = ""

        # Caches
        face_cache = st.session_state._face_cache
        sign_cache = st.session_state._sign_cache

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
                    infer_frame, rgb_frame=rgb_frame, draw_skeleton=show_hands)

                # ── 4. Face Recognition (throttled & async) ──────────────
                face_worker = st.session_state.face_worker
                async_res = face_worker.get_result()
                if async_res is not None:
                    faces_results = async_res
                    faces_results = tracker.update(faces_results)
                    face_cache = faces_results

                # If enrollment is active, we run synchronously
                if st.session_state.get("enroll_active", False):
                    faces_results, infer_frame = recognize_multiple_faces(
                        infer_frame,
                        run_mesh=True,
                        show_mesh=show_mesh,
                        show_ids=show_ids,
                        show_bbox=show_bbox,
                        run_recognition=True
                    )
                    faces_results = tracker.update(faces_results)
                    face_cache = faces_results
                else:
                    # Run Face mesh and expression/orientation detection on main thread (fast)
                    run_mesh_main = show_mesh or show_ids
                    run_expr_main = st.session_state.expr_throttle.should_run()
                    run_mesh_now = run_mesh_main or run_expr_main

                    mesh_results = []
                    if run_mesh_now:
                        mesh_results, infer_frame = recognize_multiple_faces(
                            infer_frame,
                            run_mesh=True,
                            show_mesh=show_mesh,
                            show_ids=show_ids,
                            show_bbox=False,
                            run_recognition=False
                        )

                    # Submit heavy InsightFace recognition to background thread when throttled
                    if face_throttle.should_run() and not face_worker.is_running():
                        face_worker.submit(
                            recognize_multiple_faces,
                            infer_frame.copy(),
                            run_mesh=False,
                            show_mesh=False,
                            show_ids=False,
                            show_bbox=False,
                            run_recognition=True
                        )

                    # Merge main-thread expression/orientation results into active face cached results
                    faces_to_show = []
                    for f in face_cache:
                        f_show = f.copy()
                        x1, y1, w_b, h_b = f_show["box"]
                        cx_f = x1 + w_b / 2.0
                        cy_f = y1 + h_b / 2.0
                        
                        matched_mesh = None
                        min_dist = 99999.0
                        for m in mesh_results:
                            mx1, my1, mw, mh = m["box"]
                            cx_m = mx1 + mw / 2.0
                            cy_m = my1 + mh / 2.0
                            dist = np.sqrt((cx_f - cx_m)**2 + (cy_f - cy_m)**2)
                            if dist < min_dist and dist < 100.0:
                                min_dist = dist
                                matched_mesh = m
                        
                        if matched_mesh is not None:
                            f_show["expression"] = matched_mesh["expression"]
                            f_show["expression_confidence"] = matched_mesh["expression_confidence"]
                            f_show["orientation"] = matched_mesh["orientation"]
                            f_show["box"] = matched_mesh["box"]
                        
                        faces_to_show.append(f_show)
                        
                        # Draw bbox and name on frame if enabled
                        if show_bbox:
                            x, y, w_b, h_b = f_show["box"]
                            name = f_show.get("name", "Unknown Person")
                            conf = f_show.get("confidence", 0.0)
                            match_status = f_show.get("match_status", "Unknown Person")
                            sim_pct = int(conf * 100)
                            if name == "Unknown Person":
                                color = (0, 50, 255)
                                label = "Unknown Person"
                            else:
                                color = (0, 220, 160)
                                label = f"{name} ({match_status} - {sim_pct}%)"
                            cv2.rectangle(infer_frame, (x, y), (x + w_b, y + h_b), color, 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                            cv2.rectangle(infer_frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
                            cv2.putText(infer_frame, label, (x + 2, y - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                    faces_results = faces_to_show

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
                top_label, top_conf = top_preds[0]
                
                if top_conf >= CONFIDENCE_THRESHOLD and top_label != NO_SIGN_LABEL:
                    detected_sign = top_label
                    score = top_conf
                else:
                    detected_sign = NO_SIGN_LABEL
                    score = 0.0

                # Only run translate_sign when the predicted gesture label changes or translation cache is empty
                if "last_detected_sign" not in st.session_state or detected_sign != st.session_state.last_detected_sign or not st.session_state.get("translation_text_cache"):
                    st.session_state.last_detected_sign = detected_sign
                    translation_text = translate_sign(detected_sign, lang)
                    st.session_state.translation_text_cache = translation_text
                else:
                    translation_text = st.session_state.translation_text_cache

                # ── 8. Log (throttled: 2s gap, 55% conf) ──────────────────
                now_t = time.time()
                should_log = (
                    detected_sign not in ("None", "", NO_SIGN_LABEL)
                    and score >= CONFIDENCE_THRESHOLD
                    and detected_sign != st.session_state.last_logged_sign
                    and (now_t - st.session_state.last_sign_time) > 2.0
                )
                if should_log:
                    person_id = faces_results[0]["person_id"] if faces_results else "Unknown"
                    
                    # Async daemon thread DB write
                    import threading
                    t_db = threading.Thread(
                        target=add_conversation,
                        args=(person_id, detected_sign, translation_text, lang, score),
                        daemon=True
                    )
                    t_db.start()
                    
                    st.session_state.last_logged_sign = detected_sign
                    st.session_state.last_sign_time = now_t

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
                render_tick += 1

                # Info panel: refresh every 2nd frame OR immediately when sign changes
                if render_tick % 2 == 0 or detected_sign != _last_info_sign:
                    _render_info_panel(
                        info_placeholder,
                        faces_results, detected_sign, score,
                        translation_text, top_preds, lang,
                        hands_data=hands_data
                    )
                    _last_info_sign = detected_sign

        finally:
            cap.release()
            st.session_state._face_cache = face_cache
            st.session_state._sign_cache = sign_cache

    else:
        frame_placeholder.info(
            "▶ Toggle **'Start Live Stream'** above to begin real-time ISL translation."
        )
        _render_static_widgets(info_placeholder, lang)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-components
# ─────────────────────────────────────────────────────────────────────────────

def _render_info_panel(info_pl, faces_results, detected_sign, score,
                       translation_text, top_preds, lang, hands_data=None):
    person_name = "Unknown Person"
    person_conf = 0.0
    expression = "Neutral"
    expression_conf = 1.0

    if faces_results:
        person_name = faces_results[0]["name"]
        person_conf = faces_results[0]["confidence"]
        expression = faces_results[0]["expression"]
        expression_conf = faces_results[0].get("expression_confidence", 1.0)

    # Face display string
    if person_name == "Unknown Person" or person_conf < 0.68:
        person_display = "Unknown Person"
    else:
        person_display = f"{person_name} ({int(person_conf * 100)}%)"

    # Expression display string
    expression_display = f"{expression} ({int(expression_conf * 100)}%)"

    # Sign display string & confidence
    if detected_sign == NO_SIGN_LABEL:
        sign_display = "No Gesture Detected"
        conf_display = "—"
    else:
        sign_display = detected_sign
        conf_display = f"{int(score * 100)}%"

    # Hand presence/confidence display string
    left_conf = 0.0
    right_conf = 0.0
    if hands_data and isinstance(hands_data, dict):
        left_conf = hands_data.get("left_conf", 0.0)
        right_conf = hands_data.get("right_conf", 0.0)

    if left_conf > 0.4:
        left_display = f"Detected ({int(left_conf * 100)}%)"
    else:
        left_display = "Not detected"

    if right_conf > 0.4:
        right_display = f"Detected ({int(right_conf * 100)}%)"
    else:
        right_display = "Not detected"

    with info_pl.container():
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin:0 0 10px 0;color:#3B82F6;">👤 {t('live.person', lang)}:
                <span style="color:#FFF;">{person_display}</span></h4>
            <h4 style="margin:0 0 10px 0;color:#10B981;">🤟 {t('live.sign', lang)}:
                <span style="color:#FFF;">{sign_display}</span></h4>
            <h4 style="margin:0 0 10px 0;color:#F59E0B;">📈 {t('live.confidence', lang)}:
                <span style="color:#FFF;">{conf_display}</span></h4>
            <h4 style="margin:0 0 10px 0;color:#EC4899;">🎭 {t('live.expressionContext', lang)}:
                <span style="color:#FFF;">{expression_display}</span></h4>
            <h4 style="margin:0 0 10px 0;color:#E2E8F0;">👐 Left Hand:
                <span style="color:#FFF;">{left_display}</span></h4>
            <h4 style="margin:0 0 10px 0;color:#E2E8F0;">👐 Right Hand:
                <span style="color:#FFF;">{right_display}</span></h4>
            <h4 style="margin:0 0 10px 0;color:#8B5CF6;">🌍 {t('live.language', lang)}:
                <span style="color:#FFF;">{lang.upper()}</span></h4>
        </div>
        """, unsafe_allow_html=True)

        # Top-3 predictions progress bars
        valid_preds = [
            (lbl, conf) for lbl, conf in top_preds[:3]
            if lbl not in ("", "None")
        ]
        if valid_preds:
            bars = "".join(
                f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
                f'<span style="color:#E2E8F0;font-size:12px;min-width:90px;font-weight:600;">{lbl}</span>'
                f'<div style="flex:1;background:#1E293B;border-radius:3px;height:8px;">'
                f'<div style="background:#10B981;border-radius:3px;height:8px;width:{int(max(0,min(100,conf*100)))}%;"></div></div>'
                f'<span style="color:#94A3B8;font-size:11px;min-width:30px;">{int(max(0,min(100,conf*100)))}%</span></div>'
                for lbl, conf in valid_preds
            )
            st.markdown(
                '<div class="glass-card" style="padding:16px;margin-top:10px;">'
                '<p style="margin:0 0 8px 0;color:#94A3B8;font-size:12px;font-weight:600;letter-spacing:0.5px;">'
                '📊 TOP PREDICTIONS</p>' + bars + '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"### 💬 {t('live.output', lang)}")
        st.success(translation_text if translation_text else "Awaiting sign...")

        if translation_text and detected_sign != NO_SIGN_LABEL:
            st.markdown("#### 🔊 Voice Output")
            st.html(get_tts_html(translation_text, lang))


def _render_static_widgets(info_pl, lang):
    """Renders static (non-streaming) right-panel widgets."""
    with info_pl.container():
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="margin:0 0 10px 0;color:#3B82F6;">👤 {t('live.person', lang)}: <span style="color:#FFF;">—</span></h4>
            <h4 style="margin:0 0 10px 0;color:#10B981;">🤟 {t('live.sign', lang)}: <span style="color:#FFF;">Awaiting stream...</span></h4>
            <h4 style="margin:0 0 10px 0;color:#F59E0B;">📈 {t('live.confidence', lang)}: <span style="color:#FFF;">—</span></h4>
            <h4 style="margin:0 0 10px 0;color:#EC4899;">🎭 {t('live.expressionContext', lang)}: <span style="color:#FFF;">—</span></h4>
            <h4 style="margin:0 0 10px 0;color:#E2E8F0;">👐 Left Hand: <span style="color:#FFF;">—</span></h4>
            <h4 style="margin:0 0 10px 0;color:#E2E8F0;">👐 Right Hand: <span style="color:#FFF;">—</span></h4>
        </div>
        """, unsafe_allow_html=True)
        st.info("Start the webcam stream to begin real-time sign recognition.")
