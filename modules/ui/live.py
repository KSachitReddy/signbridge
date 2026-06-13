"""
live.py — Real-time ISL sign translation page for SignBridge AI.

Pipeline (optimized, WebRTC-based):
  Browser Camera (streamlit-webrtc) → RGB convert (once) →
    MediaPipe Pose + Hands (shared RGB) →
    InsightFace Face Recog (throttled, TimeThrottle) →
    Sign Classifier (throttled, FrameThrottle) →
    Translation → TTS + Ollama (async) →
    Streamlit placeholder update

Performance Modes:
  ⚡ Performance  — 320×240 inference, face every 1.5s, no face mesh
  ⚖️ Balanced    — 480×360 inference, face every 1.0s, full face mesh ON
  🎯 Accuracy    — 640×480 inference, face every 0.5s,  full face mesh ON

Unknown Face Workflow (NON-BLOCKING):
  When an unknown face is detected, a sidebar notification appears.
  The video stream NEVER pauses. Recognition continues.
  User can [Save] (opens guided enrollment) or [Ignore] the notification.
"""

import streamlit as st
import av
import cv2
import numpy as np
import time
import io
import os
import secrets
import threading
import queue
from typing import List

import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit_webrtc import webrtc_streamer
from modules.locales import t
from modules.pose.holistic import track_and_draw_pose
from modules.hands.landmarks import track_and_draw_hands
from modules.face.face_ai import (
    validate_and_enroll_face,
    recognize_multiple_faces,
    reinit_insightface_async,
)
from modules.signs.recognizer import (
    SignSequenceBuffer,
    sign_classifier,
    NO_SIGN_LABEL,
    CONFIDENCE_THRESHOLD,
)
from modules.translation import translate_sign
from modules.speech import get_tts_html
from modules.database import (
    add_conversation,
    get_all_people,
    save_person,
    add_face_vector,
    get_all_face_vectors,
)
from modules.perf import (
    FrameThrottle,
    TimeThrottle,
    CentroidTracker,
    AsyncWorker,
    PERF_MODES,
)
from modules.perf.db_cache import invalidate_face_cache


# ─── RTC Configuration (STUN + public TURN for cloud/NAT traversal) ──────────
RTC_CONFIG = RTCConfiguration(
    iceServers=[
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
)


# ─── Shared result queue (VideoProcessor → main thread) ──────────────────────
_RESULT_QUEUE: "queue.Queue[dict]" = queue.Queue(maxsize=4)


# ─────────────────────────────────────────────────────────────────────────────
# VideoProcessor — runs inside the WebRTC worker thread
# ─────────────────────────────────────────────────────────────────────────────
class VideoProcessor:
    """
    Receives raw browser video frames via WebRTC, runs the full AI pipeline,
    annotates the frame, and pushes result metadata into a shared queue for
    the main Streamlit thread to consume for UI updates.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Per-processor state (mirrors session_state but lives in the worker thread)
        self.sequence_buffer = SignSequenceBuffer(size=20)
        self.face_cache: list = []
        self.sign_cache = [("None", 1.0), ("None", 0.0), ("None", 0.0)]
        self.last_logged_sign = ""
        self.last_sign_time = 0.0
        self.last_detected_sign = ""
        self.translation_text_cache = ""
        self.last_expr_dict: dict = {}

        # Throttles — will be re-configured from main thread via configure()
        self.face_throttle = TimeThrottle(1.0)
        self.expr_throttle = TimeThrottle(0.5)
        self.sign_throttle = FrameThrottle(3)
        self.centroid_tracker = CentroidTracker(max_disappeared=30)
        self.face_worker = AsyncWorker(max_workers=1)

        # Display/overlay options — updated via configure()
        self.show_mesh = True
        self.show_ids = False
        self.show_bbox = True
        self.show_hands = True
        self.infer_w = 480
        self.infer_h = 360

        # FPS tracking
        self._fps_counter = 0
        self._fps_last_time = time.time()
        self.fps_display = 0.0

        # Pose throttle
        self._pose_tick = 0
        self._last_pose_joints: dict = {}
        self.pose_interval = 1

        # Enrollment / unknown-person flags communicated from main thread
        self.enroll_active = False
        self.enroll_name = ""
        self.enroll_samples: dict = {}
        self.unknown_notif_shown = False
        self.unknown_notif_ignored = False

    def configure(self, cfg: dict):
        """Called from main thread to push updated config into processor."""
        with self._lock:
            self.infer_w = cfg.get("infer_w", self.infer_w)
            self.infer_h = cfg.get("infer_h", self.infer_h)
            self.show_mesh = cfg.get("show_mesh", self.show_mesh)
            self.show_ids = cfg.get("show_ids", self.show_ids)
            self.show_bbox = cfg.get("show_bbox", self.show_bbox)
            self.show_hands = cfg.get("show_hands", self.show_hands)
            self.pose_interval = cfg.get("pose_interval", self.pose_interval)
            self.enroll_active = cfg.get("enroll_active", self.enroll_active)
            self.enroll_name = cfg.get("enroll_name", self.enroll_name)
            self.enroll_samples = cfg.get("enroll_samples", self.enroll_samples)
            self.unknown_notif_shown = cfg.get("unknown_notif_shown", self.unknown_notif_shown)
            self.unknown_notif_ignored = cfg.get("unknown_notif_ignored", self.unknown_notif_ignored)

            face_int = cfg.get("face_throttle_interval", 1.0)
            expr_int = cfg.get("expr_throttle_interval", 0.5)
            sign_int = cfg.get("sign_interval", 3)
            self.face_throttle = TimeThrottle(face_int)
            self.expr_throttle = TimeThrottle(expr_int)
            self.sign_throttle = FrameThrottle(sign_int)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # ── Convert to BGR numpy (standard OpenCV format) ─────────────────
        bgr = frame.to_ndarray(format="bgr24")

        with self._lock:
            infer_w = self.infer_w
            infer_h = self.infer_h
            show_mesh = self.show_mesh
            show_ids = self.show_ids
            show_bbox = self.show_bbox
            show_hands = self.show_hands
            pose_interval = self.pose_interval

        # ── 1. Downscale for inference ────────────────────────────────────
        h0, w0 = bgr.shape[:2]
        if w0 != infer_w or h0 != infer_h:
            infer_frame = cv2.resize(bgr, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
        else:
            infer_frame = bgr.copy()

        # ── 2. Single RGB conversion shared by all AI modules ─────────────
        rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)

        # ── 3. Pose + Hands ───────────────────────────────────────────────
        self._pose_tick += 1
        if self._pose_tick % pose_interval == 0:
            pose_joints, infer_frame = track_and_draw_pose(infer_frame, rgb_frame=rgb_frame)
            self._last_pose_joints = pose_joints
        else:
            pose_joints = self._last_pose_joints

        hands_data, infer_frame = track_and_draw_hands(
            infer_frame, rgb_frame=rgb_frame, draw_skeleton=show_hands
        )

        # ── 4. Face Recognition (throttled & async) ───────────────────────
        face_worker = self.face_worker
        async_res = face_worker.get_result()
        if async_res is not None:
            faces_results, _ = async_res
            faces_results = self.centroid_tracker.update(faces_results)
            self.face_cache = faces_results

        if self.enroll_active:
            faces_results, infer_frame = recognize_multiple_faces(
                infer_frame,
                run_mesh=True,
                show_mesh=show_mesh,
                show_ids=show_ids,
                show_bbox=show_bbox,
                run_recognition=True,
            )
            faces_results = self.centroid_tracker.update(faces_results)
            self.face_cache = faces_results
        else:
            run_mesh_now = show_mesh or show_ids or self.expr_throttle.should_run()
            mesh_results = []
            if run_mesh_now:
                mesh_results, infer_frame = recognize_multiple_faces(
                    infer_frame,
                    run_mesh=True,
                    show_mesh=show_mesh,
                    show_ids=show_ids,
                    show_bbox=False,
                    run_recognition=False,
                )

            if self.face_throttle.should_run() and not face_worker.is_running():
                face_worker.submit(
                    recognize_multiple_faces,
                    infer_frame.copy(),
                    run_mesh=False,
                    show_mesh=False,
                    show_ids=False,
                    show_bbox=False,
                    run_recognition=True,
                )

            faces_to_show = []
            for f in self.face_cache:
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
                    dist = np.sqrt((cx_f - cx_m) ** 2 + (cy_f - cy_m) ** 2)
                    if dist < min_dist and dist < 100.0:
                        min_dist = dist
                        matched_mesh = m

                tid = f_show.get("track_id")
                if matched_mesh is not None:
                    f_show["expression"] = matched_mesh["expression"]
                    f_show["expression_confidence"] = matched_mesh["expression_confidence"]
                    f_show["orientation"] = matched_mesh["orientation"]
                    f_show["box"] = matched_mesh["box"]
                    if tid is not None:
                        self.last_expr_dict[tid] = (
                            matched_mesh["expression"],
                            matched_mesh["expression_confidence"],
                            matched_mesh["orientation"],
                        )
                elif tid is not None and tid in self.last_expr_dict:
                    f_show["expression"], f_show["expression_confidence"], f_show["orientation"] = (
                        self.last_expr_dict[tid]
                    )

                faces_to_show.append(f_show)

                if show_bbox:
                    x, y, w_b2, h_b2 = f_show["box"]
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
                    cv2.rectangle(infer_frame, (x, y), (x + w_b2, y + h_b2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(infer_frame, (x, y - th - 8), (x + tw + 4, y), color, -1)
                    cv2.putText(
                        infer_frame, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
                    )

            faces_results = faces_to_show

        # ── 5. Guided Enrollment Sample Capturing ─────────────────────────
        enrollment_update = None
        if self.enroll_active and faces_results:
            primary_face = faces_results[0]
            orient = primary_face.get("orientation", "Front")
            emb = primary_face.get("embedding", None)
            if orient in ["Front", "Left", "Right", "Up", "Down"] and emb is not None:
                if self.enroll_samples.get(orient) is None:
                    self.enroll_samples[orient] = emb
                    enrollment_update = {
                        "type": "sample_captured",
                        "orient": orient,
                        "samples": dict(self.enroll_samples),
                        "snap": infer_frame.copy() if orient == "Front" else None,
                    }
            if all(
                self.enroll_samples.get(o) is not None
                for o in ["Front", "Left", "Right", "Up", "Down"]
            ):
                enrollment_update = {
                    "type": "enrollment_complete",
                    "samples": dict(self.enroll_samples),
                }

        # ── 6. Sequence buffer ────────────────────────────────────────────
        self.sequence_buffer.add(
            left_hand=hands_data["left"],
            right_hand=hands_data["right"],
            pose=pose_joints,
        )

        # ── 7. Sign prediction (throttled) ────────────────────────────────
        if self.sign_throttle.should_run():
            self.sign_cache = sign_classifier.predict(self.sequence_buffer)

        top_preds = self.sign_cache
        top_label, top_conf = top_preds[0]

        if top_conf >= CONFIDENCE_THRESHOLD and top_label != NO_SIGN_LABEL:
            detected_sign = top_label
            score = top_conf
        else:
            detected_sign = NO_SIGN_LABEL
            score = 0.0

        if (
            detected_sign != self.last_detected_sign
            or not self.translation_text_cache
        ):
            self.last_detected_sign = detected_sign
            self.translation_text_cache = translate_sign(detected_sign, "en")
        translation_text = self.translation_text_cache

        # ── 8. DB log (throttled) ─────────────────────────────────────────
        now_t = time.time()
        should_log = (
            detected_sign not in ("None", "", NO_SIGN_LABEL)
            and score >= CONFIDENCE_THRESHOLD
            and detected_sign != self.last_logged_sign
            and (now_t - self.last_sign_time) > 2.0
        )
        if should_log:
            person_id = faces_results[0]["person_id"] if faces_results else "Unknown"
            t_db = threading.Thread(
                target=add_conversation,
                args=(person_id, detected_sign, translation_text, "en", score),
                daemon=True,
            )
            t_db.start()
            self.last_logged_sign = detected_sign
            self.last_sign_time = now_t

        # ── 9. Unknown face detection ─────────────────────────────────────
        has_unknown = any(r["name"] == "Unknown Person" for r in faces_results)
        trigger_unknown_notif = (
            has_unknown
            and not self.enroll_active
            and not self.unknown_notif_shown
            and not self.unknown_notif_ignored
        )
        if trigger_unknown_notif:
            self.unknown_notif_shown = True

        # ── 10. FPS ───────────────────────────────────────────────────────
        self._fps_counter += 1
        elapsed = now_t - self._fps_last_time
        if elapsed >= 1.0:
            self.fps_display = round(self._fps_counter / elapsed, 1)
            self._fps_counter = 0
            self._fps_last_time = now_t

        # ── 11. Annotate display frame ────────────────────────────────────
        display_frame = infer_frame
        fps_color = (0, 220, 100) if self.fps_display >= 10 else (0, 80, 255)
        cv2.putText(
            display_frame, f"FPS: {self.fps_display}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, fps_color, 2,
        )
        if self.unknown_notif_shown and not self.unknown_notif_ignored:
            cv2.putText(
                display_frame, "Unknown Person - Save in panel",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2,
            )
        if self.enroll_active:
            cv2.putText(
                display_frame,
                f"Enrolling: {self.enroll_name} - Rotate Head",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2,
            )

        # ── 12. Push result metadata for main thread ──────────────────────
        result = {
            "faces": faces_results,
            "detected_sign": detected_sign,
            "score": score,
            "translation_text": translation_text,
            "top_preds": top_preds,
            "fps": self.fps_display,
            "hands_data": hands_data,
            "trigger_unknown_notif": trigger_unknown_notif,
            "enrollment_update": enrollment_update,
            "enroll_samples": dict(self.enroll_samples),
        }
        try:
            _RESULT_QUEUE.put_nowait(result)
        except queue.Full:
            try:
                _RESULT_QUEUE.get_nowait()
            except queue.Empty:
                pass
            try:
                _RESULT_QUEUE.put_nowait(result)
            except queue.Full:
                pass

        # ── 13. Return annotated frame back to browser ─────────────────────
        return av.VideoFrame.from_ndarray(display_frame, format="bgr24")


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "last_logged_sign":       "",
        "last_sign_time":         0.0,
        "enroll_pending":         False,
        "enroll_name_input":      "",
        "enroll_frame_snap":      None,
        "enroll_active":          False,
        "enroll_name":            "",
        "enroll_samples":         {},
        "enroll_success_msg":     "",
        "unknown_notif_shown":    False,
        "unknown_notif_ignored":  False,
        "perf_mode":              "⚖️ Balanced",
        "_last_perf_mode":        "",
        "last_expr_dict":         {},
        # Latest pipeline results for UI rendering
        "ui_faces":               [],
        "ui_detected_sign":       NO_SIGN_LABEL,
        "ui_score":               0.0,
        "ui_translation":         "",
        "ui_top_preds":           [("None", 1.0), ("None", 0.0), ("None", 0.0)],
        "ui_fps":                 0.0,
        "ui_hands_data":          {},
        "ui_lang":                "en",
        # VideoProcessor reference
        "_processor":             None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


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
            sim = float(
                np.dot(np.array(avg_emb), np.array(ev_emb))
                / (
                    max(np.linalg.norm(avg_emb), 1e-8)
                    * max(np.linalg.norm(ev_emb), 1e-8)
                )
            )
            if sim >= 0.85:
                existing_name = people_map.get(ev["person_id"], "an existing profile")
                st.session_state.enroll_success_msg = (
                    f"Face already enrolled as '{existing_name}' ({int(sim * 100)}% match)."
                )
                _reset_enrollment_state()
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
        st.session_state.enroll_success_msg = (
            f"✅ Face profile for '{name}' enrolled successfully!"
        )
        invalidate_face_cache()
        proc: VideoProcessor = st.session_state.get("_processor")
        if proc is not None:
            proc.centroid_tracker.clear()
    _reset_enrollment_state()


def _reset_enrollment_state():
    st.session_state.enroll_active = False
    st.session_state.enroll_name = ""
    st.session_state.enroll_samples = {}
    st.session_state.enroll_frame_snap = None
    st.session_state.enroll_pending = False
    st.session_state.unknown_notif_shown = False
    st.session_state.unknown_notif_ignored = False
    proc: VideoProcessor = st.session_state.get("_processor")
    if proc is not None:
        proc.configure({
            "enroll_active": False,
            "enroll_name": "",
            "enroll_samples": {},
            "unknown_notif_shown": False,
            "unknown_notif_ignored": False,
        })


def _push_config_to_processor(mode_cfg: dict, mode_name: str):
    """Sync current UI settings into the VideoProcessor instance."""
    proc: VideoProcessor = st.session_state.get("_processor")
    if proc is None:
        return
    face_int = 1.5 if "Performance" in mode_name else 1.0 if "Balanced" in mode_name else 0.5
    expr_int = 1.0 if "Performance" in mode_name else 0.5
    proc.configure({
        "infer_w":                 mode_cfg["infer_w"],
        "infer_h":                 mode_cfg["infer_h"],
        "show_mesh":               st.session_state.get("show_mesh", mode_cfg["use_face_mesh"]),
        "show_ids":                st.session_state.get("show_ids", False),
        "show_bbox":               st.session_state.get("show_bbox", True),
        "show_hands":              st.session_state.get("show_hands", True),
        "pose_interval":           mode_cfg.get("pose_interval", 1),
        "face_throttle_interval":  face_int,
        "expr_throttle_interval":  expr_int,
        "sign_interval":           mode_cfg["sign_interval"],
        "enroll_active":           st.session_state.enroll_active,
        "enroll_name":             st.session_state.enroll_name,
        "enroll_samples":          dict(st.session_state.enroll_samples),
        "unknown_notif_shown":     st.session_state.unknown_notif_shown,
        "unknown_notif_ignored":   st.session_state.unknown_notif_ignored,
    })


def _drain_result_queue(lang: str):
    """Pull the latest result from the queue into session_state UI vars."""
    latest = None
    while True:
        try:
            latest = _RESULT_QUEUE.get_nowait()
        except queue.Empty:
            break
    if latest is None:
        return

    st.session_state.ui_faces = latest["faces"]
    st.session_state.ui_detected_sign = latest["detected_sign"]
    st.session_state.ui_score = latest["score"]
    st.session_state.ui_top_preds = latest["top_preds"]
    st.session_state.ui_fps = latest["fps"]
    st.session_state.ui_hands_data = latest.get("hands_data", {})

    # Translate with the UI-selected language (the processor uses "en" internally)
    detected_sign = latest["detected_sign"]
    st.session_state.ui_translation = translate_sign(detected_sign, lang)

    # Handle unknown person notification
    if latest.get("trigger_unknown_notif"):
        st.session_state.enroll_pending = True
        st.session_state.unknown_notif_shown = True
        frame_snap = latest.get("snap")
        if frame_snap is not None:
            st.session_state.enroll_frame_snap = frame_snap

    # Handle enrollment updates from processor
    eu = latest.get("enrollment_update")
    if eu:
        if eu["type"] == "sample_captured":
            st.session_state.enroll_samples = eu["samples"]
            if eu.get("snap") is not None and st.session_state.enroll_frame_snap is None:
                st.session_state.enroll_frame_snap = eu["snap"]
        elif eu["type"] == "enrollment_complete":
            st.session_state.enroll_samples = eu["samples"]
            _complete_enrollment()
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────

def render_live_page(lang="en"):
    _init_state()
    st.session_state.ui_lang = lang

    st.title(f"📹 {t('nav.live', lang)}")
    st.markdown("---")

    if st.session_state.get("enroll_success_msg"):
        st.success(st.session_state.enroll_success_msg)
        st.session_state.enroll_success_msg = ""

    selected_mode = st.session_state.get("perf_mode", "⚖️ Balanced")
    mode_cfg = PERF_MODES[selected_mode]

    # ── Layout ────────────────────────────────────────────────────────────────
    col_cam, col_info = st.columns([7, 3])

    with col_cam:
        st.subheader(t("live.cameraTitle", lang))

        col_cam_btn, col_cam_opts = st.columns([3, 2])
        with col_cam_btn:
            st.markdown(
                "<p style='color:#94A3B8;font-size:13px;margin-bottom:4px;'>"
                "🌐 Uses your browser camera via WebRTC</p>",
                unsafe_allow_html=True,
            )
        with col_cam_opts:
            with st.popover("⚙️ Overlays", use_container_width=True):
                show_mesh = st.checkbox(
                    "Show Face Mesh",
                    value=st.session_state.get("show_mesh", mode_cfg["use_face_mesh"]),
                    key="show_mesh",
                )
                show_ids = st.checkbox(
                    "Show Landmark IDs", value=False, key="show_ids"
                )
                show_bbox = st.checkbox(
                    "Show Face Bounding Box", value=True, key="show_bbox"
                )
                show_hands = st.checkbox(
                    "Show Hand Skeleton", value=True, key="show_hands"
                )

        # ── WebRTC Streamer ───────────────────────────────────────────────
        def _factory():
            proc = VideoProcessor()
            st.session_state["_processor"] = proc
            return proc

        ctx = webrtc_streamer(
            key="signbridge-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=_factory,
            media_stream_constraints={
                "video": {
                    "width":  {"ideal": 640, "max": 1280},
                    "height": {"ideal": 480, "max": 720},
                    "frameRate": {"ideal": 30, "max": 60},
                },
                "audio": False,
            },
            async_processing=True,
        )

        is_streaming = ctx.state.playing if ctx is not None else False

        # Keep processor config in sync every render cycle
        if is_streaming:
            _push_config_to_processor(mode_cfg, selected_mode)
            _drain_result_queue(lang)

        enroll_placeholder = st.empty()

        if st.session_state.get("enroll_active", False):
            with enroll_placeholder.container():
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left: 5px solid #2563EB; padding: 16px; margin-bottom: 15px;">
                        <h4 style="margin:0;color:#3B82F6;">👤 Guided Enrollment: Enrolling <b>{st.session_state.enroll_name}</b></h4>
                        <p style="margin:5px 0 10px 0;color:#94A3B8;font-size:13px;">Rotate head slowly to capture views:</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_states = st.columns(5)
                for idx, o in enumerate(["Front", "Left", "Right", "Up", "Down"]):
                    captured = "🟢" if st.session_state.enroll_samples.get(o) is not None else "🟡"
                    col_states[idx].markdown(
                        f"<div style='text-align:center;font-size:12px;'>{captured}<br><b>{o}</b></div>",
                        unsafe_allow_html=True,
                    )
                col_save2, col_cancel = st.columns(2)
                if col_save2.button("💾 Save Profile", use_container_width=True, key="inline_guided_save_btn"):
                    _complete_enrollment()
                    st.rerun()
                if col_cancel.button("❌ Cancel", use_container_width=True, key="inline_guided_cancel_btn"):
                    _reset_enrollment_state()
                    st.rerun()

        elif st.session_state.enroll_pending and not st.session_state.unknown_notif_ignored:
            with enroll_placeholder.container():
                st.markdown(
                    """
                    <div class="glass-card" style="border-left: 5px solid #F59E0B; padding: 16px;">
                        <h4 style="margin:0;color:#F59E0B;">👤 Unknown Person Detected</h4>
                        <p style="margin:5px 0 10px 0;color:#94A3B8;font-size:13px;">Save profile now to identify them in the future.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_in, col_sv, col_ig = st.columns([3, 2, 2])
                with col_in:
                    name_input = st.text_input(
                        "Name",
                        key="inline_enroll_name",
                        placeholder="e.g. Rahul",
                        label_visibility="collapsed",
                    )
                with col_sv:
                    if st.button("💾 Save", use_container_width=True, key="inline_save_btn"):
                        if name_input.strip():
                            st.session_state.enroll_name = name_input.strip()
                            st.session_state.enroll_active = True
                            st.session_state.enroll_samples = {
                                "Front": None,
                                "Left": None,
                                "Right": None,
                                "Up": None,
                                "Down": None,
                            }
                            st.session_state.enroll_pending = False
                            _push_config_to_processor(mode_cfg, selected_mode)
                            st.rerun()
                with col_ig:
                    if st.button("✕ Ignore", use_container_width=True, key="inline_ignore_btn"):
                        st.session_state.unknown_notif_ignored = True
                        st.session_state.enroll_pending = False
                        _push_config_to_processor(mode_cfg, selected_mode)
                        st.rerun()
        else:
            enroll_placeholder.empty()

    with col_info:
        st.subheader(t("live.detailsTitle", lang))
        info_placeholder = st.empty()

        if is_streaming:
            _render_info_panel(
                info_placeholder,
                st.session_state.ui_faces,
                st.session_state.ui_detected_sign,
                st.session_state.ui_score,
                st.session_state.ui_translation,
                st.session_state.ui_top_preds,
                lang,
                hands_data=st.session_state.ui_hands_data,
            )
        else:
            _render_static_widgets(info_placeholder, lang)

    # Auto-refresh while streaming so UI keeps updating
    if is_streaming:
        time.sleep(0.05)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Sub-components
# ─────────────────────────────────────────────────────────────────────────────

def _render_info_panel(
    info_pl, faces_results, detected_sign, score,
    translation_text, top_preds, lang, hands_data=None
):
    person_name = "Unknown Person"
    person_conf = 0.0
    expression = "Neutral"
    expression_conf = 1.0

    if faces_results:
        person_name = faces_results[0]["name"]
        person_conf = faces_results[0]["confidence"]
        expression = faces_results[0]["expression"]
        expression_conf = faces_results[0].get("expression_confidence", 1.0)

    recognized_person = person_name

    if person_name == "Unknown Person" or person_conf < 0.50:
        face_confidence = "—"
    else:
        face_confidence = f"{int(person_conf * 100)}%"

    if detected_sign == NO_SIGN_LABEL:
        detected_gesture = "No Gesture Detected"
        gesture_confidence = "—"
    else:
        detected_gesture = detected_sign
        gesture_confidence = f"{int(score * 100)}%"

    expression_display = f"{expression} ({int(expression_conf * 100)}%)"
    current_translation = translation_text if translation_text else "Awaiting sign..."

    left_conf = 0.0
    right_conf = 0.0
    if hands_data and isinstance(hands_data, dict):
        left_conf = hands_data.get("left_conf", 0.0)
        right_conf = hands_data.get("right_conf", 0.0)

    left_display = f"Detected ({int(left_conf * 100)}%)" if left_conf > 0.4 else "Not detected"
    right_display = f"Detected ({int(right_conf * 100)}%)" if right_conf > 0.4 else "Not detected"

    langs_map = {
        "en": "English",
        "hi": "Hindi (हिंदी)",
        "te": "Telugu (తెలుగు)",
        "ta": "Tamil (தமிழ்)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "ml": "Malayalam (മലയാളം)",
        "tcy": "Tulu (तुळु)",
    }
    current_language = langs_map.get(lang, lang.upper())

    with info_pl.container():
        st.markdown(
            f"""
            <div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 12px;">
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #3B82F6; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">👤 Recognized Person</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #F8FAFC; font-weight: 700;">{recognized_person}</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #60A5FA; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">🔍 Face Confidence</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #F8FAFC; font-weight: 700;">{face_confidence}</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #10B981; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">🤟 Detected Gesture</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #F8FAFC; font-weight: 700;">{detected_gesture}</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #34D399; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">📊 Gesture Confidence</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #F8FAFC; font-weight: 700;">{gesture_confidence}</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #EC4899; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">🎭 Expression</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #F8FAFC; font-weight: 700;">{expression_display}</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #E2E8F0; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">👐 Left Hand</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #F8FAFC; font-weight: 700;">{left_display}</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #E2E8F0; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">👐 Right Hand</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #F8FAFC; font-weight: 700;">{right_display}</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #8B5CF6; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">💬 Current Translation</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #F8FAFC; font-weight: 700;">{current_translation}</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"## 🌍 {current_language}")

        valid_preds = [
            (lbl, conf)
            for lbl, conf in top_preds[:3]
            if lbl not in ("", "None")
        ]
        if valid_preds:
            bars = "".join(
                f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">'
                f'<span style="color:#E2E8F0;font-size:12px;min-width:90px;font-weight:600;">{lbl}</span>'
                f'<div style="flex:1;background:#1E293B;border-radius:3px;height:8px;">'
                f'<div style="background:#10B981;border-radius:3px;height:8px;width:{int(max(0, min(100, conf * 100)))}%;"></div></div>'
                f'<span style="color:#94A3B8;font-size:11px;min-width:30px;">{int(max(0, min(100, conf * 100)))}%</span></div>'
                for lbl, conf in valid_preds
            )
            st.markdown(
                '<div class="glass-card" style="padding:16px;">'
                '<p style="margin:0 0 8px 0;color:#94A3B8;font-size:11px;font-weight:700;letter-spacing:0.8px;">'
                "📊 TOP PREDICTIONS</p>" + bars + "</div>",
                unsafe_allow_html=True,
            )

        if translation_text and detected_sign != NO_SIGN_LABEL:
            st.html(get_tts_html(translation_text, lang))


def _render_static_widgets(info_pl, lang):
    """Renders static (non-streaming) right-panel widgets."""
    langs_map = {
        "en": "English",
        "hi": "Hindi (हिंदी)",
        "te": "Telugu (తెలుగు)",
        "ta": "Tamil (தமிழ்)",
        "kn": "Kannada (ಕನ್ನಡ)",
        "ml": "Malayalam (മലയാളം)",
        "tcy": "Tulu (तुळु)",
    }
    current_language = langs_map.get(lang, lang.upper())

    with info_pl.container():
        st.markdown(
            """
            <div style="display: flex; flex-direction: column; gap: 12px; margin-bottom: 12px;">
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #3B82F6; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">👤 Recognized Person</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #94A3B8; font-weight: 700;">—</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #60A5FA; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">🔍 Face Confidence</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #94A3B8; font-weight: 700;">—</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #10B981; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">🤟 Detected Gesture</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #94A3B8; font-weight: 700;">Awaiting stream...</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #34D399; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">📊 Gesture Confidence</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #94A3B8; font-weight: 700;">—</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #EC4899; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">🎭 Expression</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #94A3B8; font-weight: 700;">—</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #E2E8F0; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">👐 Left Hand</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #94A3B8; font-weight: 700;">—</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #E2E8F0; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">👐 Right Hand</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #94A3B8; font-weight: 700;">—</h3>
                </div>
                <div class="glass-card" style="padding: 18px; border-left: 5px solid #8B5CF6; margin-bottom: 0px;">
                    <p style="margin: 0; font-size: 11px; color: #94A3B8; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase;">💬 Current Translation</p>
                    <h3 style="margin: 6px 0 0 0; font-size: 22px; color: #94A3B8; font-weight: 700;">—</h3>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"## 🌍 {current_language}")
