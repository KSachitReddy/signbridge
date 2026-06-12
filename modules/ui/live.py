"""
live.py — Real-time ISL sign translation page for SignBridge AI.

Pipeline:
  Camera → MediaPipe Holistic → Landmarks → RF Classifier → Translation → TTS + Ollama
  Speech → STT → Display → Log
"""

import streamlit as st
import cv2
import numpy as np
import time
import streamlit.components.v1 as _stv1  # kept only as compatibility shim

from modules.locales import t
from modules.camera import generate_mock_frame
from modules.pose.holistic import track_and_draw_pose
from modules.hands.landmarks import track_and_draw_hands
from modules.face.face_ai import validate_and_enroll_face, recognize_multiple_faces
from modules.signs.recognizer import SignSequenceBuffer, sign_classifier
from modules.translation import translate_sign
from modules.speech import get_tts_html, render_stt_listener
from modules.database import add_conversation, get_conversations, get_all_people
from modules.ollama import generate_response


# ── Session state initializer ─────────────────────────────────────────────────
def _init_state():
    defaults = {
        "sequence_buffer": SignSequenceBuffer(size=20),
        "last_logged_sign": "",
        "last_sign_time": 0.0,
        "current_results": None,
        "ai_response": "",
        "enroll_pending": False,
        "enroll_name_input": "",
        "enroll_frame_snap": None,
        "fps_counter": 0,
        "fps_last_time": time.time(),
        "fps_display": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_live_page(lang="en"):
    _init_state()

    st.title(f"📹 {t('nav.live', lang)}")
    st.markdown("---")

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
        ai_response_placeholder = st.empty()
        stt_placeholder = st.empty()
        history_placeholder = st.empty()

    # ── Enrollment panel (shown below camera) ─────────────────────────────────
    enroll_placeholder = st.empty()

    if st.session_state.enroll_pending:
        st.warning("⏸ Live stream paused for face enrollment.")
        snap = st.session_state.get("enroll_frame_snap")
        if snap is not None:
            img_rgb = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(img_rgb, use_container_width=True)
        with enroll_placeholder.container():
            _render_enrollment_panel(lang)
        _render_static_widgets(info_placeholder, ai_response_placeholder,
                               stt_placeholder, history_placeholder, lang)
        return

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

            if not cap.isOpened():
                st.error("❌ Could not open webcam. Check camera permissions.")
                return

            try:
                while st.session_state.get("live_toggle", False):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Unable to retrieve webcam frame.")
                        break

                    # ── FPS tracking ──────────────────────────────────────────
                    st.session_state.fps_counter += 1
                    now = time.time()
                    elapsed = now - st.session_state.fps_last_time
                    if elapsed >= 1.0:
                        st.session_state.fps_display = round(
                            st.session_state.fps_counter / elapsed, 1
                        )
                        st.session_state.fps_counter = 0
                        st.session_state.fps_last_time = now

                    # ── MediaPipe Holistic pipeline ───────────────────────────
                    # NOTE: track_and_draw_pose sets st.session_state.current_results
                    # which track_and_draw_hands reads to extract hand landmarks.
                    pose_joints, frame = track_and_draw_pose(frame)
                    hands_data, frame = track_and_draw_hands(frame)
                    faces_results, frame = recognize_multiple_faces(frame)

                    # ── Sequence buffer ───────────────────────────────────────
                    st.session_state.sequence_buffer.add(
                        left_hand=hands_data["left"],
                        right_hand=hands_data["right"],
                        pose=pose_joints
                    )

                    # ── Sign prediction ───────────────────────────────────────
                    top_preds = sign_classifier.predict(st.session_state.sequence_buffer)
                    detected_sign = top_preds[0][0]
                    score = top_preds[0][1]
                    translation_text = translate_sign(detected_sign, lang)

                    # ── Log & AI response (throttled: 2s gap, 65% confidence) ─
                    now_t = time.time()
                    should_log = (
                        detected_sign not in ("None", "")
                        and score > 0.65
                        and detected_sign != st.session_state.last_logged_sign
                        and (now_t - st.session_state.last_sign_time) > 2.0
                    )
                    if should_log:
                        person_id = faces_results[0]["person_id"] if faces_results else "Unknown"
                        person_name = faces_results[0]["name"] if faces_results else "Unknown"
                        add_conversation(person_id, detected_sign, translation_text, lang, score)

                        # Generate AI response (non-blocking call)
                        ai_resp = generate_response(detected_sign, translation_text, person_name, lang)
                        st.session_state.ai_response = ai_resp
                        if ai_resp:
                            add_conversation("System", "[AI Response]", ai_resp, lang, 1.0)

                        st.session_state.last_logged_sign = detected_sign
                        st.session_state.last_sign_time = now_t

                    # ── Unknown face enrollment trigger ───────────────────────
                    has_unknown = any(r["name"] == "Unknown" for r in faces_results)
                    if has_unknown and not st.session_state.enroll_pending:
                        st.session_state.enroll_pending = True
                        st.session_state.enroll_frame_snap = frame.copy()
                        st.rerun()

                    # ── FPS overlay on frame ──────────────────────────────────
                    fps_text = f"FPS: {st.session_state.fps_display}"
                    cv2.putText(frame, fps_text, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

                    # ── Display frame ─────────────────────────────────────────
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(img_rgb, use_container_width=True)

                    # ── Right panel updates ───────────────────────────────────
                    _render_info_panel(
                        info_placeholder, ai_response_placeholder,
                        faces_results, detected_sign, score,
                        translation_text, top_preds, lang
                    )

                    with stt_placeholder.container():
                        st.markdown(f"### {t('live.twoWayHeader', lang)}")
                        render_stt_listener(lang)

                    with history_placeholder.container():
                        _render_chat_history(lang)

                    # ── Enrollment panel ──────────────────────────────────────
                    # Handled outside loop to avoid duplicate element keys

                    time.sleep(0.04)  # ~25 FPS cap

            finally:
                cap.release()

        else:
            frame_placeholder.info(
                "▶ Toggle **'Start Live Stream'** above to begin real-time ISL translation."
            )
            _render_static_widgets(info_placeholder, ai_response_placeholder,
                                   stt_placeholder, history_placeholder, lang)

    # ═══════════════════════════════════════════════════════════════════════════
    # SIMULATED FEED MODE
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        frame = generate_mock_frame("SignBridge AI — Simulated Feed")
        pose_joints, frame = track_and_draw_pose(frame, use_mock=True)
        hands_data, frame = track_and_draw_hands(frame, use_mock=True)
        faces_results, frame = recognize_multiple_faces(frame)

        st.session_state.sequence_buffer.add(
            left_hand=hands_data["left"],
            right_hand=hands_data["right"],
            pose=pose_joints
        )

        top_preds = sign_classifier.predict(st.session_state.sequence_buffer)
        detected_sign = top_preds[0][0]
        score = top_preds[0][1]
        translation_text = translate_sign(detected_sign, lang)

        if detected_sign not in ("None", ""):
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

        if top_preds and top_preds[0][0] != "None":
            st.markdown("**Top 3 Predictions:**")
            for lbl, conf in top_preds:
                if lbl == "None":
                    continue
                c1, c2 = st.columns([2, 5])
                c1.write(f"**{lbl}**")
                c2.progress(float(conf))

        if translation_text and detected_sign != "None":
            st.markdown("#### 🔊 Voice Output")
            st.html(get_tts_html(translation_text, lang))

    # AI Response panel
    with ai_pl.container():
        ai_resp = st.session_state.get("ai_response", "")
        if ai_resp:
            st.markdown("""
            <div style="padding:12px;border-radius:10px;
                        background:rgba(16,185,129,0.12);
                        border-left:4px solid #10B981;margin-top:8px;">
                <span style="color:#10B981;font-weight:600;">🤖 AI Response</span><br>
            """ + ai_resp + "</div>", unsafe_allow_html=True)


def _render_enrollment_panel(lang):
    """Shows inline face enrollment panel when an unknown face is detected."""
    st.markdown("""
    <div style="padding:16px;border-radius:12px;
                background:rgba(245,158,11,0.12);
                border:1px solid rgba(245,158,11,0.4);margin:12px 0;">
        <span style="color:#F59E0B;font-weight:700;">👤 Unknown Person Detected</span>
    </div>
    """, unsafe_allow_html=True)

    col_name, col_btn, col_skip = st.columns([3, 2, 1])
    name_in = col_name.text_input("Enter name to enroll:", key="enroll_name_field",
                                   placeholder="e.g. Rahul")
    enroll_clicked = col_btn.button("✅ Enroll Face", use_container_width=True,
                                     key="enroll_btn")
    skip_clicked = col_skip.button("✕ Skip", key="enroll_skip_btn")

    if skip_clicked:
        st.session_state.enroll_pending = False
        st.session_state.enroll_frame_snap = None
        st.rerun()
 
    if enroll_clicked and name_in.strip():
        snap = st.session_state.get("enroll_frame_snap")
        if snap is not None:
            result = validate_and_enroll_face(snap, name_in.strip())
            if result["status"] == "success":
                st.success(result["message"])
                st.session_state.enroll_pending = False
                st.session_state.enroll_frame_snap = None
                st.session_state.last_logged_sign = ""  # reset to re-log with new name
                st.rerun()
            else:
                st.error(result["message"])
        else:
            st.warning("No frame snapshot available. Please try again.")


def _render_chat_history(lang):
    """Renders two-way conversation history as a chat-style log."""
    st.markdown("#### 💬 Conversation History")
    logs = get_conversations()[:10]
    people = get_all_people()
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
