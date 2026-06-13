import streamlit as st
import json
import requests
import cv2
from modules.locales import t
from modules.database import get_setting, save_setting, get_all_people, get_db_connection
from modules.ollama import list_installed_models, download_model, delete_model
from modules.providers import get_provider_key, save_provider_key, validate_key_format, test_connection
from modules.signs import (
    VOCABULARY,
    record_sign_sample,
    get_recorded_samples,
    delete_recorded_sample,
    retrain_sign_model
)
from modules.ollama.manage import get_ollama_endpoint

def render_settings_page(lang="en"):
    st.title(f"⚙️ {t('settings.title', lang)}")
    st.markdown("---")
    
    # 6 Specialized Tabs
    tab_general, tab_langs, tab_camera, tab_models, tab_dataset, tab_system = st.tabs([
        "⚙️ General", 
        "🌍 Languages", 
        "📷 Camera", 
        "🤖 AI Models", 
        "📊 Dataset",
        "💻 System"
    ])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Tab 1: General Settings
    # ─────────────────────────────────────────────────────────────────────────
    with tab_general:
        st.subheader("Global Settings & UI Theme")
        
        theme_mode = st.selectbox(
            t('settings.theme', lang),
            ["Standard Dark Theme", "High Contrast Dark Theme", "Large Text Mode"],
            index=["Standard Dark Theme", "High Contrast Dark Theme", "Large Text Mode"].index(
                get_setting("visual_theme", "Standard Dark Theme")
            )
        )
        save_setting("visual_theme", theme_mode)
        
        # Apply theme-specific adjustments
        if theme_mode == "High Contrast Dark Theme":
            st.markdown("<style>.main { background-color: #000000 !important; color: #FFFFFF !important; } .glass-card { border: 2px solid #FFFFFF !important; }</style>", unsafe_allow_html=True)
        elif theme_mode == "Large Text Mode":
            st.markdown("<style>body, p, button, span, label, select, input { font-size: 20px !important; }</style>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 2: Languages
    # ─────────────────────────────────────────────────────────────────────────
    with tab_langs:
        st.subheader("Language Configurations")
        
        langs_map = {
            "en": "English",
            "hi": "Hindi (हिंदी)",
            "te": "Telugu (తెలుగు)",
            "ta": "Tamil (தமிழ்)",
            "kn": "Kannada (ಕನ್ನಡ)",
            "ml": "Malayalam (മലയാളം)",
            "tcy": "Tulu (ತುಳು)"
        }
        
        selected_lang_name = st.selectbox(
            t('settings.language', lang),
            list(langs_map.values()),
            index=list(langs_map.keys()).index(lang) if lang in langs_map else 0
        )
        lang_code = [k for k, v in langs_map.items() if v == selected_lang_name][0]
        
        if st.button("💾 Apply Language Configuration", use_container_width=True):
            save_setting("ui_language", lang_code)
            st.success(t('settings.saveSuccess', lang))
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 3: Camera
    # ─────────────────────────────────────────────────────────────────────────
    with tab_camera:
        st.subheader("Camera Input Selection & Overlays")
        
        cam_source = st.selectbox(
            t('settings.camera', lang),
            ["Webcam Hardware Feed", "Simulated Camera Source"],
            index=0 if get_setting("camera_source", "Webcam Hardware Feed") == "Webcam Hardware Feed" else 1
        )
        save_setting("camera_source", cam_source)
        
        st.markdown("---")
        st.subheader("🛠️ Overlay Visualizers")
        st.caption("Toggle overlays to customize render performance:")
        
        show_mesh_val = st.checkbox("Show Face Mesh Overlay", value=st.session_state.get("show_mesh", True))
        show_ids_val = st.checkbox("Show Face Landmark Numerical IDs", value=st.session_state.get("show_ids", False))
        show_bbox_val = st.checkbox("Show Face Bounding Boxes", value=st.session_state.get("show_bbox", True))
        show_hands_val = st.checkbox("Show Hand Joint Skeletons", value=st.session_state.get("show_hands", True))
        
        # Sync immediately with st.session_state
        st.session_state.show_mesh = show_mesh_val
        st.session_state.show_ids = show_ids_val
        st.session_state.show_bbox = show_bbox_val
        st.session_state.show_hands = show_hands_val

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 4: AI Models
    # ─────────────────────────────────────────────────────────────────────────
    with tab_models:
        st.subheader("AI Models & Providers Selection")
        
        active_provider = st.selectbox(
            t('settings.aiProvider', lang),
            ["None (Offline Dictionary)", "Ollama", "OpenAI", "Gemini", "Anthropic"],
            index=["None (Offline Dictionary)", "Ollama", "OpenAI", "Gemini", "Anthropic"].index(
                get_setting("ai_provider", "None (Offline Dictionary)")
            )
        )
        save_setting("ai_provider", active_provider)
        
        # OpenAI, Gemini, Anthropic API Keys Configuration
        if active_provider in ["OpenAI", "Gemini", "Anthropic"]:
            current_key = get_provider_key(active_provider)
            input_key = st.text_input(
                f"Bring Your Own Key (BYOK) — {active_provider} API Key",
                value=current_key,
                type="password",
            )

            col_save_key, col_test_key = st.columns(2)
            if col_save_key.button("💾 Validate & Save API Key", use_container_width=True):
                if validate_key_format(active_provider, input_key):
                    save_provider_key(active_provider, input_key)
                    st.success("API Key encrypted and stored.")
                else:
                    st.error("Invalid API Key format — check key prefix.")

            if col_test_key.button("🔌 Test Connection", use_container_width=True):
                key_to_test = input_key or current_key
                if not key_to_test:
                    st.warning("Enter an API key before testing.")
                else:
                    with st.spinner(f"Testing {active_provider} connection…"):
                        ok, msg = test_connection(active_provider, key_to_test)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                        
        elif active_provider == "Ollama":
            st.markdown(f"#### {t('settings.modelsHeader', lang)}")
            
            # Configure endpoint
            current_endpoint = get_setting("ollama_endpoint", "http://localhost:11434")
            input_endpoint = st.text_input("Ollama Base Endpoint URL", value=current_endpoint)
            if input_endpoint != current_endpoint:
                save_setting("ollama_endpoint", input_endpoint)
                
            # List catalog & download
            models = list_installed_models()
            
            st.markdown("##### Installed & Available Models")
            for m in models:
                col_m1, col_m2, col_m3 = st.columns([3, 2, 2])
                col_m1.write(f"**{m['name']}** ({m['size']})")
                col_m2.info(m["status"])
                
                if m["status"] == "Installed":
                    if col_m3.button(t('settings.btnDeleteModel', lang), key=f"del_{m['name']}"):
                        success, msg = delete_model(m["name"])
                        st.toast(msg)
                        st.rerun()
                else:
                    if col_m3.button(t('settings.downloadModel', lang), key=f"pull_{m['name']}"):
                        success, msg = download_model(m["name"])
                        st.toast(msg)
                        st.rerun()
                        
            # Active model selector
            installed_names = [m["name"] for m in models if m["status"] == "Installed"]
            if installed_names:
                active_model = st.selectbox(
                    t('settings.activeModel', lang),
                    installed_names,
                    index=installed_names.index(get_setting("ollama_model", installed_names[0])) if get_setting("ollama_model", "") in installed_names else 0
                )
                save_setting("ollama_model", active_model)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 5: Dataset
    # ─────────────────────────────────────────────────────────────────────────
    with tab_dataset:
        st.subheader(t('settings.datasetsHeader', lang))
        
        # Custom sign landmarks recording
        st.markdown(f"##### {t('settings.recordSampleTitle', lang)}")
        
        people = get_all_people()
        people_options = {"Anonymous": "Unknown"}
        for p in people:
            people_options[p["name"]] = p["id"]
        rec_person = st.selectbox("Assign Sample to Profile", list(people_options.keys()))
        rec_person_id = people_options[rec_person]
        
        rec_label = st.selectbox(t('settings.labelInput', lang), VOCABULARY)
        
        # Record sample
        if st.button(t('settings.btnRecord', lang)):
            buffer_to_record = st.session_state.get("sequence_buffer", None)
            if buffer_to_record and len(buffer_to_record.buffer) > 0:
                success, msg = record_sign_sample(rec_person_id, rec_label, buffer_to_record)
            else:
                # Mock fallback
                from modules.signs import SignSequenceBuffer
                mock_buffer = SignSequenceBuffer(size=15)
                for i in range(15):
                    mock_buffer.add(
                        [{"x": 0.5, "y": 0.5 - i*0.01, "z": 0.0}] * 21,
                        [{"x": 0.5, "y": 0.5 - i*0.01, "z": 0.0}] * 21,
                        {
                            "left_shoulder": {"x": 0.38, "y": 0.52, "z": 0.0},
                            "right_shoulder": {"x": 0.62, "y": 0.52, "z": 0.0},
                            "left_elbow": {"x": 0.31, "y": 0.71, "z": 0.0},
                            "right_elbow": {"x": 0.69, "y": 0.71, "z": 0.0},
                            "left_wrist": {"x": 0.28, "y": 0.83, "z": 0.0},
                            "right_wrist": {"x": 0.72, "y": 0.83, "z": 0.0}
                        }
                    )
                success, msg = record_sign_sample(rec_person_id, rec_label, mock_buffer)
            if success:
                st.success(msg)
            else:
                st.error(msg)
                
        # Dataset view table
        samples = get_recorded_samples()
        st.markdown(f"##### Custom Saved Dataset Samples ({len(samples)})")
        if samples:
            display_samples = []
            for s in samples:
                display_samples.append({
                    "Sample ID": s["id"],
                    "Timestamp": s["timestamp"],
                    "Label": s["sign_label"],
                    "Person ID": s["person_id"],
                    "Version": s["model_version"]
                })
            st.dataframe(display_samples, use_container_width=True)
            
            del_sample_id = st.number_input("Enter Sample ID to Delete", min_value=1, step=1)
            if st.button("Delete Sign Sample"):
                if any(s["id"] == del_sample_id for s in samples):
                    delete_recorded_sample(del_sample_id)
                    st.success(f"Sample {del_sample_id} deleted successfully.")
                    st.rerun()
                else:
                    st.error("Sample ID not found.")
                    
        # Active model selector & retrain
        active_arch = st.selectbox(
            t('settings.modelVersions', lang),
            ["Phase 1: Random Forest", "Phase 2: LSTM", "Phase 3: Transformer"],
            index=["Phase 1: Random Forest", "Phase 2: LSTM", "Phase 3: Transformer"].index(
                get_setting("model_architecture", "Phase 1: Random Forest")
            )
        )
        save_setting("model_architecture", active_arch)
        
        st.markdown("##### Model Re-Training Pipeline")
        if st.button(t('settings.btnRetrain', lang)):
            success, report = retrain_sign_model()
            if success:
                st.success(f"Model Retraining Complete! Version updated to v{report['new_version']}")
                st.json(report)
            else:
                st.error(report)

    # ─────────────────────────────────────────────────────────────────────────
    # Tab 6: System Settings (Diagnostics & DB clearing)
    # ─────────────────────────────────────────────────────────────────────────
    with tab_system:
        st.subheader("System Diagnostics & Database Maintenance")
        
        # System Diagnostics Checklist
        st.markdown("#### 🛠️ Diagnostics Health Check")
        webcam_test = False
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                webcam_test = True
                cap.release()
        except Exception:
            pass
            
        ollama_test = False
        try:
            res = requests.get(f"{get_ollama_endpoint()}/api/tags", timeout=0.8)
            ollama_test = res.status_code == 200
        except Exception:
            pass
            
        db_test = False
        try:
            conn = get_db_connection()
            conn.execute("SELECT 1")
            conn.close()
            db_test = True
        except Exception:
            pass
            
        st.markdown(f"- Webcam hardware status: **{'🟢 Connected' if webcam_test else '🔴 Offline/Occupied'}**")
        st.markdown(f"- Local Ollama Inference Server: **{'🟢 Online' if ollama_test else '🔴 Offline'}**")
        st.markdown(f"- Database integrity check: **{'🟢 Healthy' if db_test else '🔴 Error'}**")
        
        st.markdown("---")
        st.markdown(f"#### 🧹 {t('settings.dbMgmt', lang)}")
        st.caption("Clean historical communications data and enrolled profiles:")
        
        col_clear_l, col_clear_r = st.columns(2)
        with col_clear_l:
            if st.button("🧹 Clear Conversation Logs Only", use_container_width=True, key="sys_clear_convs"):
                from modules.database import delete_all_conversations
                delete_all_conversations()
                st.success("Wiped conversations successfully.")
                
        with col_clear_r:
            if st.button("👥 Wipe Enrolled Profiles Registry", use_container_width=True, key="sys_wipe_profiles"):
                try:
                    conn = get_db_connection()
                    conn.execute("DELETE FROM people")
                    conn.execute("DELETE FROM face_vectors")
                    conn.commit()
                    conn.close()
                    st.success("Wiped all enrolled people registry successfully.")
                except Exception as e:
                    st.error(f"Error: {e}")
