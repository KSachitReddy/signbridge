import streamlit as st
import json
from modules.locales import t
from modules.database import get_setting, save_setting, get_all_people
from modules.ollama import list_installed_models, download_model, delete_model
from modules.providers import get_provider_key, save_provider_key, validate_key_format
from modules.signs import (
    VOCABULARY,
    record_sign_sample,
    get_recorded_samples,
    delete_recorded_sample,
    retrain_sign_model
)

def render_settings_page(lang="en"):
    st.title(f"⚙️ {t('settings.title', lang)}")
    st.markdown("---")
    
    tab_global, tab_ollama, tab_dataset = st.tabs(["Global & Keys", "Ollama Models", "ISL Dataset & Training"])
    
    # --- Tab 1: Global & Keys Config ---
    with tab_global:
        st.subheader("Global App Configuration")
        
        # 1. Target Language
        selected_lang = st.selectbox(
            t('settings.language', lang),
            ["English", "Hindi", "Telugu"],
            index=0 if lang == "en" else 1 if lang == "hi" else 2
        )
        lang_code = "en" if selected_lang == "English" else "hi" if selected_lang == "Hindi" else "te"
        
        # 2. Camera Selection
        cam_source = st.selectbox(
            t('settings.camera', lang),
            ["Simulated Camera Source", "Webcam Hardware Feed"]
        )
        save_setting("camera_source", cam_source)
        
        # 3. AI Provider Selection
        active_provider = st.selectbox(
            t('settings.aiProvider', lang),
            ["None (Offline Dictionary)", "Ollama", "OpenAI", "Gemini", "Anthropic"],
            index=["None (Offline Dictionary)", "Ollama", "OpenAI", "Gemini", "Anthropic"].index(
                get_setting("ai_provider", "None (Offline Dictionary)")
            )
        )
        save_setting("ai_provider", active_provider)
        
        # Keys input for BYOK
        if active_provider in ["OpenAI", "Gemini", "Anthropic"]:
            current_key = get_provider_key(active_provider)
            input_key = st.text_input(f"Bring Your Own Key (BYOK) - {active_provider} API Key", value=current_key, type="password")
            
            if st.button("Validate & Save API Key"):
                if validate_key_format(active_provider, input_key):
                    save_provider_key(active_provider, input_key)
                    st.success("API Key successfully encrypted and stored.")
                else:
                    st.error("Invalid API Key format pattern.")
                    
        # 4. Accessibility Settings (High Contrast, Large Text)
        theme_mode = st.selectbox(
            t('settings.theme', lang),
            ["Standard Dark Theme", "High Contrast Dark Theme", "Large Text Mode"]
        )
        save_setting("visual_theme", theme_mode)
        
        # Apply style injections
        if theme_mode == "High Contrast Dark Theme":
            st.markdown("<style>.main { background-color: #000000 !important; color: #FFFFFF !important; } .glass-card { border: 2px solid #FFFFFF !important; }</style>", unsafe_allow_html=True)
        elif theme_mode == "Large Text Mode":
            st.markdown("<style>body, p, button, span, label, select, input { font-size: 20px !important; }</style>", unsafe_allow_html=True)
            
        # 5. Database Settings
        st.markdown(f"#### {t('settings.dbMgmt', lang)}")
        if st.button("🧹 Clear All Telemetry Data"):
            from modules.database import delete_all_conversations
            delete_all_conversations()
            st.success("Cleaned database tables successfully.")
            
        if st.button("Save System Settings", use_container_width=True):
            save_setting("ui_language", lang_code)
            st.success(t('settings.saveSuccess', lang))
            st.rerun()

    # --- Tab 2: Ollama Models Config ---
    with tab_ollama:
        st.subheader(t('settings.modelsHeader', lang))
        
        # Configure endpoint
        current_endpoint = get_setting("ollama_endpoint", "http://localhost:11434")
        input_endpoint = st.text_input("Ollama Base Endpoint URL", value=current_endpoint)
        if input_endpoint != current_endpoint:
            save_setting("ollama_endpoint", input_endpoint)
            
        # List Catalog & status
        models = list_installed_models()
        
        st.markdown("#### Installed & Available Models")
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
                    
        # Active Model select
        installed_names = [m["name"] for m in models if m["status"] == "Installed"]
        if installed_names:
            active_model = st.selectbox(
                t('settings.activeModel', lang),
                installed_names,
                index=installed_names.index(get_setting("ollama_model", installed_names[0])) if get_setting("ollama_model", "") in installed_names else 0
            )
            save_setting("ollama_model", active_model)

    # --- Tab 3: Dataset & Training Config ---
    with tab_dataset:
        st.subheader(t('settings.datasetsHeader', lang))
        
        # 1. Record Sign landmarks sample simulation
        st.markdown(f"#### {t('settings.recordSampleTitle', lang)}")
        
        people = get_all_people()
        people_options = {"Anonymous": "Unknown"}
        for p in people:
            people_options[p["name"]] = p["id"]
        rec_person = st.selectbox("Assign Sample to Profile", list(people_options.keys()))
        rec_person_id = people_options[rec_person]
        
        rec_label = st.selectbox(t('settings.labelInput', lang), VOCABULARY)
        
        # Record landmarks sample sequence
        if st.button(t('settings.btnRecord', lang)):
            buffer_to_record = st.session_state.get("sequence_buffer", None)
            if buffer_to_record and len(buffer_to_record.buffer) > 0:
                success, msg = record_sign_sample(rec_person_id, rec_label, buffer_to_record)
            else:
                # Fallback to realistic mock sequence
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
                
        # 2. View/Delete Recorded Sign datasets
        samples = get_recorded_samples()
        st.markdown(f"#### Saved Dataset Samples ({len(samples)})")
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
        else:
            st.caption("No custom sign samples recorded yet.")
            
        # 3. Choose active model architecture
        active_arch = st.selectbox(
            t('settings.modelVersions', lang),
            ["Phase 1: Random Forest", "Phase 2: LSTM", "Phase 3: Transformer"],
            index=["Phase 1: Random Forest", "Phase 2: LSTM", "Phase 3: Transformer"].index(
                get_setting("model_architecture", "Phase 1: Random Forest")
            )
        )
        save_setting("model_architecture", active_arch)
        
        # Retrain Trigger
        st.markdown("#### Model Re-Training Pipeline")
        if st.button(t('settings.btnRetrain', lang)):
            success, report = retrain_sign_model()
            if success:
                st.success(f"Model Retraining Complete! Version updated to v{report['new_version']}")
                st.json(report)
            else:
                st.error(report)
