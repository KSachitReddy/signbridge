import streamlit as st
import pandas as pd
from modules.locales import t
from modules.database import get_conversations, delete_conversation, delete_all_conversations, get_all_people

def render_conversations_page(lang="en"):
    st.title(f"💬 {t('conversations.title', lang)}")
    st.markdown("---")
    
    # Filter Inputs
    st.markdown("### 🔍 Filter History")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        people = get_all_people()
        people_options = {"All": None}
        for p in people:
            people_options[p["name"]] = p["id"]
        selected_person = st.selectbox(t('conversations.filterPerson', lang), list(people_options.keys()))
        person_id = people_options[selected_person]
        
    with col2:
        date_filter = st.text_input(t('conversations.filterDate', lang), placeholder="YYYY-MM-DD")
        
    with col3:
        lang_filter = st.selectbox(t('conversations.filterLang', lang), ["All", "en", "hi", "te"])
        lang_val = None if lang_filter == "All" else lang_filter
        
    # Query Database
    logs = get_conversations(person_id=person_id, date_filter=date_filter, lang_filter=lang_val)
    
    if not logs:
        st.info(t('conversations.emptyLogs', lang))
    else:
        # Map person names for display
        people_map = {p["id"]: p["name"] for p in people}
        people_map["Unknown"] = "Unknown"
        people_map[""] = "Unknown"
        
        display_logs = []
        for l in logs:
            display_logs.append({
                "ID": l["id"],
                "Person": people_map.get(l["person_id"], "Unknown"),
                "Timestamp": l["timestamp"],
                "Sign": l["recognized_sign"],
                "Translation": l["translated_text"],
                "Language": l["language"].upper(),
                "Confidence": f"{int(l['confidence']*100)}%"
            })
            
        df = pd.DataFrame(display_logs)
        st.dataframe(df, use_container_width=True)
        
        # Actions Panel
        st.markdown("### 🛠️ Actions")
        col_act1, col_act2, col_act3 = st.columns(3)
        
        with col_act1:
            log_to_delete = st.number_input("Select Log ID to Delete", min_value=1, step=1)
            if st.button("❌ " + t('conversations.btnDeleteSelected', lang), use_container_width=True):
                # Verify if ID exists
                if any(l["id"] == log_to_delete for l in logs):
                    delete_conversation(log_to_delete)
                    st.success(f"Log ID {log_to_delete} deleted successfully!")
                    st.rerun()
                else:
                    st.error("Log ID not found.")
                    
        with col_act2:
            st.markdown("**Danger Zone**")
            confirm = st.checkbox("I confirm I want to wipe all conversation logs")
            if st.button("🗑️ " + t('conversations.btnDeleteAll', lang), use_container_width=True, disabled=not confirm):
                delete_all_conversations()
                st.success("All conversation history successfully cleared!")
                st.rerun()
                
        with col_act3:
            st.markdown("**Export Logs**")
            export_data = pd.DataFrame(logs).to_json(orient="records", indent=2)
            st.download_button(
                label="📥 " + t('conversations.btnExport', lang),
                data=export_data,
                file_name="conversations_backup.json",
                mime="application/json",
                use_container_width=True
            )
