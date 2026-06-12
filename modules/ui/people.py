import streamlit as st
import json
from modules.locales import t
from modules.database import (
    get_all_people,
    delete_person,
    update_person_name,
    export_database_json,
    import_database_json
)

def render_people_page(lang="en"):
    st.title(f"👥 {t('people.title', lang)}")
    st.markdown("---")
    
    people = get_all_people()
    
    if not people:
        st.info("No saved people profiles in the database registry yet. You can enroll faces during Live Translation.")
    else:
        # Display Cards Grid
        for p in people:
            with st.container():
                st.markdown(f"""
                <div class="glass-card">
                    <h3>👤 {p['name']}</h3>
                    <p style="margin: 5px 0; font-size:14px; color:#94A3B8;">
                        <strong>ID:</strong> {p['id']}<br>
                        <strong>{t('people.dateAdded', lang)}:</strong> {p['date_added']}<br>
                        <strong>{t('people.lastSeen', lang)}:</strong> {p['last_seen']}<br>
                        <strong>Notes:</strong> {p.get('notes', 'None')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Action Sub-buttons for this profile
                exp = st.expander(f"Manage Profile: {p['name']}")
                with exp:
                    col_rename, col_del = st.columns(2)
                    
                    with col_rename:
                        new_name = st.text_input(t('people.renameInput', lang), key=f"rename_{p['id']}")
                        if st.button("✏️ Save Name", key=f"btn_rename_{p['id']}", use_container_width=True):
                            if new_name.strip():
                                update_person_name(p["id"], new_name.strip())
                                st.success("Profile updated!")
                                st.rerun()
                            else:
                                st.error("Name cannot be empty.")
                                
                    with col_del:
                        confirm = st.checkbox("I confirm permanent deletion of all data", key=f"check_del_{p['id']}")
                        if st.button("🗑️ " + t('people.btnDelete', lang), key=f"btn_del_{p['id']}", use_container_width=True, disabled=not confirm):
                            delete_person(p["id"])
                            st.success("Profile deleted successfully!")
                            st.rerun()
                            
            st.markdown("<br>", unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader(f"📂 {t('people.importExport', lang)}")
    
    col_bak1, col_bak2 = st.columns(2)
    
    with col_bak1:
        st.markdown("#### Database Export")
        db_data = export_database_json()
        st.download_button(
            label="📥 " + t('people.btnExportDb', lang),
            data=db_data,
            file_name="signbridge_database_backup.json",
            mime="application/json",
            use_container_width=True
        )
        
    with col_bak2:
        st.markdown("#### Database Import")
        uploaded_backup = st.file_uploader(t('people.btnImportDb', lang), type=["json"])
        if uploaded_backup:
            if st.button("⚙️ " + t('people.btnRestore', lang), use_container_width=True):
                try:
                    backup_content = uploaded_backup.getvalue().decode("utf-8")
                    import_database_json(backup_content)
                    st.success("Database backup successfully restored!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error restoring backup: {e}")
