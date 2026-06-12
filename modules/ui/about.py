import streamlit as st
from modules.locales import t
from modules.translation.translator import OFFLINE_DICTIONARY

def render_about_page(lang="en"):
    """Renders the detailed About page with keyboard shortcuts and dictionary tables."""
    st.title(f"ℹ️ {t('about.title', lang)}")
    
    st.markdown("---")
    
    # Mission glass card
    st.markdown(f"""
    <div class="glass-card">
        <h3>{t('about.missionHeader', lang)}</h3>
        <p style="font-size:16px; color:#E2E8F0; line-height:1.6;">
            {t('about.missionText', lang)}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"⌨️ {t('about.shortcutsTitle', lang)}")
        # Keyboard Shortcuts table
        shortcuts = [
            {"Shortcut": "H / h", "Action": f"Go to {t('nav.home', lang)}"},
            {"Shortcut": "L / l", "Action": f"Go to {t('nav.live', lang)}"},
            {"Shortcut": "C / c", "Action": f"Go to {t('nav.conversations', lang)}"},
            {"Shortcut": "P / p", "Action": f"Go to {t('nav.people', lang)}"},
            {"Shortcut": "S / s", "Action": f"Go to {t('nav.settings', lang)}"},
            {"Shortcut": "A / a", "Action": f"Go to {t('nav.about', lang)}"}
        ]
        st.table(shortcuts)
        
    with col2:
        st.subheader(f"📖 {t('about.dictionaryTitle', lang)}")
        # Generate the supported dictionary table from OFFLINE_DICTIONARY
        dict_data = []
        for key, trans in OFFLINE_DICTIONARY.items():
            dict_data.append({
                "Sign Gesture": key.title(),
                "English Sentence": trans.get("en", ""),
                "Hindi Translation": trans.get("hi", ""),
                "Telugu Translation": trans.get("te", "")
            })
        st.dataframe(dict_data, use_container_width=True)
