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
            {"Shortcut": "H / h", "Action": "Navigate to Home Page"},
            {"Shortcut": "L / l", "Action": "Navigate to Live Translation Page"},
            {"Shortcut": "C / c", "Action": "Navigate to Conversations Log History"},
            {"Shortcut": "P / p", "Action": "Navigate to Known People Profiles"},
            {"Shortcut": "S / s", "Action": "Navigate to System Configuration Settings"},
            {"Shortcut": "A / a", "Action": "Navigate to About SignBridge AI"}
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
