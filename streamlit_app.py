import streamlit as st
from modules.locales import t
from modules.database import get_setting, save_setting
from modules.ui import (
    render_home_page,
    render_live_page,
    render_conversations_page,
    render_people_page,
    render_settings_page,
    render_about_page
)
from modules.signs import initialize_default_dataset_if_empty

# Initialize database synthetic samples and train classifier if empty
initialize_default_dataset_if_empty()

# 1. Initialize settings from DB
lang = get_setting("ui_language", "en")
if "lang" not in st.session_state:
    st.session_state.lang = lang
else:
    # Keep session state in sync with database if changed
    st.session_state.lang = lang

# Synchronize tab and page session states
if "active_tab" in st.session_state and st.session_state.active_tab != st.session_state.get("active_page"):
    st.session_state.active_page = st.session_state.active_tab
elif "active_page" in st.session_state:
    st.session_state.active_tab = st.session_state.active_page
else:
    st.session_state.active_page = "Home"
    st.session_state.active_tab = "Home"

# 2. Page Configuration
st.set_page_config(
    page_title="SignBridge AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. Apply custom dark theme & accessibility stylesheets
theme_mode = get_setting("visual_theme", "Standard Dark Theme")

css_style = """
<style>
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: border-color 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.2);
    }
</style>
"""

if theme_mode == "High Contrast Dark Theme":
    css_style += """
    <style>
        .main {
            background-color: #000000 !important;
            color: #FFFFFF !important;
        }
        .glass-card {
            background: #000000 !important;
            border: 2px solid #FFFFFF !important;
            border-radius: 8px !important;
        }
        .stButton>button {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 2px solid #FFFFFF !important;
            border-radius: 4px !important;
        }
    </style>
    """
elif theme_mode == "Large Text Mode":
    css_style += """
    <style>
        body, p, button, span, label, select, input, table, td, th, h1, h2, h3, h4, h5, h6 {
            font-size: 1.15rem !important;
        }
        h1 { font-size: 2.2rem !important; }
        h2 { font-size: 1.8rem !important; }
        h3 { font-size: 1.5rem !important; }
        .stButton>button {
            font-size: 1.15rem !important;
        }
    </style>
    """

st.markdown(css_style, unsafe_allow_html=True)

# Warm up ML models in a background thread after page config is set.
# st.cache_resource must not be called before st.set_page_config().
@st.cache_resource
def _preload_ml_models():
    from modules.face.face_ai import preload_models
    preload_models()
    return True

_preload_ml_models()

# Inject accessibility keyboard shortcuts listener
st.html("""
<script>
    const parentDoc = window.parent.document;
    if (!parentDoc._shortcut_listener_added) {
        parentDoc._shortcut_listener_added = true;
        parentDoc.addEventListener('keydown', (e) => {
            const active = parentDoc.activeElement;
            if (active && (
                active.tagName === 'INPUT' || 
                active.tagName === 'TEXTAREA' || 
                active.isContentEditable ||
                active.closest('input') ||
                active.closest('textarea')
            )) {
                return;
            }
            const key = e.key.toLowerCase();
            let index = -1;
            if (key === 'h') index = 0;
            else if (key === 'l') index = 1;
            else if (key === 'c') index = 2;
            else if (key === 'p') index = 3;
            else if (key === 's') index = 4;
            else if (key === 'a') index = 5;
            
            if (index !== -1) {
                const radioContainer = parentDoc.querySelector('div[role="radiogroup"]');
                if (radioContainer) {
                    const labels = radioContainer.querySelectorAll('label');
                    if (labels && labels.length > index) {
                        labels[index].click();
                    }
                }
            }
        });
    }
</script>
""")

# 4. Define pages and navigation mappings
pages = {
    "Home": render_home_page,
    "Live Translation": render_live_page,
    "Conversations": render_conversations_page,
    "People": render_people_page,
    "Settings": render_settings_page,
    "About": render_about_page
}

# Translate nav labels dynamically
nav_labels = {
    "Home": t("nav.home", st.session_state.lang),
    "Live Translation": t("nav.live", st.session_state.lang),
    "Conversations": t("nav.conversations", st.session_state.lang),
    "People": t("nav.people", st.session_state.lang),
    "Settings": t("nav.settings", st.session_state.lang),
    "About": t("nav.about", st.session_state.lang)
}

# Dynamic inverse mapping to translate page label to keys
label_to_key = {v: k for k, v in nav_labels.items()}

# 5. Sidebar Layout
st.sidebar.title("🤟 SignBridge AI")
st.sidebar.caption("ISL Platform")

# Target Quick Language Selector in Sidebar
st.sidebar.markdown("---")
_lang_labels = {
    "en": "🌐 Language", "hi": "🌐 भाषा",   "te": "🌐 భాష",
    "ta": "🌐 மொழி",     "kn": "🌐 ಭಾಷೆ",   "ml": "🌐 ഭാഷ",   "tcy": "🌐 ಭಾಸೆ",
}
sidebar_lang_title = _lang_labels.get(st.session_state.lang, "🌐 Language")
selected_sidebar_lang = st.sidebar.selectbox(
    sidebar_lang_title,
    ["English", "Hindi", "Telugu", "Tamil", "Kannada", "Malayalam", "Tulu"],
    index=["en", "hi", "te", "ta", "kn", "ml", "tcy"].index(st.session_state.lang) if st.session_state.lang in ["en", "hi", "te", "ta", "kn", "ml", "tcy"] else 0
)
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Tulu": "tcy"
}
sidebar_lang_code = lang_map[selected_sidebar_lang]

if sidebar_lang_code != st.session_state.lang:
    save_setting("ui_language", sidebar_lang_code)
    st.session_state.lang = sidebar_lang_code
    st.rerun()

st.sidebar.markdown("---")

_nav_section_labels = {
    "en": "Navigation",  "hi": "नेविगेशन",    "te": "నావిగేషన్",
    "ta": "வழிசெலுத்தல்", "kn": "ನ್ಯಾವಿಗೇಶನ್", "ml": "നാവിഗേഷൻ", "tcy": "ನ್ಯಾವಿಗೇಶನ್",
}
# Navigation radio list
selected_label = st.sidebar.radio(
    _nav_section_labels.get(st.session_state.lang, "Navigation"),
    list(nav_labels.values()),
    index=list(nav_labels.keys()).index(st.session_state.active_page)
)

new_active_page = label_to_key[selected_label]
if new_active_page != st.session_state.active_page:
    st.session_state.active_page = new_active_page
    st.session_state.active_tab = new_active_page
    st.rerun()

st.sidebar.markdown("---")

_mode_section_labels = {
    "en": "Mode",    "hi": "मोड",   "te": "మోడ్",
    "ta": "பயன்முறை", "kn": "ಮೋಡ್", "ml": "മോഡ്", "tcy": "ಮೋಡ್",
}
# Compact Processing Mode in Sidebar
from modules.perf import PERF_MODES
selected_mode = st.sidebar.radio(
    _mode_section_labels.get(st.session_state.lang, "Mode"),
    list(PERF_MODES.keys()),
    index=list(PERF_MODES.keys()).index(st.session_state.get("perf_mode", "⚖️ Balanced")),
    key="perf_mode_radio"
)
st.session_state.perf_mode = selected_mode

# 6. Render Active View
pages[st.session_state.active_page](lang=st.session_state.lang)
