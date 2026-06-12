import streamlit as st
from modules.locales import t

def render_home_page(lang="en"):
    """Renders the clean and modern Home page onboarding view."""
    st.title(f"🤟 {t('home.title', lang)}")
    st.subheader(t('home.tagline', lang))
    
    st.markdown("---")
    
    # Hero Button
    col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
    with col_btn_c:
        if st.button(f"🚀 {t('home.btnStart', lang)}", use_container_width=True):
            st.session_state.active_tab = "Live Translation"
            st.rerun()
            
    st.markdown("### 🌟 Features Overview")
    
    # 3x2 Grid for Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <h4>🤟 {t('home.sign', lang)}</h4>
            <p style='color:#94A3B8; font-size:14px;'>{t('home.sign_desc', lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card">
            <h4>🔌 {t('home.offline', lang)}</h4>
            <p style='color:#94A3B8; font-size:14px;'>{t('home.offline_desc', lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <h4>🔊 {t('home.voice', lang)}</h4>
            <p style='color:#94A3B8; font-size:14px;'>{t('home.voice_desc', lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card">
            <h4>🧠 {t('home.memory', lang)}</h4>
            <p style='color:#94A3B8; font-size:14px;'>{t('home.memory_desc', lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <h4>🌍 {t('home.lang', lang)}</h4>
            <p style='color:#94A3B8; font-size:14px;'>{t('home.lang_desc', lang)}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card">
            <h4>👤 {t('home.face', lang)}</h4>
            <p style='color:#94A3B8; font-size:14px;'>{t('home.face_desc', lang)}</p>
        </div>
        """, unsafe_allow_html=True)
