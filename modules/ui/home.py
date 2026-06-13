import streamlit as st
import requests
import os
from modules.locales import t
from modules.database import get_db_connection
from modules.ollama.manage import get_ollama_endpoint
 
 
def render_home_page(lang="en"):
    """Renders the premium redesigned Home page onboarding view."""
    st.title(f"🤟 {t('home.title', lang)}")
    st.subheader(t('home.tagline', lang))
 
    st.markdown("---")
 
    # ── 1. Health Status Dashboard ─────────────────────────────────────────────
 
    # Streamlit Cloud cannot access server webcams.
    # Browser camera access is handled separately in live pages.
    # We always report camera as available since browser WebRTC is used.
    webcam_ok = True
 
    # Database Counts
    db_convs = 0
    db_people = 0
    try:
        conn = get_db_connection()
        db_convs = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        db_people = conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]
        conn.close()
    except Exception:
        pass
 
    # Ollama Health Check
    ollama_ok = False
    if not (os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("SPACE_ID")):
        try:
            res = requests.get(f"{get_ollama_endpoint()}/api/tags", timeout=0.8)
            ollama_ok = res.status_code == 200
        except Exception:
            ollama_ok = False
 
    # Render Health Status Cards
    col1, col2, col3, col4 = st.columns(4)
 
    with col1:
        cam_status = "✅ Available" if webcam_ok else "❌ Unavailable"
        cam_color = "#10B981" if webcam_ok else "#EF4444"
        st.markdown(f"""
        <div class="glass-card" style="padding: 16px; text-align: center;">
            <div style="font-size: 28px;">📷</div>
            <h5 style="margin: 6px 0 4px 0; color: #E2E8F0; font-size: 14px;">Browser Camera</h5>
            <span style="color: {cam_color}; font-size: 13px; font-weight: 600;">{cam_status}</span>
        </div>
        """, unsafe_allow_html=True)
 
    with col2:
        ollama_status = "✅ Online" if ollama_ok else "❌ Offline"
        ollama_color = "#10B981" if ollama_ok else "#EF4444"
        st.markdown(f"""
        <div class="glass-card" style="padding: 16px; text-align: center;">
            <div style="font-size: 28px;">🧠</div>
            <h5 style="margin: 6px 0 4px 0; color: #E2E8F0; font-size: 14px;">Ollama LLM</h5>
            <span style="color: {ollama_color}; font-size: 13px; font-weight: 600;">{ollama_status}</span>
        </div>
        """, unsafe_allow_html=True)
 
    with col3:
        st.markdown(f"""
        <div class="glass-card" style="padding: 16px; text-align: center;">
            <div style="font-size: 28px;">💬</div>
            <h5 style="margin: 6px 0 4px 0; color: #E2E8F0; font-size: 14px;">Conversations</h5>
            <span style="color: #3B82F6; font-size: 20px; font-weight: 700;">{db_convs}</span>
        </div>
        """, unsafe_allow_html=True)
 
    with col4:
        st.markdown(f"""
        <div class="glass-card" style="padding: 16px; text-align: center;">
            <div style="font-size: 28px;">👤</div>
            <h5 style="margin: 6px 0 4px 0; color: #E2E8F0; font-size: 14px;">People</h5>
            <span style="color: #8B5CF6; font-size: 20px; font-weight: 700;">{db_people}</span>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── 2. Onboarding Mission Card ────────────────────────────────────────────
    mission_text = t('about.missionText', lang)
    st.markdown(f"""
    <div class="glass-card" style="border-left: 5px solid #8B5CF6; padding: 20px;">
        <h4 style="margin: 0 0 10px 0; color: #8B5CF6; font-size: 18px;">🌟 Our Mission</h4>
        <p style="margin: 0; color: #E2E8F0; font-size: 15px; line-height: 1.6;">
            {mission_text}
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── 3. How It Works Pipeline Grid ──────────────────────────────────────────
    st.markdown("### ⚙️ How It Works (Pipeline)")
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 25px;">
        <div class="glass-card" style="padding: 16px;">
            <div style="font-size: 24px; margin-bottom: 8px;">📷</div>
            <h5 style="margin: 0 0 8px 0; color: #3B82F6; font-size: 16px;">1. Video Input</h5>
            <p style="margin: 0; font-size: 13px; color: #94A3B8; line-height: 1.4;">Webcam captures video streams at standard 640x480 resolution.</p>
        </div>
        <div class="glass-card" style="padding: 16px;">
            <div style="font-size: 24px; margin-bottom: 8px;">✨</div>
            <h5 style="margin: 0 0 8px 0; color: #10B981; font-size: 16px;">2. Landmark Tracking</h5>
            <p style="margin: 0; font-size: 13px; color: #94A3B8; line-height: 1.4;">Extracts 468 face mesh landmarks, 21 hand joints, and poses in real-time.</p>
        </div>
        <div class="glass-card" style="padding: 16px;">
            <div style="font-size: 24px; margin-bottom: 8px;">🧠</div>
            <h5 style="margin: 0 0 8px 0; color: #F59E0B; font-size: 16px;">3. Gesture Classifier</h5>
            <p style="margin: 0; font-size: 13px; color: #94A3B8; line-height: 1.4;">A temporal classification model detects sign sequences with high accuracy.</p>
        </div>
        <div class="glass-card" style="padding: 16px;">
            <div style="font-size: 24px; margin-bottom: 8px;">🔊</div>
            <h5 style="margin: 0 0 8px 0; color: #8B5CF6; font-size: 16px;">4. Context Translation</h5>
            <p style="margin: 0; font-size: 13px; color: #94A3B8; line-height: 1.4;">Ollama LLM refines sentences and synthesizes translated voice output.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── 4. Supported Gestures and Languages ──────────────────────────────────
    col_g, col_l = st.columns(2)
 
    with col_g:
        st.markdown("""
        <div class="glass-card" style="padding: 20px; height: 100%;">
            <h4 style="margin: 0 0 12px 0; color: #EC4899; font-size: 18px;">🤟 Supported Gestures</h4>
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">👍 Thumbs Up</span>
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">👎 Thumbs Down</span>
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">👈 Point Left</span>
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">👉 Point Right</span>
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">👆 Point Up</span>
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">👇 Point Down</span>
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">✋ Open Palm</span>
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">✊ Closed Fist</span>
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">👋 Hello</span>
                <span style="background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.3); color: #F472B6; padding: 4px 10px; border-radius: 20px; font-size: 13px; font-weight: 500;">🖐️ Bye</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    with col_l:
        st.markdown("""
        <div class="glass-card" style="padding: 20px; height: 100%;">
            <h4 style="margin: 0 0 12px 0; color: #10B981; font-size: 18px;">🌍 Supported Indian Languages</h4>
            <div style="display: flex; flex-direction: column; gap: 8px;">
                <span style="color: #E2E8F0; font-size: 14px;">🇮🇳 <b>English</b> — default output</span>
                <span style="color: #E2E8F0; font-size: 14px;">🇮🇳 <b>Hindi (हिंदी)</b> — translation & speech support</span>
                <span style="color: #E2E8F0; font-size: 14px;">🇮🇳 <b>Telugu (తెలుగు)</b> — translation & speech support</span>
                <span style="color: #E2E8F0; font-size: 14px;">🇮🇳 <b>Tamil (தமிழ்)</b> — translation support</span>
                <span style="color: #E2E8F0; font-size: 14px;">🇮🇳 <b>Kannada (ಕನ್ನಡ)</b> — translation support</span>
                <span style="color: #E2E8F0; font-size: 14px;">🇮🇳 <b>Malayalam (മലയാളം)</b> — translation support</span>
                <span style="color: #E2E8F0; font-size: 14px;">🇮🇳 <b>Tulu (ತುಳು)</b> — translation support</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("<br><br>", unsafe_allow_html=True)
 
    # ── 5. Hero Button ────────────────────────────────────────────────────────
    col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
    with col_btn_c:
        if st.button(f"🚀 {t('home.btnStart', lang)}", use_container_width=True):
            st.session_state.active_tab = "Live Translation"
            st.rerun()
