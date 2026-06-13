"""
chat.py — Ollama conversation integration for SignBridge AI.

Generates contextual, empathetic responses to recognized ISL signs using
the locally running Ollama LLM. Falls back to a curated dictionary response
if Ollama is unavailable or the model is slow.
"""

import requests
import json
import os
from modules.database import get_setting
from modules.providers import get_provider_key
from modules.providers.cloud import cloud_generate_response as _cloud_response

# Curated fallback responses per sign (offline-first)
FALLBACK_RESPONSES = {
    "Hello":     "Hello! Great to meet you. How can I help you today?",
    "Thank You": "You're very welcome! Happy to be of assistance.",
    "Yes":       "Understood — yes! Please go ahead.",
    "No":        "Noted — no problem at all.",
    "Help":      "Of course, I'm here to help you. What do you need?",
    "Water":     "Sure, let me get you some water right away.",
    "Food":      "Are you hungry? Let me arrange some food for you.",
    "Mother":    "I understand — your mother. Would you like to reach her?",
    "Father":    "Your father — noted. Do you need to contact him?",
    "Brother":   "Your brother — got it. Can I help you reach him?",
    "Sister":    "Your sister — understood. Do you need to contact her?",
    "Friend":    "A friend — wonderful! Who would you like to reach?",
    "School":    "School — understood. Do you need directions or information?",
    "Teacher":   "I'll find your teacher for you right away.",
    "Hospital":  "I'll help you get to the hospital immediately. Are you okay?",
    "Doctor":    "I'll get a doctor for you immediately. Please stay calm.",
    "Emergency": "⚠️ Emergency! I'm alerting staff immediately. Please stay with me.",
    "Pain":      "I understand you're in pain. Let me get medical help right away.",
    "Medicine":  "I'll get your medicine. Can you show me which one you need?",
    "Bathroom":  "Of course — the bathroom is right this way. Let me show you.",
    "Home":      "You'd like to go home. Let me help arrange that for you.",
    "Eat":       "Let's get you something to eat. What would you like?",
    "Drink":     "Let me get you something to drink. Water, juice, or tea?",
    "Sleep":     "You need rest — that's important. Let me help you get comfortable.",
    "Stop":      "I'll stop right away. Just let me know when to continue.",
    "Come":      "Come here — I'm on my way!",
    "Go":        "Go ahead — I'll follow your lead.",
    "Good":      "That's great to hear! I'm glad everything is going well.",
    "Bad":       "I'm sorry to hear that. How can I make things better?",
    "Please":    "Of course! I'm happy to help. What do you need?",
}

DEFAULT_RESPONSE = "I received your message. How can I assist you further?"


def generate_response(sign: str, translation: str, person_name: str = "You", lang: str = "en") -> str:
    """
    Generates a natural language response to a recognized ISL sign.

    Args:
        sign: The recognized sign label (e.g., "Help")
        translation: The translated text (e.g., "I need help")
        person_name: Name of the signer (for personalization)
        lang: Target language code ("en", "hi", "te")

    Returns:
        str: The AI-generated or fallback response string
    """
    ai_provider = get_setting("ai_provider", "None (Offline Dictionary)")

    # ── Offline dictionary fallback ─────────────────────────────────────────
    if ai_provider == "None (Offline Dictionary)":
        return _offline_response(sign, person_name, lang)

    # ── Ollama LLM ──────────────────────────────────────────────────────────
    if ai_provider == "Ollama":
        response = _ollama_response(sign, translation, person_name, lang)
        if response:
            return response
        # Graceful fallback to dictionary if Ollama times out
        return _offline_response(sign, person_name, lang)

    # ── Cloud AI providers (BYOK) ───────────────────────────────────────────
    if ai_provider in {"OpenAI", "Gemini", "Anthropic"}:
        api_key = get_provider_key(ai_provider)
        offline_fallback = _offline_response(sign, person_name, lang)
        if not api_key:
            return offline_fallback
        return _cloud_response(ai_provider, sign, translation, person_name, lang,
                               api_key, offline_fallback)

    return _offline_response(sign, person_name, lang)


def _offline_response(sign: str, person_name: str, lang: str) -> str:
    """Returns a curated response from the static fallback dictionary."""
    base = FALLBACK_RESPONSES.get(sign, DEFAULT_RESPONSE)

    # Personalize
    if person_name and person_name not in ("Unknown", "System", "Synthetic"):
        base = base.replace("you", person_name, 1)

    # Language variants (simple mapping for Hindi and Telugu)
    if lang == "hi":
        lang_note = " (हिंदी में: " + _translate_response_hi(sign) + ")"
        return base + lang_note
    elif lang == "te":
        lang_note = " (తెలుగులో: " + _translate_response_te(sign) + ")"
        return base + lang_note

    return base


def _translate_response_hi(sign: str) -> str:
    hi_map = {
        "Hello": "नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूँ?",
        "Thank You": "आपका बहुत-बहुत स्वागत है!",
        "Help": "मैं यहाँ आपकी मदद के लिए हूँ।",
        "Water": "मैं अभी आपके लिए पानी लाता हूँ।",
        "Food": "क्या आपको भूख लगी है? मैं खाना मंगवाता हूँ।",
        "Doctor": "मैं अभी डॉक्टर को बुलाता हूँ।",
        "Emergency": "⚠️ आपातकाल! मैं तुरंत सहायता बुला रहा हूँ।",
        "Pain": "मुझे खेद है, मैं तुरंत डॉक्टरी सहायता दिलाता हूँ।",
    }
    return hi_map.get(sign, "समझ गया।")


def _translate_response_te(sign: str) -> str:
    te_map = {
        "Hello": "నమస్కారం! నేను మీకు ఎలా సహాయం చేయగలను?",
        "Thank You": "మీకు స్వాగతం!",
        "Help": "నేను మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను.",
        "Water": "నేను మీకు వెంటనే నీళ్ళు తీసుకొస్తాను.",
        "Food": "మీకు ఆకలిగా ఉందా? నేను వెంటనే ఏర్పాటు చేస్తాను.",
        "Doctor": "నేను వెంటనే డాక్టర్‌ను పిలుస్తాను.",
        "Emergency": "⚠️ అత్యవసరం! నేను వెంటనే సహాయం పిలుస్తున్నాను.",
        "Pain": "మీకు నొప్పిగా ఉందా? నేను వెంటనే వైద్య సహాయం తీసుకొస్తాను.",
    }
    return te_map.get(sign, "అర్థమైంది.")


def _ollama_response(sign: str, translation: str, person_name: str, lang: str) -> str | None:
    """Calls local Ollama API and returns the response string, or None on failure."""
    if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("SPACE_ID"):
        return None
    endpoint = get_setting("ollama_endpoint", "http://localhost:11434").rstrip("/")
    model = get_setting("ollama_model", "deepseek-r1:1.5b")

    lang_name = {"en": "English", "hi": "Hindi", "te": "Telugu"}.get(lang, "English")

    system_prompt = (
        "You are SignBridge AI, a compassionate communication assistant helping "
        "deaf and hard-of-hearing people communicate through Indian Sign Language (ISL). "
        "Respond briefly (1-2 sentences), warmly, and helpfully in " + lang_name + ". "
        "Never use jargon. Be direct and empathetic."
    )

    user_prompt = (
        f"{person_name} just signed '{sign}' which means '{translation}'. "
        f"Please respond naturally and helpfully to this communication."
    )

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 80  # Keep responses short
            }
        }
        res = requests.post(
            f"{endpoint}/api/chat",
            json=payload,
            timeout=8.0  # Fail fast if model is slow
        )
        if res.status_code == 200:
            data = res.json()
            content = data.get("message", {}).get("content", "").strip()
            # Strip DeepSeek <think> tags if present
            if "<think>" in content:
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content:
                return content
    except Exception as e:
        print(f"[OllamaChat] Request failed: {e}")

    return None


def get_conversation_summary(recent_signs: list, person_name: str = "User", lang: str = "en") -> str:
    """
    Generates a summary of the recent conversation using Ollama.
    Falls back to a simple formatted list if Ollama is unavailable.
    """
    if not recent_signs:
        return "No recent signs to summarize."

    sign_list = ", ".join(f"'{s}'" for s in recent_signs[-5:])

    ai_provider = get_setting("ai_provider", "None (Offline Dictionary)")
    if ai_provider == "Ollama":
        if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("SPACE_ID"):
            return f"{person_name} communicated: {sign_list}."
        endpoint = get_setting("ollama_endpoint", "http://localhost:11434").rstrip("/")
        model = get_setting("ollama_model", "deepseek-r1:1.5b")
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content":
                    f"{person_name} recently signed: {sign_list}. "
                    f"Summarize what they communicated in one sentence."}],
                "stream": False,
                "options": {"temperature": 0.5, "num_predict": 60}
            }
            res = requests.post(f"{endpoint}/api/chat", json=payload, timeout=6.0)
            if res.status_code == 200:
                content = res.json().get("message", {}).get("content", "").strip()
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content:
                    return content
        except Exception:
            pass

    return f"{person_name} communicated: {sign_list}."
