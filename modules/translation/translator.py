import requests
import os

from modules.database import get_setting
from modules.providers import get_provider_key
from modules.providers.cloud import cloud_translate as _cloud_api_translate


OFFLINE_DICTIONARY = {
    "hello": {
        "en": "Hello", 
        "hi": "नमस्ते", 
        "te": "నమస్కారం",
        "ta": "வணக்கம்",
        "kn": "ನಮಸ್ಕಾರ",
        "ml": "ഹലോ / നമസ്കാരം",
        "tcy": "നമസ്കാര"
    },
    "bye": {
        "en": "Goodbye", 
        "hi": "अलविदा", 
        "te": "సెలవు / వీడ్కోలు",
        "ta": "சென்று வருகிறேன் (பை)",
        "kn": "ಹೋಗಿ ಬರುತ್ತೇನೆ (ಬೈ)",
        "ml": "പോയി വരാം (ബൈ)",
        "tcy": "പോതു ബർപെ (ബൈ)"
    },
    "thumbs up": {
        "en": "Thumbs Up", 
        "hi": "अंगूठा ऊपर (बहुत बढ़िया)", 
        "te": "అభినందనలు (థంబ్స్ అప్)",
        "ta": "சிறந்தது (தம்ஸ் அப்)",
        "kn": "ಮೆಚ್ಚುಗೆ (ಥಂಬ್ಸ್ ಅಪ್)",
        "ml": "അഭിനന്ദനം (തംസ് അപ്)",
        "tcy": "ಎಡ್ಡೆ (ಥಂಬ್ಸ್ ಅಪ್)"
    },
    "thumbs down": {
        "en": "Thumbs Down", 
        "hi": "अंगूठा नीचे (असहमत)", 
        "te": "అసమ్మతి (థంబ్స్ డౌన్)",
        "ta": "அசம்மதி (தம்ஸ் டவுன்)",
        "kn": "ಅಸಮ್ಮತಿ (ಥಂಬ್ಸ್ ಡೌನ್)",
        "ml": "അവഗണന (തംസ് ഡൗൺ)",
        "tcy": "ഒಪ್ಪಂದ ಇದ್ದಿ (ಥಂಬ್ಸ್ ഡೌನ್)"
    },
    "point left": {
        "en": "Point Left", 
        "hi": "बाईं ओर इशारा", 
        "te": "ఎడమ వైపు చూపించు",
        "ta": "இடது பக்கம் காட்டு",
        "kn": "ಎಡಭಾಗ ತೋರಿಸು",
        "ml": "ഇടത്തോട്ട് ചൂണ്ടുക",
        "tcy": "ಎಡತ್ತ್ ತೋಜಾಲೆ"
    },
    "point right": {
        "en": "Point Right", 
        "hi": "दाईं ओर इशारा", 
        "te": "కుడి వైపు చూపించు",
        "ta": "வலது பக்கம் காட்டு",
        "kn": "ಬಲಭಾಗ ತೋರಿಸು",
        "ml": "വലത്തോട്ട് ചൂണ്ടുക",
        "tcy": "ಬലತ್ತ್ ತೋಜಾಲೆ"
    },
    "point up": {
        "en": "Point Up", 
        "hi": "ऊपर की ओर इशारा", 
        "te": "పైకి చూపించు",
        "ta": "மேலே காட்டு",
        "kn": "ಮೇಲಕ್ಕೆ ತೋರಿಸು",
        "ml": "മുകളിലോട്ട് ചൂണ്ടുക",
        "tcy": "ಮಿತ್ತ್ ತೋಜಾಲೆ"
    },
    "point down": {
        "en": "Point Down", 
        "hi": "नीचे की ओर इशारा", 
        "te": "క్రిందికి చూపించు",
        "ta": "கீழே காட்டு",
        "kn": "ಕೆಳಗೆ ತೋರಿಸು",
        "ml": "ತಾഴോട്ട് ചൂണ്ടുക",
        "tcy": "ತಿರ್ತ್ ತೋಜಾಲೆ"
    },
    "open palm": {
        "en": "Open Palm", 
        "hi": "खुली हथेली", 
        "te": "తెరచిన చేయి",
        "ta": "திறந்த கை",
        "kn": "ತೆರೆದ ಕೈ",
        "ml": "തുറന്ന കൈപ്പത്തി",
        "tcy": "ദെಪ್ಪುದ ಕೈ"
    },
    "closed fist": {
        "en": "Closed Fist", 
        "hi": "बंद मुट्ठी", 
        "te": "మూసివున్న పిడికిలి",
        "ta": "மூடிய முஷ்டி",
        "kn": "ಮುಚ್ಚಿದ ಮುಷ್ಟಿ",
        "ml": "അടഞ്ഞ മുഷ്ടി",
        "tcy": "ಮುಚ್ಚಿದ ಮುಷ್ಟಿ"
    }
}


def _fallback(sign_label, lang):
    if lang == "hi":
        return "\u0938\u0902\u0915\u0947\u0924: " + sign_label
    if lang == "te":
        return "\u0c38\u0c02\u0c1c\u0c4d\u0c1e: " + sign_label
    if lang == "ta":
        return "அடையாளம்: " + sign_label
    if lang == "kn":
        return "ಸಂಕೇತ: " + sign_label
    if lang == "ml":
        return "അടയാളം: " + sign_label
    if lang == "tcy":
        return "ಸಂಕೇತ: " + sign_label
    return sign_label


def _ollama_translate(sign_label, lang):
    if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("SPACE_ID"):
        return ""
    endpoint = get_setting("ollama_endpoint", "http://localhost:11434")
    model = get_setting("ollama_model", "llama3")
    try:
        res = requests.post(
            endpoint.rstrip("/") + "/api/chat",
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": (
                        "Translate the Indian Sign Language sign "
                        f"'{sign_label}' into a short natural sentence in {lang}. "
                        "Return only the translation."
                    ),
                }],
                "stream": False,
            },
            timeout=1.5,
        )
        if res.status_code == 200:
            return res.json().get("message", {}).get("content", "").strip()
    except Exception:
        return ""
    return ""


def _cloud_translate(provider, sign_label, lang, fallback_text):
    api_key = get_provider_key(provider)
    if not api_key:
        return fallback_text
    return _cloud_api_translate(provider, sign_label, lang, api_key, fallback_text)


def translate_sign(sign_label, target_lang="en"):
    if not sign_label or sign_label in ("None", "No Sign Detected", "No Gesture Detected"):
        return ""

    lang = (target_lang or "en").lower().strip()
    if lang not in {"en", "hi", "te", "ta", "kn", "ml", "tcy"}:
        lang = "en"

    clean_sign = sign_label.lower().strip()
    fallback_text = OFFLINE_DICTIONARY.get(clean_sign, {}).get(lang, _fallback(sign_label, lang))
    provider = get_setting("ai_provider", "None (Offline Dictionary)")

    if provider == "Ollama":
        return _ollama_translate(sign_label, lang) or fallback_text
    if provider in {"OpenAI", "Gemini", "Anthropic"}:
        return _cloud_translate(provider, sign_label, lang, fallback_text)
    return fallback_text
