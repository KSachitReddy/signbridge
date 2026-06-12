import requests

from modules.database import get_setting
from modules.providers import get_provider_key
from modules.providers.cloud import cloud_translate as _cloud_api_translate


OFFLINE_DICTIONARY = {
    "hello": {"en": "Hello", "hi": "नमस्ते", "te": "నమస్కారం"},
    "bye": {"en": "Goodbye", "hi": "अलविदा", "te": "సెలవు / వీడ్కోలు"},
    "thumbs up": {"en": "Thumbs Up", "hi": "अंगूठा ऊपर (बहुत बढ़िया)", "te": "అభినందనలు (థంబ్స్ అప్)"},
    "thumbs down": {"en": "Thumbs Down", "hi": "अंगूठा नीचे (असहमत)", "te": "అసమ్మతి (థంబ్స్ డౌన్)"},
    "point left": {"en": "Point Left", "hi": "बाईं ओर इशारा", "te": "ఎడమ వైపు చూపించు"},
    "point right": {"en": "Point Right", "hi": "दाईं ओर इशारा", "te": "కుడి వైపు చూపించు"},
    "point up": {"en": "Point Up", "hi": "ऊपर की ओर इशारा", "te": "పైకి చూపించు"},
    "point down": {"en": "Point Down", "hi": "नीचे की ओर इशारा", "te": "క్రిందికి చూపించు"},
    "open palm": {"en": "Open Palm", "hi": "खुली हथेली", "te": "తెరచిన చేయి"},
    "closed fist": {"en": "Closed Fist", "hi": "बंद मुट्ठी", "te": "మూసివున్న పిడికిలి"}
}


def _fallback(sign_label, lang):
    if lang == "hi":
        return "\u0938\u0902\u0915\u0947\u0924: " + sign_label
    if lang == "te":
        return "\u0c38\u0c02\u0c1c\u0c4d\u0c1e: " + sign_label
    return sign_label


def _ollama_translate(sign_label, lang):
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

    lang = (target_lang or "en").lower()[:2]
    if lang not in {"en", "hi", "te"}:
        lang = "en"

    clean_sign = sign_label.lower().strip()
    fallback_text = OFFLINE_DICTIONARY.get(clean_sign, {}).get(lang, _fallback(sign_label, lang))
    provider = get_setting("ai_provider", "None (Offline Dictionary)")

    if provider == "Ollama":
        return _ollama_translate(sign_label, lang) or fallback_text
    if provider in {"OpenAI", "Gemini", "Anthropic"}:
        return _cloud_translate(provider, sign_label, lang, fallback_text)
    return fallback_text
