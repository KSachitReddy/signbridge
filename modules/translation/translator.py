import requests

from modules.database import get_setting
from modules.providers import get_provider_key
from modules.providers.cloud import cloud_translate as _cloud_api_translate


OFFLINE_DICTIONARY = {
    "hello": {"en": "Hello", "hi": "\u0928\u092e\u0938\u094d\u0924\u0947", "te": "\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02"},
    "thank you": {"en": "Thank you", "hi": "\u0927\u0928\u094d\u092f\u0935\u093e\u0926", "te": "\u0c27\u0c28\u0c4d\u0c2f\u0c35\u0c3e\u0c26\u0c3e\u0c32\u0c41"},
    "yes": {"en": "Yes", "hi": "\u0939\u093e\u0901", "te": "\u0c05\u0c35\u0c41\u0c28\u0c41"},
    "no": {"en": "No", "hi": "\u0928\u0939\u0940\u0902", "te": "\u0c35\u0c26\u0c4d\u0c26\u0c41"},
    "help": {"en": "I need help", "hi": "\u092e\u0941\u091d\u0947 \u0938\u0939\u093e\u092f\u0924\u093e \u091a\u093e\u0939\u093f\u090f", "te": "\u0c28\u0c3e\u0c15\u0c41 \u0c38\u0c39\u0c3e\u0c2f\u0c02 \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f"},
    "water": {"en": "I need water", "hi": "\u092e\u0941\u091d\u0947 \u092a\u093e\u0928\u0940 \u091a\u093e\u0939\u093f\u090f", "te": "\u0c28\u0c3e\u0c15\u0c41 \u0c28\u0c40\u0c30\u0c41 \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f"},
    "food": {"en": "I need food", "hi": "\u092e\u0941\u091d\u0947 \u092d\u094b\u091c\u0928 \u091a\u093e\u0939\u093f\u090f", "te": "\u0c28\u0c3e\u0c15\u0c41 \u0c06\u0c39\u0c3e\u0c30\u0c02 \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f"},
    "mother": {"en": "She is my mother", "hi": "\u0935\u0939 \u092e\u0947\u0930\u0940 \u092e\u093e\u0901 \u0939\u0948", "te": "\u0c06\u0c2e\u0c46 \u0c28\u0c3e \u0c24\u0c32\u0c4d\u0c32\u0c3f"},
    "father": {"en": "He is my father", "hi": "\u0935\u0939 \u092e\u0947\u0930\u0947 \u092a\u093f\u0924\u093e \u0939\u0948\u0902", "te": "\u0c05\u0c24\u0c28\u0c41 \u0c28\u0c3e \u0c24\u0c02\u0c21\u0c4d\u0c30\u0c3f"},
    "brother": {"en": "He is my brother", "hi": "\u0935\u0939 \u092e\u0947\u0930\u093e \u092d\u093e\u0908 \u0939\u0948", "te": "\u0c05\u0c24\u0c28\u0c41 \u0c28\u0c3e \u0c38\u0c4b\u0c26\u0c30\u0c41\u0c21\u0c41"},
    "sister": {"en": "She is my sister", "hi": "\u0935\u0939 \u092e\u0947\u0930\u0940 \u092c\u0939\u0928 \u0939\u0948", "te": "\u0c06\u0c2e\u0c46 \u0c28\u0c3e \u0c38\u0c4b\u0c26\u0c30\u0c3f"},
    "friend": {"en": "He is my friend", "hi": "\u0935\u0939 \u092e\u0947\u0930\u093e \u0926\u094b\u0938\u094d\u0924 \u0939\u0948", "te": "\u0c05\u0c24\u0c28\u0c41 \u0c28\u0c3e \u0c38\u0c4d\u0c28\u0c47\u0c39\u0c3f\u0c24\u0c41\u0c21\u0c41"},
    "school": {"en": "I need to go to school", "hi": "\u092e\u0941\u091d\u0947 \u0938\u094d\u0915\u0942\u0932 \u091c\u093e\u0928\u093e \u0939\u0948", "te": "\u0c28\u0c47\u0c28\u0c41 \u0c2a\u0c3e\u0c20\u0c36\u0c3e\u0c32\u0c15\u0c41 \u0c35\u0c46\u0c33\u0c4d\u0c32\u0c3e\u0c32\u0c3f"},
    "teacher": {"en": "I need my teacher", "hi": "\u092e\u0941\u091d\u0947 \u0905\u092a\u0928\u0947 \u0936\u093f\u0915\u094d\u0937\u0915 \u0915\u0940 \u091c\u0930\u0942\u0930\u0924 \u0939\u0948", "te": "\u0c28\u0c3e\u0c15\u0c41 \u0c28\u0c3e \u0c09\u0c2a\u0c3e\u0c27\u0c4d\u0c2f\u0c3e\u0c2f\u0c41\u0c21\u0c41 \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f"},
    "doctor": {"en": "Call a doctor", "hi": "\u0921\u0949\u0915\u094d\u091f\u0930 \u0915\u094b \u092c\u0941\u0932\u093e\u0907\u090f", "te": "\u0c21\u0c3e\u0c15\u0c4d\u0c1f\u0c30\u0c4d\u0c28\u0c41 \u0c2a\u0c3f\u0c32\u0c35\u0c02\u0c21\u0c3f"},
    "hospital": {"en": "Take me to the hospital", "hi": "\u092e\u0941\u091d\u0947 \u0905\u0938\u094d\u092a\u0924\u093e\u0932 \u0932\u0947 \u091a\u0932\u094b", "te": "\u0c28\u0c28\u0c4d\u0c28\u0c41 \u0c06\u0c38\u0c41\u0c2a\u0c24\u0c4d\u0c30\u0c3f\u0c15\u0c3f \u0c24\u0c40\u0c38\u0c41\u0c15\u0c46\u0c33\u0c4d\u0c32\u0c02\u0c21\u0c3f"},
    "emergency": {"en": "This is an emergency", "hi": "\u092f\u0939 \u0906\u092a\u093e\u0924\u0915\u093e\u0932 \u0939\u0948", "te": "\u0c07\u0c26\u0c3f \u0c05\u0c24\u0c4d\u0c2f\u0c35\u0c38\u0c30\u0c02"},
    "pain": {"en": "I am in pain", "hi": "\u092e\u0941\u091d\u0947 \u0926\u0930\u094d\u0926 \u0939\u0948", "te": "\u0c28\u0c3e\u0c15\u0c41 \u0c28\u0c4a\u0c2a\u0c4d\u0c2a\u0c3f\u0c17\u0c3e \u0c09\u0c02\u0c26\u0c3f"},
    "medicine": {"en": "I need medicine", "hi": "\u092e\u0941\u091d\u0947 \u0926\u0935\u093e\u0908 \u091a\u093e\u0939\u093f\u090f", "te": "\u0c28\u0c3e\u0c15\u0c41 \u0c2e\u0c02\u0c26\u0c41 \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f"},
    "bathroom": {"en": "I need the bathroom", "hi": "\u092e\u0941\u091d\u0947 \u0936\u094c\u091a\u093e\u0932\u092f \u091c\u093e\u0928\u093e \u0939\u0948", "te": "\u0c28\u0c3e\u0c15\u0c41 \u0c2c\u0c3e\u0c25\u0c4d\u0c30\u0c42\u0c2e\u0c4d \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f"},
    "home": {"en": "I want to go home", "hi": "\u092e\u0948\u0902 \u0918\u0930 \u091c\u093e\u0928\u093e \u091a\u093e\u0939\u0924\u093e \u0939\u0942\u0901", "te": "\u0c28\u0c47\u0c28\u0c41 \u0c07\u0c02\u0c1f\u0c3f\u0c15\u0c3f \u0c35\u0c46\u0c33\u0c4d\u0c32\u0c3e\u0c32\u0c28\u0c41\u0c15\u0c41\u0c02\u0c1f\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c28\u0c41"},
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
    if not sign_label or sign_label in ("None", "No Sign Detected"):
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
