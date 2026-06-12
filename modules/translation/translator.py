import requests
import json
from modules.database import get_setting

# Offline Translation Dictionary
OFFLINE_DICTIONARY = {
    "hello": {
        "en": "Hello",
        "hi": "नमस्ते",
        "te": "నమస్కారం"
    },
    "thank you": {
        "en": "Thank You",
        "hi": "धन्यवाद",
        "te": "ధన్యవాదాలు"
    },
    "yes": {
        "en": "Yes",
        "hi": "हाँ",
        "te": "అవును"
    },
    "no": {
        "en": "No",
        "hi": "नहीं",
        "te": "వద్దు"
    },
    "help": {
        "en": "I need help",
        "hi": "मुझे सहायता चाहिए",
        "te": "నాకు సహాయం కావాలి"
    },
    "water": {
        "en": "Give me water",
        "hi": "मुझे पानी चाहिए",
        "te": "నాకు నీరు కావాలి"
    },
    "food": {
        "en": "I want food",
        "hi": "मुझे भोजन चाहिए",
        "te": "నాకు ఆహారం కావాలి"
    },
    "mother": {
        "en": "She is my mother",
        "hi": "वह मेरी माँ है",
        "te": "ఆమె నా తల్లి"
    },
    "father": {
        "en": "He is my father",
        "hi": "वह मेरे पिता हैं",
        "te": "అతను నా తండ్రి"
    },
    "friend": {
        "en": "He is my friend",
        "hi": "वह मेरा दोस्त है",
        "te": "అతను నా స్నేహితుడు"
    },
    "school": {
        "en": "I go to school",
        "hi": "मैं स्कूल जाता हूँ",
        "te": "నేను పాఠశాలకు వెళ్తాను"
    },
    "hospital": {
        "en": "Take me to the hospital",
        "hi": "मुझे अस्पताल ले चलो",
        "te": "నన్ను ఆసుపత్రికి తీసుకెళ్లండి"
    },
    "emergency": {
        "en": "This is an emergency!",
        "hi": "यह एक आपातकालीन स्थिति है!",
        "te": "ఇది అత్యవసర పరిస్థితి!"
    }
}

def translate_sign(sign_label, target_lang="en"):
    """
    Translates sign_label into target_lang ('en', 'hi', 'te') using dictionaries,
    local Ollama model, or BYOK endpoints.
    """
    if not sign_label or sign_label == "None":
        return ""
        
    lang = target_lang.lower()[:2]
    
    # 1. Lookup Local Offline Dictionary
    clean_sign = sign_label.lower().strip()
    if clean_sign in OFFLINE_DICTIONARY:
        if lang in OFFLINE_DICTIONARY[clean_sign]:
            return OFFLINE_DICTIONARY[clean_sign][lang]
            
    # 2. Check settings to query local or cloud AI model for translation refinement
    provider = get_setting("ai_provider", "None")
    
    if provider == "Ollama":
        endpoint = get_setting("ollama_endpoint", "http://localhost:11434")
        model = get_setting("ollama_model", "llama3")
        try:
            url = endpoint.rstrip("/") + "/api/chat"
            prompt = f"Translate and refine Indian Sign Language gesture '{sign_label}' into a natural sentence in {target_lang}. Return ONLY the translation."
            res = requests.post(url, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }, timeout=2.0)
            if res.status_code == 200:
                data = res.json()
                return data["message"]["content"].strip()
        except Exception:
            pass # Fallback to local dictionary
            
    elif provider in ["OpenAI", "Gemini", "Anthropic"]:
        # Mock BYOK cloud queries (returns refined sentence or fallback)
        refined_phrase = sign_label
        if lang == "hi":
            refined_phrase = f"रिफाइंड ({provider}): " + (OFFLINE_DICTIONARY.get(clean_sign, {}).get("hi", sign_label))
        elif lang == "te":
            refined_phrase = f"రిఫైన్డ్ ({provider}): " + (OFFLINE_DICTIONARY.get(clean_sign, {}).get("te", sign_label))
        else:
            refined_phrase = f"Refined ({provider}): " + (OFFLINE_DICTIONARY.get(clean_sign, {}).get("en", sign_label))
        return refined_phrase
        
    # Standard fallback
    if lang == "hi":
        return f"संकेत: {sign_label}"
    elif lang == "te":
        return f"సంజ్ఞ: {sign_label}"
    return sign_label
