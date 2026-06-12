import json
import os

LOCALES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "locales"))

class I18nManager:
    def __init__(self):
        self.translations = {}
        self.load_translations()
        
    def load_translations(self):
        for lang in ["en", "hi", "te", "ta", "kn", "ml", "tcy"]:
            path = os.path.join(LOCALES_DIR, f"{lang}.json")
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        self.translations[lang] = json.load(f)
                except Exception as e:
                    print(f"Error loading translation for {lang}: {e}")
                    self.translations[lang] = {}
            else:
                self.translations[lang] = {}

    def translate(self, key, lang="en"):
        if not lang:
            lang = "en"
        lang = lang.lower()[:3] if lang.lower().startswith("tcy") else lang.lower()[:2]
        if lang not in ["en", "hi", "te", "ta", "kn", "ml", "tcy"]:
            lang = "en"
            
        parts = key.split(".")
        val = self.translations.get(lang, {})
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return self.translate_en(key)
        return val if isinstance(val, str) else key
        
    def translate_en(self, key):
        parts = key.split(".")
        val = self.translations.get("en", {})
        for part in parts:
            if isinstance(val, dict) and part in val:
                val = val[part]
            else:
                return key
        return val if isinstance(val, str) else key

# Global instance
translator = I18nManager()

def t(key, lang="en"):
    return translator.translate(key, lang)
