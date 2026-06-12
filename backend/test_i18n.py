import unittest
import os
import sys

# Add root directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.locales.i18n import translator, t

class TestI18n(unittest.TestCase):
    def test_translation_en(self):
        # Test basic translation in English
        self.assertEqual(t("nav.home", "en"), "Home")
        self.assertEqual(t("home.title", "en"), "SignBridge AI")

    def test_translation_te(self):
        # Test basic translation in Telugu
        self.assertEqual(t("nav.home", "te"), "హోమ్")

    def test_fallback_en(self):
        # Test fallback to English for missing key
        self.assertEqual(t("non.existent.key", "hi"), "non.existent.key")

    def test_invalid_lang_fallback(self):
        # Test fallback for invalid language code
        self.assertEqual(t("nav.home", "invalid"), "Home")

    def test_case_insensitivity(self):
        # Test language code case insensitivity
        self.assertEqual(t("nav.home", "EN"), "Home")
        self.assertEqual(t("nav.home", "Te"), "హోమ్")

if __name__ == "__main__":
    unittest.main()
