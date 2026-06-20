# Supported Languages

SignBridge AI is committed to making communication accessible to everyone in India. We currently support the following languages for both the UI and the translated sign-to-speech output.

## Language Catalog

| Language      | ISO Code | Script                  | Region                    |
| :------------ | :------- | :---------------------- | :------------------------ |
| **English**   | `en`     | Latin                   | Global                    |
| **Telugu**    | `te`     | Telugu                  | Andhra Pradesh, Telangana |
| **Hindi**     | `hi`     | Devanagari              | National                  |
| **Tamil**     | `ta`     | Tamil                   | Tamil Nadu, Puducherry    |
| **Kannada**   | `kn`     | Kannada                 | Karnataka                 |
| **Malayalam** | `ml`     | Malayalam               | Kerala, Lakshadweep       |
| **Tulu**      | `tcy`    | Kannada / Tulu-Tigalari | Coastal Karnataka         |

## How to Add a New Language

If you would like to contribute a new language:

1. Create a new JSON file in the `locales/` directory (e.g., `mr.json` for Marathi).
2. Follow the structure of `en.json`.
3. Add the language code to `modules/locales/i18n.py` and `streamlit_app.py`.
4. Submit a Pull Request!
