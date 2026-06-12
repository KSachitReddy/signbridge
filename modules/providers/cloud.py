"""
cloud.py — Real BYOK API clients for OpenAI, Gemini, and Anthropic.

All calls use the `requests` library only (no provider SDKs required).
Every function returns (result_str, error_str | None).
A None error means success; a non-empty error means the call failed.
"""
import requests

# Timeouts: generous for translation (10 s), strict for connection test (5 s).
_TRANSLATE_TIMEOUT = 10.0
_RESPONSE_TIMEOUT = 12.0
_TEST_TIMEOUT = 5.0

# ─────────────────────────────────────────────────────────────────────────────
# Low-level callers
# ─────────────────────────────────────────────────────────────────────────────

def _call_openai(messages: list, api_key: str, timeout: float = _TRANSLATE_TIMEOUT) -> tuple[str, str | None]:
    """POST to OpenAI Chat Completions. Returns (content, error)."""
    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": messages,
                "max_tokens": 120,
                "temperature": 0.4,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return content, None
        err = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
        return "", f"OpenAI error: {err}"
    except requests.exceptions.Timeout:
        return "", "OpenAI: request timed out"
    except Exception as e:
        return "", f"OpenAI: {e}"


def _call_gemini(prompt: str, api_key: str, timeout: float = _TRANSLATE_TIMEOUT) -> tuple[str, str | None]:
    """POST to Gemini generateContent. Returns (content, error)."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={api_key}"
    )
    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 120, "temperature": 0.4},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            if content:
                return content, None
            return "", "Gemini: empty response"
        err = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
        return "", f"Gemini error: {err}"
    except requests.exceptions.Timeout:
        return "", "Gemini: request timed out"
    except Exception as e:
        return "", f"Gemini: {e}"


def _call_anthropic(system: str, user: str, api_key: str, timeout: float = _TRANSLATE_TIMEOUT) -> tuple[str, str | None]:
    """POST to Anthropic Messages API. Returns (content, error)."""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 120,
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": 0.4,
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            content = resp.json()["content"][0]["text"].strip()
            return content, None
        err = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
        return "", f"Anthropic error: {err}"
    except requests.exceptions.Timeout:
        return "", "Anthropic: request timed out"
    except Exception as e:
        return "", f"Anthropic: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Connection test  (called from Settings UI)
# ─────────────────────────────────────────────────────────────────────────────

def test_connection(provider: str, api_key: str) -> tuple[bool, str]:
    """
    Sends a minimal live request to verify the key works.
    Returns (success: bool, message: str).
    """
    if not api_key:
        return False, f"{provider}: no API key configured."

    ping_msg = "Reply with the single word OK."

    if provider == "OpenAI":
        content, err = _call_openai(
            [{"role": "user", "content": ping_msg}], api_key, timeout=_TEST_TIMEOUT
        )
        if err:
            return False, err
        return True, f"OpenAI ✅  response: \"{content[:60]}\""

    if provider == "Gemini":
        content, err = _call_gemini(ping_msg, api_key, timeout=_TEST_TIMEOUT)
        if err:
            return False, err
        return True, f"Gemini ✅  response: \"{content[:60]}\""

    if provider == "Anthropic":
        content, err = _call_anthropic(
            "You are a minimal test assistant.", ping_msg, api_key, timeout=_TEST_TIMEOUT
        )
        if err:
            return False, err
        return True, f"Anthropic ✅  response: \"{content[:60]}\""

    return False, f"Unknown provider: {provider}"


# ─────────────────────────────────────────────────────────────────────────────
# Translation  (called from modules/translation/translator.py)
# ─────────────────────────────────────────────────────────────────────────────

def cloud_translate(provider: str, sign_label: str, lang: str, api_key: str, fallback: str) -> str:
    """
    Calls the selected cloud provider to translate a sign into a natural sentence.
    Falls back to `fallback` on any error.
    """
    lang_name = {"en": "English", "hi": "Hindi", "te": "Telugu"}.get(lang, "English")
    prompt = (
        f"Translate the Indian Sign Language sign '{sign_label}' into a single short, "
        f"natural sentence in {lang_name}. Return only the translation, nothing else."
    )

    if provider == "OpenAI":
        content, err = _call_openai(
            [{"role": "user", "content": prompt}], api_key
        )
        if err:
            print(f"[CloudTranslate] {err}")
            return fallback
        return content or fallback

    if provider == "Gemini":
        content, err = _call_gemini(prompt, api_key)
        if err:
            print(f"[CloudTranslate] {err}")
            return fallback
        return content or fallback

    if provider == "Anthropic":
        content, err = _call_anthropic(
            "You are a concise ISL translation assistant. Return only the translation sentence.",
            prompt,
            api_key,
        )
        if err:
            print(f"[CloudTranslate] {err}")
            return fallback
        return content or fallback

    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Conversational response  (called from modules/ollama/chat.py)
# ─────────────────────────────────────────────────────────────────────────────

def cloud_generate_response(
    provider: str,
    sign: str,
    translation: str,
    person_name: str,
    lang: str,
    api_key: str,
    fallback: str,
) -> str:
    """
    Generates a short empathetic conversational reply via the selected cloud provider.
    Falls back to `fallback` on any error.
    """
    lang_name = {"en": "English", "hi": "Hindi", "te": "Telugu"}.get(lang, "English")
    system = (
        "You are SignBridge AI, a compassionate communication assistant helping "
        "deaf and hard-of-hearing people communicate through Indian Sign Language. "
        f"Respond briefly (1-2 sentences), warmly, and helpfully in {lang_name}. "
        "Never use jargon. Be direct and empathetic."
    )
    user = (
        f"{person_name} just signed '{sign}' which means '{translation}'. "
        "Please respond naturally and helpfully to this communication."
    )

    if provider == "OpenAI":
        content, err = _call_openai(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            api_key,
            timeout=_RESPONSE_TIMEOUT,
        )
        if err:
            print(f"[CloudResponse] {err}")
            return fallback
        return content or fallback

    if provider == "Gemini":
        full_prompt = f"{system}\n\n{user}"
        content, err = _call_gemini(full_prompt, api_key, timeout=_RESPONSE_TIMEOUT)
        if err:
            print(f"[CloudResponse] {err}")
            return fallback
        return content or fallback

    if provider == "Anthropic":
        content, err = _call_anthropic(system, user, api_key, timeout=_RESPONSE_TIMEOUT)
        if err:
            print(f"[CloudResponse] {err}")
            return fallback
        return content or fallback

    return fallback
