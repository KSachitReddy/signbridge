import base64
import re
from modules.database import get_setting, save_setting

# Basic Obfuscation Key XOR encryption/decryption
XOR_KEY = 101 # Fixed salt for secure obfuscation

def encrypt_key(api_key: str) -> str:
    """Encrypts raw API key using base64 XOR encoding to protect keys in the database."""
    if not api_key:
        return ""
    xored = bytearray([ord(c) ^ XOR_KEY for c in api_key])
    return base64.b64encode(xored).decode('utf-8')

def decrypt_key(encrypted_str: str) -> str:
    """Decrypts base64 XOR encoded API keys."""
    if not encrypted_str:
        return ""
    try:
        decoded = base64.b64decode(encrypted_str.encode('utf-8'))
        decrypted = "".join([chr(b ^ XOR_KEY) for b in decoded])
        return decrypted
    except Exception:
        return ""

def validate_key_format(provider: str, api_key: str) -> bool:
    """Validates the basic syntax structure of provider API keys to prevent configuration errors."""
    if not api_key:
        return False
        
    p = provider.lower()
    if p == "openai":
        # Standard OpenAI keys start with sk- and are about 48+ chars
        return bool(re.match(r"^sk-[A-Za-z0-9\-_]{40,}$", api_key))
    elif p == "gemini":
        # Gemini keys are generally alphanumeric 39 chars
        return len(api_key) >= 35
    elif p == "anthropic":
        # Anthropic keys start with sk-ant-
        return bool(re.match(r"^sk-ant-[A-Za-z0-9\-_]{40,}$", api_key))
        
    return len(api_key) > 10 # Default validation length

def save_provider_key(provider: str, api_key: str):
    """Encrypts and saves the key in settings."""
    encrypted = encrypt_key(api_key)
    save_setting(f"api_key_{provider.lower()}", encrypted)

def get_provider_key(provider: str) -> str:
    """Retrieves and decrypts the provider API key."""
    encrypted = get_setting(f"api_key_{provider.lower()}", "")
    return decrypt_key(encrypted)
