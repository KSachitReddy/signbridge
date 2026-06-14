"""
Local LLM inference via llama-cpp-python (GGUF models).
Provides a single answer() function that sends a RAG prompt to the model.

Model is cached after first load so repeated calls don't reload weights.
Supports an Ollama fallback if llama-cpp-python is not installed.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

_model_lock = threading.Lock()
_llm_instance = None
_loaded_model_path: Optional[str] = None

DEFAULT_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "gguf")
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the context "
    "provided below. If the answer is not in the context, say 'I don't know.' "
    "Be concise and accurate."
)


def _find_gguf(model_dir: str) -> Optional[str]:
    """Return the first .gguf file found in model_dir, or None."""
    if not os.path.isdir(model_dir):
        return None
    for name in sorted(os.listdir(model_dir)):
        if name.endswith(".gguf"):
            return os.path.join(model_dir, name)
    return None


def _load_llama(model_path: str, n_ctx: int = 4096, n_threads: int = 4):
    global _llm_instance, _loaded_model_path
    with _model_lock:
        if _llm_instance is not None and _loaded_model_path == model_path:
            return _llm_instance
        from llama_cpp import Llama
        _llm_instance = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            verbose=False,
        )
        _loaded_model_path = model_path
        return _llm_instance


def _format_prompt(question: str, context: str, system: str) -> str:
    # ChatML format compatible with Qwen2.5 / Mistral / LLaMA-3 GGUF models
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def answer_with_llama(
    question: str,
    context: str,
    model_path: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.1,
    n_ctx: int = 4096,
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Run local GGUF inference. Returns generated answer text."""
    if model_path is None:
        model_path = _find_gguf(DEFAULT_MODEL_DIR)
    if model_path is None or not os.path.isfile(model_path):
        return (
            "No GGUF model found. Run `python scripts/download_gguf_model.py` "
            "to download a model, or place a .gguf file in models/gguf/."
        )
    llm = _load_llama(model_path, n_ctx=n_ctx)
    prompt = _format_prompt(question, context, system)
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<|im_end|>", "<|endoftext|>"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def answer_with_ollama(
    question: str,
    context: str,
    model: str = "llama3.2",
    system: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Fallback: use Ollama HTTP API if llama-cpp-python is unavailable."""
    import requests
    payload = {
        "model": model,
        "prompt": (
            f"{system}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        ),
        "stream": False,
    }
    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as exc:
        return f"Ollama request failed: {exc}"


def answer(
    question: str,
    context: str,
    model_path: Optional[str] = None,
    use_ollama_fallback: bool = True,
    ollama_model: str = "llama3.2",
    **kwargs,
) -> str:
    """
    Generate an answer for question using context.
    Tries llama-cpp-python first; falls back to Ollama if requested.
    """
    try:
        return answer_with_llama(question, context, model_path=model_path, **kwargs)
    except ImportError:
        if use_ollama_fallback:
            return answer_with_ollama(question, context, model=ollama_model)
        return (
            "llama-cpp-python is not installed. "
            "Run: pip install llama-cpp-python"
        )
