"""Example sentence generation using open-source models.

Supports two backends:
  - "ollama"       (default) — local models via Ollama, no account needed
  - "huggingface"  — HF's free serverless Inference API

Generates natural Japanese example sentences that incorporate vocabulary
the user has already studied, reinforcing retention.
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.request

from kado.debug import debug_print
from kado.ollama_utils import OLLAMA_URL, ollama_available_models, ollama_resolve_model

# HF models available on the free serverless inference (no token needed).
HF_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "HuggingFaceH4/zephyr-7b-beta",
]

# Ollama models to try, largest first. Override with KADO_OLLAMA_MODEL env var.
OLLAMA_MODELS = [
    "qwen2.5:32b",
    "qwen2.5:14b",
    "qwen2.5:7b",
    "gemma2:27b",
    "gemma2:9b",
    "llama3.1:8b",
    "mistral:7b",
]


def resolve_model_name(
    *,
    provider: str = "ollama",
    model: str | None = None,
    ollama_url: str = "",
) -> str:
    """Resolve which model will be used, without generating anything.

    Returns a human-readable model name like "qwen2.5:7b" or "Qwen/Qwen2.5-72B-Instruct".
    Returns the provider name if no specific model can be determined yet.
    """
    if provider == "ollama":
        base_url = ollama_url or os.environ.get("KADO_OLLAMA_URL", OLLAMA_URL)
        override = model or os.environ.get("KADO_OLLAMA_MODEL", "")
        models_to_try = [override] if override else OLLAMA_MODELS

        available = ollama_available_models(base_url)
        if available is None:
            return "ollama — not running, start with: ollama serve"

        for m in models_to_try:
            resolved = ollama_resolve_model(m, available)
            if resolved:
                return resolved
        recommended = models_to_try[2] if len(models_to_try) > 2 else models_to_try[0]
        return f"ollama — no supported model, run: ollama pull {recommended}"

    elif provider == "huggingface":
        if model:
            return model
        return HF_MODELS[0] if HF_MODELS else "huggingface"

    return provider


def generate_example(
    word: str,
    reading: str,
    meaning: str,
    known_vocab: list[str],
    *,
    provider: str = "ollama",
    model: str | None = None,
    ollama_url: str = "",
) -> tuple[str, str, str]:
    """Generate a Japanese example sentence + English translation.

    Args:
        provider: "ollama" (default) or "huggingface"
        model: Pin a specific model name (overrides auto-selection)
        ollama_url: Ollama base URL (default: http://localhost:11434)

    Returns (japanese_sentence, english_translation, model_used).
    Falls back to empty strings if generation fails.
    """
    context_words = _pick_context_words(known_vocab, max_words=5)
    context_hint = ""
    if context_words:
        context_hint = (
            f"\nTry to naturally incorporate one or two of these previously "
            f"studied words: {', '.join(context_words)}."
        )

    prompt = (
        f"Generate ONE natural Japanese example sentence using the word "
        f"「{word}」 ({reading}, meaning: {meaning}).{context_hint}\n\n"
        f"Requirements:\n"
        f"- The sentence should be natural, everyday Japanese\n"
        f"- Aim for JLPT N4 grammar and vocabulary level\n"
        f"- Keep it under 30 characters\n"
        f"- Respond ONLY with two lines, nothing else:\n"
        f"Line 1: The Japanese sentence\n"
        f"Line 2: The English translation"
    )

    if provider == "ollama":
        result = _generate_via_ollama(prompt, model=model, ollama_url=ollama_url)
        if result:
            ja, en, model_used = result
            return ja, en, model_used
        _warn_ollama_missing(model=model, ollama_url=ollama_url)
        return "", "", ""

    elif provider == "huggingface":
        result = _generate_via_hf(prompt, model=model)
        if result:
            ja, en, model_used = result
            return ja, en, model_used
        return "", "", ""

    return "", "", ""


# ── Ollama backend ─────────────────────────────────────────────────


def _warn_ollama_missing(*, model: str | None = None, ollama_url: str = "") -> None:
    """Print a user-friendly warning explaining why Ollama generation failed."""
    base_url = ollama_url or os.environ.get("KADO_OLLAMA_URL", OLLAMA_URL)
    available = ollama_available_models(base_url)

    if available is None:
        print(
            "   Ollama is not running. Start it with:\n     ollama serve",
            file=sys.stderr,
        )
        return

    if model:
        # User pinned a specific model
        print(
            f"   Model '{model}' is not installed. Install it with:\n"
            f"     ollama pull {model}",
            file=sys.stderr,
        )
        return

    # No pinned model — show what we tried vs what's available
    tried = OLLAMA_MODELS
    installed = sorted(available)
    recommended = tried[2] if len(tried) > 2 else tried[0]  # qwen2.5:7b

    print(
        f"   No compatible Ollama model found.\n"
        f"   Installed: {', '.join(installed) if installed else '(none)'}\n"
        f"   Supported: {', '.join(tried)}\n"
        f"   Install one with:\n"
        f"     ollama pull {recommended}",
        file=sys.stderr,
    )


def _generate_via_ollama(
    prompt: str,
    *,
    model: str | None = None,
    ollama_url: str = "",
) -> tuple[str, str, str] | None:
    """Generate a sentence using a local Ollama model.

    Returns (japanese, english, model_name) or None.
    """
    base_url = ollama_url or os.environ.get("KADO_OLLAMA_URL", OLLAMA_URL)
    override = model or os.environ.get("KADO_OLLAMA_MODEL", "")

    models_to_try = [override] if override else OLLAMA_MODELS

    available = ollama_available_models(base_url)
    if available is None:
        return None  # Ollama not running

    for m in models_to_try:
        resolved = ollama_resolve_model(m, available)
        if not resolved:
            continue

        try:
            payload = json.dumps(
                {
                    "model": resolved,
                    "messages": [{"role": "user", "content": f"/no_think\n{prompt}"}],
                    "stream": False,
                    "options": {
                        "num_ctx": 4096,
                        "num_predict": 200,
                        "temperature": 0.7,
                        "think": False,
                    },
                }
            ).encode()

            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                text = data.get("message", {}).get("content", "").strip()
                if text:
                    ja, en = _parse_response(text)
                    return ja, en, resolved
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as e:
            debug_print(f"Ollama {resolved}: {e}")
            continue

    return None


# ── HuggingFace backend ────────────────────────────────────────────


def _generate_via_hf(
    prompt: str, *, model: str | None = None
) -> tuple[str, str, str] | None:
    """Generate a sentence using HF's free serverless Inference API.

    Returns (japanese, english, model_name) or None.
    """
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        debug_print("huggingface_hub not installed, run: pip install huggingface-hub")
        return None

    models_to_try = [model] if model else HF_MODELS
    last_error = None

    for m in models_to_try:
        try:
            client = InferenceClient(model=m)
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            ja, en = _parse_response(text)
            return ja, en, m
        except (OSError, KeyError, IndexError, ValueError) as e:
            last_error = f"{m}: {e}"
            continue

    if last_error:
        debug_print(f"HF: {last_error}")
    return None


# ── Shared helpers ─────────────────────────────────────────────────


def _parse_response(text: str) -> tuple[str, str]:
    """Extract Japanese sentence and English translation from model output."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]

    cleaned = []
    for line in lines:
        for prefix in ("1.", "2.", "Line 1:", "Line 2:", "Japanese:", "English:", "- "):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
        cleaned.append(line)

    if len(cleaned) >= 2:
        return cleaned[0], cleaned[1]
    elif len(cleaned) == 1:
        return cleaned[0], ""
    return "", ""


def _pick_context_words(known: list[str], max_words: int = 5) -> list[str]:
    """Pick a random sample of known vocab to use as sentence context."""
    if not known:
        return []
    sample_size = min(max_words, len(known))
    return random.sample(known, sample_size)
