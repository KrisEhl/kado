"""Shared Ollama helpers used by multiple kado modules."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"

# Vision models tried in order of preference when auto-selecting.
# Override with KADO_OLLAMA_VISION_MODEL env var to pin a specific model.
# Preference order favours NON-THINKING OCR models. OCR is pure transcription —
# there is nothing to reason about — so a "thinking" VL model (qwen3-vl) just
# burns thousands of tokens reasoning before the answer, which is slow and can
# return empty content when the reasoning exhausts the output budget. qwen2.5vl
# is instruct-tuned (no thinking) and is the reliable default for Japanese tables.
OLLAMA_VISION_MODELS = [
    "qwen2.5vl:32b",        # best accuracy, non-thinking (~21GB Q4)
    "qwen2.5vl:7b",         # fast, non-thinking, reliable (~6GB)
    "qwen2.5vl:72b",        # non-thinking, heaviest
    "glm-ocr",              # ~2-4GB, #1 OmniDocBench, Japanese support
    "deepseek-ocr:3b",      # ~6-8GB, 100+ langs, confirmed Japanese
    "minicpm-v",
    "llama3.2-vision:11b",
    "qwen3-vl:8b",          # thinking model — works (num_predict:-1) but slow; last resort
    # llava intentionally excluded: it can't reliably read Japanese kanji tables
    # and hallucinates plausible-but-wrong vocabulary instead of transcribing.
]


def ollama_available_models(base_url: str) -> set[str] | None:
    """Return set of installed Ollama model names, or None if Ollama isn't running."""
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            return {m["name"] for m in data.get("models", [])}
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def ollama_resolve_model(requested: str, available: set[str]) -> str | None:
    """Resolve a model name to an installed Ollama model.

    Handles cases like requesting 'llava:13b' when 'llava:latest' is installed,
    or 'qwen2.5:7b' matching 'qwen2.5:latest' — returns the actual installed
    name so the API call succeeds.
    """
    if not requested:
        return None
    if requested in available:
        return requested
    # Short-name match: 'llava' matches 'llava:latest', 'qwen2.5' matches 'qwen2.5:7b'
    short = requested.split(":")[0]
    for name in available:
        if name.split(":")[0] == short:
            return name
    return None
