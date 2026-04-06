"""Shared Ollama helpers used by multiple kado modules."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"

# Vision models tried in order of preference when auto-selecting.
# Override with KADO_OLLAMA_VISION_MODEL env var to pin a specific model.
OLLAMA_VISION_MODELS = [
    "qwen2.5vl:72b",
    "qwen2.5vl:7b",
    "llama3.2-vision:11b",
    "llava:34b",
    "llava:13b",
    "llava:7b",
    "minicpm-v",
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
