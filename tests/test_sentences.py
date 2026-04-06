"""Tests for sentences.py — model resolution and response parsing."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kado.ollama_utils import ollama_resolve_model as _resolve_ollama_model
from kado.sentences import (
    _parse_response,
    resolve_model_name,
)


# ── _resolve_ollama_model (now in ollama_utils) ───────────────────────


def test_resolve_exact_match():
    available = {"qwen2.5:7b", "llama3.1:8b"}
    assert _resolve_ollama_model("qwen2.5:7b", available) == "qwen2.5:7b"


def test_resolve_short_name_match():
    available = {"qwen2.5:latest", "llama3.1:8b"}
    assert _resolve_ollama_model("qwen2.5:7b", available) == "qwen2.5:latest"


def test_resolve_no_match_returns_none():
    available = {"llama3.1:8b"}
    assert _resolve_ollama_model("qwen2.5:7b", available) is None


def test_resolve_empty_requested_returns_none():
    available = {"qwen2.5:7b"}
    assert _resolve_ollama_model("", available) is None


def test_resolve_empty_available_returns_none():
    assert _resolve_ollama_model("qwen2.5:7b", set()) is None


def test_resolve_picks_first_matching_short_name():
    # If multiple variants exist, returns whichever is found first in set iteration
    available = {"qwen2.5:14b", "qwen2.5:7b"}
    result = _resolve_ollama_model("qwen2.5:32b", available)
    # Should return one of the qwen2.5 variants
    assert result is not None
    assert result.startswith("qwen2.5:")


# ── resolve_model_name ────────────────────────────────────────────────


def test_resolve_model_name_ollama_not_running():
    with patch("kado.sentences.ollama_available_models", return_value=None):
        result = resolve_model_name(provider="ollama", ollama_url="http://localhost:11434")
    assert "not running" in result


def test_resolve_model_name_ollama_model_found():
    with patch("kado.sentences.ollama_available_models", return_value={"qwen2.5:7b"}):
        result = resolve_model_name(
            provider="ollama",
            model="qwen2.5:7b",
            ollama_url="http://localhost:11434",
        )
    assert result == "qwen2.5:7b"


def test_resolve_model_name_ollama_auto_selects():
    with patch("kado.sentences.ollama_available_models", return_value={"llama3.1:8b"}):
        result = resolve_model_name(provider="ollama", ollama_url="http://localhost:11434")
    assert result == "llama3.1:8b"


def test_resolve_model_name_ollama_no_supported_model():
    with patch("kado.sentences.ollama_available_models", return_value={"custom:model"}):
        result = resolve_model_name(provider="ollama", ollama_url="http://localhost:11434")
    assert "ollama" in result and "no supported model" in result


def test_resolve_model_name_huggingface_explicit_model():
    result = resolve_model_name(provider="huggingface", model="Qwen/Qwen2.5-7B-Instruct")
    assert result == "Qwen/Qwen2.5-7B-Instruct"


def test_resolve_model_name_huggingface_auto():
    result = resolve_model_name(provider="huggingface")
    # Should return the first HF_MODELS entry
    assert "Qwen" in result or "llama" in result.lower() or "mistral" in result.lower()


def test_resolve_model_name_none_provider():
    result = resolve_model_name(provider="none")
    assert result == "none"


# ── _parse_response ───────────────────────────────────────────────────


def test_parse_response_two_lines():
    text = "彼は毎朝コーヒーを飲む。\nHe drinks coffee every morning."
    ja, en = _parse_response(text)
    assert ja == "彼は毎朝コーヒーを飲む。"
    assert en == "He drinks coffee every morning."


def test_parse_response_strips_numbered_prefixes():
    text = "1. 彼は毎朝コーヒーを飲む。\n2. He drinks coffee every morning."
    ja, en = _parse_response(text)
    assert ja == "彼は毎朝コーヒーを飲む。"
    assert en == "He drinks coffee every morning."


def test_parse_response_strips_line_labels():
    text = "Line 1: 東京は大きい。\nLine 2: Tokyo is large."
    ja, en = _parse_response(text)
    assert ja == "東京は大きい。"
    assert en == "Tokyo is large."


def test_parse_response_strips_language_labels():
    text = "Japanese: 東京は大きい。\nEnglish: Tokyo is large."
    ja, en = _parse_response(text)
    assert ja == "東京は大きい。"
    assert en == "Tokyo is large."


def test_parse_response_single_line_returns_empty_translation():
    text = "東京は大きい。"
    ja, en = _parse_response(text)
    assert ja == "東京は大きい。"
    assert en == ""


def test_parse_response_empty_text():
    ja, en = _parse_response("")
    assert ja == ""
    assert en == ""


def test_parse_response_skips_blank_lines():
    text = "\n\n彼は毎朝コーヒーを飲む。\n\nHe drinks coffee every morning.\n\n"
    ja, en = _parse_response(text)
    assert ja == "彼は毎朝コーヒーを飲む。"
    assert en == "He drinks coffee every morning."


def test_parse_response_dash_prefix_stripped():
    text = "- 東京は大きい。\n- Tokyo is large."
    ja, en = _parse_response(text)
    assert ja == "東京は大きい。"
    assert en == "Tokyo is large."
