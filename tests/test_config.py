"""Tests for KadoConfig load/save/defaults."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import toml

from kado.config import KadoConfig


def test_defaults():
    cfg = KadoConfig()
    assert cfg.anki_url == "http://localhost:8765"
    assert cfg.anki_deck == "Japanese::Vocabulary"
    assert cfg.anki_model == "Kado-Japanese"
    assert cfg.sentence_provider == "ollama"
    assert cfg.hf_model == ""
    assert cfg.ollama_url == "http://localhost:11434"
    assert cfg.ollama_model == ""
    assert cfg.audio_enabled is True
    assert cfg.audio_lang == "ja"


def test_load_returns_defaults_when_no_file(tmp_path):
    fake_config = tmp_path / "config.toml"
    with patch("kado.config.CONFIG_PATH", fake_config):
        cfg = KadoConfig.load()
    assert cfg.anki_url == "http://localhost:8765"
    assert cfg.anki_deck == "Japanese::Vocabulary"
    assert cfg.audio_enabled is True


def test_load_reads_values_from_file(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        toml.dumps({
            "anki": {
                "url": "http://custom:1234",
                "deck": "MyDeck",
                "model": "MyModel",
            },
            "sentences": {
                "provider": "huggingface",
                "hf_model": "Qwen/Qwen2.5-7B",
                "ollama_url": "http://other:11434",
                "ollama_model": "llama3.1:8b",
            },
            "audio": {
                "enabled": False,
                "lang": "en",
            },
        })
    )
    with patch("kado.config.CONFIG_PATH", config_file):
        cfg = KadoConfig.load()

    assert cfg.anki_url == "http://custom:1234"
    assert cfg.anki_deck == "MyDeck"
    assert cfg.anki_model == "MyModel"
    assert cfg.sentence_provider == "huggingface"
    assert cfg.hf_model == "Qwen/Qwen2.5-7B"
    assert cfg.ollama_url == "http://other:11434"
    assert cfg.ollama_model == "llama3.1:8b"
    assert cfg.audio_enabled is False
    assert cfg.audio_lang == "en"


def test_load_partial_file_uses_defaults_for_missing_keys(tmp_path):
    config_file = tmp_path / "config.toml"
    config_file.write_text(toml.dumps({"anki": {"deck": "PartialDeck"}}))
    with patch("kado.config.CONFIG_PATH", config_file):
        cfg = KadoConfig.load()

    assert cfg.anki_deck == "PartialDeck"
    assert cfg.anki_url == "http://localhost:8765"  # default
    assert cfg.sentence_provider == "ollama"         # default


def test_save_roundtrip(tmp_path):
    config_file = tmp_path / "config.toml"
    audio_dir = tmp_path / "audio"

    cfg = KadoConfig(
        anki_url="http://save-test:9999",
        anki_deck="SaveDeck",
        anki_model="SaveModel",
        sentence_provider="none",
        hf_model="some/model",
        ollama_url="http://ollama-test:11434",
        ollama_model="gemma2:9b",
        audio_enabled=False,
        audio_lang="en",
    )

    with patch("kado.config.CONFIG_PATH", config_file), \
         patch("kado.config.CONFIG_DIR", tmp_path), \
         patch("kado.config.AUDIO_DIR", audio_dir):
        cfg.save()

    assert config_file.exists()
    data = toml.load(config_file)
    assert data["anki"]["url"] == "http://save-test:9999"
    assert data["anki"]["deck"] == "SaveDeck"
    assert data["sentences"]["provider"] == "none"
    assert data["sentences"]["ollama_model"] == "gemma2:9b"
    assert data["audio"]["enabled"] is False
    assert data["audio"]["lang"] == "en"


def test_save_then_load_roundtrip(tmp_path):
    config_file = tmp_path / "config.toml"
    audio_dir = tmp_path / "audio"

    original = KadoConfig(
        anki_deck="RoundtripDeck",
        sentence_provider="huggingface",
        audio_enabled=False,
    )

    with patch("kado.config.CONFIG_PATH", config_file), \
         patch("kado.config.CONFIG_DIR", tmp_path), \
         patch("kado.config.AUDIO_DIR", audio_dir):
        original.save()

    with patch("kado.config.CONFIG_PATH", config_file):
        loaded = KadoConfig.load()

    assert loaded.anki_deck == "RoundtripDeck"
    assert loaded.sentence_provider == "huggingface"
    assert loaded.audio_enabled is False
