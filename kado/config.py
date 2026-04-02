"""Configuration management for kado.

Config lives at ~/.kado/config.toml and stores:
- Anki deck name
- Anki model/note type name
- AnkiConnect URL
- HF model for sentence generation
- Audio settings
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import toml

CONFIG_DIR = Path.home() / ".kado"
CONFIG_PATH = CONFIG_DIR / "config.toml"
AUDIO_DIR = CONFIG_DIR / "audio"


@dataclass
class KadoConfig:
    anki_url: str = "http://localhost:8765"
    anki_deck: str = "Japanese::Vocabulary"
    anki_model: str = "Kado-Japanese"
    sentence_provider: str = "huggingface"  # "huggingface" or "none"
    hf_model: str = ""  # empty = auto-select from defaults
    audio_enabled: bool = True
    audio_lang: str = "ja"

    @classmethod
    def load(cls) -> KadoConfig:
        """Load config from disk, falling back to defaults."""
        cfg = cls()
        if CONFIG_PATH.exists():
            data = toml.load(CONFIG_PATH)
            anki = data.get("anki", {})
            cfg.anki_url = anki.get("url", cfg.anki_url)
            cfg.anki_deck = anki.get("deck", cfg.anki_deck)
            cfg.anki_model = anki.get("model", cfg.anki_model)

            sentences = data.get("sentences", {})
            cfg.sentence_provider = sentences.get("provider", cfg.sentence_provider)
            cfg.hf_model = sentences.get("hf_model", cfg.hf_model)

            audio = data.get("audio", {})
            cfg.audio_enabled = audio.get("enabled", cfg.audio_enabled)
            cfg.audio_lang = audio.get("lang", cfg.audio_lang)

        return cfg

    def save(self) -> None:
        """Persist current config to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        data = {
            "anki": {
                "url": self.anki_url,
                "deck": self.anki_deck,
                "model": self.anki_model,
            },
            "sentences": {
                "provider": self.sentence_provider,
                "hf_model": self.hf_model,
            },
            "audio": {
                "enabled": self.audio_enabled,
                "lang": self.audio_lang,
            },
        }

        with open(CONFIG_PATH, "w") as f:
            toml.dump(data, f)
