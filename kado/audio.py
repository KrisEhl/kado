"""TTS audio generation for Japanese words."""

from __future__ import annotations

import hashlib

from kado.config import AUDIO_DIR


def generate_audio(word: str, lang: str = "ja") -> str:
    """Generate an mp3 pronunciation for a Japanese word using gTTS.

    Returns the path to the generated audio file. Files are cached by
    content hash so the same word won't be regenerated.
    """
    from gtts import gTTS

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    # Deterministic filename based on word and language
    slug = hashlib.md5(f"{lang}:{word}".encode()).hexdigest()
    filename = f"kado_{slug}.mp3"
    path = AUDIO_DIR / filename

    if path.exists():
        return str(path)

    tts = gTTS(text=word, lang=lang)
    tts.save(str(path))
    return str(path)
