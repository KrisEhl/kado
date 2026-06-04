"""TTS audio generation for Japanese words.

Primary backend: VOICEVOX (local server, high-quality Japanese TTS).
Fallback: gTTS (Google TTS, requires internet).

VOICEVOX setup:
  1. Download from https://voicevox.hiroshiba.jp/ or run via Docker:
       docker run --rm -p 50021:50021 voicevox/voicevox_engine:cpu-latest
  2. Set voicevox_url in ~/.kado/config.toml:
       [audio]
       voicevox_url = "http://localhost:50021"
       voicevox_speaker = 1
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request

from kado.config import AUDIO_DIR


def generate_audio(
    text: str,
    lang: str = "ja",
    voicevox_url: str = "",
    voicevox_speaker: int = 1,
) -> str:
    """Generate an audio file for the given text.

    Tries VOICEVOX first if a URL is configured, falls back to gTTS.
    Returns the path to the generated audio file (cached by content hash).
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if voicevox_url:
        path = _generate_voicevox(text, voicevox_url, voicevox_speaker)
        if path:
            return path

    return _generate_gtts(text, lang)


def backend_label(path: str) -> str:
    """Return the TTS backend that produced *path*, inferred from its extension.

    VOICEVOX writes .wav; the gTTS fallback writes .mp3.
    """
    return "VOICEVOX" if str(path).lower().endswith(".wav") else "gTTS"


def voicevox_available(url: str) -> bool:
    """Return True if the VOICEVOX engine is reachable at the given URL."""
    try:
        req = urllib.request.Request(f"{url}/version", method="GET")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except (urllib.error.URLError, OSError):
        return False


def voicevox_speakers(url: str) -> list[dict] | None:
    """Return the list of available VOICEVOX speakers, or None on error."""
    try:
        req = urllib.request.Request(f"{url}/speakers", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


# ── VOICEVOX backend ──────────────────────────────────────────────────


def _generate_voicevox(text: str, url: str, speaker: int) -> str | None:
    """Generate audio via VOICEVOX engine. Returns path or None on failure."""
    slug = hashlib.md5(f"voicevox:{speaker}:{text}".encode()).hexdigest()
    path = AUDIO_DIR / f"kado_{slug}.wav"

    if path.exists():
        return str(path)

    try:
        # Step 1: create audio query
        query_url = f"{url}/audio_query?text={urllib.parse.quote(text)}&speaker={speaker}"
        req = urllib.request.Request(query_url, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            query = json.loads(resp.read())

        # Step 2: synthesize WAV
        payload = json.dumps(query).encode()
        synth_url = f"{url}/synthesis?speaker={speaker}"
        req = urllib.request.Request(
            synth_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            wav_bytes = resp.read()

        path.write_bytes(wav_bytes)
        return str(path)

    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError):
        return None


# ── gTTS fallback ─────────────────────────────────────────────────────


def _generate_gtts(text: str, lang: str = "ja") -> str:
    """Generate audio via gTTS. Returns path to .mp3."""
    from gtts import gTTS

    slug = hashlib.md5(f"{lang}:{text}".encode()).hexdigest()
    path = AUDIO_DIR / f"kado_{slug}.mp3"

    if path.exists():
        return str(path)

    tts = gTTS(text=text, lang=lang)
    tts.save(str(path))
    return str(path)


