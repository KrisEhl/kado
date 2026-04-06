"""Dictionary lookup via Jisho API."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request

from kado.models import VocabCard

JISHO_API = "https://jisho.org/api/v1/search/words"


def lookup(word: str) -> VocabCard:
    """Look up a Japanese word on Jisho and return an enriched VocabCard.

    Raises ValueError if the word is not found.
    """
    params = urllib.parse.urlencode({"keyword": word})
    url = f"{JISHO_API}?{params}"

    req = urllib.request.Request(url, headers={"User-Agent": "kado/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not reach Jisho (network error): {e.reason}"
        ) from e
    except socket.timeout as e:
        raise ConnectionError("Could not reach Jisho (request timed out)") from e

    data = body.get("data", [])
    if not data:
        raise ValueError(f"No results found for '{word}'")

    # Pick the best match — prefer exact kanji/kana match
    entry = _best_match(word, data)

    # Extract reading
    jp = entry.get("japanese", [{}])[0]
    kanji = jp.get("word", word)
    reading = jp.get("reading", "")

    # Extract meanings and POS
    senses = entry.get("senses", [])
    meanings: list[str] = []
    pos_parts: list[str] = []
    for sense in senses[:3]:  # first 3 senses max
        defs = sense.get("english_definitions", [])
        if defs:
            meanings.append("; ".join(defs))
        for p in sense.get("parts_of_speech", []):
            if p and p not in pos_parts:
                pos_parts.append(p)

    meaning_str = " / ".join(meanings) if meanings else ""
    pos_str = ", ".join(pos_parts) if pos_parts else ""

    # Collect tags (JLPT level, common word, etc.)
    tags: list[str] = list(entry.get("tags", []))
    tags.extend(entry.get("jlpt", []))
    if entry.get("is_common"):
        tags.append("common")

    return VocabCard(
        word=kanji,
        reading=reading,
        meaning=meaning_str,
        part_of_speech=pos_str,
        tags=tags,
    )


def _best_match(word: str, data: list[dict]) -> dict:
    """Pick the Jisho result that best matches the query."""
    # Exact match on kanji or reading
    for entry in data:
        for jp in entry.get("japanese", []):
            if jp.get("word") == word or jp.get("reading") == word:
                return entry
    # Fallback: first common result, or just the first result
    for entry in data:
        if entry.get("is_common"):
            return entry
    return data[0]
