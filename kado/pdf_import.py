"""Parse Japanese vocabulary from PDF files (e.g. J.Bridge vocab lists).

Supports both text-based and scanned/image PDFs.
- Text PDFs: pdfplumber for direct table extraction
- Scanned PDFs (best):  HF vision model via free `hf auth login`
- Scanned PDFs (local): tesseract OCR with spatial column reconstruction
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

from kado.models import VocabCard
from kado.debug import debug_print
from kado.ollama_utils import (
    OLLAMA_URL,
    OLLAMA_VISION_MODELS,
    ollama_available_models,
    ollama_resolve_model,
)

# Na-adjective endings that should NOT be canonicalized to word(な)
_NA_ADJ_EXCEPTIONS = frozenset({"みんな", "こんな", "そんな", "あんな", "どんな"})


def parse_vocab_pdf(
    path: str,
    use_vision: bool = True,
    llm_cleanup: bool = True,
    pages: set[int] | None = None,
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
    ollama_vision_model: str = "",
) -> list[VocabCard]:
    """Extract vocabulary cards from a PDF file.

    For text PDFs: pdfplumber extracts directly.
    For scanned PDFs: runs all available sources and merges results.

    Args:
        provider: "ollama" (default) or "huggingface" — which LLM backend for cleanup/reconstruction
        ollama_url: Override Ollama base URL
        ollama_model: Pin a specific Ollama model for LLM cleanup
        ollama_vision_model: Pin a specific Ollama model for vision extraction
    """
    cards = _extract_from_text(path, pages=pages)
    if cards:
        return cards

    # Scanned PDF — run all sources and merge
    return _extract_scanned(
        path,
        use_vision=use_vision,
        llm_cleanup=llm_cleanup,
        pages=pages,
        provider=provider,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        ollama_vision_model=ollama_vision_model,
    )


def _extract_scanned(
    path: str,
    use_vision: bool,
    llm_cleanup: bool,
    pages: set[int] | None,
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
    ollama_vision_model: str = "",
) -> list[VocabCard]:
    """Run all available extraction methods on a scanned PDF and merge results.

    Priority for overlapping words (best data wins):
      vision > llm > ocr

    Source labels:
      - "vision"  = found by vision model (may also be in llm/ocr)
      - "ocr"     = found by OCR column parsing + confirmed by LLM
      - "llm"     = reconstructed from German by LLM only
      - "ocr"     = found by OCR only (when LLM is disabled)
    """
    # Collect results from each source
    vision_cards: list[VocabCard] = []
    llm_cards: list[VocabCard] = []
    ocr_cards: list[VocabCard] = []

    # 1. Vision model first — runs before LLM so the vision model loads into VRAM
    # without having to evict the text model first (Ollama loads one model at a time)
    if use_vision:
        vision_cards = _extract_via_vision(
            path, pages=pages, provider=provider, ollama_url=ollama_url,
            ollama_vision_model=ollama_vision_model,
        )

    # 2. OCR column parsing (always runs — it's local and fast-ish)
    try:
        ocr_cards = _extract_via_ocr_columns(path, pages=pages)
    except RuntimeError as e:
        debug_print(f"OCR unavailable: {e}")

    # 3. LLM reconstruction from raw OCR text
    if llm_cleanup and ocr_cards is not None:
        try:
            page_dumps = _get_raw_ocr_pages(path, pages=pages)
            llm_cards = _llm_reconstruct_from_ocr(
                page_dumps,
                provider=provider,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
            )
        except RuntimeError:
            pass  # tesseract not available

    # Merge: vision > llm > ocr
    return _merge_sources(
        ocr_cards,
        llm_cards,
        vision_cards,
        provider=provider,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )


def _merge_sources(
    ocr_cards: list[VocabCard],
    llm_cards: list[VocabCard],
    vision_cards: list[VocabCard],
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
) -> list[VocabCard]:
    """Merge cards from all sources. Higher-quality source wins on overlap.

    Vision > LLM > OCR for data quality.
    Source label reflects how the word was found:
      - "vision" if vision found it
      - "ocr" if both OCR and LLM found it (confirmed)
      - "llm" if only LLM found it
      - "ocr" if only OCR found it
    """
    # Translate vision German meanings to English before merging
    if vision_cards:
        vision_cards = _llm_translate_vision_cards(
            vision_cards,
            provider=provider,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )

    # Normalize words for matching (full-width ↔ half-width brackets, する/な forms)
    ocr_norm = {_dedup_key(c.word) for c in ocr_cards}

    # All non-vision words — used to detect truncated vision hallucinations
    other_words = [_normalize_word(c.word) for c in ocr_cards + llm_cards if c.word]

    merged: list[VocabCard] = []
    seen: set[str] = set()  # dedup keys (base forms, する/な stripped)
    merged_words: set[str] = set()  # full normalized words already accepted

    # Pass 1: vision cards — best quality, normalize word to canonical form.
    for card in vision_cards:
        if not card.word:
            continue
        card.word = _normalize_word(card.word)
        card.reading = _clean_reading(card.reading)
        key = _dedup_key(card.word)
        if key in seen:
            continue
        if _is_garbage_meaning(card.meaning):
            debug_print(f"Vision: discarding '{card.word}' — garbage meaning: '{card.meaning}'")
            continue
        # Discard if this looks like a truncated prefix of a longer word found elsewhere
        if any(w.startswith(card.word) and len(w) > len(card.word) for w in other_words):
            debug_print(f"Vision: discarding '{card.word}' — truncated prefix of a longer word")
            continue
        card.source = "vision"
        seen.add(key)
        merged_words.add(card.word)
        merged.append(card)

    # Pass 2: LLM cards — clean English meanings
    for card in llm_cards:
        if not card.word:
            continue
        norm = _normalize_word(card.word)
        key = _dedup_key(norm)
        if key in seen:
            continue
        # Skip if this word is already covered as a substring of a longer accepted word
        # e.g. 半球 is redundant if 南半球 was already added
        if any(norm in w and len(w) > len(norm) for w in merged_words):
            debug_print(f"LLM: discarding '{norm}' — substring of already-accepted word")
            continue
        card.word = norm
        card.reading = _clean_reading(card.reading)
        card.source = "ocr" if key in ocr_norm else "llm"
        seen.add(key)
        merged_words.add(norm)
        merged.append(card)

    # Pass 3: OCR-only cards — clean up with LLM before including
    ocr_leftovers = []
    for card in ocr_cards:
        if not card.word:
            continue
        key = _dedup_key(card.word)
        if key in seen:
            continue
        ocr_leftovers.append(card)

    if ocr_leftovers:
        cleaned = _llm_fix_ocr_cards(
            ocr_leftovers,
            provider=provider,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )
        for card in cleaned:
            card.reading = _clean_reading(card.reading)
            key = _dedup_key(card.word)
            if key in seen:
                continue
            if any(card.word in w and len(w) > len(card.word) for w in merged_words):
                debug_print(f"OCR: discarding '{card.word}' — substring of already-accepted word")
                continue
            if _is_garbage_meaning(card.meaning):
                debug_print(f"OCR: discarding '{card.word}' — garbage meaning after LLM cleanup")
                continue
            seen.add(key)
            merged_words.add(card.word)
            merged.append(card)

    return merged


def _llm_fix_ocr_cards(
    cards: list[VocabCard],
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
) -> list[VocabCard]:
    """Quick LLM pass to fix OCR-only cards: correct kanji and translate German → English."""
    entries = []
    for i, c in enumerate(cards):
        entries.append(
            {
                "id": i,
                "ocr_word": c.word,
                "ocr_reading": c.reading,
                "ocr_meaning": c.meaning,
            }
        )

    batch_json = json.dumps(entries, ensure_ascii=False)

    prompt = (
        "These are Japanese vocabulary entries from OCR with errors.\n"
        "For each entry:\n"
        "- Fix the Japanese word (correct wrong kanji, e.g. 天主党→天主堂)\n"
        "- Fix the hiragana reading\n"
        "- The meaning is noisy German — translate it to clean, concise ENGLISH\n"
        "- If the entry is garbage (not a real word), set word to empty string\n\n"
        f"Entries: {batch_json}\n\n"
        "Return ONLY a JSON array, no markdown:\n"
        '[{"id": 0, "word": "浦上天主堂", "reading": "うらかみてんしゅどう", "meaning": "Urakami Cathedral"}, ...]'
    )

    text = _llm_chat(
        prompt, provider=provider, ollama_url=ollama_url, ollama_model=ollama_model
    )
    if not text:
        return cards

    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return cards
        data = json.loads(match.group())

        result: list[VocabCard] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            word = str(entry.get("word", "")).strip()
            reading = str(entry.get("reading", "")).strip()
            meaning = str(entry.get("meaning", "")).strip()
            if word and _has_japanese(word):
                result.append(
                    VocabCard(word=word, reading=reading, meaning=meaning, source="ocr")
                )
        return result if result else cards
    except (json.JSONDecodeError, KeyError):
        return cards


def _llm_translate_vision_cards(
    cards: list[VocabCard],
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
) -> list[VocabCard]:
    """Translate German meanings from vision cards to English via LLM."""
    if not cards:
        return cards

    entries = []
    for i, c in enumerate(cards):
        entries.append(
            {
                "id": i,
                "word": c.word,
                "reading": c.reading,
                "german_meaning": c.meaning,
            }
        )

    batch_json = json.dumps(entries, ensure_ascii=False)

    prompt = (
        "These are Japanese vocabulary entries with GERMAN meanings.\n"
        "Translate each meaning to concise ENGLISH.\n"
        "Keep the Japanese word and reading exactly as-is.\n\n"
        f"Entries: {batch_json}\n\n"
        "Return ONLY a JSON array, no markdown:\n"
        '[{"id": 0, "word": "発展(する)", "reading": "はってん", "meaning": "development; to develop"}, ...]'
    )

    text = _llm_chat(
        prompt, provider=provider, ollama_url=ollama_url, ollama_model=ollama_model
    )
    if not text:
        return cards

    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            return cards
        data = json.loads(match.group())

        result: list[VocabCard] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("id", -1)
            meaning = str(entry.get("meaning", "")).strip()
            if 0 <= idx < len(cards):
                card = cards[idx]
                if meaning:
                    card.meaning = meaning
                result.append(card)
            else:
                word = str(entry.get("word", "")).strip()
                reading = str(entry.get("reading", "")).strip()
                if word and meaning:
                    result.append(
                        VocabCard(
                            word=word, reading=reading, meaning=meaning, source="vision"
                        )
                    )

        # Return translated cards + any originals not in response
        returned_ids = {
            entry.get("id", -1) for entry in data if isinstance(entry, dict)
        }
        for i, c in enumerate(cards):
            if i not in returned_ids:
                result.append(c)
        return result
    except (json.JSONDecodeError, KeyError):
        return cards


# ── Text-based extraction ────────────────────────────────────────────


def _extract_from_text(path: str, pages: set[int] | None = None) -> list[VocabCard]:
    import pdfplumber

    cards: list[VocabCard] = []
    seen: set[str] = set()

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            if pages and page_num not in pages:
                continue
            if not page.chars:
                continue
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                for card in _parse_table(table):
                    if card.word and card.word not in seen:
                        card.source = "text"
                        seen.add(card.word)
                        cards.append(card)
    return cards


def _parse_table(table: list[list[str | None]]) -> list[VocabCard]:
    if len(table) < 2:
        return []
    header = [_clean(c) for c in table[0]]
    col_map = _detect_text_columns(header)
    cards: list[VocabCard] = []

    for row in table[1:]:
        if not row or all(not _clean(c) for c in row):
            continue
        cells = [_clean(c) for c in row]
        while len(cells) < 5:
            cells.append("")

        word = cells[col_map.get("word", 2)]
        reading = cells[col_map.get("reading", 3)]
        meaning = cells[col_map.get("meaning", 4)]

        if not word and not reading:
            continue
        if word in ("単語", "単語（漢字）", "単語(漢字)"):
            continue
        if not word and reading:
            word, reading = reading, ""
        if word and _is_all_kana(word) and reading == word:
            reading = ""
        meaning = re.sub(r"\s*\d+\s*$", "", meaning).strip()
        if word:
            cards.append(VocabCard(word=word, reading=reading, meaning=meaning))
    return cards


def _detect_text_columns(header: list[str]) -> dict[str, int]:
    col_map: dict[str, int] = {}
    for i, h in enumerate(header):
        if not h:
            continue
        if "単語" in h and ("漢字" in h or "読" not in h):
            col_map["word"] = i
        elif "読み" in h or "読み方" in h:
            col_map["reading"] = i
        elif "意味" in h and "漢字" not in h:
            col_map["meaning"] = i
    col_map.setdefault("word", 2)
    col_map.setdefault("reading", 3)
    col_map.setdefault("meaning", 4)
    return col_map


# ── Vision-based extraction (HF free account) ───────────────────────


def _extract_via_vision(
    path: str,
    pages: set[int] | None = None,
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_vision_model: str = "",
) -> list[VocabCard]:
    """Use an HF vision model to extract vocab from scanned PDF pages.

    Requires a free HF account: run `hf auth login` or `huggingface-cli login`.
    No payment, no API key purchase — just a free account.

    Args:
        provider: "ollama" (skip HF vision) or "huggingface" (try Ollama first, fall back to HF)
        ollama_url: Override Ollama base URL
    """
    import base64
    import tempfile

    try:
        from huggingface_hub import InferenceClient
        from pdf2image import convert_from_path
    except ImportError:
        return []

    images = convert_from_path(path, dpi=150)
    all_cards: list[VocabCard] = []
    seen: set[str] = set()

    prompt = (
        "This image is a Japanese vocabulary list table from a textbook.\n"
        "Extract vocabulary entries that are CLEARLY VISIBLE in the table.\n\n"
        "The table columns are: 漢字 | 漢字の意味 | 単語（漢字）| 単語（読み方）| 単語の意味\n"
        "I need columns 3, 4, and 5: the word, reading, and meaning.\n\n"
        "For each entry, extract:\n"
        "- word: the Japanese word (単語) in kanji or katakana from column 3\n"
        "- reading: the hiragana reading (読み方) from column 4, empty string if none\n"
        "- meaning: the meaning/translation (単語の意味) from column 5\n\n"
        "IMPORTANT RULES:\n"
        "- Only include rows where you can clearly read BOTH the word AND the meaning\n"
        "- If a meaning cell contains an image, graphic, or is unclear, omit that row entirely\n"
        "- Do NOT guess, infer, or hallucinate — only transcribe what you can actually see\n"
        "- Include katakana-only words (like エジプト, ガイドブック)\n\n"
        "Return ONLY a JSON array, no markdown fences, no explanation:\n"
        '[{"word": "観光する", "reading": "かんこうする", "meaning": "Besichtigen v. Sehenswürdigkeiten"}, ...]'
    )

    hf_models = [
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ]

    # Ollama vision models to try
    ollama_vision_model = (
        ollama_vision_model
        or os.environ.get("KADO_OLLAMA_VISION_MODEL", "")
    )
    ollama_vision_models = [ollama_vision_model] if ollama_vision_model else OLLAMA_VISION_MODELS

    for page_num, image in enumerate(images, 1):
        if pages and page_num not in pages:
            continue
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_name = tmp.name
            image.save(tmp_name, "JPEG", quality=85)
        try:
            with open(tmp_name, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        finally:
            os.unlink(tmp_name)
        debug_print(f"Vision image: {len(b64) // 1024}KB base64")

        # Try Ollama vision first
        page_cards = _try_ollama_vision(b64, prompt, ollama_vision_models, ollama_url)
        if page_cards:
            for card in page_cards:
                if card.word and card.word not in seen:
                    card.source = "vision"
                    seen.add(card.word)
                    all_cards.append(card)
            continue  # next page

        # Fall back to HuggingFace vision (unless provider is "ollama")
        if provider == "huggingface":
            for model in hf_models:
                try:
                    client = InferenceClient(model=model)
                    response = client.chat_completion(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{b64}"
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                        max_tokens=4000,
                        temperature=0.1,
                    )
                    text = response.choices[0].message.content.strip()
                    page_cards = _parse_llm_json(text)
                    for card in page_cards:
                        if card.word and card.word not in seen:
                            card.source = "vision"
                            seen.add(card.word)
                            all_cards.append(card)
                    break  # success, move to next page
                except (OSError, KeyError, IndexError, ValueError):
                    continue

    return all_cards


def _try_ollama_vision(
    b64_image: str, prompt: str, models: list[str], ollama_url: str = ""
) -> list[VocabCard]:
    """Try to extract vocab from an image using Ollama's vision API."""
    base_url = ollama_url or os.environ.get("KADO_OLLAMA_URL", OLLAMA_URL)

    available_names = ollama_available_models(base_url)
    if available_names is None:
        return []

    for model in models:
        resolved = ollama_resolve_model(model, available_names)
        if not resolved:
            continue

        try:
            payload = json.dumps(
                {
                    "model": resolved,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [b64_image],
                        }
                    ],
                    "stream": False,
                    "options": {
                        "num_ctx": 8192,
                        "num_predict": 2048,
                        "temperature": 0.1,
                    },
                }
            ).encode()

            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read())
                text = data.get("message", {}).get("content", "").strip()
                if text:
                    debug_print(f"Ollama vision ({resolved}): OK")
                    return _parse_llm_json(text)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            debug_print(f"Ollama vision {resolved}: HTTP {e.code} — {body}")
            continue
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as e:
            debug_print(f"Ollama vision {resolved}: {e}")
            continue

    return []


# Table headers and structural text that should never be vocab entries
_TABLE_NOISE = {
    "漢字",
    "漢字の意味",
    "単語",
    "単語の意味",
    "読み方",
    "単語(漢字)",
    "単語(読み方)",
    "単語（漢字）",
    "単語（読み方）",
    "ステップ",
    "課",
}


def _is_table_noise(word: str) -> bool:
    """Check if a word is a table header or structural text, not actual vocab."""
    norm = _normalize_word(word)
    return (
        norm in _TABLE_NOISE or _normalize_word(word.replace("　", "")) in _TABLE_NOISE
    )


def _parse_llm_json(text: str) -> list[VocabCard]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    cards: list[VocabCard] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        word = str(entry.get("word", "")).strip()
        reading = str(entry.get("reading", "")).strip()
        meaning = str(entry.get("meaning", "")).strip()
        if word and _has_japanese(word) and not _is_table_noise(word):
            cards.append(VocabCard(word=word, reading=reading, meaning=meaning))
    return cards


# ── LLM backend ─────────────────────────────────────────────────────

_CLEANUP_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]

# Ollama models to try (local), largest first. Override with KADO_OLLAMA_MODEL env var.
_OLLAMA_MODELS = [
    "qwen2.5:32b",
    "qwen2.5:14b",
    "qwen2.5:7b",
    "llama3.1:70b",
    "llama3.1:8b",
    "gemma2:27b",
    "gemma2:9b",
    "mistral:7b",
]


def _llm_chat(
    prompt: str,
    *,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
) -> str | None:
    """Send a chat prompt to an LLM backend.

    Args:
        provider: "ollama" (default, local) or "huggingface" (remote API)
        ollama_url: Override Ollama URL (default: http://localhost:11434)
        ollama_model: Pin a specific Ollama model

    Returns the response text, or None if the backend fails.
    """
    if provider == "ollama":
        result = _try_ollama(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )
        if result is not None:
            return result
        _warn_ollama_missing(ollama_url=ollama_url, ollama_model=ollama_model)
        return None

    elif provider == "huggingface":
        try:
            from huggingface_hub import InferenceClient
        except ImportError:
            debug_print(
                "huggingface_hub not installed, run: pip install huggingface-hub"
            )
            return None

        for model in _CLEANUP_MODELS:
            try:
                client = InferenceClient(model=model)
                response = client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except (OSError, KeyError, IndexError, ValueError) as e:
                debug_print(f"HF model {model}: {e}")
                continue
        return None

    return None


def _warn_ollama_missing(*, ollama_url: str = "", ollama_model: str = "") -> None:
    """Print a user-friendly warning explaining why Ollama failed."""
    base_url = ollama_url or os.environ.get("KADO_OLLAMA_URL", OLLAMA_URL)
    available = ollama_available_models(base_url)

    if available is None:
        print(
            "   Ollama is not running. Start it with:\n     ollama serve",
            file=sys.stderr,
        )
        return

    override = ollama_model or os.environ.get("KADO_OLLAMA_MODEL", "")
    if override:
        print(
            f"   Model '{override}' is not installed. Install it with:\n"
            f"     ollama pull {override}",
            file=sys.stderr,
        )
        return

    installed = sorted(available)
    recommended = "qwen2.5:7b"
    print(
        f"   No compatible Ollama model found.\n"
        f"   Installed: {', '.join(installed) if installed else '(none)'}\n"
        f"   Supported: {', '.join(_OLLAMA_MODELS)}\n"
        f"   Install one with:\n"
        f"     ollama pull {recommended}",
        file=sys.stderr,
    )


def _try_ollama(
    prompt: str,
    *,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    ollama_url: str = "",
    ollama_model: str = "",
) -> str | None:
    """Try to get a response from a local Ollama instance."""
    base_url = ollama_url or os.environ.get("KADO_OLLAMA_URL", OLLAMA_URL)
    override_model = ollama_model or os.environ.get("KADO_OLLAMA_MODEL", "")

    models_to_try = [override_model] if override_model else _OLLAMA_MODELS

    available_names = ollama_available_models(base_url)
    if available_names is None:
        return None  # Ollama not running

    for model in models_to_try:
        resolved = ollama_resolve_model(model, available_names)
        if not resolved:
            continue

        try:
            payload = json.dumps(
                {
                    "model": resolved,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "num_ctx": 8192,
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                }
            ).encode()

            req = urllib.request.Request(
                f"{base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                text = data.get("message", {}).get("content", "").strip()
                if text:
                    debug_print(f"Ollama ({resolved}): OK")
                    return text
        except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError) as e:
            debug_print(f"Ollama model {resolved}: {e}")
            continue

    return None


def _llm_reconstruct_from_ocr(
    page_dumps: list[str],
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
) -> list[VocabCard]:
    """Feed raw OCR text to an LLM and reconstruct the full vocabulary table.

    The German meanings survive OCR much better than the Japanese, so the
    LLM uses them as anchors to figure out every Japanese word on the page
    — including ones that tesseract couldn't parse into table rows at all.
    """
    all_cards: list[VocabCard] = []
    seen: set[str] = set()

    debug_print(
        f"Reconstructing vocab from OCR text with LLM ({len(page_dumps)} pages)..."
    )

    for page_num, raw_text in enumerate(page_dumps, 1):
        if not raw_text.strip():
            continue
        debug_print(f"... page {page_num}/{len(page_dumps)}")
        page_cards = _llm_reconstruct_page(
            raw_text,
            provider=provider,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )
        for card in page_cards:
            if card.word and card.word not in seen:
                card.source = "llm"
                seen.add(card.word)
                all_cards.append(card)

    if all_cards:
        debug_print(f"LLM reconstructed {len(all_cards)} words")
    else:
        debug_print("LLM reconstruction: no results (models may be busy)")

    return all_cards


def _llm_reconstruct_page(
    raw_text: str,
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
) -> list[VocabCard]:
    """Send one page of raw OCR text to an LLM to extract vocab entries."""
    # Truncate very long pages to stay in token limits
    if len(raw_text) > 4000:
        raw_text = raw_text[:4000]

    prompt = (
        "Below is raw OCR text from a scanned page of a Japanese vocabulary table "
        "(J.Bridge textbook). The table has 5 columns:\n"
        "  漢字 | 漢字の意味 | 単語（漢字）| 単語（読み方）| 単語の意味\n\n"
        "The OCR is very noisy — the Japanese is often garbled, but the German "
        "meanings (column 5, 単語の意味) come through much more clearly.\n\n"
        "Your task: reconstruct the vocabulary entries from column 3 (単語/漢字).\n"
        "Each ROW in the table is ONE vocabulary entry — a complete word or phrase "
        "that a student would study (e.g. 異国情緒豊か(な), 移民(する), 江戸時代).\n\n"
        "RULES:\n"
        "- Each entry is a COMPLETE word/phrase from column 3 — do NOT decompose "
        "compound words into parts (江戸時代 is ONE entry, not 江 + 戸 + 時代)\n"
        "- Do NOT invent words that aren't in the table — only extract what's there\n"
        "- Use the German meaning (column 5) to identify what the Japanese word must be\n"
        "- Use fragments of hiragana reading (column 4) to confirm your guess\n"
        "- Include する in parentheses for suru-verbs: 移民(する), not just 移民\n"
        "- Include な in parentheses for na-adjectives: 異国情緒豊か(な)\n"
        "- Translate the German meaning into concise ENGLISH\n"
        "- Skip header rows, page numbers, and textbook metadata\n"
        "- Each table row has exactly ONE vocabulary entry\n\n"
        "Raw OCR text:\n"
        "---\n"
        f"{raw_text}\n"
        "---\n\n"
        "Return ONLY a JSON array, no markdown fences, no explanation:\n"
        '[{"word": "異国情緒豊か(な)", "reading": "いこくじょうちょゆたか(な)", '
        '"meaning": "exotic; having an exotic atmosphere"}, ...]'
    )

    text = _llm_chat(
        prompt,
        max_tokens=4000,
        provider=provider,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )
    if text:
        return _parse_llm_json(text)
    return []


def _get_raw_ocr_pages(path: str, pages: set[int] | None = None) -> list[str]:
    """Run tesseract on each page and return the raw text per page."""
    _ensure_ocr_deps()
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(path, dpi=300)
    result: list[str] = []

    for page_num, image in enumerate(images, 1):
        if pages and page_num not in pages:
            continue
        data = pytesseract.image_to_data(
            image,
            lang="jpn+deu+eng",
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
        # Build a spatial text dump: group words into rows by y-position
        ocr_words = _build_ocr_words(data)
        rows = _group_into_rows(ocr_words, tolerance=35)
        lines = []
        for row_words in rows:
            line = " ".join(w.text for w in sorted(row_words, key=lambda w: w.left))
            lines.append(line)
        result.append("\n".join(lines))

    return result


# ── OCR-based extraction (tesseract fallback) ────────────────────────


@dataclass
class _OcrWord:
    text: str
    left: int
    top: int
    width: int
    height: int

    @property
    def center_x(self) -> int:
        return self.left + self.width // 2

    @property
    def center_y(self) -> int:
        return self.top + self.height // 2


@dataclass
class _PartialRow:
    """Accumulator for merging split table rows."""

    word: str = ""
    reading: str = ""
    meaning: str = ""

    def is_empty(self) -> bool:
        return not self.word and not self.reading and not self.meaning

    def merge(self, other: _PartialRow) -> None:
        if other.word and not self.word:
            self.word = other.word
        if other.reading and not self.reading:
            self.reading = other.reading
        if other.meaning and not self.meaning:
            self.meaning = other.meaning
        # Append meaning if both have it (multi-line meaning)
        elif other.meaning and self.meaning:
            self.meaning = f"{self.meaning} {other.meaning}"

    def to_card(self) -> VocabCard | None:
        word = _clean_ocr_jp(self.word)
        reading = _clean_ocr_jp(self.reading)
        meaning = _clean_ocr_meaning(self.meaning)

        if not word and reading:
            word, reading = reading, ""
        if not word or not _has_japanese(word):
            return None
        if any(
            h in word for h in ("単語", "漢字", "読み方", "意味", "J.Bridge", "語彙")
        ):
            return None
        if word and _is_all_kana(word) and reading == word:
            reading = ""
        if not meaning:
            return None
        # Filter out single-character words (likely OCR fragments)
        jp_chars = [c for c in word if _has_japanese(c)]
        if len(jp_chars) < 2 and not _is_all_kana(word):
            return None
        # Filter out words with too much non-Japanese noise
        noise = sum(
            1 for c in word if c in "|[]{}=<>_!1lI" or (c.isascii() and c.isalpha())
        )
        if noise > len(word) * 0.3:
            return None
        # Filter out unbalanced parentheses (OCR truncation)
        if word.count("(") != word.count(")"):
            return None
        return VocabCard(word=word, reading=reading, meaning=meaning)


def _ensure_ocr_deps() -> None:
    """Check tesseract + poppler are installed."""
    try:
        import pytesseract
        from pdf2image import convert_from_path  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Scanned PDF detected. Install OCR dependencies:\n"
            "  brew install tesseract tesseract-lang poppler  (macOS)\n"
            "  apt install tesseract-ocr tesseract-ocr-jpn poppler-utils  (Linux)"
        )
    try:
        pytesseract.get_tesseract_version()
    except (OSError, EnvironmentError):
        raise RuntimeError(
            "tesseract not found. Install it:\n"
            "  macOS:  brew install tesseract tesseract-lang\n"
            "  Linux:  apt install tesseract-ocr tesseract-ocr-jpn"
        )


def _extract_via_ocr_columns(
    path: str, pages: set[int] | None = None
) -> list[VocabCard]:
    """Pure tesseract fallback with spatial column detection and row merging."""
    _ensure_ocr_deps()
    from pdf2image import convert_from_path
    import pytesseract

    images = convert_from_path(path, dpi=300)
    cards: list[VocabCard] = []
    seen: set[str] = set()

    for page_num, image in enumerate(images, 1):
        if pages and page_num not in pages:
            continue
        data = pytesseract.image_to_data(
            image,
            lang="jpn+deu+eng",
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
        ocr_words = _build_ocr_words(data)
        if not ocr_words:
            continue

        rows = _group_into_rows(ocr_words, tolerance=35)
        col_bounds = _jbridge_column_bounds(ocr_words, image.width)

        partials = []
        for row_words in rows:
            p = _row_to_partial(row_words, col_bounds)
            partials.append(p)

        merged = _merge_partial_rows(partials)

        for partial in merged:
            card = partial.to_card()
            if card and card.word not in seen:
                card.source = "ocr"
                seen.add(card.word)
                cards.append(card)

    return cards


def _build_ocr_words(data: dict) -> list[_OcrWord]:
    words = []
    n = len(data["text"])
    for i in range(n):
        text = str(data["text"][i]).strip()
        conf = int(data["conf"][i])
        if not text or conf < 15:
            continue
        words.append(
            _OcrWord(
                text=text,
                left=int(data["left"][i]),
                top=int(data["top"][i]),
                width=int(data["width"][i]),
                height=int(data["height"][i]),
            )
        )
    return words


def _group_into_rows(
    words: list[_OcrWord], tolerance: int = 35
) -> list[list[_OcrWord]]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w.top, w.left))
    rows: list[list[_OcrWord]] = []
    current_row: list[_OcrWord] = [sorted_words[0]]
    current_y = sorted_words[0].center_y

    for word in sorted_words[1:]:
        if abs(word.center_y - current_y) <= tolerance:
            current_row.append(word)
        else:
            rows.append(sorted(current_row, key=lambda w: w.left))
            current_row = [word]
            current_y = word.center_y

    if current_row:
        rows.append(sorted(current_row, key=lambda w: w.left))

    return rows


def _jbridge_column_bounds(
    words: list[_OcrWord], page_width: int
) -> dict[str, tuple[int, int]]:
    """J.Bridge table column boundaries as page-width ratios."""
    return {
        "word": (int(page_width * 0.28), int(page_width * 0.48)),
        "reading": (int(page_width * 0.48), int(page_width * 0.66)),
        "meaning": (int(page_width * 0.66), int(page_width * 0.95)),
    }


def _row_to_partial(
    row_words: list[_OcrWord], col_bounds: dict[str, tuple[int, int]]
) -> _PartialRow:
    def col_text(col: str) -> str:
        start, end = col_bounds[col]
        return " ".join(w.text for w in row_words if start <= w.center_x <= end)

    return _PartialRow(
        word=col_text("word"),
        reading=col_text("reading"),
        meaning=col_text("meaning"),
    )


def _merge_partial_rows(partials: list[_PartialRow]) -> list[_PartialRow]:
    """Merge consecutive partial rows that belong to the same table entry.

    Tesseract often splits a single table row into 2-3 screen rows.
    Strategy: accumulate partials until we see a new word starting.
    """
    if not partials:
        return []

    merged: list[_PartialRow] = []
    current = _PartialRow()

    for p in partials:
        has_jp_word = bool(p.word) and _has_japanese(p.word)

        if has_jp_word and not current.is_empty() and current.word:
            # New entry starting — flush the current one
            merged.append(current)
            current = _PartialRow()

        current.merge(p)

    if not current.is_empty():
        merged.append(current)

    return merged


# ── Debug ────────────────────────────────────────────────────────────


def dump_ocr_debug(
    path: str,
    pages: set[int] | None = None,
    use_vision: bool = True,
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
    ollama_vision_model: str = "",
) -> list[VocabCard]:
    """Full debug: raw OCR → column parsing → LLM reconstruction."""
    import pytesseract
    from pdf2image import convert_from_path

    images = convert_from_path(path, dpi=300)

    # ── Phase 1: Raw OCR + column parsing ──
    page_dumps: list[str] = []
    ocr_cards: list[VocabCard] = []
    col_parsed_words: set[str] = set()
    ocr_seen: set[str] = set()

    for page_num, image in enumerate(images, 1):
        if pages and page_num not in pages:
            continue
        print(f"\n{'=' * 60}")
        print(f"PAGE {page_num}  (size: {image.width} x {image.height})")
        print(f"{'=' * 60}")

        data = pytesseract.image_to_data(
            image,
            lang="jpn+deu+eng",
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
        ocr_words = _build_ocr_words(data)
        rows = _group_into_rows(ocr_words, tolerance=35)
        col_bounds = _jbridge_column_bounds(ocr_words, image.width)

        print(f"\nColumn bounds (page width={image.width}):")
        for name, (start, end) in col_bounds.items():
            print(f"  {name:15s}: {start:4d} - {end:4d}")

        # Show merged column-parsed results
        partials = [_row_to_partial(r, col_bounds) for r in rows]
        merged = _merge_partial_rows(partials)

        print(f"\nColumn-parsed entries: {len(merged)}")
        print(f"{'-' * 60}")
        for i, p in enumerate(merged):
            card = p.to_card()
            status = "✓" if card else "·"
            if card:
                col_parsed_words.add(_dedup_key(card.word))
                if card.word not in ocr_seen:
                    card.source = "ocr"
                    ocr_seen.add(card.word)
                    ocr_cards.append(card)
            print(
                f"  {status} {i:2d}: word=[{p.word[:20]:20s}] reading=[{p.reading[:15]:15s}] meaning=[{p.meaning[:35]:35s}]"
            )

        # Collect raw text for LLM phase
        lines = []
        for row_words in rows:
            line = " ".join(w.text for w in sorted(row_words, key=lambda w: w.left))
            lines.append(line)
        page_dumps.append("\n".join(lines))

    # ── Phase 2: Vision model (runs before LLM so the large vision model loads
    # into VRAM before the text model, avoiding a costly model swap) ──
    vision_cards: list[VocabCard] = []
    vision_words: set[str] = set()

    if use_vision:
        print(f"\n{'=' * 60}")
        print("VISION MODEL")
        print(f"{'=' * 60}")

        vision_cards = (
            _extract_via_vision(
                path,
                pages=pages,
                provider=provider,
                ollama_url=ollama_url,
                ollama_vision_model=ollama_vision_model,
            )
            or []
        )
        vision_words = {_dedup_key(c.word) for c in vision_cards}

        if vision_cards:
            print(f"\nVision extracted {len(vision_cards)} entries:")
            print(f"{'-' * 60}")
            for i, c in enumerate(vision_cards):
                key = _dedup_key(c.word)
                tag = "ocr" if key in col_parsed_words else "vision"
                print(
                    f"  ✓ {i:2d} [{tag:12s}]: word=[{c.word:20s}] reading=[{c.reading:15s}] meaning=[{c.meaning[:35]:35s}]"
                )
        else:
            print("\n  (no results — vision model may still be loading, or pull a vision model with: ollama pull qwen2.5vl:7b)")

    # ── Phase 3: LLM reconstruction ──
    print(f"\n{'=' * 60}")
    print("LLM RECONSTRUCTION")
    print(f"{'=' * 60}")

    llm_cards = (
        _llm_reconstruct_from_ocr(
            page_dumps,
            provider=provider,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )
        or []
    )
    llm_words = {_dedup_key(c.word) for c in llm_cards}

    if llm_cards:
        print(f"\nLLM reconstructed {len(llm_cards)} entries:")
        print(f"{'-' * 60}")
        for i, c in enumerate(llm_cards):
            key = _dedup_key(c.word)
            tags = []
            if key in col_parsed_words:
                tags.append("ocr")
            if key in vision_words:
                tags.append("vision")
            tag = "+".join(tags) if tags else "llm"
            print(
                f"  ✓ {i:2d} [{tag:7s}]: word=[{c.word:20s}] reading=[{c.reading:15s}] meaning=[{c.meaning[:35]:35s}]"
            )
    else:
        print("\n  (no results — LLM models may be busy, try again)")

    # ── Summary ──
    all_words = col_parsed_words | llm_words | vision_words
    in_all = col_parsed_words & llm_words & vision_words
    in_ocr_llm = (col_parsed_words & llm_words) - vision_words
    in_ocr_vision = (col_parsed_words & vision_words) - llm_words
    in_llm_vision = (llm_words & vision_words) - col_parsed_words
    ocr_only = col_parsed_words - llm_words - vision_words
    llm_only = llm_words - col_parsed_words - vision_words
    vision_only = vision_words - col_parsed_words - llm_words

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  OCR column parsing:   {len(col_parsed_words):3d} words")
    print(f"  LLM reconstruction:   {len(llm_words):3d} words")
    print(f"  Vision model:         {len(vision_words):3d} words")
    print("  ────────────────────────────────────")
    print(f"  All three:            {len(in_all):3d}  (highest confidence)")
    if in_ocr_llm:
        print(f"  OCR + LLM:            {len(in_ocr_llm):3d}")
    if in_ocr_vision:
        print(f"  OCR + Vision:         {len(in_ocr_vision):3d}")
    if in_llm_vision:
        print(f"  LLM + Vision:         {len(in_llm_vision):3d}")
    if ocr_only:
        print(f"  OCR only:             {len(ocr_only):3d}")
    if llm_only:
        print(f"  LLM only:             {len(llm_only):3d}")
    if vision_only:
        print(f"  Vision only:          {len(vision_only):3d}")
    print("  ────────────────────────────────────")
    print(f"  Total unique:         {len(all_words):3d}")

    # Show words unique to each source (key by normalized form)
    if vision_only:
        print("\n  Vision-only words:")
        vision_by_norm = {_dedup_key(c.word): c for c in vision_cards}
        for w in sorted(vision_only):
            c_v: VocabCard | None = vision_by_norm.get(w)
            if c_v:
                c = c_v
                print(f"    + {c.word} 【{c.reading}】 — {c.meaning[:50]}")

    if llm_only:
        print("\n  LLM-only words (from German):")
        llm_by_norm = {_dedup_key(c.word): c for c in llm_cards}
        for w in sorted(llm_only):
            c_l: VocabCard | None = llm_by_norm.get(w)
            if c_l:
                c = c_l
                print(f"    + {c.word} 【{c.reading}】 — {c.meaning[:50]}")

    if ocr_only:
        print("\n  OCR-only words:")
        for w in sorted(ocr_only):
            print(f"    · {w}")

    # Return merged cards so the pipeline doesn't need to re-run everything
    return _merge_sources(
        ocr_cards,
        llm_cards,
        vision_cards,
        provider=provider,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _clean(cell: str | None) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", str(cell)).strip()


def _clean_ocr_jp(text: str) -> str:
    """Clean OCR artifacts from Japanese text."""
    if not text:
        return ""
    # Remove stray Latin chars, numbers, punctuation from table borders
    text = re.sub(r"^[|!l1I\[\](){}<>=_\-\s]+", "", text)
    text = re.sub(r"[|!l1I\[\](){}<>=_\-\s]+$", "", text)
    # Remove isolated Latin words that are OCR noise (but keep katakana loan words)
    text = re.sub(r"\b[A-Za-z]{1,3}\b", "", text)
    # Collapse spaces within Japanese text
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _clean_ocr_meaning(text: str) -> str:
    """Clean OCR artifacts from the meaning column."""
    if not text:
        return ""
    text = re.sub(r"\s*\d+\s*$", "", text)
    text = re.sub(r"[|]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(".,;:/ _=")
    # Remove short garbage strings
    if len(text) < 3:
        return ""
    return text


_VOWELS = set("aeiouäöüAEIOUÄÖÜ")


def _is_garbage_meaning(text: str) -> bool:
    """Return True if a meaning string looks like a vision hallucination.

    Detects strings like "BQSOUL", "Trnbeeland", "Klelhn", "am" that the
    vision model produces when it can't find real text in the meaning cell
    (e.g., the cell contains a flag or graphic).
    """
    if not text:
        return True
    # Too short to be a real meaning
    if len(text) < 3:
        return True
    # Single token (no spaces) — real German meanings are almost always phrases
    if " " not in text:
        # All-uppercase alphabetic string (e.g. "BQSOUL") — never a real meaning
        if len(text) >= 4 and text.isupper():
            return True
        if len(text) >= 5:
            # Very consonant-heavy: fewer than 25% vowels (catches "Klelhn" at 1/6)
            vowel_ratio = sum(1 for c in text if c in _VOWELS) / len(text)
            if vowel_ratio < 0.25:
                return True
            # Long consecutive consonant run (catches "Trnbeeland": T-r-n-b = 4)
            # Also checks trailing consonant cluster (catches "Kleelhn": -lhn = 3)
            max_run = run = trailing_run = 0
            for c in text:
                if c.isalpha() and c not in _VOWELS:
                    run += 1
                    max_run = max(max_run, run)
                    trailing_run += 1
                else:
                    run = 0
                    trailing_run = 0
            if max_run >= 4 or trailing_run >= 3:
                return True
    return False


def _clean_reading(reading: str) -> str:
    """Return empty string if a reading contains romaji/latin characters.

    Real Japanese readings are hiragana only. If OCR produces something like
    'nekutai pin' for ネクタイピン, it's noise and should be discarded.
    """
    if not reading:
        return reading
    reading = reading.replace("（", "(").replace("）", ")")
    # Any latin letter means it's romaji or noise, not a valid hiragana reading
    if any(c.isascii() and c.isalpha() for c in reading):
        return ""
    return reading


def _dedup_key(text: str) -> str:
    """Return a base form used only for deduplication.

    Strips (する)/(な) suffixes so that 発展 and 発展(する) are treated as the same
    word regardless of which source found them.
    """
    norm = _normalize_word(text)
    if norm.endswith("(する)"):
        norm = norm[: -len("(する)")]
    if norm.endswith("(な)"):
        norm = norm[: -len("(な)")]
    return norm


def _normalize_word(text: str) -> str:
    """Normalize a word for matching: full-width brackets → half-width, strip spaces.

    Also canonicalizes する/な verb/adjective suffixes:
      発展する  → 発展(する)
      静か(な)  stays as-is (already canonical)
      静かな    → 静か(な)
    """
    text = text.replace("（", "(").replace("）", ")").replace("　", "")
    text = text.replace("【", "[").replace("】", "]")
    text = text.strip()
    # Canonicalize する verbs: 発展する → 発展(する) if not already bracketed
    if text.endswith("する") and not text.endswith("(する)"):
        text = text[:-2] + "(する)"
    # Canonicalize な adjectives: 静かな → 静か(な) if not already bracketed.
    # Only apply when the character before な is a CJK ideograph — this avoids
    # corrupting words like みんな, 奈良, etc. that happen to end in な.
    if (
        text.endswith("な")
        and not text.endswith("(な)")
        and text not in _NA_ADJ_EXCEPTIONS
        and len(text) >= 2
        and "\u4e00" <= text[-2] <= "\u9fff"
    ):
        text = text[:-1] + "(な)"
    return text


def _is_all_kana(text: str) -> bool:
    return bool(
        re.match(r"^[\u3040-\u309F\u30A0-\u30FF\u30FC\u3000-\u303F\s・ー]+$", text)
    )


def _has_japanese(text: str) -> bool:
    return bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text))
