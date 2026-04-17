"""AI-powered grammar point generation for Japanese grammar cards."""

from __future__ import annotations

import json
import re

from kado.debug import debug_print
from kado.models import GrammarCard
from kado.ollama_utils import OLLAMA_URL, ollama_available_models, ollama_resolve_model

OLLAMA_MODELS = [
    "qwen3.5:27b-q8_0",
    "qwen2.5:32b",
    "qwen2.5:14b",
    "qwen2.5:7b",
]

_PROMPT = """\
You are a Japanese language teacher. Generate a structured explanation for the grammar point: {pattern}

Respond with ONLY valid JSON in this exact format:
{{
  "meaning": "brief English meaning (under 15 words)",
  "note": "important nuance, exception, or contrast with similar patterns (or empty string)",
  "jlpt": "N1/N2/N3/N4/N5 or empty string if unknown",
  "formation": [
    {{"type": "Verb", "rule": "formation rule", "example_ja": "Japanese example", "example_en": "English translation", "note": "exception if any"}},
    {{"type": "い-Adj", "rule": "formation rule", "example_ja": "Japanese example", "example_en": "English translation", "note": ""}},
    {{"type": "な-Adj", "rule": "formation rule", "example_ja": "Japanese example", "example_en": "English translation", "note": ""}},
    {{"type": "Noun", "rule": "formation rule", "example_ja": "Japanese example", "example_en": "English translation", "note": ""}}
  ],
  "examples": [
    {{"ja": "natural Japanese sentence", "en": "English translation"}},
    {{"ja": "natural Japanese sentence", "en": "English translation"}},
    {{"ja": "natural Japanese sentence", "en": "English translation"}}
  ]
}}

Rules:
- Include only applicable word types in "formation" (omit types where the grammar cannot apply)
- "examples" should have 3-4 sentences covering the applicable word types, one per type
- Examples should be natural everyday Japanese at JLPT N4-N3 level
- "note" for a formation entry should only include irregular forms (e.g. いい→よさそう)
- All example sentences must actually use the grammar point {pattern}
"""


def generate_grammar_card(
    pattern: str,
    *,
    provider: str = "ollama",
    ollama_url: str = "",
    ollama_model: str = "",
) -> GrammarCard | None:
    """Generate a GrammarCard using an LLM.

    Returns None if generation fails.
    """
    prompt = _PROMPT.format(pattern=pattern)

    if provider == "ollama":
        result = _generate_via_ollama(prompt, ollama_url=ollama_url, ollama_model=ollama_model)
    else:
        return None

    if not result:
        return None

    return _parse_response(pattern, result)


# ── Ollama backend ─────────────────────────────────────────────────────


def _generate_via_ollama(
    prompt: str,
    *,
    ollama_url: str = "",
    ollama_model: str = "",
) -> str | None:
    import urllib.error
    import urllib.request

    base_url = ollama_url or OLLAMA_URL
    override = ollama_model

    available = ollama_available_models(base_url)
    if available is None:
        debug_print("Ollama not running for grammar generation")
        return None

    models_to_try = [override] if override else OLLAMA_MODELS
    for m in models_to_try:
        resolved = ollama_resolve_model(m, available)
        if not resolved:
            continue
        try:
            payload = json.dumps({
                "model": resolved,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {"num_ctx": 4096, "num_predict": 1200, "temperature": 0.3},
            }).encode()
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
                    debug_print(f"Grammar LLM ({resolved}): {len(text)} chars")
                    return text
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            continue

    return None


# ── Response parsing ───────────────────────────────────────────────────


def _parse_response(pattern: str, text: str) -> GrammarCard | None:
    """Parse LLM JSON response into a GrammarCard."""
    # Strip <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Extract JSON
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        debug_print("Grammar: no JSON found in response")
        return None

    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as e:
        debug_print(f"Grammar: JSON parse error: {e}")
        return None

    meaning = data.get("meaning", "")
    note = data.get("note", "")
    jlpt = data.get("jlpt", "")

    # Build formation text and collect one example per applicable type.
    # Formation entries are the authoritative source — guarantees coverage.
    formation_lines = []
    examples: list[tuple[str, str]] = []
    seen_ja: set[str] = set()

    # Build as HTML table so Anki renders it correctly
    formation_rows = []
    for entry in data.get("formation", []):
        wtype = entry.get("type", "")
        rule = entry.get("rule", "")
        ex_ja = entry.get("example_ja", "").strip()
        ex_en = entry.get("example_en", "").strip()
        ex_note = entry.get("note", "")
        note_html = f'<span class="fnote">※ {ex_note}</span>' if ex_note else ""
        example_html = f"<br><span class='fex'>{ex_ja}</span> <span class='ftrans'>({ex_en})</span> {note_html}" if ex_ja else ""
        formation_rows.append(
            f"<tr><td class='ftype'>{wtype}</td><td>{rule}{example_html}</td></tr>"
        )
        if ex_ja and ex_ja not in seen_ja and len(examples) < 4:
            examples.append((ex_ja, ex_en))
            seen_ja.add(ex_ja)
    formation = f"<table class='formation'>{''.join(formation_rows)}</table>"

    # Append any extra examples from the top-level list (up to cap of 4)
    for ex in data.get("examples", []):
        if len(examples) >= 4:
            break
        ja = ex.get("ja", "").strip()
        en = ex.get("en", "").strip()
        if ja and en and ja not in seen_ja:
            examples.append((ja, en))
            seen_ja.add(ja)

    if not meaning or not examples:
        debug_print("Grammar: missing meaning or examples in response")
        return None

    return GrammarCard(
        pattern=pattern,
        meaning=meaning,
        formation=formation,
        note=note,
        jlpt=jlpt,
        examples=examples,
    )
