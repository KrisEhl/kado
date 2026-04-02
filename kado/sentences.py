"""Example sentence generation using open-source models via Hugging Face.

Uses HF's free serverless Inference API — no API key or token required.
Generates natural Japanese example sentences that incorporate vocabulary
the user has already studied, reinforcing retention.
"""

from __future__ import annotations

import random
import sys

# Models available on HF's free serverless inference (no token needed).
# These are regularly updated — if one stops working, the next is tried.
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "HuggingFaceH4/zephyr-7b-beta",
]


def generate_example(
    word: str,
    reading: str,
    meaning: str,
    known_vocab: list[str],
    model: str | None = None,
) -> tuple[str, str]:
    """Generate a Japanese example sentence + English translation.

    Uses Hugging Face's free serverless Inference API with an open-source
    model. No API key required.

    The sentence tries to incorporate words from *known_vocab* so the
    user practises previously-learnt vocabulary alongside the new word.

    Returns (japanese_sentence, english_translation).
    Falls back to empty strings if generation fails.
    """
    from huggingface_hub import InferenceClient

    context_words = _pick_context_words(known_vocab, max_words=5)
    context_hint = ""
    if context_words:
        context_hint = (
            f"\nTry to naturally incorporate one or two of these previously "
            f"studied words: {', '.join(context_words)}."
        )

    prompt = (
        f"Generate ONE natural Japanese example sentence using the word "
        f"「{word}」 ({reading}, meaning: {meaning}).{context_hint}\n\n"
        f"Requirements:\n"
        f"- The sentence should be natural, everyday Japanese\n"
        f"- Aim for JLPT N4-N3 grammar level\n"
        f"- Keep it under 30 characters\n"
        f"- Respond ONLY with two lines, nothing else:\n"
        f"Line 1: The Japanese sentence\n"
        f"Line 2: The English translation"
    )

    models_to_try = [model] if model else DEFAULT_MODELS
    last_error = None

    for m in models_to_try:
        try:
            client = InferenceClient(model=m)
            response = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            return _parse_response(text)
        except Exception as e:
            last_error = f"{m}: {e}"
            continue  # try next model

    # Print the last error to stderr so the user can see what went wrong
    if last_error:
        print(f"  [debug] {last_error}", file=sys.stderr)

    return "", ""


def _parse_response(text: str) -> tuple[str, str]:
    """Extract Japanese sentence and English translation from model output."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    # Strip common prefixes like "1.", "Line 1:", "Japanese:", etc.
    cleaned = []
    for line in lines:
        for prefix in ("1.", "2.", "Line 1:", "Line 2:", "Japanese:", "English:", "- "):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
        cleaned.append(line)

    if len(cleaned) >= 2:
        return cleaned[0], cleaned[1]
    elif len(cleaned) == 1:
        return cleaned[0], ""
    return "", ""


def _pick_context_words(known: list[str], max_words: int = 5) -> list[str]:
    """Pick a random sample of known vocab to use as sentence context."""
    if not known:
        return []
    sample_size = min(max_words, len(known))
    return random.sample(known, sample_size)
