"""Tests for pdf_import pure-logic functions."""

from __future__ import annotations

import pytest

from kado.models import VocabCard
from kado.pdf_import import _normalize_word, _parse_llm_json, _dedup_key, _is_garbage_meaning, _clean_reading


# ── _normalize_word ───────────────────────────────────────────────────


def test_normalize_word_fullwidth_brackets():
    assert _normalize_word("食べる（たべる）") == "食べる(たべる)"


def test_normalize_word_fullwidth_square_brackets():
    assert _normalize_word("食べる【verb】") == "食べる[verb]"


def test_normalize_word_strips_ideographic_space():
    assert _normalize_word("食べる　") == "食べる"


def test_normalize_word_strips_whitespace():
    assert _normalize_word("  食べる  ") == "食べる"


def test_normalize_word_suru_verb_canonicalized():
    assert _normalize_word("発展する") == "発展(する)"


def test_normalize_word_suru_already_canonical():
    assert _normalize_word("発展(する)") == "発展(する)"


def test_normalize_word_na_adj_canonicalized():
    # Only applies when the char before な is a CJK ideograph (kanji).
    # 便利な → 便利(な) because 利 is kanji.
    assert _normalize_word("便利な") == "便利(な)"


def test_normalize_word_na_already_canonical():
    assert _normalize_word("便利(な)") == "便利(な)"


def test_normalize_word_na_hiragana_prefix_unchanged():
    # 静かな: the char before な is か (hiragana), not kanji — no canonicalization.
    assert _normalize_word("静かな") == "静かな"


def test_normalize_word_plain_kanji_unchanged():
    assert _normalize_word("東京") == "東京"


def test_normalize_word_fullwidth_bracket_suru():
    # Full-width brackets first converted then suru rule applied
    result = _normalize_word("発展する")
    assert result == "発展(する)"


# ── _parse_llm_json ───────────────────────────────────────────────────


def test_parse_llm_json_basic():
    text = '[{"word": "東京", "reading": "とうきょう", "meaning": "Tokyo"}]'
    cards = _parse_llm_json(text)
    assert len(cards) == 1
    assert cards[0].word == "東京"
    assert cards[0].reading == "とうきょう"
    assert cards[0].meaning == "Tokyo"


def test_parse_llm_json_multiple_entries():
    text = (
        '[{"word": "食べる", "reading": "たべる", "meaning": "to eat"},'
        ' {"word": "飲む", "reading": "のむ", "meaning": "to drink"}]'
    )
    cards = _parse_llm_json(text)
    assert len(cards) == 2
    assert cards[0].word == "食べる"
    assert cards[1].word == "飲む"


def test_parse_llm_json_with_markdown_wrapper():
    text = "Here is the result:\n```json\n[{\"word\": \"学校\", \"reading\": \"がっこう\", \"meaning\": \"school\"}]\n```"
    cards = _parse_llm_json(text)
    assert len(cards) == 1
    assert cards[0].word == "学校"


def test_parse_llm_json_strips_non_japanese():
    # Entry without Japanese characters should be filtered out
    text = '[{"word": "hello", "reading": "", "meaning": "english word"}]'
    cards = _parse_llm_json(text)
    assert cards == []


def test_parse_llm_json_empty_word_skipped():
    text = '[{"word": "", "reading": "", "meaning": "nothing"}]'
    cards = _parse_llm_json(text)
    assert cards == []


def test_parse_llm_json_no_array_returns_empty():
    cards = _parse_llm_json("Sorry, I cannot help with that.")
    assert cards == []


def test_parse_llm_json_malformed_json_returns_empty():
    cards = _parse_llm_json("[{word: broken json}]")
    assert cards == []


def test_parse_llm_json_mixed_valid_invalid():
    text = (
        '[{"word": "山", "reading": "やま", "meaning": "mountain"},'
        ' {"word": "not-japanese", "reading": "", "meaning": "skip me"}]'
    )
    cards = _parse_llm_json(text)
    # "山" is kanji (Japanese), "not-japanese" has no Japanese chars
    assert len(cards) == 1
    assert cards[0].word == "山"


def test_parse_llm_json_non_dict_entries_skipped():
    text = '["just a string", {"word": "川", "reading": "かわ", "meaning": "river"}]'
    cards = _parse_llm_json(text)
    assert len(cards) == 1
    assert cards[0].word == "川"


# ── _dedup_key ────────────────────────────────────────────────────────


def test_dedup_key_strips_suru_suffix():
    # 発展する normalizes to 発展(する), then strips (する) → 発展
    assert _dedup_key("発展する") == "発展"


def test_dedup_key_strips_already_canonical_suru():
    assert _dedup_key("発展(する)") == "発展"


def test_dedup_key_strips_na_suffix():
    # 便利な normalizes to 便利(な), then strips (な) → 便利
    assert _dedup_key("便利な") == "便利"


def test_dedup_key_strips_already_canonical_na():
    assert _dedup_key("便利(な)") == "便利"


def test_dedup_key_plain_kanji_unchanged():
    assert _dedup_key("東京") == "東京"


def test_dedup_key_suru_and_na_forms_collide():
    # Both forms of the same word should produce the same key
    assert _dedup_key("発展する") == _dedup_key("発展(する)") == _dedup_key("発展")


def test_dedup_key_na_forms_collide():
    assert _dedup_key("便利な") == _dedup_key("便利(な)") == _dedup_key("便利")


def test_dedup_key_normalizes_fullwidth_brackets_first():
    # Full-width brackets get converted before stripping
    assert _dedup_key("発展（する）") == "発展"


# ── _is_garbage_meaning ──────────────────────────────────────────────


def test_garbage_meaning_all_caps_no_spaces():
    assert _is_garbage_meaning("BQSOUL") is True


def test_garbage_meaning_consonant_heavy():
    assert _is_garbage_meaning("Trnbeeland") is True  # 4 consecutive consonants
    assert _is_garbage_meaning("Klelhn") is True      # 1/6 vowels < 0.25
    assert _is_garbage_meaning("Kleelhn") is True     # trailing -lhn = 3 consonants



def test_garbage_meaning_too_short():
    assert _is_garbage_meaning("am") is True
    assert _is_garbage_meaning("") is True


def test_garbage_meaning_real_german_phrase():
    assert _is_garbage_meaning("Entwicklung; sich entwickeln") is False
    assert _is_garbage_meaning("Park des Friedens") is False  # 3 words, passes


def test_garbage_meaning_real_single_word():
    assert _is_garbage_meaning("England") is False
    assert _is_garbage_meaning("Amerika") is False
    assert _is_garbage_meaning("tram") is False
    assert _is_garbage_meaning("Chinatown") is False


# ── _clean_reading ───────────────────────────────────────────────────


def test_clean_reading_removes_romaji():
    assert _clean_reading("nekutai pin") == ""


def test_clean_reading_keeps_hiragana():
    assert _clean_reading("ねくたいぴん") == "ねくたいぴん"


def test_clean_reading_keeps_empty():
    assert _clean_reading("") == ""


def test_clean_reading_normalizes_brackets():
    assert _clean_reading("かんこう（する）") == "かんこう(する)"


# ── _merge_sources (via pure logic, no LLM) ───────────────────────────
# We test the deduplication logic by calling _merge_sources with empty
# vision_cards so the LLM translation path is skipped.


def test_merge_sources_deduplicates_by_normalized_word():
    from unittest.mock import patch
    from kado.pdf_import import _merge_sources

    ocr = [VocabCard(word="発展する", reading="はってんする", meaning="development", source="ocr")]
    llm = [VocabCard(word="発展(する)", reading="はってんする", meaning="to develop", source="llm")]

    # Patch out the LLM fix call so we can test just the dedup logic
    with patch("kado.pdf_import._llm_fix_ocr_cards", side_effect=lambda cards, **kw: cards):
        merged = _merge_sources(ocr_cards=ocr, llm_cards=llm, vision_cards=[])

    words = [c.word for c in merged]
    # Both normalize to 発展(する) — only one should survive
    assert len(words) == 1


def test_merge_sources_vision_wins_over_llm():
    from unittest.mock import patch
    from kado.pdf_import import _merge_sources

    vision = [VocabCard(word="東京", reading="とうきょう", meaning="Tokyo (vision)", source="vision")]
    llm = [VocabCard(word="東京", reading="とうきょう", meaning="Tokyo (llm)", source="llm")]

    with patch("kado.pdf_import._llm_fix_ocr_cards", side_effect=lambda cards, **kw: cards):
        merged = _merge_sources(ocr_cards=[], llm_cards=llm, vision_cards=vision)

    assert len(merged) == 1
    assert merged[0].source == "vision"
    assert "vision" in merged[0].meaning


def test_merge_sources_llm_only_card_gets_llm_source():
    from unittest.mock import patch
    from kado.pdf_import import _merge_sources

    llm = [VocabCard(word="夕焼け", reading="ゆうやけ", meaning="sunset", source="")]

    with patch("kado.pdf_import._llm_fix_ocr_cards", side_effect=lambda cards, **kw: cards):
        merged = _merge_sources(ocr_cards=[], llm_cards=llm, vision_cards=[])

    assert len(merged) == 1
    assert merged[0].source == "llm"


def test_merge_sources_llm_confirmed_by_ocr_gets_ocr_source():
    from unittest.mock import patch
    from kado.pdf_import import _merge_sources

    # OCR and LLM both have 夕焼け (in normalized form OCR has 夕焼け)
    ocr = [VocabCard(word="夕焼け", reading="ゆうやけ", meaning="sunset ocr")]
    llm = [VocabCard(word="夕焼け", reading="ゆうやけ", meaning="sunset llm")]

    with patch("kado.pdf_import._llm_fix_ocr_cards", side_effect=lambda cards, **kw: cards):
        merged = _merge_sources(ocr_cards=ocr, llm_cards=llm, vision_cards=[])

    # LLM processes first among non-vision; since OCR also has it, source = "ocr"
    assert len(merged) == 1
    assert merged[0].source == "ocr"


def test_merge_sources_empty_inputs():
    from unittest.mock import patch
    from kado.pdf_import import _merge_sources

    with patch("kado.pdf_import._llm_fix_ocr_cards", side_effect=lambda cards, **kw: cards):
        merged = _merge_sources(ocr_cards=[], llm_cards=[], vision_cards=[])

    assert merged == []
