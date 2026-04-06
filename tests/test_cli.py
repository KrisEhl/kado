"""CLI integration tests using typer.testing.CliRunner with mocked externals."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from kado.cli import app, _parse_pages
from kado.config import KadoConfig

runner = CliRunner()


# ── helpers ───────────────────────────────────────────────────────────


def _default_cfg(**overrides) -> KadoConfig:
    cfg = KadoConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── config --show ─────────────────────────────────────────────────────


def test_config_show_displays_table():
    with patch("kado.cli.KadoConfig.load", return_value=_default_cfg()):
        result = runner.invoke(app, ["config", "--show"])
    assert result.exit_code == 0
    assert "http://localhost:8765" in result.output
    assert "Japanese::Vocabulary" in result.output
    assert "ollama" in result.output


def test_config_show_reflects_custom_values():
    cfg = _default_cfg(
        anki_url="http://custom:1234",
        anki_deck="MyDeck",
        sentence_provider="huggingface",
        audio_enabled=False,
    )
    with patch("kado.cli.KadoConfig.load", return_value=cfg):
        result = runner.invoke(app, ["config", "--show"])
    assert result.exit_code == 0
    assert "http://custom:1234" in result.output
    assert "MyDeck" in result.output
    assert "huggingface" in result.output
    assert "disabled" in result.output


# ── status ────────────────────────────────────────────────────────────


def test_status_anki_reachable():
    cfg = _default_cfg(sentence_provider="none")
    mock_anki = MagicMock()
    mock_anki.ping.return_value = True
    mock_anki.get_existing_vocab.return_value = ["食べる", "飲む"]

    with patch("kado.cli.KadoConfig.load", return_value=cfg), \
         patch("kado.anki.AnkiConnect", return_value=mock_anki), \
         patch("kado.cli.AnkiConnect", return_value=mock_anki, create=True):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "AnkiConnect reachable" in result.output


def test_status_anki_unreachable_exits_1():
    cfg = _default_cfg(sentence_provider="none")
    mock_anki = MagicMock()
    mock_anki.ping.return_value = False

    with patch("kado.cli.KadoConfig.load", return_value=cfg), \
         patch("kado.anki.AnkiConnect", return_value=mock_anki, create=True):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 1
    assert "Cannot reach AnkiConnect" in result.output


def test_status_shows_deck_word_count():
    cfg = _default_cfg(sentence_provider="none")
    mock_anki = MagicMock()
    mock_anki.ping.return_value = True
    mock_anki.get_existing_vocab.return_value = ["食べる", "飲む", "学ぶ"]

    with patch("kado.cli.KadoConfig.load", return_value=cfg), \
         patch("kado.anki.AnkiConnect", return_value=mock_anki, create=True):
        result = runner.invoke(app, ["status"])

    assert "3 words" in result.output


def test_status_ollama_provider_shows_model():
    cfg = _default_cfg(sentence_provider="ollama", ollama_model="qwen2.5:7b")
    mock_anki = MagicMock()
    mock_anki.ping.return_value = True
    mock_anki.get_existing_vocab.return_value = []

    with patch("kado.cli.KadoConfig.load", return_value=cfg), \
         patch("kado.anki.AnkiConnect", return_value=mock_anki, create=True), \
         patch("kado.sentences.ollama_available_models", return_value={"qwen2.5:7b"}):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "ollama" in result.output


def test_status_huggingface_provider():
    cfg = _default_cfg(sentence_provider="huggingface", hf_model="Qwen/Qwen2.5-7B")
    mock_anki = MagicMock()
    mock_anki.ping.return_value = True
    mock_anki.get_existing_vocab.return_value = []

    with patch("kado.cli.KadoConfig.load", return_value=cfg), \
         patch("kado.anki.AnkiConnect", return_value=mock_anki, create=True):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "huggingface" in result.output


def test_status_none_provider_shows_disabled():
    cfg = _default_cfg(sentence_provider="none")
    mock_anki = MagicMock()
    mock_anki.ping.return_value = True
    mock_anki.get_existing_vocab.return_value = []

    with patch("kado.cli.KadoConfig.load", return_value=cfg), \
         patch("kado.anki.AnkiConnect", return_value=mock_anki, create=True):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "disabled" in result.output


# ── add --dry-run ─────────────────────────────────────────────────────


def test_add_dry_run_skips_anki():
    from kado.models import VocabCard

    cfg = _default_cfg(sentence_provider="none", audio_enabled=False)
    fake_card = VocabCard(word="食べる", reading="たべる", meaning="to eat")

    with patch("kado.cli.KadoConfig.load", return_value=cfg), \
         patch("kado.dictionary.lookup", return_value=fake_card), \
         patch("kado.anki.AnkiConnect", create=True) as mock_anki_cls:
        result = runner.invoke(app, ["add", "食べる", "--dry-run", "--no-audio", "--no-sentence"])

    assert result.exit_code == 0
    assert "Dry run" in result.output
    # AnkiConnect should not have been used to add a card
    mock_anki_cls.return_value.add_card.assert_not_called()


def test_add_word_not_found_exits_1():
    cfg = _default_cfg(sentence_provider="none", audio_enabled=False)

    with patch("kado.cli.KadoConfig.load", return_value=cfg), \
         patch("kado.dictionary.lookup", side_effect=ValueError("Not found")):
        result = runner.invoke(app, ["add", "zzz", "--no-audio", "--no-sentence"])

    assert result.exit_code == 1
    assert "Not found" in result.output


# ── _parse_pages ──────────────────────────────────────────────────────


def test_parse_pages_single():
    assert _parse_pages("3") == {3}


def test_parse_pages_range():
    assert _parse_pages("1-4") == {1, 2, 3, 4}


def test_parse_pages_list():
    assert _parse_pages("1,3,5") == {1, 3, 5}


def test_parse_pages_mixed():
    assert _parse_pages("1-3,5,7-8") == {1, 2, 3, 5, 7, 8}


def test_parse_pages_single_with_spaces():
    assert _parse_pages(" 2 ") == {2}
