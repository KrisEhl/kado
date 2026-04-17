"""AnkiConnect integration — create models, add notes, query existing vocab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import urllib.request

from kado.config import KadoConfig
from kado.models import GrammarCard, VocabCard

# Kado note model field definitions
KADO_FIELDS = [
    "Word",
    "Reading",
    "Meaning",
    "PartOfSpeech",
    "ExampleJA",
    "ExampleEN",
    "Audio",
    "ExampleAudio",
]

CARD_FRONT_TEMPLATE = """
<div class="card front">
  <div class="word">{{Word}}</div>
</div>
""".strip()

CARD_BACK_TEMPLATE = """
<div class="card back">
  <div class="word">{{Word}}</div>
  <div class="reading">{{Reading}}</div>
  <hr>
  <div class="meaning">{{Meaning}}</div>
  <div class="pos">{{PartOfSpeech}}</div>
  {{#Audio}}{{Audio}}{{/Audio}}{{#ExampleAudio}}{{ExampleAudio}}{{/ExampleAudio}}
  {{#ExampleJA}}
  <hr>
  <div class="example-ja">{{ExampleJA}}</div>
  <div class="example-en">{{ExampleEN}}</div>
  {{/ExampleJA}}
</div>
""".strip()

CARD_CSS = """
.card { font-family: "Hiragino Sans", "Yu Gothic", "Noto Sans JP", sans-serif; text-align: center; padding: 20px; }
.word { font-size: 48px; font-weight: bold; margin: 20px 0; }
.reading { font-size: 28px; color: #666; }
.meaning { font-size: 24px; margin: 10px 0; }
.pos { font-size: 14px; color: #999; font-style: italic; }
.example-ja { font-size: 24px; margin: 10px 0; }
.example-en { font-size: 24px; color: #555; }
hr { border: none; border-top: 1px solid #ddd; margin: 15px 0; }
""".strip()


GRAMMAR_MODEL_NAME = "Kado-Grammar"

GRAMMAR_FIELDS = [
    "Pattern",
    "Meaning",
    "Formation",
    "Note",
    "JLPT",
    "Example1JA", "Example1EN", "Example1Audio",
    "Example2JA", "Example2EN", "Example2Audio",
    "Example3JA", "Example3EN", "Example3Audio",
    "Example4JA", "Example4EN", "Example4Audio",
]

# JS picks the same example on front and back using a daily seed keyed on the pattern.
_GRAMMAR_EXAMPLE_JS = """
<script>
(function() {
  var items = document.querySelectorAll('.ex-item');
  if (!items.length) return;
  var seed = '{{Pattern}}' + new Date().toDateString();
  var h = 0;
  for (var i = 0; i < seed.length; i++) {
    h = Math.imul(31, h) + seed.charCodeAt(i) | 0;
  }
  var item = items[Math.abs(h) % items.length];
  var jaEl = document.getElementById('ex-ja');
  if (jaEl) jaEl.textContent = item.dataset.ja;
  var enEl = document.getElementById('ex-en');
  if (enEl) enEl.textContent = item.dataset.en;
  var fname = item.dataset.audio;
  if (fname) { try { new Audio(fname).play(); } catch(e) {} }
})();
</script>
""".strip()

# Hidden data store — Anki processes {{...}} before JS runs
_GRAMMAR_EXAMPLE_DATA = """
<div style="display:none">
{{#Example1JA}}<span class="ex-item" data-ja="{{Example1JA}}" data-en="{{Example1EN}}" data-audio="{{Example1Audio}}"></span>{{/Example1JA}}
{{#Example2JA}}<span class="ex-item" data-ja="{{Example2JA}}" data-en="{{Example2EN}}" data-audio="{{Example2Audio}}"></span>{{/Example2JA}}
{{#Example3JA}}<span class="ex-item" data-ja="{{Example3JA}}" data-en="{{Example3EN}}" data-audio="{{Example3Audio}}"></span>{{/Example3JA}}
{{#Example4JA}}<span class="ex-item" data-ja="{{Example4JA}}" data-en="{{Example4EN}}" data-audio="{{Example4Audio}}"></span>{{/Example4JA}}
</div>
""".strip()

GRAMMAR_FRONT_TEMPLATE = f"""
<div class="card front">
  <div id="ex-ja" class="example-ja"></div>
</div>
{_GRAMMAR_EXAMPLE_DATA}
{_GRAMMAR_EXAMPLE_JS}
""".strip()

GRAMMAR_BACK_TEMPLATE = f"""
<div class="card back">
  <div id="ex-ja" class="example-ja"></div>
  <div id="ex-en" class="example-en"></div>
  <hr>
  <div class="pattern">{{{{Pattern}}}}</div>
  <div class="meaning">{{{{Meaning}}}}</div>
  {{{{#Note}}}}<div class="note">{{{{Note}}}}</div>{{{{/Note}}}}
  {{{{#JLPT}}}}<div class="jlpt">{{{{JLPT}}}}</div>{{{{/JLPT}}}}
  <hr>
  <div class="formation">{{{{Formation}}}}</div>
</div>
{_GRAMMAR_EXAMPLE_DATA}
{_GRAMMAR_EXAMPLE_JS}
""".strip()

GRAMMAR_CSS = """
.card { font-family: "Hiragino Sans", "Yu Gothic", "Noto Sans JP", sans-serif; text-align: center; padding: 20px; }
.pattern { font-size: 36px; font-weight: bold; margin: 10px 0; }
.meaning { font-size: 24px; margin: 8px 0; }
.note { font-size: 16px; color: #e67e22; font-style: italic; margin: 6px 0; }
.jlpt { font-size: 13px; color: #999; margin: 4px 0; }
.formation { font-size: 16px; text-align: left; display: inline-block; margin: 10px auto; border-collapse: collapse; }
.formation td { padding: 4px 12px; vertical-align: top; }
.ftype { font-weight: bold; color: #444; white-space: nowrap; }
.fex { font-size: 18px; }
.ftrans { color: #777; }
.fnote { color: #e67e22; font-style: italic; }
.example-ja { font-size: 28px; margin: 16px 0 4px; }
.example-en { font-size: 20px; color: #555; margin-bottom: 10px; }
hr { border: none; border-top: 1px solid #ddd; margin: 12px 0; }
""".strip()


class AnkiConnectError(Exception):
    pass


class AnkiConnect:
    """Client for the AnkiConnect REST API."""

    def __init__(self, cfg: KadoConfig):
        self.url = cfg.anki_url
        self.deck = cfg.anki_deck
        self.model = cfg.anki_model
        self._version = 6

    # ------------------------------------------------------------------
    # Low-level
    # ------------------------------------------------------------------

    def _invoke(self, action: str, **params) -> Any:
        payload = {"action": action, "version": self._version}
        if params:
            payload["params"] = params
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        # Use a no-proxy handler for localhost
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=10) as resp:
                body = json.loads(resp.read().decode())
        except (ConnectionRefusedError, OSError):
            raise AnkiConnectError(
                "Cannot connect to AnkiConnect. Make sure Anki is running "
                "and the AnkiConnect add-on is installed (code 2055492159)."
            )
        if body.get("error"):
            raise AnkiConnectError(body["error"])
        return body.get("result")

    # ------------------------------------------------------------------
    # Deck discovery
    # ------------------------------------------------------------------

    def list_decks(self) -> list[str]:
        """Return all deck names from Anki."""
        return self._invoke("deckNames") or []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def ensure_model(self) -> None:
        """Create the Kado note model if it doesn't exist, or migrate fields/template if it does."""
        models = self._invoke("modelNames")
        if self.model not in models:
            self._invoke(
                "createModel",
                modelName=self.model,
                inOrderFields=KADO_FIELDS,
                css=CARD_CSS,
                cardTemplates=[
                    {
                        "Name": "Kado Card",
                        "Front": CARD_FRONT_TEMPLATE,
                        "Back": CARD_BACK_TEMPLATE,
                    }
                ],
            )
            return

        # Model exists — add any fields that are missing (e.g. ExampleAudio added later)
        existing_fields = self._invoke("modelFieldNames", modelName=self.model)
        for field in KADO_FIELDS:
            if field not in existing_fields:
                self._invoke("modelFieldAdd", modelName=self.model, fieldName=field)

        # Keep the card template and CSS up to date
        self._invoke(
            "updateModelTemplates",
            model={
                "name": self.model,
                "templates": {
                    "Kado Card": {
                        "Front": CARD_FRONT_TEMPLATE,
                        "Back": CARD_BACK_TEMPLATE,
                    }
                },
            },
        )
        self._invoke(
            "updateModelStyling",
            model={"name": self.model, "css": CARD_CSS},
        )

    def ensure_deck(self) -> None:
        """Create the target deck if it doesn't exist."""
        self._invoke("createDeck", deck=self.deck)

    def setup(self) -> None:
        """Run all first-time setup (deck + model)."""
        self.ensure_deck()
        self.ensure_model()

    # ------------------------------------------------------------------
    # Cards
    # ------------------------------------------------------------------

    def add_card(self, card: VocabCard) -> int:
        """Add a VocabCard to Anki. Returns the new note ID."""
        note: dict = {
            "deckName": self.deck,
            "modelName": self.model,
            "fields": {
                "Word": card.word,
                "Reading": card.reading,
                "Meaning": card.meaning,
                "PartOfSpeech": card.part_of_speech,
                "ExampleJA": card.example_ja,
                "ExampleEN": card.example_en,
            },
            "tags": card.tags or [],
            "options": {"allowDuplicate": False},
        }

        note_id = self._invoke("addNote", note=note)

        # Store audio file separately and reference it in the Audio field
        if card.audio_path and Path(card.audio_path).exists():
            filename = Path(card.audio_path).name
            self._invoke("storeMediaFile", filename=filename, path=str(card.audio_path))
            self._invoke(
                "updateNoteFields",
                note={"id": note_id, "fields": {"Audio": f"[sound:{filename}]"}},
            )

        if card.sentence_audio_path and Path(card.sentence_audio_path).exists():
            filename = Path(card.sentence_audio_path).name
            self._invoke("storeMediaFile", filename=filename, path=str(card.sentence_audio_path))
            self._invoke(
                "updateNoteFields",
                note={"id": note_id, "fields": {"ExampleAudio": f"[sound:{filename}]"}},
            )

        return note_id

    def get_existing_vocab(self, limit: int = 200) -> list[str]:
        """Return a list of words already in the deck (for sentence context)."""
        query = f'"deck:{self.deck}"'
        note_ids = self._invoke("findNotes", query=query)
        if not note_ids:
            return []

        # Take a sample if the deck is large
        ids_to_fetch = note_ids[-limit:]
        notes_info = self._invoke("notesInfo", notes=ids_to_fetch)
        words = []
        for note in notes_info:
            fields = note.get("fields", {})
            word = fields.get("Word", {}).get("value", "")
            if word:
                words.append(word)
        return words

    def find_word(self, word: str) -> int | None:
        """Find a note ID for a word in the deck. Returns None if not found."""
        escaped = word.replace('"', '\\"')
        query = f'"deck:{self.deck}" Word:"{escaped}"'
        ids = self._invoke("findNotes", query=query)
        return ids[0] if ids else None

    def has_word(self, word: str) -> bool:
        """Check if a word already exists in the deck."""
        return self.find_word(word) is not None

    def update_card(self, card: VocabCard, note_id: int | None = None) -> int:
        """Update an existing card in Anki. Returns the note ID.

        If *note_id* is provided it is used directly, avoiding a redundant
        ``find_word`` lookup (useful when the caller already holds the id).
        """
        if note_id is None:
            note_id = self.find_word(card.word)
        if not note_id:
            raise AnkiConnectError(f"Note not found for '{card.word}'")

        fields = {
            "Word": card.word,
            "Reading": card.reading,
            "Meaning": card.meaning,
            "PartOfSpeech": card.part_of_speech,
            "ExampleJA": card.example_ja,
            "ExampleEN": card.example_en,
        }

        self._invoke("updateNoteFields", note={"id": note_id, "fields": fields})

        # Update audio separately if we have a file
        if card.audio_path and Path(card.audio_path).exists():
            filename = Path(card.audio_path).name
            self._invoke(
                "updateNoteFields",
                note={"id": note_id, "fields": {"Audio": f"[sound:{filename}]"}},
            )
            self._invoke("storeMediaFile", filename=filename, path=str(card.audio_path))

        if card.sentence_audio_path and Path(card.sentence_audio_path).exists():
            filename = Path(card.sentence_audio_path).name
            self._invoke(
                "updateNoteFields",
                note={"id": note_id, "fields": {"ExampleAudio": f"[sound:{filename}]"}},
            )
            self._invoke("storeMediaFile", filename=filename, path=str(card.sentence_audio_path))

        # Update tags
        if card.tags:
            self._invoke("addTags", notes=[note_id], tags=" ".join(card.tags))

        return note_id

    # ------------------------------------------------------------------
    # Grammar cards
    # ------------------------------------------------------------------

    def ensure_grammar_model(self) -> None:
        """Create or migrate the Kado-Grammar note type."""
        models = self._invoke("modelNames")
        if GRAMMAR_MODEL_NAME not in models:
            self._invoke(
                "createModel",
                modelName=GRAMMAR_MODEL_NAME,
                inOrderFields=GRAMMAR_FIELDS,
                css=GRAMMAR_CSS,
                cardTemplates=[
                    {
                        "Name": "Grammar Card",
                        "Front": GRAMMAR_FRONT_TEMPLATE,
                        "Back": GRAMMAR_BACK_TEMPLATE,
                    }
                ],
            )
            return

        existing_fields = self._invoke("modelFieldNames", modelName=GRAMMAR_MODEL_NAME)
        for f in GRAMMAR_FIELDS:
            if f not in existing_fields:
                self._invoke("modelFieldAdd", modelName=GRAMMAR_MODEL_NAME, fieldName=f)

        self._invoke(
            "updateModelTemplates",
            model={
                "name": GRAMMAR_MODEL_NAME,
                "templates": {
                    "Grammar Card": {
                        "Front": GRAMMAR_FRONT_TEMPLATE,
                        "Back": GRAMMAR_BACK_TEMPLATE,
                    }
                },
            },
        )
        self._invoke(
            "updateModelStyling",
            model={"name": GRAMMAR_MODEL_NAME, "css": GRAMMAR_CSS},
        )

    def find_grammar_pattern(self, pattern: str) -> int | None:
        """Return the note ID for a grammar pattern, or None if not found."""
        escaped = pattern.replace('"', '\\"')
        query = f'"deck:{self.deck}" "note:{GRAMMAR_MODEL_NAME}" Pattern:"{escaped}"'
        ids = self._invoke("findNotes", query=query)
        return ids[0] if ids else None

    def has_grammar_pattern(self, pattern: str) -> bool:
        """Check if a grammar pattern already exists in the deck."""
        return self.find_grammar_pattern(pattern) is not None

    def add_grammar_card(self, card: GrammarCard) -> int:
        """Add a GrammarCard to Anki. Returns the new note ID."""
        examples = card.examples[:4]  # cap at 4
        audios = card.example_audio_paths[:4]

        fields: dict[str, str] = {
            "Pattern": card.pattern,
            "Meaning": card.meaning,
            "Formation": card.formation,
            "Note": card.note,
            "JLPT": card.jlpt,
        }
        for i, (ja, en) in enumerate(examples, 1):
            fields[f"Example{i}JA"] = ja
            fields[f"Example{i}EN"] = en

        note = {
            "deckName": self.deck,
            "modelName": GRAMMAR_MODEL_NAME,
            "fields": fields,
            "tags": card.tags or [],
            "options": {"allowDuplicate": False},
        }
        note_id = self._invoke("addNote", note=note)

        for i, audio_path in enumerate(audios, 1):
            if audio_path and Path(audio_path).exists():
                filename = Path(audio_path).name
                self._invoke("storeMediaFile", filename=filename, path=str(audio_path))
                # Store just the filename — JS plays it via new Audio(filename).
                # Do NOT use [sound:...] here: Anki auto-plays every [sound:] tag
                # it finds in the rendered HTML, which would play all examples at once.
                self._invoke(
                    "updateNoteFields",
                    note={"id": note_id, "fields": {f"Example{i}Audio": filename}},
                )

        return note_id

    def export_deck(
        self, deck: str, path: str, include_scheduling: bool = False
    ) -> str:
        """Export *deck* to an .apkg file at *path*.

        Args:
            deck: Deck name to export.
            path: Absolute filesystem path for the output .apkg file.
            include_scheduling: When ``True``, scheduling information is
                included in the export (default ``False``).

        Returns:
            The result string returned by AnkiConnect (typically ``None``
            on success, which AnkiConnect represents as a JSON null).
        """
        return self._invoke(
            "exportPackage",
            deck=deck,
            path=path,
            includeSched=include_scheduling,
        )

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Check if AnkiConnect is reachable."""
        try:
            self._invoke("version")
            return True
        except AnkiConnectError:
            return False
