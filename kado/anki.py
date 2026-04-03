"""AnkiConnect integration — create models, add notes, query existing vocab."""

from __future__ import annotations

from pathlib import Path

import json
import urllib.request

from kado.config import KadoConfig
from kado.models import VocabCard

# Kado note model field definitions
KADO_FIELDS = [
    "Word",
    "Reading",
    "Meaning",
    "PartOfSpeech",
    "ExampleJA",
    "ExampleEN",
    "Audio",
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
  {{#ExampleJA}}
  <hr>
  <div class="example-ja">{{ExampleJA}}</div>
  <div class="example-en">{{ExampleEN}}</div>
  {{/ExampleJA}}
  {{#Audio}}
  <div class="audio">{{Audio}}</div>
  {{/Audio}}
</div>
""".strip()

CARD_CSS = """
.card { font-family: "Hiragino Sans", "Yu Gothic", "Noto Sans JP", sans-serif; text-align: center; padding: 20px; }
.word { font-size: 48px; font-weight: bold; margin: 20px 0; }
.reading { font-size: 28px; color: #666; }
.meaning { font-size: 24px; margin: 10px 0; }
.pos { font-size: 14px; color: #999; font-style: italic; }
.example-ja { font-size: 20px; margin: 10px 0; }
.example-en { font-size: 16px; color: #555; }
hr { border: none; border-top: 1px solid #ddd; margin: 15px 0; }
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

    def _invoke(self, action: str, **params) -> dict:
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
        """Create the Kado note model if it doesn't already exist."""
        models = self._invoke("modelNames")
        if self.model in models:
            return
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

        # Attach audio if we have a file
        if card.audio_path and Path(card.audio_path).exists():
            filename = Path(card.audio_path).name
            note["audio"] = [
                {
                    "path": str(card.audio_path),
                    "filename": filename,
                    "fields": ["Audio"],
                }
            ]

        return self._invoke("addNote", note=note)

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
        query = f'"deck:{self.deck}" Word:{word}'
        ids = self._invoke("findNotes", query=query)
        return ids[0] if ids else None

    def has_word(self, word: str) -> bool:
        """Check if a word already exists in the deck."""
        return self.find_word(word) is not None

    def update_card(self, card: VocabCard) -> int:
        """Update an existing card in Anki. Returns the note ID."""
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
            fields["Audio"] = f"[sound:{filename}]"
            self._invoke(
                "updateNoteFields",
                note={"id": note_id, "fields": {"Audio": f"[sound:{filename}]"}},
            )
            self._invoke("storeMediaFile", filename=filename, path=str(card.audio_path))

        # Update tags
        if card.tags:
            self._invoke("addTags", notes=[note_id], tags=" ".join(card.tags))

        return note_id

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
