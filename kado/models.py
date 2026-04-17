"""Data models for kado."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VocabCard:
    """A single vocabulary card with all enriched fields."""

    word: str  # e.g. 食べる
    reading: str = ""  # e.g. たべる
    meaning: str = ""  # e.g. to eat
    part_of_speech: str = ""  # e.g. Godan verb
    example_ja: str = ""  # Japanese example sentence
    example_en: str = ""  # English translation of example
    audio_path: str | None = None          # path to word audio .mp3
    sentence_audio_path: str | None = None  # path to example sentence audio .mp3
    tags: list[str] = field(default_factory=list)
    source: str = ""  # "ocr", "llm", "vision", "text" — how this card was extracted

    @property
    def summary(self) -> str:
        parts = [f"{self.word} 【{self.reading}】" if self.reading else self.word]
        if self.meaning:
            parts.append(self.meaning)
        return " — ".join(parts)

    @property
    def has_example(self) -> bool:
        return bool(self.example_ja)


@dataclass
class GrammarCard:
    """A Japanese grammar point card."""

    pattern: str               # e.g. ～そうです
    meaning: str               # e.g. Looks like ~; seems ~
    formation: str             # Formation rules (plain text, one rule per line)
    note: str = ""             # Nuances, exceptions, contrasts
    jlpt: str = ""             # e.g. N4
    # Up to 4 examples (verb / i-adj / na-adj / noun), (ja, en) pairs
    examples: list[tuple[str, str]] = field(default_factory=list)
    example_audio_paths: list[str | None] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
