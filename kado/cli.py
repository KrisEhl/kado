"""kado CLI — Japanese Anki vocab card generator."""

from __future__ import annotations

from typing import Optional

import typer
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from kado.config import KadoConfig

app = typer.Typer(
    name="kado",
    help="Japanese Anki vocab card generator. Look up, enrich, and sync cards to Anki.",
    no_args_is_help=True,
)


# ── add ──────────────────────────────────────────────────────────────


@app.command()
def add(
    word: str = typer.Argument(help="Japanese word to add (e.g. 食べる)"),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip audio generation"),
    no_sentence: bool = typer.Option(
        False, "--no-sentence", help="Skip example sentence"
    ),
    tag: Optional[list[str]] = typer.Option(None, "--tag", "-t", help="Extra tags"),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-w", help="Overwrite if word already exists"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview card without adding to Anki"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="LLM provider: ollama (default) or huggingface"
    ),
    debug: bool = typer.Option(False, "--debug", help="Show detailed LLM/OCR debug logs"),
):
    """Look up a word, generate audio + example sentence, and add to Anki."""
    from kado.anki import AnkiConnect, AnkiConnectError
    from kado.audio import generate_audio
    from kado.debug import set_debug
    from kado.dictionary import lookup
    from kado.sentences import generate_example, resolve_model_name

    set_debug(debug)
    cfg = KadoConfig.load()
    active_provider = provider or cfg.sentence_provider

    # 1. Dictionary lookup
    rprint(f"[bold]🔍 Looking up [cyan]{word}[/cyan] on Jisho...[/bold]")
    try:
        card = lookup(word)
    except ValueError as e:
        rprint(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)

    rprint(f"   [green]✓[/green] {card.word} 【{card.reading}】 — {card.meaning}")

    # 2. Extra tags
    if tag:
        card.tags.extend(tag)

    # 3. Example sentence
    if not no_sentence and active_provider != "none":
        from rich.console import Console

        console = Console()
        known_vocab: list[str] = []
        if not dry_run:
            try:
                anki = AnkiConnect(cfg)
                known_vocab = anki.get_existing_vocab()
            except AnkiConnectError:
                pass  # will connect later anyway

        # Pick the right model arg based on provider
        model_arg = None
        if active_provider == "huggingface" and cfg.hf_model:
            model_arg = cfg.hf_model
        elif active_provider == "ollama" and cfg.ollama_model:
            model_arg = cfg.ollama_model

        resolved_name = resolve_model_name(
            provider=active_provider,
            model=model_arg,
            ollama_url=cfg.ollama_url,
        )
        with console.status(f"[bold]💬 Generating example sentence ({resolved_name})...[/bold]"):
            ja, en, model_used = generate_example(
                word=card.word,
                reading=card.reading,
                meaning=card.meaning,
                known_vocab=known_vocab,
                provider=active_provider,
                model=model_arg,
                ollama_url=cfg.ollama_url,
            )
        if ja:
            card.example_ja = ja
            card.example_en = en
            rprint(f"   [green]✓[/green] {card.example_ja}  [dim]({model_used})[/dim]")
            rprint(f"     {card.example_en}")
        else:
            rprint(
                f"   [yellow]⚠ Sentence generation failed ({active_provider} may be unavailable)[/yellow]"
            )

    # 4. Audio
    if not no_audio and cfg.audio_enabled:
        from rich.console import Console

        console = Console()
        with console.status("[bold]🔊 Generating audio (gTTS)...[/bold]"):
            try:
                card.audio_path = generate_audio(card.word, lang=cfg.audio_lang)
            except Exception as e:
                card.audio_path = None
                rprint(f"   [yellow]⚠ Audio failed: {e}[/yellow]")
        if card.audio_path:
            rprint("   [green]✓[/green] Audio saved [dim](gTTS)[/dim]")

    # 5. Preview
    _print_card_preview(card)

    # 6. Add to Anki
    if dry_run:
        rprint("[dim]Dry run — card was not added to Anki.[/dim]")
        return

    rprint(f"[bold]📤 Syncing to Anki deck [cyan]{cfg.anki_deck}[/cyan] (AnkiConnect @ {cfg.anki_url})...[/bold]")
    try:
        anki = AnkiConnect(cfg)
        anki.setup()

        exists = anki.has_word(card.word)

        if exists and not overwrite:
            rprint(
                f"[yellow]⚠ '{card.word}' already exists. Use --overwrite / -w to replace it.[/yellow]"
            )
            raise typer.Exit(0)

        if exists:
            note_id = anki.update_card(card)
            rprint(f"[green]✓ Updated! Note ID: {note_id}[/green]")
        else:
            note_id = anki.add_card(card)
            rprint(f"[green]✓ Added! Note ID: {note_id}[/green]")
    except AnkiConnectError as e:
        rprint(f"[red]✗ AnkiConnect error: {e}[/red]")
        raise typer.Exit(1)


# ── batch ────────────────────────────────────────────────────────────


@app.command()
def batch(
    file: str = typer.Argument(help="Text file with one word per line"),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip audio generation"),
    no_sentence: bool = typer.Option(
        False, "--no-sentence", help="Skip example sentences"
    ),
    tag: Optional[list[str]] = typer.Option(None, "--tag", "-t", help="Extra tags"),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-w", help="Overwrite existing words"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="LLM provider: ollama (default) or huggingface"
    ),
    debug: bool = typer.Option(False, "--debug", help="Show detailed LLM/OCR debug logs"),
):
    """Add multiple words from a text file (one word per line)."""
    from pathlib import Path

    from kado.debug import set_debug

    set_debug(debug)
    words_file = Path(file)
    if not words_file.exists():
        rprint(f"[red]✗ File not found: {file}[/red]")
        raise typer.Exit(1)

    words = [w.strip() for w in words_file.read_text().splitlines() if w.strip()]
    rprint(f"[bold]Processing {len(words)} words...[/bold]\n")

    for i, word in enumerate(words, 1):
        rprint(f"[dim]── [{i}/{len(words)}] ──[/dim]")
        try:
            add(
                word=word,
                no_audio=no_audio,
                no_sentence=no_sentence,
                tag=tag,
                overwrite=overwrite,
                dry_run=False,
                provider=provider,
                debug=debug,
            )
        except SystemExit:
            pass  # typer.Exit from add — continue with next word
        rprint()

    rprint("[bold green]✓ Batch complete![/bold green]")


# ── import ───────────────────────────────────────────────────────────


@app.command("import")
def import_pdf(
    file: str = typer.Argument(help="PDF file with vocabulary table"),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip audio generation"),
    no_sentence: bool = typer.Option(
        False, "--no-sentence", help="Skip example sentences"
    ),
    tag: Optional[list[str]] = typer.Option(None, "--tag", "-t", help="Extra tags"),
    overwrite: bool = typer.Option(
        False, "--overwrite", "-w", help="Overwrite existing words"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview extracted words without adding"
    ),
    use_pdf_meaning: bool = typer.Option(
        True,
        "--pdf-meaning/--jisho-meaning",
        help="Use meaning from PDF (default) or look up on Jisho",
    ),
    no_vision: bool = typer.Option(
        False, "--no-vision", help="Skip vision model, use tesseract OCR only"
    ),
    no_llm_cleanup: bool = typer.Option(
        False, "--no-llm-cleanup", help="Skip LLM cleanup of noisy OCR results"
    ),
    pages: Optional[str] = typer.Option(
        None,
        "--pages",
        "-p",
        help="Page selection: range (1-4), list (1,3,5), or mix (1-3,5)",
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Show detailed LLM/OCR debug logs and raw OCR output"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", help="LLM provider: ollama (default) or huggingface"
    ),
):
    """Import vocabulary from a PDF file (e.g. J.Bridge 語彙リスト)."""
    from pathlib import Path

    from kado.debug import set_debug
    from kado.pdf_import import parse_vocab_pdf, dump_ocr_debug

    set_debug(debug)
    cfg = KadoConfig.load()
    active_provider = provider or cfg.sentence_provider

    pdf_path = Path(file)
    if not pdf_path.exists():
        rprint(f"[red]✗ File not found: {file}[/red]")
        raise typer.Exit(1)

    # Parse page selection
    page_set = _parse_pages(pages) if pages else None

    from kado.sentences import resolve_model_name

    resolved_name = resolve_model_name(
        provider=active_provider,
        model=cfg.ollama_model if active_provider == "ollama" else cfg.hf_model or None,
        ollama_url=cfg.ollama_url,
    )

    if debug:
        rprint(
            f"[bold]📄 Debug: dumping OCR data for [cyan]{pdf_path.name}[/cyan] (LLM: {resolved_name})...[/bold]"
        )
        cards = dump_ocr_debug(
            str(pdf_path),
            pages=page_set,
            use_vision=not no_vision,
            provider=active_provider,
            ollama_url=cfg.ollama_url,
            ollama_model=cfg.ollama_model,
        )
        rprint()
    else:
        rprint(
            f"[bold]📄 Parsing [cyan]{pdf_path.name}[/cyan] (LLM: {resolved_name})...[/bold]"
        )
        cards = parse_vocab_pdf(
            str(pdf_path),
            use_vision=not no_vision,
            llm_cleanup=not no_llm_cleanup,
            pages=page_set,
            provider=active_provider,
            ollama_url=cfg.ollama_url,
            ollama_model=cfg.ollama_model,
        )

    if not cards:
        rprint("[red]✗ No vocabulary found in the PDF.[/red]")
        raise typer.Exit(1)

    rprint(f"   [green]✓[/green] Found {len(cards)} words\n")

    # Preview table
    _SOURCE_LABELS = {
        "text": ("[green]text[/green]", "extracted from PDF text layer"),
        "vision": ("[blue]vision[/blue]", "extracted by HF vision model"),
        "ocr": ("[green]ocr[/green]", "found by OCR, cleaned by LLM"),
        "llm": ("[yellow]llm[/yellow]", "reconstructed from German via LLM"),
    }

    preview_table = Table(title=f"Vocabulary from {pdf_path.name}", show_lines=False)
    preview_table.add_column("#", style="dim", width=4)
    preview_table.add_column("Source", width=8)
    preview_table.add_column("Word", style="bold")
    preview_table.add_column("Reading", style="cyan")
    preview_table.add_column("Meaning")
    for i, card in enumerate(cards, 1):
        source_label = _SOURCE_LABELS.get(card.source, ("[dim]?[/dim]", ""))[0]
        preview_table.add_row(
            str(i), source_label, card.word, card.reading, card.meaning
        )
    rprint(preview_table)

    # Source summary
    from collections import Counter

    source_counts = Counter(c.source for c in cards)
    parts = []
    for src, (label, desc) in _SOURCE_LABELS.items():
        if src in source_counts:
            parts.append(f"{label}: {source_counts[src]} ({desc})")
    if parts:
        rprint("\n" + "\n".join(f"  {p}" for p in parts))
    rprint()

    if dry_run:
        rprint("[dim]Dry run — no cards were added to Anki.[/dim]")
        return

    # Confirm before importing
    if not typer.confirm(f"Add {len(cards)} words to Anki?", default=True):
        rprint("[dim]Cancelled.[/dim]")
        return

    rprint()

    from kado.anki import AnkiConnect, AnkiConnectError
    from kado.audio import generate_audio
    from kado.dictionary import lookup
    from kado.sentences import generate_example

    anki = AnkiConnect(cfg)

    try:
        anki.setup()
    except AnkiConnectError as e:
        rprint(f"[red]✗ AnkiConnect error: {e}[/red]")
        raise typer.Exit(1)

    # Fetch known vocab once (not per word — it's the same query each time)
    known_vocab: list[str] = []
    if not no_sentence and active_provider != "none":
        try:
            known_vocab = anki.get_existing_vocab()
        except AnkiConnectError:
            pass

    # Pick model arg based on provider
    model_arg = None
    if active_provider == "huggingface" and cfg.hf_model:
        model_arg = cfg.hf_model
    elif active_provider == "ollama" and cfg.ollama_model:
        model_arg = cfg.ollama_model

    added = 0
    updated = 0
    skipped = 0

    for i, card in enumerate(cards, 1):
        rprint(f"[dim]── [{i}/{len(cards)}] {card.word} ──[/dim]")

        # Optionally enrich with Jisho data
        if not use_pdf_meaning:
            try:
                jisho_card = lookup(card.word)
                card.reading = jisho_card.reading or card.reading
                card.meaning = jisho_card.meaning or card.meaning
                card.part_of_speech = jisho_card.part_of_speech
                card.tags = jisho_card.tags
            except (ValueError, Exception):
                pass  # keep PDF data

        # Fill in reading from Jisho if PDF didn't have one
        if not card.reading:
            try:
                jisho_card = lookup(card.word)
                card.reading = jisho_card.reading
                if not card.part_of_speech:
                    card.part_of_speech = jisho_card.part_of_speech
                if not card.tags:
                    card.tags = jisho_card.tags
            except (ValueError, Exception):
                pass

        # Extra tags
        if tag:
            card.tags.extend(tag)

        # Example sentence
        if not no_sentence and active_provider != "none":
            ja, en, model_used = generate_example(
                word=card.word,
                reading=card.reading,
                meaning=card.meaning,
                known_vocab=known_vocab,
                provider=active_provider,
                model=model_arg,
                ollama_url=cfg.ollama_url,
            )
            if ja:
                card.example_ja = ja
                card.example_en = en
                rprint(f"   💬 {card.example_ja}  [dim]({model_used})[/dim]")

        # Audio
        if not no_audio and cfg.audio_enabled:
            try:
                card.audio_path = generate_audio(card.word)
            except Exception:
                pass

        # Add or update
        try:
            exists = anki.has_word(card.word)
            if exists and not overwrite:
                rprint("   [yellow]⚠ Exists, skipping[/yellow]")
                skipped += 1
            elif exists:
                anki.update_card(card)
                rprint("   [green]✓ Updated[/green]")
                updated += 1
            else:
                anki.add_card(card)
                rprint("   [green]✓ Added[/green]")
                added += 1
        except AnkiConnectError as e:
            rprint(f"   [red]✗ {e}[/red]")

    rprint(
        f"\n[bold green]✓ Import complete![/bold green] Added: {added}, Updated: {updated}, Skipped: {skipped}"
    )


# ── lookup ───────────────────────────────────────────────────────────


@app.command("lookup")
def lookup_cmd(
    word: str = typer.Argument(help="Japanese word to look up"),
):
    """Look up a word without adding it to Anki."""
    from kado.dictionary import lookup

    rprint(f"[bold]🔍 Looking up [cyan]{word}[/cyan]...[/bold]")
    try:
        card = lookup(word)
    except ValueError as e:
        rprint(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)

    _print_card_preview(card)


# ── config ───────────────────────────────────────────────────────────


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current config"),
):
    """View or edit kado configuration."""
    cfg = KadoConfig.load()

    if show:
        table = Table(title="kado config")
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        table.add_row("AnkiConnect URL", cfg.anki_url)
        table.add_row("Deck", cfg.anki_deck)
        table.add_row("Note model", cfg.anki_model)
        table.add_row("LLM provider", cfg.sentence_provider)
        table.add_row("Ollama URL", cfg.ollama_url)
        table.add_row("Ollama model", cfg.ollama_model or "(auto)")
        table.add_row("HF model", cfg.hf_model or "(auto)")
        table.add_row("Audio", "enabled" if cfg.audio_enabled else "disabled")
        rprint(table)
        return

    # Interactive setup
    rprint("[bold]kado setup[/bold]\n")

    # Fuzzy deck picker
    cfg.anki_deck = _pick_deck(cfg)

    use_sentences = typer.confirm("Enable AI example sentences?", default=True)
    if use_sentences:
        provider_choice = typer.prompt(
            "LLM provider (ollama = local, huggingface = remote API)",
            default="ollama",
        )
        cfg.sentence_provider = (
            provider_choice
            if provider_choice in ("ollama", "huggingface")
            else "ollama"
        )
    else:
        cfg.sentence_provider = "none"

    cfg.audio_enabled = typer.confirm("Enable audio generation?", default=True)

    cfg.save()
    rprint("\n[green]✓ Config saved to ~/.kado/config.toml[/green]")


# ── status ───────────────────────────────────────────────────────────


@app.command()
def status():
    """Check connection to Anki and show deck info."""
    from kado.anki import AnkiConnect, AnkiConnectError

    cfg = KadoConfig.load()
    anki = AnkiConnect(cfg)

    rprint("[bold]kado status[/bold]\n")

    if anki.ping():
        rprint(f"[green]✓[/green] AnkiConnect reachable at {cfg.anki_url}")
    else:
        rprint(f"[red]✗[/red] Cannot reach AnkiConnect at {cfg.anki_url}")
        rprint("  Make sure Anki is running with the AnkiConnect add-on installed.")
        raise typer.Exit(1)

    try:
        vocab = anki.get_existing_vocab()
        rprint(f"[green]✓[/green] Deck '{cfg.anki_deck}' has {len(vocab)} words")
    except AnkiConnectError as e:
        rprint(f"[yellow]⚠[/yellow] Deck info: {e}")

    from kado.sentences import resolve_model_name, _ollama_available_models

    prov = cfg.sentence_provider

    if prov == "ollama":
        resolved = resolve_model_name(
            provider="ollama",
            model=cfg.ollama_model or None,
            ollama_url=cfg.ollama_url,
        )
        rprint(f"\n  LLM provider:        [green]ollama[/green] → [bold]{resolved}[/bold]")

        # Show all installed Ollama models
        available = _ollama_available_models(cfg.ollama_url)
        if available is not None:
            rprint(f"  Installed models:    {', '.join(sorted(available))}")
        else:
            rprint("  Ollama:              [red]not running[/red] — start with: ollama serve")
    elif prov == "huggingface":
        resolved = resolve_model_name(
            provider="huggingface",
            model=cfg.hf_model or None,
        )
        rprint(f"\n  LLM provider:        [green]huggingface[/green] → [bold]{resolved}[/bold]")
    else:
        rprint("\n  Sentence generation: [dim]disabled[/dim]")

    rprint(
        f"  Audio generation:    {'[green]gTTS[/green]' if cfg.audio_enabled else '[dim]disabled[/dim]'}"
    )


# ── export ───────────────────────────────────────────────────────────


@app.command()
def export(
    output: str = typer.Argument("kado_deck.apkg", help="Output .apkg filename"),
):
    """Export the Anki deck as a .apkg file (backup / sharing)."""
    from kado.anki import AnkiConnect, AnkiConnectError

    cfg = KadoConfig.load()
    anki = AnkiConnect(cfg)

    rprint(f"[bold]Exporting deck '{cfg.anki_deck}'...[/bold]")
    try:
        path = anki._invoke(
            "exportPackage", deck=cfg.anki_deck, path=output, includeSched=False
        )
        rprint(f"[green]✓ Exported to {path or output}[/green]")
    except AnkiConnectError as e:
        rprint(f"[red]✗ Export failed: {e}[/red]")
        raise typer.Exit(1)


# ── helpers ──────────────────────────────────────────────────────────


def _print_card_preview(card) -> None:
    """Pretty-print a VocabCard as a Rich panel."""
    lines = [
        f"[bold]{card.word}[/bold]  【{card.reading}】",
        f"[dim]{card.part_of_speech}[/dim]" if card.part_of_speech else "",
        f"\n{card.meaning}",
    ]
    if card.example_ja:
        lines.append(f"\n[italic]{card.example_ja}[/italic]")
        lines.append(f"[dim]{card.example_en}[/dim]")
    if card.audio_path:
        lines.append(f"\n🔊 [dim]{card.audio_path}[/dim]")
    if card.tags:
        lines.append(f"\n[dim]tags: {', '.join(card.tags)}[/dim]")

    rprint(
        Panel(
            "\n".join(line for line in lines if line),
            title="📇 Card Preview",
            border_style="cyan",
        )
    )


# ── fuzzy deck picker ────────────────────────────────────────────────


def _pick_deck(cfg) -> str:
    """Interactive fuzzy deck picker. Falls back to text prompt if Anki is unreachable."""
    from InquirerPy import inquirer
    from kado.anki import AnkiConnect, AnkiConnectError

    try:
        anki = AnkiConnect(cfg)
        decks = sorted(anki.list_decks())
    except AnkiConnectError:
        rprint("[yellow]⚠ Cannot reach Anki — type the deck name manually.[/yellow]")
        return typer.prompt("Anki deck name", default=cfg.anki_deck)

    if not decks:
        rprint("[yellow]⚠ No decks found in Anki.[/yellow]")
        return typer.prompt("Anki deck name", default=cfg.anki_deck)

    # Default selection to current deck if it exists
    default = cfg.anki_deck if cfg.anki_deck in decks else None

    selected = inquirer.fuzzy(
        message="Select deck:",
        choices=decks,
        default=default,
        max_height="60%",
    ).execute()

    rprint(f"[green]✓ Selected deck: {selected}[/green]")
    return selected


def _parse_pages(spec: str) -> set[int]:
    """Parse a page spec like '1-3,5,7-9' into a set of 1-based page numbers."""
    result: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(part))
    return result


if __name__ == "__main__":
    app()
