# kado カード

Japanese Anki vocab card generator. Look up a word, auto-fill reading/meaning, generate an example sentence using your existing vocab, add TTS audio, and sync straight to Anki.

No API keys required — example sentences and OCR cleanup are powered by open-source models, either locally via Ollama or through Hugging Face's free inference API.

## Quick start

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone / cd into the project, then:
uv sync

# First-time setup — picks your Anki deck via fuzzy search
uv run kado config

# Add a word
uv run kado add 食べる

# Import a full vocab list from PDF
uv run kado import 語彙リスト2-2.pdf
```

## Prerequisites

### Required

**Anki** with the **AnkiConnect** add-on:

1. Open Anki
2. Go to `Tools → Add-ons → Get Add-ons`
3. Paste code `2055492159`, click OK
4. Restart Anki

AnkiConnect runs a local server on port 8765 that kado talks to. Anki must be open whenever you use kado.

**uv** (Python package manager):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Required for PDF import (`kado import`)

The `import` command parses vocabulary from scanned PDF files (like J.Bridge 語彙リスト). If your PDF has a text layer (not scanned), kado extracts the table directly and none of these extra dependencies are needed.

For scanned/image PDFs, kado runs up to three extraction methods and merges results for best coverage. Each word in the output is tagged with its source (ocr, llm, or vision).

#### Recommended: Ollama (local, free, no account needed)

Run LLMs locally for OCR cleanup, vocab reconstruction, and translation. This is the fastest and most reliable option since it doesn't depend on external APIs or credits.

```bash
# Install Ollama: https://ollama.com
# Then pull a model:
ollama pull qwen2.5:7b          # text model (~4.7GB) — OCR cleanup & reconstruction
ollama pull llava:7b             # vision model (optional) — reads page images directly
```

Kado auto-detects Ollama on `localhost:11434`. Override with env vars if needed:

```bash
KADO_OLLAMA_MODEL=qwen2.5:7b          # force a specific text model
KADO_OLLAMA_VISION_MODEL=llava:13b     # force a specific vision model
KADO_OLLAMA_URL=http://localhost:11434  # custom Ollama URL
```

#### Alternative: Hugging Face inference API

If you don't want to run models locally, kado falls back to Hugging Face's free serverless API. For vision support, you need a free HF account (no payment):

```bash
pip install -U huggingface-hub
huggingface-cli login
```

Note: HF free tier has monthly credit limits. If you hit them, switch to Ollama.

#### Poppler (required for all scanned PDF methods)

Converts PDF pages to images for OCR and vision models:

```bash
# macOS
brew install poppler

# Ubuntu / Debian
sudo apt install poppler-utils

# Arch
sudo pacman -S poppler

# Windows
# Download from https://github.com/oschwartz10612/poppler-windows/releases
# and add the bin/ folder to your PATH
```

#### Tesseract OCR (recommended alongside Ollama)

Local OCR provides the initial text extraction that LLM cleanup improves:

```bash
# macOS
brew install tesseract tesseract-lang

# Ubuntu / Debian
sudo apt install tesseract-ocr tesseract-ocr-jpn tesseract-ocr-deu

# Arch
sudo pacman -S tesseract tesseract-data-jpn tesseract-data-deu

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
# Select Japanese and German language packs during installation
```

If your vocab lists use English meanings instead of German, replace `tesseract-ocr-deu` with `tesseract-ocr-eng`.

## Commands

| Command | Description |
|---------|-------------|
| `kado add <word>` | Look up, enrich, and add a card to Anki |
| `kado add <word> -w` | Overwrite an existing card |
| `kado import <pdf>` | Import vocabulary from a PDF file |
| `kado batch <file>` | Add words from a text file (one per line) |
| `kado lookup <word>` | Look up a word without adding to Anki |
| `kado config` | Interactive setup (deck picker, options) |
| `kado config --show` | Show current configuration |
| `kado status` | Check Anki connection and deck info |
| `kado export [file]` | Export deck as .apkg file |

## Usage

### Adding a single word

```bash
uv run kado add 食べる
```

This will:
1. Look up 食べる on Jisho → reading, meaning, part of speech, JLPT tags
2. Fetch your existing vocab from Anki, then generate an example sentence via an open-source LLM that reuses words you already know
3. Generate TTS audio pronunciation
4. Sync the card directly to your Anki deck

### Importing from PDF

```bash
# Preview what kado finds in the PDF
uv run kado import 語彙リスト2-2.pdf --dry-run

# Import all words
uv run kado import 語彙リスト2-2.pdf

# Import specific pages only
uv run kado import 語彙リスト2-2.pdf -p 3-5,8

# Debug mode — shows all extraction sources and overlap analysis
uv run kado import 語彙リスト2-2.pdf --debug -p 5

# Import with English meanings from Jisho instead of PDF meanings
uv run kado import 語彙リスト2-2.pdf --jisho-meaning

# Fast import — skip sentences and audio
uv run kado import 語彙リスト2-2.pdf --no-sentence --no-audio

# Tag all imported cards
uv run kado import 語彙リスト2-2.pdf -t jbridge -t lesson2

# Use only tesseract OCR, no vision or LLM
uv run kado import 語彙リスト2-2.pdf --no-vision --no-llm-cleanup
```

### Batch add from text file

Create a text file with one word per line:

```
食べる
飲む
読む
```

Then:

```bash
uv run kado batch words.txt
```

### Options for `add` / `batch` / `import`

| Flag | Description |
|------|-------------|
| `--no-audio` | Skip TTS audio generation |
| `--no-sentence` | Skip example sentence generation |
| `--tag / -t` | Add extra tags (repeatable) |
| `--overwrite / -w` | Overwrite if word already exists in deck |
| `--no-vision` | Skip vision model, use tesseract OCR only (import) |
| `--no-llm-cleanup` | Skip LLM reconstruction from OCR text (import) |
| `--pages / -p` | Select pages to import, e.g. `1-3,5,8` (import) |
| `--debug` | Show detailed extraction breakdown per source (import) |
| `--provider` | LLM provider: `ollama` (default, local) or `huggingface` (remote) |
| `--dry-run` | Preview without adding to Anki |

## How example sentences work

When you add a word, kado fetches your existing vocabulary from Anki and asks an open-source LLM to write a natural example sentence that reuses words you've already studied. This creates reinforcement loops — every new card connects to things you know.

By default kado uses Ollama (local models). It auto-detects installed models and picks the best one. You can also switch to Hugging Face's free serverless API with `--provider huggingface` or by setting it in the config.

## Config

Stored at `~/.kado/config.toml`. Edit directly or run `kado config`.

```toml
[anki]
url = "http://localhost:8765"
deck = "Japanese::Vocabulary"
model = "Kado-Japanese"

[sentences]
provider = "ollama"        # "ollama" (default), "huggingface", or "none"
ollama_url = "http://localhost:11434"
ollama_model = ""          # empty = auto-select from installed models
hf_model = ""              # empty = auto-select (only used when provider = "huggingface")

[audio]
enabled = true
lang = "ja"
```

## Project structure

```
kado/
├── pyproject.toml          # dependencies and project metadata
├── kado/
│   ├── cli.py              # all CLI commands
│   ├── config.py           # ~/.kado/config.toml management
│   ├── dictionary.py       # Jisho API lookup
│   ├── anki.py             # AnkiConnect integration
│   ├── audio.py            # gTTS audio generation
│   ├── sentences.py        # LLM sentence generation (Ollama / HuggingFace)
│   ├── pdf_import.py       # PDF vocab extraction (text + OCR)
│   └── models.py           # VocabCard data model
```
