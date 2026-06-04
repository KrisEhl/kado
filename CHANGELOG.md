# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-04

### Added

- **Ubuntu / Linux support for local setup.** New `make setup-local` runs a
  one-command provisioning flow on Linux (including WSL2): `uv sync`, installs
  poppler, checks Tesseract + language packs, ensures Ollama is running and
  pulls the text (`qwen2.5:7b`) and vision (`llava:7b`) models, starts VOICEVOX
  in Docker, and points `voicevox_url` at the local engine in `~/.kado/config.toml`
  (without clobbering a custom value).

### Changed

- `make local-start` / `make local-stop` are now OS-aware (Linux and macOS)
  instead of macOS-only — on Linux they manage VOICEVOX without trying to
  open/quit Docker Desktop.
- `kado add` now labels the TTS backend it actually used (VOICEVOX vs gTTS)
  instead of always reporting gTTS.

### Fixed

- **Anki audio sync across hosts.** Media is now uploaded to AnkiConnect as
  base64 `data` rather than a local filesystem `path`, so audio syncs correctly
  when Anki runs on a different host than kado (e.g. Anki on Windows with kado
  under WSL2).
- Generated example sentences no longer include leaked `Line 1:` / `Line 2:`
  (and similar) prefixes from the model output.
- Vision-card meaning translation no longer fabricates spurious vocab cards
  from out-of-range model response ids (e.g. echoing the prompt's example word).

## [0.1.0]

- Initial release.
