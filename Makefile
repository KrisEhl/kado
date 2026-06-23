UNAME_S := $(shell uname -s)
# Container CLI. Override with `make CONTAINER=nerdctl local-start`.
CONTAINER ?= docker
# Prefer Rancher Desktop's moby context when present — avoids the login-gated
# Docker Desktop daemon. Empty otherwise (docker uses its active context).
DOCKER_CONTEXT ?= $(shell docker context ls --format '{{.Name}}' 2>/dev/null | grep -qx rancher-desktop && echo rancher-desktop)
export DOCKER_CONTEXT
# macOS GUI app backing the runtime (auto-opened by local-start).
RUNTIME_APP ?= $(if $(DOCKER_CONTEXT),Rancher Desktop,Docker)
OLLAMA_TEXT_MODEL ?= qwen2.5:7b
OLLAMA_VISION_MODEL ?= llava:7b

.PHONY: setup-local start stop local-start local-stop

# One-shot local setup: Python deps, OCR tooling, Ollama models, and TTS engine.
# Idempotent — safe to re-run. Targets a Linux host (Docker runs as a daemon).
setup-local:
	@echo "→ [1/5] Installing Python dependencies (uv sync)..."
	@uv sync
	@echo "→ [2/5] Checking poppler (PDF → image for scanned imports)..."
	@command -v pdftoppm > /dev/null \
		&& echo "  poppler already installed" \
		|| (echo "  installing poppler-utils (sudo)..."; sudo apt-get update -qq && sudo apt-get install -y poppler-utils)
	@echo "→ [3/5] Checking tesseract OCR + jpn/deu/eng language packs..."
	@command -v tesseract > /dev/null \
		&& echo "  tesseract installed (langs: $$(tesseract --list-langs 2>&1 | tail -n +2 | tr '\n' ' '))" \
		|| (echo "  installing tesseract + language packs (sudo)..."; sudo apt-get install -y tesseract-ocr tesseract-ocr-jpn tesseract-ocr-deu tesseract-ocr-eng)
	@echo "→ [4/5] Ensuring Ollama is running and pulling models..."
	@curl -sf localhost:11434/api/tags > /dev/null 2>&1 \
		&& echo "  Ollama already running" \
		|| (echo "  starting ollama serve..."; ollama serve > /dev/null 2>&1 & until curl -sf localhost:11434/api/tags > /dev/null 2>&1; do sleep 1; done)
	@ollama list | grep -q "$(OLLAMA_TEXT_MODEL)" \
		&& echo "  text model $(OLLAMA_TEXT_MODEL) already present" \
		|| (echo "  pulling text model $(OLLAMA_TEXT_MODEL) (~4.7GB)..."; ollama pull $(OLLAMA_TEXT_MODEL))
	@ollama list | grep -q "$(OLLAMA_VISION_MODEL)" \
		&& echo "  vision model $(OLLAMA_VISION_MODEL) already present" \
		|| (echo "  pulling vision model $(OLLAMA_VISION_MODEL) (~4.7GB)..."; ollama pull $(OLLAMA_VISION_MODEL))
	@echo "  Pointing kado config at local Ollama (localhost:11434)..."
	@uv run python -c "from kado.config import KadoConfig; c = KadoConfig.load(); c.ollama_url = 'http://localhost:11434'; c.save(); print('  ollama_url = ' + c.ollama_url)"
	@echo "→ [5/5] Starting local VOICEVOX (docker, :50021)..."
	@docker info > /dev/null 2>&1 || (echo "  ✗ Docker is not running — start Docker and re-run 'make setup-local'"; exit 1)
	@docker ps --filter "publish=50021" --filter "status=running" --quiet | grep -q . \
		&& echo "  VOICEVOX already running" \
		|| docker run -d --rm -p 50021:50021 voicevox/voicevox_engine:cpu-latest > /dev/null
	@echo "  Pointing kado config at local VOICEVOX (only if not already set)..."
	@uv run python -c "from kado.config import KadoConfig; c = KadoConfig.load(); c.voicevox_url = c.voicevox_url or 'http://localhost:50021'; c.save(); print('  voicevox_url = ' + c.voicevox_url)"
	@echo "✓ Local setup complete."
	@echo "  Next: run 'uv run kado config' to pick your Anki deck."
	@echo "  Verify services with 'uv run kado status'."

start:
	@echo "→ Opening SSH tunnel to cluster Ollama (localhost:11435 → omarchyd:11434)..."
	@pgrep -f "ssh -N -L 11435" > /dev/null \
		&& echo "  Ollama tunnel already active" \
		|| (ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 11435:localhost:11434 kristian@omarchyd &)
	@echo "  Pointing kado config at the tunnel (localhost:11435)..."
	@uv run python -c "from kado.config import KadoConfig; c = KadoConfig.load(); c.ollama_url = 'http://localhost:11435'; c.save(); print('  ollama_url = ' + c.ollama_url)"
	@echo "→ Starting VOICEVOX on cluster (omarchyd:50021)..."
	@ssh kristian@omarchyd "docker ps --filter 'publish=50021' --filter 'status=running' --quiet | grep -q . \
		&& echo '  VOICEVOX already running' \
		|| docker run -d --rm -p 50021:50021 voicevox/voicevox_engine:cpu-latest"
	@echo "→ Opening SSH tunnel to cluster VOICEVOX (localhost:50021 → omarchyd:50021)..."
	@pgrep -f "ssh -N -L 50021" > /dev/null \
		&& echo "  VOICEVOX tunnel already active" \
		|| (ssh -N -L 50021:localhost:50021 kristian@omarchyd &)
	@echo "✓ All services started. Run 'uv run kado status' to verify."

stop:
	@echo "→ Killing SSH tunnels..."
	@pkill -f "ssh -N -L 11435" 2>/dev/null && echo "  Ollama tunnel stopped" || echo "  Ollama tunnel not running"
	@pkill -f "ssh -N -L 50021" 2>/dev/null && echo "  VOICEVOX tunnel stopped" || echo "  VOICEVOX tunnel not running"
	@echo "→ Stopping VOICEVOX on cluster..."
	@ssh kristian@omarchyd "docker ps --filter 'publish=50021' --filter 'status=running' --quiet | grep -q . \
		&& docker stop \$$(docker ps --filter 'publish=50021' --quiet) \
		|| echo '  VOICEVOX not running'"
	@echo "✓ All services stopped."

local-start:
	@echo "→ Ensuring container runtime ($(CONTAINER)) is running..."
ifeq ($(UNAME_S),Darwin)
	@$(CONTAINER) info > /dev/null 2>&1 || open -a "$(RUNTIME_APP)"
	@echo "→ Waiting for $(CONTAINER) to be ready..."
	@until $(CONTAINER) info > /dev/null 2>&1; do sleep 2; done
else
	@$(CONTAINER) info > /dev/null 2>&1 \
		|| (echo "  ✗ $(CONTAINER) daemon not running — start your runtime and re-run"; exit 1)
endif
	@echo "→ Starting VOICEVOX locally..."
	@$(CONTAINER) ps --filter "publish=50021" --filter "status=running" --quiet | grep -q . \
		&& echo "  VOICEVOX already running" \
		|| $(CONTAINER) run -d --rm -p 50021:50021 voicevox/voicevox_engine:cpu-latest > /dev/null
	@echo "✓ Local services started."

local-stop:
	@echo "→ Stopping local VOICEVOX..."
	@$(CONTAINER) ps --filter "publish=50021" --filter "status=running" --quiet | grep -q . \
		&& $(CONTAINER) stop $$($(CONTAINER) ps --filter "publish=50021" --quiet) && echo "  VOICEVOX stopped" \
		|| echo "  VOICEVOX not running"
	@echo "✓ Local services stopped."
