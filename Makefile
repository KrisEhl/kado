start:
	@echo "→ Opening SSH tunnel to cluster Ollama (localhost:11435 → omarchyd:11434)..."
	@pgrep -f "ssh -N -L 11435" > /dev/null \
		&& echo "  Ollama tunnel already active" \
		|| (ssh -N -L 11435:localhost:11434 kristian@omarchyd &)
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
	@echo "→ Starting Docker Desktop..."
	@open -a Docker
	@echo "→ Waiting for Docker to be ready..."
	@until docker info > /dev/null 2>&1; do sleep 1; done
	@echo "→ Starting VOICEVOX locally..."
	@docker ps --filter "publish=50021" --filter "status=running" --quiet | grep -q . \
		&& echo "  VOICEVOX already running" \
		|| docker run -d --rm -p 50021:50021 voicevox/voicevox_engine:cpu-latest
	@echo "✓ Local services started."

local-stop:
	@echo "→ Stopping local VOICEVOX..."
	@docker ps --filter "publish=50021" --filter "status=running" --quiet | grep -q . \
		&& docker stop $$(docker ps --filter "publish=50021" --quiet) && echo "  VOICEVOX stopped" \
		|| echo "  VOICEVOX not running"
	@echo "→ Quitting Docker Desktop..."
	@osascript -e 'quit app "Docker Desktop"' 2>/dev/null && echo "  Docker Desktop quit" || echo "  Docker Desktop not running"
	@echo "✓ Local services stopped."
