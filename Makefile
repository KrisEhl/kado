services:
	@echo "→ Starting Docker Desktop..."
	@open -a Docker
	@echo "→ Waiting for Docker to be ready..."
	@until docker info > /dev/null 2>&1; do sleep 1; done
	@echo "→ Starting VOICEVOX audio server..."
	@docker ps --filter "publish=50021" --filter "status=running" --quiet | grep -q . \
		&& echo "  VOICEVOX already running" \
		|| docker run -d --rm -p 50021:50021 voicevox/voicevox_engine:cpu-latest
	@echo "→ Opening SSH tunnel to cluster (localhost:11435 → omarchyd:11434)..."
	@pgrep -f "ssh -N -L 11435" > /dev/null \
		&& echo "  Tunnel already active" \
		|| (ssh -N -L 11435:localhost:11434 kristian@omarchyd &)
	@echo "✓ All services started. Run 'uv run kado status' to verify."

