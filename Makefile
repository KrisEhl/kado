#On the cluster — install & start Ollama:
# SSH in first
#ssh kristian@omarchyd

# Install Ollama
#curl -fsSL https://ollama.com/install.sh | sh

# Start it (listens on localhost:11434 by default)
#ollama serve &

# Pull a model — qwen2.5:7b is good for Japanese, fits easily in 24GB VRAM
#ollama pull qwen2.5:7b

#On your Mac — open the tunnel (run once per session):
connect-cluster:
	ssh -N -L 11435:localhost:11434 kristian@omarchyd
#This forwards localhost:11435 on your Mac → localhost:11434 on the cluster.

#Configure kado to use it:
cluster-configure:
	uv run kado config --set ollama-url http://localhost:11435

audio-server:
	docker run -d --rm -p 50021:50021 voicevox/voicevox_engine:cpu-latest

# Start all required services:
#   1. Docker Desktop (if not already running)
#   2. VOICEVOX audio engine
#   3. SSH tunnel to Ollama on the cluster (runs in background)
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

# To make the tunnel persistent/auto-start, add this to ~/.ssh/config:
# Host omarchyd
# 		HostName omarchyd
# 		User kristian
# 		LocalForward 11435 localhost:11434
# 		ServerAliveInterval 60
# Then just ssh omarchyd and the tunnel opens automatically when you connect.
#
