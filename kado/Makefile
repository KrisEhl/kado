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

#Or run kado status to verify after the tunnel is up — it will show the installed models on the cluster.

# To make the tunnel persistent/auto-start, add this to ~/.ssh/config:
# Host omarchyd
# 		HostName omarchyd
# 		User kristian
# 		LocalForward 11435 localhost:11434
# 		ServerAliveInterval 60
# Then just ssh omarchyd and the tunnel opens automatically when you connect.
#
