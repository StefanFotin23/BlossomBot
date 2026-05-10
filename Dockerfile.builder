# Nume fișier: Dockerfile.builder
FROM python:3.10-slim

# Instalăm dependențele de sistem o singură dată
RUN apt-get update && apt-get install -y \
    zstd git git-lfs build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

# Instalăm Ollama CLI
RUN curl -fsSL https://ollama.com/install.sh | sh

# Instalăm dependențele Python grele pentru llama.cpp
RUN pip install --no-cache-dir torch numpy sentencepiece gguf