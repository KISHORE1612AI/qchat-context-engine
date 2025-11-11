# Dockerfile for Q Chat Context Engine (FastAPI + Ollama + ChromaDB)
FROM ollama/ollama:latest

SHELL ["/bin/bash", "-lc"]

# Install system build deps (minimal)
RUN apt-get update && apt-get install -y \
    python3-venv python3-dev build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create and use a venv to avoid PEP 668 "externally managed environment" errors
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install into venv
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code after dependencies (better caching)
COPY . /app

# Ensure start script has LF endings and is executable
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Persistence directory (Render / Railway will mount a disk here)
ENV PERSIST_DIRECTORY=/app/storage
RUN mkdir -p /app/storage

EXPOSE 8080

CMD ["/app/start.sh"]
