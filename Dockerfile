# Dockerfile for Q Chat Context Engine (FastAPI + Ollama + ChromaDB)

FROM ollama/ollama:latest

# System deps for Python + curl
RUN apt-get update && apt-get install -y python3 python3-pip curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Python deps
RUN pip3 install --no-cache-dir -r requirements.txt

# Persist Chroma data (Render/Railway will mount a disk here)
ENV PERSIST_DIRECTORY=/app/storage
RUN mkdir -p /app/storage

# Make start script executable
RUN chmod +x /app/start.sh

# Expose the app port (Render/Railway will set PORT env)
EXPOSE 8080

# Start Ollama + FastAPI via your script
CMD ["/app/start.sh"]
