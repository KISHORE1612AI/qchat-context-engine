# Dockerfile for Q Chat Context Engine (FastAPI + Ollama + ChromaDB)
FROM ollama/ollama:latest

# Use bash for better script behavior
SHELL ["/bin/bash", "-lc"]

# System deps for Python, builds, and networking tools
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure 'python' and 'pip' aliases exist
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip/setuptools/wheel for smoother installs
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the source
COPY . /app

# Make start script executable (handles Windows CRLF safely)
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Persist Chroma data (Render/Railway will mount a disk here)
ENV PERSIST_DIRECTORY=/app/storage
RUN mkdir -p /app/storage

# Expose the app port (Render/Railway will set PORT env)
EXPOSE 8080

# Start Ollama + FastAPI via your script
CMD ["/app/start.sh"]
