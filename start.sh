#!/usr/bin/env bash
set -e

echo "🚀 Starting Q Chat Context Engine..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

export ANONYMIZED_TELEMETRY=False
export CHROMA_TELEMETRY=False

# Start Ollama in background
ollama serve &

# Wait until Ollama API is ready
until curl -s http://localhost:11434/api/tags >/dev/null 2>&1; do
  echo "⌛ Waiting for Ollama..."
  sleep 1
done
echo "✅ Ollama is running"

# Ensure model exists
if ! ollama list | grep -q "embeddinggemma"; then
  echo "📦 Pulling embeddinggemma model..."
  ollama pull embeddinggemma
fi
echo "✅ embeddinggemma ready"

# Ensure template exists
test -f templates/index.html || { echo "❌ templates/index.html not found"; exit 1; }

# Install requirements (safe for cloud build)


echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Starting FastAPI server..."
echo "🌐 Listening on port ${PORT:-8080}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start FastAPI using uvicorn
exec uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8080}
