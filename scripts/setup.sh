#!/bin/bash
set -e

echo "========================================"
echo "  LLM Evaluation Platform Setup"
echo "========================================"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "[1/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/6] Pulling default Ollama model..."
ollama pull llama3.2
echo "Local Ollama models:"
ollama list

echo "[4/6] Checking GPU availability..."
if nvidia-smi &>/dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
    echo "GPU is ready for LLM acceleration."
else
    echo "No NVIDIA GPU detected. Running on CPU."
fi

echo "[5/6] Initializing PostgreSQL database..."
sudo -u postgres psql -c "CREATE USER llmeval WITH PASSWORD 'llmeval';" 2>/dev/null || true
sudo -u postgres psql -c "CREATE DATABASE llmevaldb OWNER llmeval;" 2>/dev/null || true
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE llmevaldb TO llmeval;" 2>/dev/null || true

echo "[6/6] Setup complete!"
echo ""
echo "To start the platform:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "Or with Docker Compose:"
echo "  docker compose up -d"
echo ""
echo "API Docs: http://localhost:8000/docs"
echo "MLflow UI: http://localhost:5001"
echo "Grafana: http://localhost:3000 (admin/admin)"
