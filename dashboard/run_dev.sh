#!/usr/bin/env bash
# Run the MARTA dashboard backend directly with the marta conda env.
# This gives you ALL Python deps (torch, rich, arch, stable-baselines3, etc.)
# and Ollama is reachable at localhost:11434 — so live debate just works.
#
# Usage:
#   cd dashboard && ./run_dev.sh          # backend on :8001
#   cd dashboard && ./run_dev.sh --port 8002
#
# The Next.js frontend is served separately — either keep the Docker container
# running or start it with:
#   cd frontend && npm run dev            # http://localhost:3000

set -e

MARTA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND_DIR="$MARTA_ROOT/dashboard/backend"
PORT="${2:-8001}"

echo "▶  MARTA Dashboard — dev backend"
echo "   MARTA_ROOT : $MARTA_ROOT"
echo "   Backend    : $BACKEND_DIR"
echo "   Port       : $PORT"
echo "   Env        : marta (conda)"
echo ""

cd "$BACKEND_DIR"

PYTHONPATH="$MARTA_ROOT:$BACKEND_DIR" \
MARTA_ROOT="$MARTA_ROOT" \
OLLAMA_HOST="http://localhost:11434" \
conda run -n marta --no-capture-output \
  uvicorn main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --reload \
    --reload-dir "$BACKEND_DIR"
