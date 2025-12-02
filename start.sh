#!/bin/bash

### -------------------------
### MMAudio Auto-Start Script
### -------------------------

API_DIR="$(dirname "$0")"
CONDA_BASE="/workspace/miniconda"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="mmaudio"

WORKER_LOG="$API_DIR/worker.log"
SERVER_LOG="$API_DIR/server.log"

echo ""
echo "-----------------------------------"
echo "Starting MMAudio at $(date)"
echo "-----------------------------------"

### Load conda
if [ -f "$CONDA_SH" ]; then
    echo "[OK] Loading conda..."
    source "$CONDA_SH"
else
    echo "[ERROR] conda.sh not found!"
    exit 1
fi

### Activate environment
echo "[OK] Activating conda env: $ENV_NAME"
conda activate "$ENV_NAME"

### Kill old processes
echo "[INFO] Killing old worker/server processes..."
pkill -f "worker.py" || true
pkill -f "server.py" || true
sleep 1

### -------------------------
### Start worker in background
### -------------------------
echo "[START] Launching worker.py (background)..."

nohup python3 "$API_DIR/worker.py" \
    >> "$WORKER_LOG" 2>&1 &

sleep 2
WORKER_PID=$(pgrep -f "worker.py")
echo "[OK] Worker running PID: $WORKER_PID"
echo "Worker log: $WORKER_LOG"

### -------------------------
### Start server in background
### -------------------------
echo "[START] Launching FastAPI server (background)..."

nohup uvicorn server:app \
    --host 0.0.0.0 \
    --port 8080 \
    >> "$SERVER_LOG" 2>&1 &

sleep 2
SERVER_PID=$(pgrep -f "uvicorn server:app")
echo "[OK] Server running PID: $SERVER_PID"
echo "Server log: $SERVER_LOG"

echo "-----------------------------------"
echo "MMAudio Startup Complete (Background Mode)"
echo "-----------------------------------"
