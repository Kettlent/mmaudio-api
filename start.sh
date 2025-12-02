#!/bin/bash
set -e

### ----------------------------------------
### PATHS
### ----------------------------------------
CONDA_BASE="/workspace/miniconda"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="mmaudio"
API_DIR="$(dirname "$0")"

WORKER_LOG="$API_DIR/worker.log"
SERVER_LOG="$API_DIR/server.log"

### ----------------------------------------
### LOAD CONDA + ACTIVATE ENV
### ----------------------------------------
echo "[START] Loading conda..."
source "$CONDA_SH"

echo "[START] Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

### ----------------------------------------
### STOP OLD PROCESSES
### ----------------------------------------
echo "[START] Killing old worker/server processes..."
pkill -f "worker.py" || true
pkill -f "uvicorn server:app" || true

sleep 1

### ----------------------------------------
### START WORKER (BACKGROUND)
### ----------------------------------------
echo "[START] Launching worker.py in background..."
nohup python3 "$API_DIR/worker.py" > "$WORKER_LOG" 2>&1 &

sleep 2

WORKER_PID=$(pgrep -f "worker.py")
echo "[OK] Worker started with PID: $WORKER_PID"
echo "Worker log: $WORKER_LOG"

### ----------------------------------------
### START SERVER (FOREGROUND, BLOCKING)
### ----------------------------------------
echo "[START] Launching FastAPI server on port 8080..."
echo "Server log: $SERVER_LOG"
echo "---------------------------------------------"

# Run uvicorn in foreground without using pipes
uvicorn server:app --host 0.0.0.0 --port 8080 | tee -a "$SERVER_LOG"
