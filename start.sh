#!/bin/bash

##########################################################################
#  MMAudio Full Background Startup + Watchdog (RunPod Safe)
##########################################################################

set -e

CONDA_BASE="/workspace/miniconda"
CONDA_SH="$CONDA_BASE/etc/profile.d/conda.sh"
ENV_NAME="mmaudio"
API_DIR="$(dirname "$0")"

WORKER="$API_DIR/worker.py"
SERVER="$API_DIR/server.py"

WORKER_LOG="$API_DIR/worker.log"
SERVER_LOG="$API_DIR/server.log"
WATCHDOG_LOG="$API_DIR/watchdog.log"

echo ""
echo "-------------------------------------------------" | tee -a "$WATCHDOG_LOG"
echo "[BOOT] Starting MMAudio Background System at $(date)" | tee -a "$WATCHDOG_LOG"
echo "-------------------------------------------------" | tee -a "$WATCHDOG_LOG"

##########################################################################
# LOAD CONDA ENVIRONMENT
##########################################################################
echo "[INFO] Loading conda..." | tee -a "$WATCHDOG_LOG"
source "$CONDA_SH"

echo "[INFO] Activating environment: $ENV_NAME" | tee -a "$WATCHDOG_LOG"
conda activate "$ENV_NAME"

##########################################################################
# KILL OLD PROCESSES
##########################################################################
echo "[INFO] Stopping old worker/server processes..." | tee -a "$WATCHDOG_LOG"
pkill -f "worker.py" || true
pkill -f "server.py" || true

sleep 1

##########################################################################
# START WORKER IN BACKGROUND (nohup)
##########################################################################
start_worker() {
    echo "[START] Launching worker..." | tee -a "$WATCHDOG_LOG"
    nohup python3 "$WORKER" >> "$WORKER_LOG" 2>&1 &
    sleep 2
}

##########################################################################
# START SERVER IN BACKGROUND (nohup)
##########################################################################
start_server() {
    echo "[START] Launching server..." | tee -a "$WATCHDOG_LOG"
    nohup python3 "$SERVER" >> "$SERVER_LOG" 2>&1 &
    sleep 2
}

##########################################################################
# INITIAL STARTUP
##########################################################################
start_worker
start_server

##########################################################################
# WATCHDOG — RUNS FOREGROUND (required by RunPod)
##########################################################################
echo "[INFO] Watchdog is running and supervising processes..." | tee -a "$WATCHDOG_LOG"
echo "[INFO] Close SSH safely — system stays alive." | tee -a "$WATCHDOG_LOG"
echo "-------------------------------------------------" | tee -a "$WATCHDOG_LOG"

while true; do
    # Check worker
    if ! pgrep -f "worker.py" > /dev/null; then
        echo "[WARN] Worker died — restarting..." | tee -a "$WATCHDOG_LOG"
        start_worker
    fi

    # Check server
    if ! pgrep -f "server.py" > /dev/null; then
        echo "[WARN] Server died — restarting..." | tee -a "$WATCHDOG_LOG"
        start_server
    fi

    sleep 3
done
