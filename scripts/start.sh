#!/bin/bash
# Start ComfyUI and API wrapper
# Designed for fleet orchestration - clean startup/shutdown
#
# Watchdog: monitors ComfyUI health every 30s.
# If ComfyUI dies (OOM, crash, etc.), it auto-relaunches without
# touching the wrapper or the container.

set -e

echo "================================================"
echo "  ComfyUI Pipeline Service"
echo "================================================"
echo "  Models path: ${COMFYUI_MODELS_PATH:-/models}"
echo "  Workflows:   ${WORKFLOWS_DIR:-/app/workflows}"
echo "================================================"

# Optional: Check models before starting
if [ "${CHECK_MODELS:-false}" = "true" ]; then
    echo "Checking required models..."
    python /app/check_models.py --models-path "${COMFYUI_MODELS_PATH:-/models}" || true
    echo ""
fi

# ── ComfyUI launcher (reusable) ──────────────────────────────────────────

COMFYUI_PID=""

start_comfyui() {
    echo "[watchdog] Starting ComfyUI on port ${COMFYUI_PORT:-8188}..."
    cd /app/ComfyUI
    python main.py \
        --listen "${COMFYUI_LISTEN:-0.0.0.0}" \
        --port "${COMFYUI_PORT:-8188}" \
        --preview-method auto \
        &
    COMFYUI_PID=$!
    echo "[watchdog] ComfyUI PID: $COMFYUI_PID"
}

wait_for_comfyui() {
    local max_wait=${1:-180}
    local waited=0
    echo "[watchdog] Waiting for ComfyUI to initialize..."
    while [ $waited -lt $max_wait ]; do
        if curl -s http://localhost:${COMFYUI_PORT:-8188}/system_stats > /dev/null 2>&1; then
            echo "[watchdog] ✓ ComfyUI ready! (${waited}s)"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        if [ $((waited % 10)) -eq 0 ]; then
            echo "[watchdog]   Still initializing... (${waited}s)"
        fi
    done
    echo "[watchdog] ⚠ ComfyUI not ready after ${max_wait}s"
    return 1
}

# ── Initial startup ──────────────────────────────────────────────────────

start_comfyui
wait_for_comfyui 180

# Start API wrapper
echo "Starting API wrapper on port ${WRAPPER_PORT:-8189}..."
cd /app
python api_wrapper.py &
WRAPPER_PID=$!

# Wait a moment for wrapper to start
sleep 2

echo ""
echo "================================================"
echo "  Services Running"
echo "================================================"
echo "  ComfyUI UI:  http://localhost:${COMFYUI_PORT:-8188}"
echo "  Wrapper API: http://localhost:${WRAPPER_PORT:-8189}"
echo "  Health:      http://localhost:${WRAPPER_PORT:-8189}/health"
echo "  Workflows:   http://localhost:${WRAPPER_PORT:-8189}/workflows"
echo "  Watchdog:    enabled (30s interval)"
echo "================================================"
echo ""

# ── Shutdown handler ─────────────────────────────────────────────────────

SHUTDOWN_REQUESTED=false

shutdown() {
    echo ""
    echo "Shutting down services..."
    SHUTDOWN_REQUESTED=true
    kill $WRAPPER_PID 2>/dev/null || true
    # Find and kill any ComfyUI process (wrapper may have restarted it)
    pkill -f "python.*main.py.*--listen" 2>/dev/null || true
    wait $WRAPPER_PID 2>/dev/null || true
    echo "Shutdown complete."
    exit 0
}

trap shutdown SIGTERM SIGINT

# ── Watchdog loop ────────────────────────────────────────────────────────
# Monitors both processes. If ComfyUI dies, relaunch it.
# If the wrapper dies, shut everything down.
#
# This handles OOM kills, segfaults, and any other unexpected ComfyUI death.
# The wrapper's /vram/purge endpoint also kills and relaunches ComfyUI;
# the watchdog detects the new PID on the next cycle and tracks it.

WATCHDOG_INTERVAL=${WATCHDOG_INTERVAL:-30}
CONSECUTIVE_FAILURES=0
MAX_RESTART_FAILURES=5

watchdog() {
    while true; do
        sleep $WATCHDOG_INTERVAL

        # Exit if shutdown was requested
        if [ "$SHUTDOWN_REQUESTED" = true ]; then
            return
        fi

        # Check wrapper first — if it's dead, we're done
        if ! kill -0 $WRAPPER_PID 2>/dev/null; then
            echo "[watchdog] Wrapper (PID $WRAPPER_PID) died — shutting down"
            shutdown
            return
        fi

        # Check ComfyUI by PID and HTTP
        comfyui_alive=false

        # First check: is the process alive?
        if [ -n "$COMFYUI_PID" ] && kill -0 $COMFYUI_PID 2>/dev/null; then
            # Process exists — verify HTTP responds
            if curl -s --max-time 5 http://localhost:${COMFYUI_PORT:-8188}/system_stats > /dev/null 2>&1; then
                comfyui_alive=true
            fi
        fi

        # Also check if wrapper restarted ComfyUI (different PID)
        if [ "$comfyui_alive" = false ]; then
            # Look for any ComfyUI process
            new_pid=$(pgrep -f "python.*main.py.*--listen.*--port.*${COMFYUI_PORT:-8188}" 2>/dev/null | head -1)
            if [ -n "$new_pid" ] && [ "$new_pid" != "$COMFYUI_PID" ]; then
                echo "[watchdog] ComfyUI restarted by wrapper (old PID: $COMFYUI_PID, new PID: $new_pid)"
                COMFYUI_PID=$new_pid
                comfyui_alive=true
                CONSECUTIVE_FAILURES=0
            fi
        fi

        if [ "$comfyui_alive" = true ]; then
            CONSECUTIVE_FAILURES=0
            continue
        fi

        # ComfyUI is dead — relaunch
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        echo "[watchdog] ✗ ComfyUI is down (attempt $CONSECUTIVE_FAILURES/$MAX_RESTART_FAILURES)"

        if [ $CONSECUTIVE_FAILURES -ge $MAX_RESTART_FAILURES ]; then
            echo "[watchdog] ✗ ComfyUI failed $MAX_RESTART_FAILURES consecutive restarts — giving up"
            echo "[watchdog] Container will need manual restart"
            # Don't shut down the wrapper — it's still serving health checks
            # and can report the problem. Just stop trying to restart.
            break
        fi

        # Clean up any zombie process
        kill $COMFYUI_PID 2>/dev/null || true
        pkill -f "python.*main.py.*--listen" 2>/dev/null || true
        sleep 2

        # Relaunch
        echo "[watchdog] Relaunching ComfyUI..."
        start_comfyui
        wait_for_comfyui 120

        if curl -s --max-time 5 http://localhost:${COMFYUI_PORT:-8188}/system_stats > /dev/null 2>&1; then
            echo "[watchdog] ✓ ComfyUI recovered successfully"
            CONSECUTIVE_FAILURES=0
        else
            echo "[watchdog] ⚠ ComfyUI started but not responding yet"
        fi
    done
}

# Run watchdog in background
watchdog &
WATCHDOG_PID=$!

# Monitor the WRAPPER process.
# If the wrapper dies, the watchdog loop will catch it and call shutdown.
wait $WRAPPER_PID
EXIT_CODE=$?

echo "Wrapper exited with code $EXIT_CODE — shutting down"
SHUTDOWN_REQUESTED=true
kill $WATCHDOG_PID 2>/dev/null || true
shutdown