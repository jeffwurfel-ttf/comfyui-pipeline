#!/bin/bash
# Start ComfyUI and API wrapper
# Designed for fleet orchestration - clean startup/shutdown

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

# Start ComfyUI in background
echo "Starting ComfyUI on port ${COMFYUI_PORT:-8188}..."
cd /app/ComfyUI
python main.py \
    --listen "${COMFYUI_LISTEN:-0.0.0.0}" \
    --port "${COMFYUI_PORT:-8188}" \
    --preview-method auto \
    &
COMFYUI_PID=$!

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to initialize..."
MAX_WAIT=180
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:${COMFYUI_PORT:-8188}/system_stats > /dev/null 2>&1; then
        echo "✓ ComfyUI ready! (${WAITED}s)"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    if [ $((WAITED % 10)) -eq 0 ]; then
        echo "  Still initializing... (${WAITED}s)"
    fi
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "⚠ Warning: ComfyUI may not be fully ready after ${MAX_WAIT}s"
fi

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
echo "================================================"
echo ""

# Handle shutdown gracefully
shutdown() {
    echo ""
    echo "Shutting down services..."
    kill $WRAPPER_PID 2>/dev/null || true
    kill $COMFYUI_PID 2>/dev/null || true
    wait $WRAPPER_PID 2>/dev/null || true
    wait $COMFYUI_PID 2>/dev/null || true
    echo "Shutdown complete."
    exit 0
}

trap shutdown SIGTERM SIGINT

# Wait for either process to exit
wait -n $COMFYUI_PID $WRAPPER_PID
EXIT_CODE=$?

echo "A service exited with code $EXIT_CODE"
shutdown