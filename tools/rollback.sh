#!/usr/bin/env bash
# rollback.sh <tag> — point :latest at <tag> and restart the container.
#
# Run with no args to list available rollback targets.
# The current :latest is preserved as :pre-rollback before the swap, so
# the rollback itself is reversible: ./rollback.sh pre-rollback undoes it.
#
# The compose service name is "comfyui" (container name is "comfyui-pipeline").

set -euo pipefail

cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
    echo "Available tags to roll back to:"
    docker images comfyui-pipeline --format 'table {{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' \
        | grep -v '^TAG\|^latest'
    echo ""
    echo "Usage: $0 <tag>"
    exit 0
fi

TAG="$1"

if ! docker image inspect "comfyui-pipeline:$TAG" >/dev/null 2>&1; then
    echo "ERROR: comfyui-pipeline:$TAG not found locally"
    echo ""
    echo "Available tags:"
    docker images comfyui-pipeline --format 'table {{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'
    exit 1
fi

echo "================================================================"
echo "  Rollback comfyui-pipeline"
echo "================================================================"
echo ""
echo "Current :latest:"
docker images comfyui-pipeline:latest --format '  ID={{.ID}}  Created={{.CreatedSince}}  Size={{.Size}}'
echo ""
echo "Target  :$TAG:"
docker images "comfyui-pipeline:$TAG" --format '  ID={{.ID}}  Created={{.CreatedSince}}  Size={{.Size}}'
echo ""

# Same-image guard
CUR_ID=$(docker inspect --format='{{.Id}}' comfyui-pipeline:latest)
TGT_ID=$(docker inspect --format='{{.Id}}' "comfyui-pipeline:$TAG")
if [ "$CUR_ID" = "$TGT_ID" ]; then
    echo "Note: :latest and :$TAG already point at the same image. Nothing to roll back."
    exit 0
fi

read -p "Roll back to :$TAG? [y/N] " -n 1 -r
echo
[[ ! $REPLY =~ ^[Yy]$ ]] && { echo "Aborted."; exit 1; }

# Save current :latest as :pre-rollback so this rollback is itself reversible.
# If a previous :pre-rollback exists, it gets overwritten — that's intentional;
# we only support undoing the most recent rollback.
docker tag comfyui-pipeline:latest comfyui-pipeline:pre-rollback
echo "Tagged current :latest as :pre-rollback (this rollback is reversible)"

# Repoint :latest at the target tag
docker tag "comfyui-pipeline:$TAG" comfyui-pipeline:latest
echo "Repointed :latest -> :$TAG"

# Recreate the container with the new image. Service name is "comfyui".
docker compose -f docker-compose.rocky.yaml up -d --no-build comfyui

echo ""
echo "Container restarted. Waiting 60s for ComfyUI initialization..."
sleep 60

# Verify wrapper responds
if curl -sfS --max-time 10 http://localhost:8189/health >/dev/null 2>&1; then
    echo "✓ Wrapper responding on :8189"
else
    echo "✗ Wrapper not responding — check 'docker logs comfyui-pipeline'"
    exit 1
fi

# Verify ComfyUI responds (may still be initializing if just starting)
if curl -sfS --max-time 10 http://localhost:8188/system_stats >/dev/null 2>&1; then
    echo "✓ ComfyUI responding on :8188"
else
    echo "⚠ ComfyUI not yet responding (may still be initializing — check in 60s)"
fi

echo ""
echo "Health snapshot:"
curl -s http://localhost:8189/health | python3 -m json.tool 2>/dev/null || echo "(no JSON yet)"

echo ""
echo "To undo this rollback:"
echo "  $0 pre-rollback"
