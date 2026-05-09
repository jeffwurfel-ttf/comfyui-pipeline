#!/usr/bin/env bash
# build.sh — Build comfyui-pipeline with git-sha tag.
#
# Tags both :latest and :<sha> so we can roll back. Prunes old SHA tags
# to fit on the tight /home filesystem (154GB total, images are ~91GB
# each so we can only keep the current build + one rollback target).
#
# Usage:
#   ./tools/build.sh
#
# Special tags that are never auto-pruned:
#   - latest        (current production)
#   - pre-hardening (the original safety tag from the hardening pass)
#   - pre-rollback  (created by rollback.sh so rollbacks are reversible)

set -euo pipefail

cd "$(dirname "$0")/.."

# Determine tag from git, with dirty-tree handling
if git rev-parse --short HEAD >/dev/null 2>&1; then
    SHA=$(git rev-parse --short HEAD)
    if ! git diff-index --quiet HEAD --; then
        SHA="${SHA}-dirty"
        echo "WARN: working tree has uncommitted changes; tagging as $SHA"
    fi
else
    SHA="ts-$(date +%Y%m%d-%H%M%S)"
    echo "WARN: not a git repo; using timestamp tag: $SHA"
fi

echo "================================================================"
echo "  Pre-build disk state (/home)"
echo "================================================================"
df -h /home
echo ""
echo "================================================================"
echo "  Existing comfyui-pipeline images"
echo "================================================================"
docker images comfyui-pipeline --format 'table {{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'
echo ""

# Sanity-check: refuse to build if there's clearly not enough room
FREE_GB=$(df --output=avail -BG /home | tail -1 | tr -d ' G')
if [ "$FREE_GB" -lt 50 ]; then
    echo "ERROR: only ${FREE_GB}GB free on /home — need at least 50GB for safe build."
    echo "Try: docker builder prune -af"
    exit 1
fi

echo "================================================================"
echo "  Building comfyui-pipeline:latest and comfyui-pipeline:$SHA"
echo "================================================================"
docker build -t comfyui-pipeline:latest -t "comfyui-pipeline:$SHA" .

echo ""
echo "================================================================"
echo "  Pruning old SHA tags"
echo "================================================================"
# Disk-tight retention policy on this 154GB box:
#   Keep: latest, pre-hardening, pre-rollback (if exists), most recent SHA
#   Drop: any other SHA tags
PROTECTED='^(latest|pre-hardening|pre-rollback)$'

# Get all SHA tags (anything not protected) sorted by image creation, newest first
SHA_TAGS=$(docker images comfyui-pipeline --format '{{.Tag}} {{.ID}}' \
    | awk '{print $1}' \
    | grep -vE "$PROTECTED" \
    | head -1)

# Drop everything except the most recent SHA we just built
docker images comfyui-pipeline --format '{{.Tag}}' \
    | grep -vE "$PROTECTED" \
    | grep -v "^${SHA}$" \
    | while read old_tag; do
        echo "  Removing: comfyui-pipeline:$old_tag"
        docker rmi "comfyui-pipeline:$old_tag" 2>/dev/null || true
    done

echo ""
echo "================================================================"
echo "  Final state"
echo "================================================================"
docker images comfyui-pipeline --format 'table {{.Tag}}\t{{.Size}}\t{{.CreatedSince}}'
echo ""
df -h /home
echo ""
echo "Build complete. To deploy:"
echo "  docker compose -f docker-compose.rocky.yaml up -d"
echo ""
echo "To roll back:"
echo "  ./tools/rollback.sh <tag>"
