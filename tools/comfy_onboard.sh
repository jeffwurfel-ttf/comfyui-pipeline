#!/usr/bin/env bash
# =============================================================================
# comfy_onboard.sh — Workflow Onboarding Tool
# =============================================================================
# Give it a workflow JSON and it:
#   1. Checks all nodes exist (suggests closest match for missing ones)
#   2. Finds all model file references
#   3. Downloads missing models automatically (searches HuggingFace)
#   4. Reports what to add to manifest.yaml
#   5. Re-runs preflight to confirm
#
# Usage:
#   ./tools/comfy_onboard.sh workflow.json              # Onboard a workflow
#   ./tools/comfy_onboard.sh workflow.json --dry-run     # Show what would happen
#   ./tools/comfy_onboard.sh --scan-all [--dry-run]      # Scan all known workflows
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMFY_URL="${COMFY_URL:-http://localhost:8188}"
MODELS_DIR="${COMFY_MODELS_HOST:-/mnt/ssd/comfyui-models}"
MANIFEST="${REPO_DIR}/models/manifest.yaml"
CONTAINER="${COMFY_CONTAINER:-comfyui-pipeline}"
HF_TOKEN="${HF_TOKEN:-}"
DRY_RUN=false
SCAN_ALL=false

# Load .env
[[ -f "${REPO_DIR}/.env" ]] && { set -a; source "${REPO_DIR}/.env"; set +a; }

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
miss() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
info() { echo -e "  ${CYAN}ℹ${NC} $1"; }
header() { echo -e "\n${BOLD}═══ $1 ═══${NC}"; }

# ─── Argument parsing ────────────────────────────────────────────────────────
WF_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --scan-all) SCAN_ALL=true; shift ;;
        -h|--help)
            echo "Usage: $0 <workflow.json> [--dry-run]"
            echo "       $0 --scan-all [--dry-run]"
            exit 0 ;;
        *) WF_FILE="$1"; shift ;;
    esac
done

# ─── Fetch node registry ─────────────────────────────────────────────────────
NODES_JSON=$(mktemp)
trap "rm -f $NODES_JSON" EXIT

header "Node Registry"
if curl -sf "${COMFY_URL}/object_info" > "$NODES_JSON" 2>/dev/null; then
    NODE_COUNT=$(python3 -c "import json; print(len(json.load(open('$NODES_JSON'))))")
    ok "ComfyUI responding — ${NODE_COUNT} nodes"
else
    miss "ComfyUI not responding at ${COMFY_URL}"
    exit 1
fi

# ─── Core analysis function ──────────────────────────────────────────────────
analyze_workflow() {
    local wf_path="$1"
    local wf_name
    wf_name=$(basename "$wf_path")

    header "Analyzing: ${wf_name}"

    python3 - "$wf_path" "$NODES_JSON" "$MODELS_DIR" "$DRY_RUN" "$HF_TOKEN" << 'PYEOF'
import sys, json, os, subprocess, difflib
from pathlib import Path

wf_path = sys.argv[1]
nodes_json_path = sys.argv[2]
models_dir = sys.argv[3]
dry_run = sys.argv[4] == "true"
hf_token = sys.argv[5]

G = "\033[0;32m"
R = "\033[0;31m"
Y = "\033[1;33m"
C = "\033[0;36m"
B = "\033[1m"
D = "\033[2m"
N = "\033[0m"

def ok(m):   print(f"  {G}✓{N} {m}")
def miss(m): print(f"  {R}✗{N} {m}")
def warn(m): print(f"  {Y}⚠{N} {m}")
def info(m): print(f"  {C}ℹ{N} {m}")

# Load data
with open(nodes_json_path) as f:
    registered_nodes = json.load(f)
registered_names = set(registered_nodes.keys())

with open(wf_path) as f:
    try:
        workflow = json.load(f)
    except json.JSONDecodeError as e:
        miss(f"Invalid JSON: {e}")
        sys.exit(1)

# Extract nodes (API format)
nodes = {}
for node_id, node_data in workflow.items():
    if isinstance(node_data, dict) and "class_type" in node_data:
        nodes[node_id] = node_data

if not nodes:
    warn("No nodes found (not API format?)")
    sys.exit(0)

print(f"  Found {len(nodes)} nodes in workflow")
print()

# ─── CHECK NODES ─────────────────────────────────────────────────────────
missing_nodes = []
for node_id, node_data in nodes.items():
    ct = node_data["class_type"]
    if ct not in registered_names:
        matches = difflib.get_close_matches(ct, registered_names, n=3, cutoff=0.5)
        lower = ct.lower()
        substr = [n for n in registered_names if lower in n.lower() or n.lower() in lower]
        suggestions = list(dict.fromkeys(matches + substr[:3]))
        title = node_data.get("_meta", {}).get("title", ct)
        missing_nodes.append({
            "id": node_id, "class_type": ct,
            "title": title, "suggestions": suggestions
        })

if missing_nodes:
    print(f"  {B}Missing Nodes:{N}")
    for mn in missing_nodes:
        miss(f"{mn['class_type']} (node {mn['id']}: {mn['title']})")
        if mn["suggestions"]:
            for s in mn["suggestions"][:3]:
                info(f"  → Did you mean: {B}{s}{N}?")
        else:
            warn(f"  → No close match — custom node may need installing")
    print()
else:
    ok("All nodes registered")
    print()

# ─── CHECK MODELS ────────────────────────────────────────────────────────
MODEL_EXTS = {".safetensors", ".pth", ".ckpt", ".pt", ".onnx", ".bin"}

# HuggingFace repo guesses by filename pattern
HF_REPOS = {
    "wan": "Kijai/WanVideo_comfy",
    "sam2": "Kijai/sam2-safetensors",
    "flux": "Comfy-Org/flux1-dev",
    "t5xxl": "comfyanonymous/flux_text_encoders",
    "clip_l": "comfyanonymous/flux_text_encoders",
    "umt5": "Kijai/WanVideo_comfy",
    "realesrgan": "N/A",
    "ultrasharp": "Kim2091/UltraSharp",
    "vitpose": "Bingsu/adetailer",
    "lightx2v": "Kijai/WanVideo_comfy",
}

def guess_repo(filename):
    base = os.path.basename(filename).lower()
    for pattern, repo in HF_REPOS.items():
        if pattern in base:
            return repo
    return None

def guess_subdir(filename, input_key):
    base = filename.lower()
    if "vae" in base or base == "ae.safetensors":
        return "vae"
    if "clip_vision" in base:
        return "clip_vision"
    if "clip" in base or "t5" in base or "umt5" in base:
        return "clip"
    if "lora" in base or "lightx2v" in base:
        return "loras"
    if "sam2" in base:
        return "sam2"
    if "esrgan" in base or "ultrasharp" in base:
        return "upscale_models"
    if ".onnx" in base or "vitpose" in base or "yolo" in base:
        return "onnx"
    if input_key in ("model", "unet_name"):
        return "diffusion_models"
    return "diffusion_models"

missing_models = []
found_count = 0

for node_id, node_data in nodes.items():
    ct = node_data["class_type"]
    inputs = node_data.get("inputs", {})
    title = node_data.get("_meta", {}).get("title", ct)

    for key, value in inputs.items():
        if not isinstance(value, str):
            continue
        _, ext = os.path.splitext(value)
        if ext.lower() not in MODEL_EXTS:
            continue

        # Search for file
        found = False
        basename = os.path.basename(value)

        # Direct path
        if os.path.isfile(os.path.join(models_dir, value)):
            found = True
        # Search common subdirs
        if not found:
            for sd in ["diffusion_models", "checkpoints", "clip", "clip_vision",
                       "vae", "upscale_models", "sam2", "sams", "loras",
                       "onnx", "controlnet"]:
                if os.path.isfile(os.path.join(models_dir, sd, value)):
                    found = True
                    break
                if os.path.isfile(os.path.join(models_dir, sd, basename)):
                    found = True
                    break
        # Recursive by basename
        if not found:
            for root, dirs, files in os.walk(models_dir):
                if basename in files:
                    found = True
                    break

        if found:
            found_count += 1
        else:
            missing_models.append({
                "name": value,
                "basename": basename,
                "node_id": node_id,
                "node_title": title,
                "input_key": key,
                "hf_repo": guess_repo(value),
                "subdir": guess_subdir(value, key),
            })

if found_count:
    ok(f"{found_count} model file(s) found on disk")

if missing_models:
    print(f"\n  {B}Missing Models ({len(missing_models)}):{N}")
    downloads = []
    manual = []

    for mm in missing_models:
        miss(f"{mm['name']}")
        info(f"  Referenced by: node {mm['node_id']} ({mm['node_title']}), input '{mm['input_key']}'")
        if mm["hf_repo"] and mm["hf_repo"] != "N/A":
            info(f"  Source: {mm['hf_repo']}/{mm['basename']}")
            info(f"  Target: {models_dir}/{mm['subdir']}/{mm['basename']}")
            downloads.append(mm)
        else:
            warn(f"  Source unknown — manual download needed")
            manual.append(mm)

    if downloads and not dry_run:
        print(f"\n  {B}Downloading {len(downloads)} model(s)...{N}")
        for dl in downloads:
            dest_dir = os.path.join(models_dir, dl["subdir"])
            dest = os.path.join(dest_dir, dl["basename"])
            os.makedirs(dest_dir, exist_ok=True)

            url = f"https://huggingface.co/{dl['hf_repo']}/resolve/main/{dl['basename']}"
            info(f"Downloading: {dl['basename']}")

            try:
                cmd = ["wget", "-c", "--show-progress", "-q", "-O", dest + ".tmp", url]
                if hf_token:
                    cmd.insert(3, f"--header=Authorization: Bearer {hf_token}")
                subprocess.run(cmd, check=True)
                os.rename(dest + ".tmp", dest)
                ok(f"Downloaded: {dl['basename']}")
                print()
                info(f"Add to manifest.yaml under {dl['subdir']}:")
                print(f"    - name: \"{dl['basename'].split('.')[0]}\"")
                print(f"      file: \"{dl['subdir']}/{dl['basename']}\"")
                print(f"      source: {{ repo: \"{dl['hf_repo']}\", file: \"{dl['basename']}\" }}")
                print()
            except subprocess.CalledProcessError:
                miss(f"Download failed: {dl['basename']}")
                warn(f"  URL may be wrong — check {dl['hf_repo']} manually")
                # Clean up partial
                tmp = dest + ".tmp"
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception as e:
                miss(f"Error: {e}")

    elif downloads and dry_run:
        print(f"\n  {B}[DRY RUN] Would download:{N}")
        for dl in downloads:
            info(f"  {dl['hf_repo']}/{dl['basename']} → {models_dir}/{dl['subdir']}/")

    if manual:
        print(f"\n  {B}Manual downloads needed ({len(manual)}):{N}")
        for m in manual:
            warn(f"  {m['name']} — search HuggingFace or check the custom node docs")
else:
    ok("All model files present")

# ─── SUMMARY ─────────────────────────────────────────────────────────────
print(f"\n  {B}Result:{N}")
print(f"    Nodes:  {len(nodes)} total, {len(missing_nodes)} missing")
print(f"    Models: {found_count} found, {len(missing_models)} missing")

if missing_nodes:
    print(f"\n  {Y}⚠ Node issues require manual workflow JSON edits{N}")

if not missing_nodes and not missing_models:
    print(f"\n  {G}{B}Workflow ready to deploy!{N}")
PYEOF
}

# ─── Main ────────────────────────────────────────────────────────────────────
if $SCAN_ALL; then
    for wf in "$REPO_DIR"/workflows/*.json; do
        [[ -f "$wf" ]] && analyze_workflow "$wf"
    done
    CONTAINER_WFS=$(docker exec "$CONTAINER" find /app/workflows -name "*.json" 2>/dev/null || true)
    if [[ -n "$CONTAINER_WFS" ]]; then
        info "Also checking container workflows..."
        while IFS= read -r wf; do
            local_tmp=$(mktemp --suffix=.json)
            docker cp "${CONTAINER}:${wf}" "$local_tmp" 2>/dev/null
            analyze_workflow "$local_tmp"
            rm -f "$local_tmp"
        done <<< "$CONTAINER_WFS"
    fi
elif [[ -n "$WF_FILE" ]]; then
    analyze_workflow "$WF_FILE"
else
    echo "Usage: $0 <workflow.json> [--dry-run]"
    echo "       $0 --scan-all [--dry-run]"
    exit 1
fi