#!/usr/bin/env bash
# =============================================================================
# comfy_models.sh — Unified ComfyUI Model Manager
# =============================================================================
# Single tool for downloading, verifying, and managing all ComfyUI models.
# Reads from models/manifest.yaml as the single source of truth.
#
# Replaces: download_models.sh, download_models_rocky.sh, download_wan.ps1,
#           download_ttf_models.ps1, download_models.ps1, bootstrap.sh
#
# Usage:
#   ./tools/comfy_models.sh status                     # Show installed vs missing
#   ./tools/comfy_models.sh status --workflow wan-t2v   # What does wan-t2v need?
#   ./tools/comfy_models.sh download                    # Download ALL missing models
#   ./tools/comfy_models.sh download --workflow wan-t2v # Download only wan-t2v deps
#   ./tools/comfy_models.sh verify                      # Check sizes, detect corruption
#   ./tools/comfy_models.sh cleanup                     # Find orphans & broken files
#   ./tools/comfy_models.sh init                        # First-time: create dirs + symlinks
#
# Requires: bash 4+, python3 + PyYAML, wget or curl
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${REPO_DIR}/models/manifest.yaml"

# Load .env if present
[[ -f "${REPO_DIR}/.env" ]] && { set -a; source "${REPO_DIR}/.env"; set +a; }

MODELS_DIR="${COMFY_MODELS_HOST:-/mnt/ssd/comfyui-models}"
HF_TOKEN="${HF_TOKEN:-}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Counters
INSTALLED=0
MISSING=0
DOWNLOADED=0
SKIPPED=0
FAILED=0

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
miss() { echo -e "  ${RED}✗${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
info() { echo -e "  ${CYAN}ℹ${NC} $1"; }
header() { echo -e "\n${BOLD}═══ $1 ═══${NC}"; }

check_deps() {
    if ! command -v python3 &>/dev/null; then
        echo "ERROR: python3 required"
        exit 1
    fi
    if ! python3 -c "import yaml" 2>/dev/null; then
        echo "ERROR: PyYAML required. Install: pip3 install pyyaml"
        exit 1
    fi
    if [[ ! -f "$MANIFEST" ]]; then
        echo "ERROR: Manifest not found: $MANIFEST"
        exit 1
    fi
}

# Parse manifest using PyYAML
# Outputs tab-separated: name\tfile\tsource_type\tsource_url\tsize_gb\tworkflows
parse_manifest() {
    local workflow_filter="${1:-}"
    python3 <<PYEOF
import yaml, sys

with open("${MANIFEST}") as f:
    data = yaml.safe_load(f)

workflow_filter = "${workflow_filter}"

for category, items in data.items():
    if not isinstance(items, list):
        continue
    for item in items:
        if not isinstance(item, dict):
            continue

        name = item.get("name", "?")
        filepath = item.get("file", "")
        size_gb = item.get("size_gb", 0)
        workflows = item.get("workflows", [])

        # Resolve source
        src = item.get("source", {})
        if isinstance(src, dict):
            if src.get("snapshot"):
                source_type = "snapshot"
                source_url = src.get("repo", "")
            elif src.get("repo"):
                source_type = "hf"
                source_url = src["repo"] + "/" + src.get("file", "")
            elif src.get("url"):
                source_type = "url"
                source_url = src["url"]
            else:
                source_type = "unknown"
                source_url = ""
        else:
            source_type = "unknown"
            source_url = str(src)

        wf_str = ",".join(str(w) for w in workflows) if workflows else "shared"

        # Apply workflow filter
        if workflow_filter:
            wf_strs = [str(w) for w in workflows]
            if workflow_filter not in wf_strs:
                # Also try prefix match for --profile
                if not any(w.startswith(workflow_filter) for w in wf_strs):
                    continue

        print(f"{name}\t{filepath}\t{source_type}\t{source_url}\t{size_gb}\t{wf_str}")
PYEOF
}

download_hf() {
    local repo="$1"
    local filename="$2"
    local dest="$3"

    local url="https://huggingface.co/${repo}/resolve/main/${filename}"

    if command -v wget &>/dev/null; then
        local wget_opts=(-c --show-progress -q -O "${dest}.tmp")
        if [[ -n "$HF_TOKEN" ]]; then
            wget_opts+=(--header="Authorization: Bearer ${HF_TOKEN}")
        fi
        wget "${wget_opts[@]}" "$url" && mv "${dest}.tmp" "$dest"
    elif command -v curl &>/dev/null; then
        local curl_opts=(-L -C - --progress-bar -o "${dest}.tmp")
        [[ -n "$HF_TOKEN" ]] && curl_opts+=(-H "Authorization: Bearer ${HF_TOKEN}")
        curl "${curl_opts[@]}" "$url" && mv "${dest}.tmp" "$dest"
    else
        echo "ERROR: Neither wget nor curl found"
        return 1
    fi
}

download_url() {
    local url="$1"
    local dest="$2"

    if command -v wget &>/dev/null; then
        wget -c --show-progress -q -O "${dest}.tmp" "$url" && mv "${dest}.tmp" "$dest"
    else
        curl -L -C - --progress-bar -o "${dest}.tmp" "$url" && mv "${dest}.tmp" "$dest"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

cmd_status() {
    local workflow_filter="${1:-}"
    header "Model Status — ${MODELS_DIR}"
    [[ -n "$workflow_filter" ]] && info "Filtered to workflow: ${workflow_filter}"
    echo ""

    local current_section=""
    while IFS=$'\t' read -r name filepath source_type source_url size_gb workflows; do
        local full_path="${MODELS_DIR}/${filepath}"

        # Section headers based on directory
        local section
        section=$(dirname "$filepath" | cut -d/ -f1)
        if [[ "$section" != "$current_section" ]]; then
            [[ -n "$current_section" ]] && echo ""
            echo -e "  ${BOLD}${section}/${NC}"
            current_section="$section"
        fi

        if [[ -f "$full_path" ]]; then
            local actual_mb
            actual_mb=$(du -m "$full_path" 2>/dev/null | cut -f1)
            echo -e "    ${GREEN}✓${NC} ${name} ${DIM}(${actual_mb}MB)${NC} ${DIM}[${workflows}]${NC}"
            ((INSTALLED++)) || true
        else
            echo -e "    ${RED}✗${NC} ${name} ${DIM}(~${size_gb}GB needed)${NC} ${DIM}[${workflows}]${NC}"
            ((MISSING++)) || true
        fi
    done < <(parse_manifest "$workflow_filter")

    echo ""
    echo -e "  ${BOLD}Summary:${NC} ${GREEN}${INSTALLED} installed${NC}, ${RED}${MISSING} missing${NC}"

    if [[ -d "$MODELS_DIR" ]]; then
        local total avail
        total=$(du -sh "$MODELS_DIR" 2>/dev/null | cut -f1)
        avail=$(df -h "$MODELS_DIR" 2>/dev/null | tail -1 | awk '{print $4}')
        info "Disk: ${total} used, ${avail} available"
    fi
}

cmd_download() {
    local workflow_filter="${1:-}"
    header "Download Missing Models"
    [[ -n "$workflow_filter" ]] && info "Filtered to workflow: ${workflow_filter}"
    [[ -n "$HF_TOKEN" ]] && info "HuggingFace token: set" || warn "HF_TOKEN not set (gated repos will fail)"
    echo ""

    while IFS=$'\t' read -r name filepath source_type source_url size_gb workflows; do
        local full_path="${MODELS_DIR}/${filepath}"

        # Skip if already present
        if [[ -f "$full_path" ]]; then
            ((SKIPPED++)) || true
            continue
        fi

        echo -e "  ${CYAN}↓${NC} ${name} (~${size_gb}GB)"
        mkdir -p "$(dirname "$full_path")"

        case "$source_type" in
            hf)
                local hf_repo hf_file
                hf_repo=$(echo "$source_url" | cut -d/ -f1-2)
                hf_file=$(echo "$source_url" | cut -d/ -f3-)
                if download_hf "$hf_repo" "$hf_file" "$full_path"; then
                    ok "Downloaded: ${name}"
                    ((DOWNLOADED++)) || true
                else
                    miss "FAILED: ${name}"
                    ((FAILED++)) || true
                fi
                ;;
            url)
                if download_url "$source_url" "$full_path"; then
                    ok "Downloaded: ${name}"
                    ((DOWNLOADED++)) || true
                else
                    miss "FAILED: ${name}"
                    ((FAILED++)) || true
                fi
                ;;
            snapshot)
                info "Snapshot download: ${name}"
                info "  Use: huggingface-cli download ${source_url} --local-dir ${MODELS_DIR}/sam3d/hf"
                warn "Snapshot downloads not automated — run manually"
                ;;
            *)
                warn "Unknown source type '${source_type}' for ${name}"
                ;;
        esac
    done < <(parse_manifest "$workflow_filter")

    echo ""
    header "Download Summary"
    echo -e "  Downloaded: ${GREEN}${DOWNLOADED}${NC}"
    echo -e "  Skipped:    ${CYAN}${SKIPPED}${NC} (already exist)"
    [[ $FAILED -gt 0 ]] && echo -e "  Failed:     ${RED}${FAILED}${NC}"
}

cmd_verify() {
    header "Verify Model Files"

    # Check for suspiciously small files
    echo -e "  ${BOLD}Checking for broken downloads...${NC}"
    local broken
    broken=$(find "$MODELS_DIR" \( -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" \) -size -1k 2>/dev/null || true)
    if [[ -n "$broken" ]]; then
        while IFS= read -r f; do
            local size
            size=$(stat -c%s "$f" 2>/dev/null || echo "?")
            miss "Broken (${size} bytes): $f"
        done <<< "$broken"
    else
        ok "No broken model files"
    fi

    # Check for broken symlinks
    local broken_links
    broken_links=$(find "$MODELS_DIR" -xtype l 2>/dev/null || true)
    if [[ -n "$broken_links" ]]; then
        while IFS= read -r f; do
            miss "Broken symlink: $f → $(readlink "$f")"
        done <<< "$broken_links"
    else
        ok "No broken symlinks"
    fi

    # Check manifest entries exist
    echo ""
    echo -e "  ${BOLD}Manifest vs disk:${NC}"
    while IFS=$'\t' read -r name filepath source_type source_url size_gb workflows; do
        local full_path="${MODELS_DIR}/${filepath}"
        if [[ -f "$full_path" ]]; then
            local actual_mb
            actual_mb=$(du -m "$full_path" 2>/dev/null | cut -f1)
            local actual_gb
            actual_gb=$(echo "$actual_mb" | awk '{printf "%.2f", $1/1024}')
            ok "${name}: ${actual_gb}GB on disk (expected ~${size_gb}GB)"
        else
            miss "${name}: NOT FOUND — ${filepath}"
        fi
    done < <(parse_manifest)
}

cmd_cleanup() {
    header "Cleanup Scan"

    # Get all manifest paths
    local manifest_files
    manifest_files=$(mktemp)
    parse_manifest | cut -f2 > "$manifest_files"

    # Find model files not in manifest
    echo -e "  ${BOLD}Orphaned model files (not in manifest):${NC}"
    local found_orphan=false
    while IFS= read -r filepath; do
        local relpath="${filepath#${MODELS_DIR}/}"
        # Skip sam3d subdirectory (snapshot-managed)
        [[ "$relpath" == sam3d/* ]] && continue
        if ! grep -qF "$relpath" "$manifest_files" 2>/dev/null; then
            local size_mb
            size_mb=$(du -m "$filepath" 2>/dev/null | cut -f1)
            warn "Not in manifest: ${relpath} (${size_mb}MB)"
            found_orphan=true
        fi
    done < <(find "$MODELS_DIR" \( -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" -o -name "*.onnx" \) 2>/dev/null)
    $found_orphan || ok "No orphaned files"

    rm -f "$manifest_files"

    # Check for cache cruft
    echo ""
    echo -e "  ${BOLD}Cache / cruft directories:${NC}"
    local found_cruft=false
    for cruft_dir in \
        "$MODELS_DIR/_hf_staging" \
        "$MODELS_DIR/.hf_cache" \
        "$MODELS_DIR/sam3d/hf/.cache" \
        "$MODELS_DIR/sam3d/hf/doc" \
        "$MODELS_DIR/diffusion_models/Wan22Animate"; do
        if [[ -d "$cruft_dir" ]]; then
            local size
            size=$(du -sh "$cruft_dir" 2>/dev/null | cut -f1)
            warn "Removable: ${cruft_dir} (${size})"
            found_cruft=true
        fi
    done
    $found_cruft || ok "No cache cruft found"

    # Check for stray files in repo root
    echo ""
    echo -e "  ${BOLD}Stray model files in repo:${NC}"
    local stray
    stray=$(find "$REPO_DIR" -maxdepth 1 \( -name "*.safetensors" -o -name "*.pth" \) 2>/dev/null || true)
    if [[ -n "$stray" ]]; then
        while IFS= read -r f; do
            local size_mb
            size_mb=$(du -m "$f" 2>/dev/null | cut -f1)
            warn "STRAY: $(basename "$f") (${size_mb}MB)"
        done <<< "$stray"
    else
        ok "No stray model files in repo"
    fi

    # Empty directories
    echo ""
    echo -e "  ${BOLD}Empty model directories:${NC}"
    for dir in "$MODELS_DIR"/*/; do
        [[ ! -d "$dir" ]] && continue
        local dirname
        dirname=$(basename "$dir")
        [[ "$dirname" == .* ]] && continue
        local count
        count=$(find "$dir" \( -name "*.safetensors" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.pt" -o -name "*.onnx" -o -name "*.yaml" \) 2>/dev/null | wc -l)
        if [[ "$count" -eq 0 ]]; then
            info "${dirname}/ — empty"
        fi
    done
}

cmd_init() {
    header "Initialize Directory Structure"

    # Verify SSD
    if mountpoint -q /mnt/ssd 2>/dev/null; then
        ok "SSD mounted at /mnt/ssd"
    else
        warn "/mnt/ssd may not be a mount point"
    fi

    # Create all directories
    local dirs=(checkpoints clip clip_vision controlnet diffusion_models dwpose ipadapter loras onnx sam2 sam3d sams upscale_models vae)
    for dir in "${dirs[@]}"; do
        mkdir -p "${MODELS_DIR}/${dir}"
    done
    ok "Model directories created (${#dirs[@]} dirs)"

    # Create input/output
    mkdir -p /mnt/ssd/comfyui-input /mnt/ssd/comfyui-output
    ok "Input/Output directories created"

    # Symlinks
    for name in comfyui-models comfyui-input comfyui-output; do
        local target="/mnt/ssd/${name}"
        local link="${HOME}/${name}"
        if [[ -L "$link" ]]; then
            ok "Symlink exists: ${link}"
        elif [[ ! -e "$link" ]]; then
            ln -s "$target" "$link"
            ok "Created symlink: ${link} → ${target}"
        else
            warn "${link} exists but is not a symlink"
        fi
    done

    local avail
    avail=$(df -h /mnt/ssd 2>/dev/null | tail -1 | awk '{print $4}')
    info "SSD available: ${avail}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [options]

Commands:
  status    Show installed vs missing models
  download  Download missing models
  verify    Check model file integrity
  cleanup   Find orphaned/broken files
  init      First-time directory setup

Options:
  --workflow <id>   Filter to specific workflow (e.g. wan-t2v, flux-txt2img)
  --profile <n>  Filter by prefix: sdxl, flux, wan, esrgan, detection, sam3d

Examples:
  $(basename "$0") status                     # Full inventory
  $(basename "$0") status --workflow wan-t2v   # Just wan-t2v deps
  $(basename "$0") download --profile wan      # Download all wan models
  $(basename "$0") cleanup                     # Find orphaned files
EOF
}

COMMAND="${1:-}"
shift || true

WORKFLOW_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --workflow) WORKFLOW_FILTER="$2"; shift 2 ;;
        --profile) WORKFLOW_FILTER="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

check_deps

echo -e "${BOLD}ComfyUI Model Manager${NC}"
echo -e "  Models: ${MODELS_DIR}"
echo -e "  Manifest: ${MANIFEST}"

case "$COMMAND" in
    status)   cmd_status "$WORKFLOW_FILTER" ;;
    download) cmd_download "$WORKFLOW_FILTER" ;;
    verify)   cmd_verify ;;
    cleanup)  cmd_cleanup ;;
    init)     cmd_init ;;
    "")       usage; exit 1 ;;
    *)        echo "Unknown command: $COMMAND"; usage; exit 1 ;;
esac