#!/usr/bin/env python3
"""
ComfyUI Model Manager

Downloads, verifies, and manages models based on manifest.yaml.

Usage:
    python manager.py status              # Show what's installed vs missing
    python manager.py download            # Download all enabled models
    python manager.py download --all      # Download ALL models (even disabled)
    python manager.py verify              # Verify checksums of installed models
    python manager.py export              # Export installed models list
"""
import argparse
import hashlib
import os
import sys
import json
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import shutil

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).parent
MANIFEST_PATH = SCRIPT_DIR / "manifest.yaml"
MODELS_BASE_PATH = Path(os.environ.get("COMFYUI_MODELS_PATH", "/models"))

# CivitAI API (optional, for downloading from CivitAI)
CIVITAI_API_KEY = os.environ.get("CIVITAI_API_KEY", "")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelInfo:
    """Information about a model from the manifest."""
    name: str
    enabled: bool
    source: str  # huggingface, civitai, url
    dest: str
    size_gb: float
    sha256: Optional[str] = None
    
    # Source-specific fields
    repo: Optional[str] = None  # HuggingFace
    filename: Optional[str] = None  # HuggingFace
    url: Optional[str] = None  # Direct URL
    model_id: Optional[int] = None  # CivitAI
    
    @property
    def full_path(self) -> Path:
        return MODELS_BASE_PATH / self.dest
    
    @property
    def is_installed(self) -> bool:
        return self.full_path.exists()
    
    @property
    def installed_size_gb(self) -> float:
        if self.is_installed:
            return self.full_path.stat().st_size / (1024**3)
        return 0


# =============================================================================
# Manifest Loading
# =============================================================================

def load_manifest() -> Dict[str, List[ModelInfo]]:
    """Load and parse the manifest file."""
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)
    
    with open(MANIFEST_PATH, 'r') as f:
        if YAML_AVAILABLE:
            data = yaml.safe_load(f)
        else:
            # Fallback: try to parse as JSON (won't work for YAML)
            print("WARNING: PyYAML not installed, trying JSON fallback")
            data = json.load(f)
    
    models = {}
    
    # Parse each category
    categories = [
        'checkpoints', 'vae', 'controlnet', 'clip', 'upscale_models',
        'loras', 'ipadapter', 'clip_vision', 'animatediff'
    ]
    
    for category in categories:
        models[category] = []
        for item in data.get(category, []):
            models[category].append(ModelInfo(
                name=item.get('name', 'Unknown'),
                enabled=item.get('enabled', True),
                source=item.get('source', 'url'),
                dest=item.get('dest', ''),
                size_gb=item.get('size_gb', 0),
                sha256=item.get('sha256'),
                repo=item.get('repo'),
                filename=item.get('filename'),
                url=item.get('url'),
                model_id=item.get('model_id'),
            ))
    
    return models


def save_manifest(models: Dict[str, List[ModelInfo]]):
    """Save manifest with updated checksums."""
    if not YAML_AVAILABLE:
        print("WARNING: Cannot save manifest without PyYAML")
        return
    
    with open(MANIFEST_PATH, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update checksums
    for category, model_list in models.items():
        if category in data:
            for i, model in enumerate(model_list):
                if i < len(data[category]) and model.sha256:
                    data[category][i]['sha256'] = model.sha256
    
    with open(MANIFEST_PATH, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Download Functions
# =============================================================================

def download_progress_hook(block_num, block_size, total_size):
    """Progress hook for urllib downloads."""
    if TQDM_AVAILABLE:
        return  # tqdm handles progress
    
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent:.1f}%")
        sys.stdout.flush()


def download_from_url(model: ModelInfo) -> bool:
    """Download model from direct URL."""
    if not model.url:
        print(f"  ERROR: No URL specified for {model.name}")
        return False
    
    dest_path = model.full_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  Downloading from URL...")
    print(f"  URL: {model.url}")
    
    try:
        # Download with progress
        if TQDM_AVAILABLE:
            # Use tqdm for better progress
            import urllib.request
            with tqdm(unit='B', unit_scale=True, desc=model.name[:30]) as pbar:
                def hook(b, bsize, tsize):
                    pbar.total = tsize
                    pbar.update(bsize)
                urllib.request.urlretrieve(model.url, dest_path, reporthook=hook)
        else:
            urllib.request.urlretrieve(model.url, dest_path, download_progress_hook)
            print()  # New line after progress
        
        return True
    except Exception as e:
        print(f"  ERROR: Download failed: {e}")
        return False


def download_from_huggingface(model: ModelInfo) -> bool:
    """Download model from HuggingFace Hub."""
    if not HF_AVAILABLE:
        print("  ERROR: huggingface_hub not installed")
        print("  Run: pip install huggingface_hub")
        return False
    
    if not model.repo or not model.filename:
        print(f"  ERROR: Missing repo or filename for {model.name}")
        return False
    
    dest_path = model.full_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  Downloading from HuggingFace...")
    print(f"  Repo: {model.repo}")
    print(f"  File: {model.filename}")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=model.repo,
            filename=model.filename,
            local_dir=dest_path.parent,
            local_dir_use_symlinks=False,
        )
        
        # Move to correct location if needed
        downloaded = Path(downloaded_path)
        if downloaded != dest_path:
            if downloaded.exists():
                shutil.move(str(downloaded), str(dest_path))
        
        return True
    except Exception as e:
        print(f"  ERROR: Download failed: {e}")
        return False


def download_from_civitai(model: ModelInfo) -> bool:
    """Download model from CivitAI."""
    if not model.model_id:
        print(f"  ERROR: No model_id specified for {model.name}")
        return False
    
    # CivitAI API endpoint
    api_url = f"https://civitai.com/api/v1/models/{model.model_id}"
    
    print(f"  Downloading from CivitAI...")
    print(f"  Model ID: {model.model_id}")
    
    try:
        # Get model info
        req = urllib.request.Request(api_url)
        if CIVITAI_API_KEY:
            req.add_header("Authorization", f"Bearer {CIVITAI_API_KEY}")
        
        with urllib.request.urlopen(req) as response:
            model_data = json.loads(response.read())
        
        # Get download URL from latest version
        versions = model_data.get('modelVersions', [])
        if not versions:
            print("  ERROR: No versions found")
            return False
        
        # Get primary file from first version
        files = versions[0].get('files', [])
        if not files:
            print("  ERROR: No files found")
            return False
        
        download_url = files[0].get('downloadUrl')
        if not download_url:
            print("  ERROR: No download URL")
            return False
        
        # Add API key if available
        if CIVITAI_API_KEY:
            download_url += f"?token={CIVITAI_API_KEY}"
        
        # Download the file
        model.url = download_url
        return download_from_url(model)
        
    except Exception as e:
        print(f"  ERROR: CivitAI download failed: {e}")
        return False


def download_model(model: ModelInfo) -> bool:
    """Download a model based on its source."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model.name}")
    print(f"  Destination: {model.dest}")
    print(f"  Size: {model.size_gb:.2f} GB")
    
    if model.is_installed:
        print(f"  Status: Already installed ({model.installed_size_gb:.2f} GB)")
        return True
    
    success = False
    
    if model.source == 'huggingface':
        success = download_from_huggingface(model)
    elif model.source == 'civitai':
        success = download_from_civitai(model)
    elif model.source == 'url':
        success = download_from_url(model)
    else:
        print(f"  ERROR: Unknown source type: {model.source}")
    
    if success:
        print(f"  ✓ Downloaded successfully")
        # Calculate checksum
        if model.is_installed:
            print(f"  Calculating checksum...")
            model.sha256 = calculate_checksum(model.full_path)
            print(f"  SHA256: {model.sha256[:16]}...")
    else:
        print(f"  ✗ Download failed")
    
    return success


# =============================================================================
# Verification Functions
# =============================================================================

def calculate_checksum(file_path: Path, algorithm='sha256') -> str:
    """Calculate checksum of a file."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b''):  # 8MB chunks
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def verify_model(model: ModelInfo) -> tuple[bool, str]:
    """Verify a model's checksum."""
    if not model.is_installed:
        return False, "Not installed"
    
    if not model.sha256:
        return True, "No checksum (skipped)"
    
    actual = calculate_checksum(model.full_path)
    if actual == model.sha256:
        return True, "Valid"
    else:
        return False, f"Mismatch (expected {model.sha256[:8]}..., got {actual[:8]}...)"


# =============================================================================
# Commands
# =============================================================================

def cmd_status(models: Dict[str, List[ModelInfo]], show_all: bool = False):
    """Show status of all models."""
    print("\n" + "="*70)
    print("COMFYUI MODEL STATUS")
    print("="*70)
    print(f"Models directory: {MODELS_BASE_PATH}")
    print()
    
    total_enabled = 0
    total_installed = 0
    total_size_needed = 0
    total_size_installed = 0
    
    for category, model_list in models.items():
        if not model_list:
            continue
        
        # Filter based on show_all
        display_list = model_list if show_all else [m for m in model_list if m.enabled]
        if not display_list:
            continue
        
        print(f"\n{category.upper()}")
        print("-" * 50)
        
        for model in display_list:
            status = "✓ Installed" if model.is_installed else "✗ Missing"
            enabled = "" if model.enabled else " [disabled]"
            size = f"{model.size_gb:.2f}GB"
            
            print(f"  {status} {model.name}{enabled}")
            print(f"           Size: {size} | Path: {model.dest}")
            
            if model.enabled:
                total_enabled += 1
                total_size_needed += model.size_gb
                if model.is_installed:
                    total_installed += 1
                    total_size_installed += model.installed_size_gb
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Enabled models:   {total_installed}/{total_enabled} installed")
    print(f"  Space needed:     {total_size_needed:.2f} GB")
    print(f"  Space used:       {total_size_installed:.2f} GB")
    print(f"  Space remaining:  {total_size_needed - total_size_installed:.2f} GB to download")


def cmd_download(models: Dict[str, List[ModelInfo]], download_all: bool = False):
    """Download missing models."""
    print("\n" + "="*70)
    print("DOWNLOADING MODELS")
    print("="*70)
    
    to_download = []
    for category, model_list in models.items():
        for model in model_list:
            if (download_all or model.enabled) and not model.is_installed:
                to_download.append(model)
    
    if not to_download:
        print("\nAll enabled models are already installed!")
        return
    
    total_size = sum(m.size_gb for m in to_download)
    print(f"\nModels to download: {len(to_download)}")
    print(f"Total size: {total_size:.2f} GB")
    print("\nStarting downloads...")
    
    success_count = 0
    for model in to_download:
        if download_model(model):
            success_count += 1
    
    print("\n" + "="*70)
    print(f"COMPLETE: {success_count}/{len(to_download)} models downloaded")
    print("="*70)
    
    # Save updated checksums
    save_manifest(models)


def cmd_verify(models: Dict[str, List[ModelInfo]]):
    """Verify checksums of installed models."""
    print("\n" + "="*70)
    print("VERIFYING MODEL CHECKSUMS")
    print("="*70)
    
    results = []
    for category, model_list in models.items():
        for model in model_list:
            if model.is_installed:
                print(f"\nVerifying: {model.name}...")
                valid, message = verify_model(model)
                results.append((model.name, valid, message))
                print(f"  Result: {message}")
    
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    valid_count = sum(1 for _, v, _ in results if v)
    print(f"  Valid: {valid_count}/{len(results)}")
    
    invalid = [(n, m) for n, v, m in results if not v]
    if invalid:
        print("\n  INVALID MODELS:")
        for name, msg in invalid:
            print(f"    - {name}: {msg}")


def cmd_export(models: Dict[str, List[ModelInfo]]):
    """Export list of installed models."""
    installed = []
    
    for category, model_list in models.items():
        for model in model_list:
            if model.is_installed:
                installed.append({
                    "category": category,
                    "name": model.name,
                    "path": model.dest,
                    "size_gb": model.installed_size_gb,
                    "sha256": model.sha256,
                })
    
    export_path = SCRIPT_DIR / "installed_models.json"
    with open(export_path, 'w') as f:
        json.dump(installed, f, indent=2)
    
    print(f"\nExported {len(installed)} models to: {export_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ComfyUI Model Manager")
    parser.add_argument('command', choices=['status', 'download', 'verify', 'export'],
                        help='Command to run')
    parser.add_argument('--all', action='store_true',
                        help='Include disabled models')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not YAML_AVAILABLE:
        print("WARNING: PyYAML not installed. Install with: pip install pyyaml")
    
    # Load manifest
    models = load_manifest()
    
    # Run command
    if args.command == 'status':
        cmd_status(models, args.all)
    elif args.command == 'download':
        cmd_download(models, args.all)
    elif args.command == 'verify':
        cmd_verify(models)
    elif args.command == 'export':
        cmd_export(models)


if __name__ == '__main__':
    main()
