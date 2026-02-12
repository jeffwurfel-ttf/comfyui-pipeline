# ComfyUI Container

A maintainable, portable ComfyUI setup with proper model management.

## Key Features

- **Model Manifest**: YAML file defines exactly what models are needed
- **Model Manager**: Python script downloads/verifies models automatically
- **Named Volume**: Models persist in Docker volume (not container)
- **Portable**: Move to new machine by copying volume or re-running download

## Quick Start

### 1. Build the Container

```bash
cd comfyui-container
docker build -t comfyui-pipeline:latest .
```

### 2. Create Network (if not exists)

```bash
docker network create ai-network
```

### 3. Check Model Status

```bash
docker-compose run --rm comfyui python /app/models/manager.py status
```

### 4. Download Models

```bash
docker-compose run --rm comfyui python /app/models/manager.py download
```

### 5. Run ComfyUI

```bash
docker-compose up -d
```

Access at: http://localhost:8188

## Model Management

### View Status
```bash
docker-compose run --rm comfyui python /app/models/manager.py status
```

### Download Missing Models
```bash
docker-compose run --rm comfyui python /app/models/manager.py download
```

### Download ALL Models (including disabled)
```bash
docker-compose run --rm comfyui python /app/models/manager.py download --all
```

### Verify Model Integrity
```bash
docker-compose run --rm comfyui python /app/models/manager.py verify
```

### Export Installed Models List
```bash
docker-compose run --rm comfyui python /app/models/manager.py export
```

## Adding New Models

1. Edit `models/manifest.yaml`
2. Add entry with name, source, destination
3. Run `docker-compose run --rm comfyui python /app/models/manager.py download`

Example entry:
```yaml
checkpoints:
  - name: "My New Model"
    enabled: true
    source: huggingface
    repo: "author/model-name"
    filename: "model.safetensors"
    dest: "checkpoints/my_model.safetensors"
    size_gb: 6.5
    sha256: null  # Auto-filled on download
```

## Moving to Another Machine

### Option A: Re-download (Simplest)

On new machine:
```bash
# Copy project files
scp -r comfyui-container/ newmachine:~/

# On new machine
cd comfyui-container
docker build -t comfyui-pipeline:latest .
docker-compose run --rm comfyui python /app/models/manager.py download
docker-compose up -d
```

### Option B: Backup/Restore Volume (Faster)

On source machine:
```bash
# Backup models volume
docker run --rm \
  -v comfyui_models:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/models_backup.tar.gz -C /data .
```

On new machine:
```bash
# Create volume
docker volume create comfyui_models

# Restore
docker run --rm \
  -v comfyui_models:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/models_backup.tar.gz -C /data
```

### Option C: Use External Storage

Mount models from NAS/shared storage:

```yaml
# In docker-compose.yml
volumes:
  - //nas/models:/models  # Windows
  - /mnt/nas/models:/models  # Linux
```

## Directory Structure

```
comfyui-container/
├── docker-compose.yml    # Container orchestration
├── Dockerfile            # Container image definition
├── models/
│   ├── manifest.yaml     # Model definitions
│   └── manager.py        # Download/verify script
├── scripts/
│   └── entrypoint.sh     # Container startup
├── custom_nodes/         # Custom nodes (mounted)
├── workflows/            # Saved workflows (mounted)
├── output/               # Generated images (mounted)
└── input/                # Input images (mounted)
```

## Volume Layout

The `comfyui_models` volume contains:
```
/models/
├── checkpoints/          # Main models (SD, SDXL)
├── vae/                  # VAE models
├── controlnet/           # ControlNet models
├── clip/                 # CLIP models
├── upscale_models/       # ESRGAN, etc.
├── loras/                # LoRA models
├── ipadapter/            # IP-Adapter models
├── clip_vision/          # CLIP Vision models
└── animatediff_models/   # AnimateDiff motion modules
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMFYUI_PORT` | 8188 | Port to listen on |
| `CIVITAI_API_KEY` | - | CivitAI API key for downloads |
| `HF_TOKEN` | - | HuggingFace token for private models |
| `AUTODOWNLOAD_MODELS` | false | Auto-download on startup |
| `UPDATE_CUSTOM_NODES` | false | Update custom nodes on startup |

## Custom Nodes

Place custom nodes in `./custom_nodes/`. They're mounted into the container.

To install a custom node:
```bash
cd custom_nodes
git clone https://github.com/author/ComfyUI-CustomNode.git
```

Restart ComfyUI to load.

## Troubleshooting

### "No models found"
Run the download command:
```bash
docker-compose run --rm comfyui python /app/models/manager.py download
```

### "CUDA out of memory"
Add to docker-compose.yml environment:
```yaml
environment:
  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### "Model not loading"
Check the symlink:
```bash
docker-compose exec comfyui ls -la /app/ComfyUI/models
# Should show: models -> /models
```

### Check logs
```bash
docker-compose logs -f comfyui
```

## Integration with AI Gateway

This ComfyUI container integrates with the AI Gateway:

1. Both use `ai-network` Docker network
2. Gateway calls ComfyUI via `http://comfyui:8188`
3. ComfyUI workflows can be triggered via the Gateway's `/workflow` endpoints

## Backup Strategy

1. **Workflows**: Stored in `./workflows/` - commit to git
2. **Custom Nodes**: Stored in `./custom_nodes/` - track in git
3. **Models**: In Docker volume - backup periodically or re-download
4. **Outputs**: Stored in `./output/` - backup as needed
