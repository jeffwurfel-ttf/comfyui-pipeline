# ComfyUI Helper Script (PowerShell)
#
# Usage: .\comfyui.ps1 <command>
#
# Commands:
#   build     - Build the Docker image
#   start     - Start ComfyUI
#   stop      - Stop ComfyUI
#   restart   - Restart ComfyUI
#   logs      - View logs
#   shell     - Open shell in container
#   status    - Show model status
#   download  - Download missing models
#   backup    - Backup models volume
#   restore   - Restore models volume

param(
    [Parameter(Position=0)]
    [string]$Command,
    
    [Parameter(Position=1)]
    [string]$Arg1
)

$ErrorActionPreference = "Stop"

function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

switch ($Command) {
    "build" {
        Write-ColorOutput Green "Building ComfyUI image..."
        docker build -t comfyui-pipeline:latest .
    }
    
    "start" {
        Write-ColorOutput Green "Starting ComfyUI..."
        docker-compose up -d
        Write-ColorOutput Green "ComfyUI available at: http://localhost:8188"
    }
    
    "stop" {
        Write-ColorOutput Yellow "Stopping ComfyUI..."
        docker-compose down
    }
    
    "restart" {
        Write-ColorOutput Yellow "Restarting ComfyUI..."
        docker-compose restart
    }
    
    "logs" {
        docker-compose logs -f comfyui
    }
    
    "shell" {
        Write-ColorOutput Green "Opening shell in ComfyUI container..."
        docker-compose exec comfyui bash
    }
    
    "status" {
        Write-ColorOutput Green "Checking model status..."
        docker-compose run --rm comfyui python /app/models/manager.py status
    }
    
    "download" {
        Write-ColorOutput Green "Downloading missing models..."
        docker-compose run --rm comfyui python /app/models/manager.py download
    }
    
    "backup" {
        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $backupFile = "models_backup_$timestamp.tar.gz"
        Write-ColorOutput Green "Backing up models to $backupFile..."
        
        docker run --rm `
            -v comfyui_models:/data:ro `
            -v "${PWD}:/backup" `
            alpine tar czf "/backup/$backupFile" -C /data .
        
        Write-ColorOutput Green "Backup complete: $backupFile"
        Get-ChildItem $backupFile | Format-Table Name, Length
    }
    
    "restore" {
        if (-not $Arg1) {
            Write-ColorOutput Red "Usage: .\comfyui.ps1 restore <backup_file.tar.gz>"
            exit 1
        }
        if (-not (Test-Path $Arg1)) {
            Write-ColorOutput Red "Backup file not found: $Arg1"
            exit 1
        }
        
        Write-ColorOutput Yellow "Restoring models from $Arg1..."
        Write-ColorOutput Yellow "WARNING: This will overwrite existing models!"
        $confirm = Read-Host "Continue? (y/n)"
        
        if ($confirm -eq 'y' -or $confirm -eq 'Y') {
            docker volume create comfyui_models 2>$null
            docker run --rm `
                -v comfyui_models:/data `
                -v "${PWD}:/backup" `
                alpine sh -c "rm -rf /data/* && tar xzf /backup/$Arg1 -C /data"
            Write-ColorOutput Green "Restore complete!"
        }
    }
    
    default {
        Write-Output @"
ComfyUI Helper Script

Usage: .\comfyui.ps1 <command>

Commands:
  build     - Build the Docker image
  start     - Start ComfyUI
  stop      - Stop ComfyUI
  restart   - Restart ComfyUI
  logs      - View logs
  shell     - Open shell in container
  status    - Show model status
  download  - Download missing models
  backup    - Backup models volume
  restore   - Restore models from backup
"@
    }
}
