#!/bin/bash
# ComfyUI Helper Script
#
# Usage: ./comfyui.sh <command>
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

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

case "$1" in
    build)
        echo -e "${GREEN}Building ComfyUI image...${NC}"
        docker build -t comfyui-pipeline:latest .
        ;;
    
    start)
        echo -e "${GREEN}Starting ComfyUI...${NC}"
        docker-compose up -d
        echo -e "${GREEN}ComfyUI available at: http://localhost:8188${NC}"
        ;;
    
    stop)
        echo -e "${YELLOW}Stopping ComfyUI...${NC}"
        docker-compose down
        ;;
    
    restart)
        echo -e "${YELLOW}Restarting ComfyUI...${NC}"
        docker-compose restart
        ;;
    
    logs)
        docker-compose logs -f comfyui
        ;;
    
    shell)
        echo -e "${GREEN}Opening shell in ComfyUI container...${NC}"
        docker-compose exec comfyui bash
        ;;
    
    status)
        echo -e "${GREEN}Checking model status...${NC}"
        docker-compose run --rm comfyui python /app/models/manager.py status
        ;;
    
    download)
        echo -e "${GREEN}Downloading missing models...${NC}"
        docker-compose run --rm comfyui python /app/models/manager.py download
        ;;
    
    backup)
        BACKUP_FILE="models_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
        echo -e "${GREEN}Backing up models to $BACKUP_FILE...${NC}"
        docker run --rm \
            -v comfyui_models:/data:ro \
            -v "$(pwd)":/backup \
            alpine tar czf "/backup/$BACKUP_FILE" -C /data .
        echo -e "${GREEN}Backup complete: $BACKUP_FILE${NC}"
        ls -lh "$BACKUP_FILE"
        ;;
    
    restore)
        if [ -z "$2" ]; then
            echo -e "${RED}Usage: $0 restore <backup_file.tar.gz>${NC}"
            exit 1
        fi
        if [ ! -f "$2" ]; then
            echo -e "${RED}Backup file not found: $2${NC}"
            exit 1
        fi
        echo -e "${YELLOW}Restoring models from $2...${NC}"
        echo -e "${YELLOW}WARNING: This will overwrite existing models!${NC}"
        read -p "Continue? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker volume create comfyui_models 2>/dev/null || true
            docker run --rm \
                -v comfyui_models:/data \
                -v "$(pwd)":/backup \
                alpine sh -c "rm -rf /data/* && tar xzf /backup/$2 -C /data"
            echo -e "${GREEN}Restore complete!${NC}"
        fi
        ;;
    
    *)
        echo "ComfyUI Helper Script"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  build     - Build the Docker image"
        echo "  start     - Start ComfyUI"
        echo "  stop      - Stop ComfyUI"
        echo "  restart   - Restart ComfyUI"
        echo "  logs      - View logs"
        echo "  shell     - Open shell in container"
        echo "  status    - Show model status"
        echo "  download  - Download missing models"
        echo "  backup    - Backup models volume"
        echo "  restore   - Restore models from backup"
        ;;
esac
