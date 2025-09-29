#!/bin/bash

# Script to sync runs from remote server, excluding checkpoint directories
# This saves bandwidth and disk space by skipping large model checkpoints

# Default configuration
DEFAULT_REMOTE_HOST="ubuntu@129.146.20.190"
DEFAULT_REMOTE_PATH="lancy/lancy_thesis/runs/"
DEFAULT_LOCAL_PATH="./runs/"

# Parse command line arguments
REMOTE_HOST="${1:-$DEFAULT_REMOTE_HOST}"
REMOTE_PATH="${2:-$DEFAULT_REMOTE_PATH}"
LOCAL_PATH="${3:-$DEFAULT_LOCAL_PATH}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [REMOTE_HOST] [REMOTE_PATH] [LOCAL_PATH]"
    echo ""
    echo "Sync training runs from remote server, excluding checkpoint directories"
    echo ""
    echo "Arguments:"
    echo "  REMOTE_HOST   Remote host (default: $DEFAULT_REMOTE_HOST)"
    echo "  REMOTE_PATH   Remote path (default: $DEFAULT_REMOTE_PATH)"
    echo "  LOCAL_PATH    Local destination (default: $DEFAULT_LOCAL_PATH)"
    echo ""
    echo "Examples:"
    echo "  # Use all defaults"
    echo "  $0"
    echo ""
    echo "  # Custom host"
    echo "  $0 user@192.168.1.100"
    echo ""
    echo "  # Custom host and paths"
    echo "  $0 user@host /path/to/runs/ ./local_runs/"
    echo ""
    echo "What gets synced:"
    echo "  ✓ Logs (training.log, args.json, etc.)"
    echo "  ✓ TensorBoard event files"
    echo "  ✓ Final metrics and evaluations"
    echo "  ✓ Configuration files"
    echo "  ✗ Checkpoint directories (checkpoint-*)"
    echo "  ✗ Model weights (*.safetensors, *.bin)"
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    print_usage
    exit 0
fi

echo -e "${BLUE}===========================================${NC}"
echo -e "${BLUE}     Sync Training Runs from Server${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""
echo -e "${GREEN}Remote:${NC} ${REMOTE_HOST}:${REMOTE_PATH}"
echo -e "${GREEN}Local:${NC}  ${LOCAL_PATH}"
echo ""

# Create local directory if it doesn't exist
if [ ! -d "$LOCAL_PATH" ]; then
    echo -e "${YELLOW}Creating local directory: ${LOCAL_PATH}${NC}"
    mkdir -p "$LOCAL_PATH"
fi

# Build rsync command with exclusions
echo -e "${BLUE}Starting sync (excluding checkpoints)...${NC}"
echo ""

# Run rsync with progress and exclusions
rsync -avh \
    --progress \
    --stats \
    --exclude='checkpoint-*/' \
    --exclude='checkpoint-final/' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='pytorch_model.bin' \
    --exclude='model.safetensors' \
    "${REMOTE_HOST}:${REMOTE_PATH}" \
    "${LOCAL_PATH}"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}===========================================${NC}"
    echo -e "${GREEN}✓ Sync completed successfully!${NC}"
    echo -e "${GREEN}===========================================${NC}"
    
    # Show summary of what was synced
    echo ""
    echo -e "${BLUE}Summary:${NC}"
    echo "Local runs directory: ${LOCAL_PATH}"
    
    # Count experiments
    if [ -d "$LOCAL_PATH" ]; then
        num_experiments=$(find "$LOCAL_PATH" -maxdepth 1 -type d | grep -v "^${LOCAL_PATH}$" | wc -l | tr -d ' ')
        echo "Number of experiments: $num_experiments"
        
        # Show recent experiments
        echo ""
        echo -e "${BLUE}Recent experiments:${NC}"
        ls -lt "$LOCAL_PATH" | head -6 | tail -5
    fi
else
    echo ""
    echo -e "${RED}===========================================${NC}"
    echo -e "${RED}✗ Sync failed!${NC}"
    echo -e "${RED}===========================================${NC}"
    echo ""
    echo "Please check:"
    echo "  1. SSH connection to ${REMOTE_HOST}"
    echo "  2. Remote path exists: ${REMOTE_PATH}"
    echo "  3. You have read permissions on remote"
    exit 1
fi

# Optional: Show disk usage saved
echo ""
echo -e "${YELLOW}Disk space saved by excluding checkpoints:${NC}"
echo "(Estimate: ~2.5GB per checkpoint × number of checkpoints)"

# Offer to show detailed structure
echo ""
echo -e "${BLUE}To view synced structure:${NC}"
echo "  tree -L 2 ${LOCAL_PATH}  # If tree is installed"
echo "  ls -la ${LOCAL_PATH}      # Basic listing"