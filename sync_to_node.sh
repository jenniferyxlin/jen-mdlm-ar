#!/bin/bash
# Sync files to Prime Intellect node using rsync
# This will copy all project files to the remote node

# Configuration
REMOTE_HOST="prime-intellect-box0"
REMOTE_USER="root"
REMOTE_PATH="~/jen-mdlm-ar"  # Change this to match where you want the files on the remote node
LOCAL_PATH="."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Syncing files to ${REMOTE_HOST}...${NC}"
echo "Local path: $(pwd)"
echo "Remote path: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"
echo ""

# Use rsync to sync files (excludes .git, __pycache__, etc.)
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'venv' \
    --exclude 'checkpoints' \
    --exclude '*.log' \
    --exclude '.pytest_cache' \
    "${LOCAL_PATH}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Files synced successfully!${NC}"
    echo ""
    echo "To connect and run training:"
    echo "  ssh ${REMOTE_HOST}"
    echo "  cd ${REMOTE_PATH}"
    echo "  ./run_wikitext2.sh"
else
    echo ""
    echo -e "${YELLOW}✗ Sync failed. Check your SSH connection and paths.${NC}"
    exit 1
fi

