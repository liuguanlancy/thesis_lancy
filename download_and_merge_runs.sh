#!/bin/bash
#
# Script to download and merge runs folders from multiple servers
# Excludes all checkpoint directories (checkpoints, checkpoint-final, etc.)
#

# Define servers
SERVER1="ubuntu@129.153.73.125"
SERVER2="ubuntu@104.171.202.235"
REMOTE_PATH="/home/ubuntu/lancy/lancy_thesis/runs"
LOCAL_RUNS="./runs"

# Create timestamp for backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "============================================"
echo "Download and Merge Runs from Remote Servers"
echo "============================================"
echo "Servers:"
echo "  1. ${SERVER1}"
echo "  2. ${SERVER2}"
echo ""
echo "Remote path: ${REMOTE_PATH}"
echo "Local path: ${LOCAL_RUNS}"
echo "Excluding: All checkpoint directories"
echo "============================================"
echo ""

# Backup existing runs folder if it exists
if [ -d "${LOCAL_RUNS}" ]; then
    BACKUP_NAME="runs_backup_before_merge_${TIMESTAMP}"
    echo "📦 Backing up existing runs folder to ${BACKUP_NAME}..."
    mv "${LOCAL_RUNS}" "${BACKUP_NAME}"
    echo "✅ Backup created: ${BACKUP_NAME}"
    echo ""
fi

# Create fresh runs directory
mkdir -p "${LOCAL_RUNS}"

# Function to download from a server
download_from_server() {
    local SERVER=$1
    local SERVER_NAME=$2
    local TEMP_DIR="runs_${SERVER_NAME}_${TIMESTAMP}"

    echo "============================================"
    echo "Downloading from ${SERVER_NAME} (${SERVER})"
    echo "============================================"

    # Create temporary directory for this server's data
    mkdir -p "${TEMP_DIR}"

    # Use rsync to download, excluding checkpoint directories
    # -a: archive mode (preserves permissions, timestamps, etc.)
    # -v: verbose
    # -z: compression for transfer
    # --progress: show progress
    # --exclude: patterns to exclude
    echo "Starting rsync..."
    rsync -avz --progress \
        --exclude="*/checkpoints/" \
        --exclude="*/checkpoints" \
        --exclude="*/checkpoint-*/" \
        --exclude="*/checkpoint-*" \
        --exclude="*/checkpoint_*/" \
        --exclude="*/checkpoint_*" \
        "${SERVER}:${REMOTE_PATH}/" \
        "${TEMP_DIR}/"

    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded from ${SERVER_NAME}"

        # Count what was downloaded
        local DIR_COUNT=$(find "${TEMP_DIR}" -type d | wc -l)
        local FILE_COUNT=$(find "${TEMP_DIR}" -type f | wc -l)
        echo "  Downloaded: ${DIR_COUNT} directories, ${FILE_COUNT} files"

        # Merge into main runs directory
        echo "📂 Merging into main runs directory..."
        rsync -av "${TEMP_DIR}/" "${LOCAL_RUNS}/"

        if [ $? -eq 0 ]; then
            echo "✅ Successfully merged ${SERVER_NAME} data"
        else
            echo "❌ Error merging ${SERVER_NAME} data"
        fi
    else
        echo "❌ Error downloading from ${SERVER_NAME}"
        echo "   Please check:"
        echo "   - SSH connection: ssh ${SERVER}"
        echo "   - Remote path exists: ${REMOTE_PATH}"
    fi

    echo ""
}

# Download from both servers
download_from_server "${SERVER1}" "server1_129"
download_from_server "${SERVER2}" "server2_104"

# Clean up temporary directories
echo "============================================"
echo "Cleanup"
echo "============================================"
echo "Removing temporary directories..."
rm -rf "runs_server1_129_${TIMESTAMP}"
rm -rf "runs_server2_104_${TIMESTAMP}"
echo "✅ Cleanup complete"
echo ""

# Final statistics
echo "============================================"
echo "📊 Final Statistics"
echo "============================================"
FINAL_DIR_COUNT=$(find "${LOCAL_RUNS}" -type d | wc -l)
FINAL_FILE_COUNT=$(find "${LOCAL_RUNS}" -type f | wc -l)
CHECKPOINT_CHECK=$(find "${LOCAL_RUNS}" -type d -name "*checkpoint*" | wc -l)

echo "Merged runs directory:"
echo "  Total directories: ${FINAL_DIR_COUNT}"
echo "  Total files: ${FINAL_FILE_COUNT}"
echo "  Checkpoint directories (should be 0): ${CHECKPOINT_CHECK}"

if [ ${CHECKPOINT_CHECK} -gt 0 ]; then
    echo ""
    echo "⚠️  Warning: Found ${CHECKPOINT_CHECK} checkpoint directories!"
    echo "Listing them:"
    find "${LOCAL_RUNS}" -type d -name "*checkpoint*"
fi

# Calculate total size
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    TOTAL_SIZE=$(du -sh "${LOCAL_RUNS}" | awk '{print $1}')
else
    # Linux
    TOTAL_SIZE=$(du -sh "${LOCAL_RUNS}" | awk '{print $1}')
fi

echo "  Total size: ${TOTAL_SIZE}"
echo ""
echo "============================================"
echo "✅ Download and merge complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Check the merged runs directory: ls -la ${LOCAL_RUNS}/"
echo "2. If you backed up existing runs, it's at: runs_backup_before_merge_${TIMESTAMP}/"
echo "3. To create a zip backup: ./zip_runs.sh"
echo "============================================"