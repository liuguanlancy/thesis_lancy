#!/bin/bash
#
# Script to zip the runs folder excluding checkpoint-12000 directories
# Creates a timestamped zip file in the current directory
#

# Get current timestamp for the zip filename
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ZIP_NAME="runs_backup_${TIMESTAMP}.zip"

# Check if runs directory exists
if [ ! -d "runs" ]; then
    echo "Error: 'runs' directory not found in current location."
    echo "Please run this script from the repository root."
    exit 1
fi

echo "=========================================="
echo "Zipping runs folder"
echo "=========================================="
echo "Output file: ${ZIP_NAME}"
echo "Excluding: All checkpoint-12000 directories"
echo ""

# Count total directories and checkpoint-12000 directories for reporting
TOTAL_DIRS=$(find runs -type d | wc -l)
CHECKPOINT_DIRS=$(find runs -type d -name "checkpoint-12000" | wc -l)

echo "Statistics:"
echo "  Total directories: ${TOTAL_DIRS}"
echo "  checkpoint-12000 directories to exclude: ${CHECKPOINT_DIRS}"
echo ""
echo "Starting compression..."
echo "----------------------------------------"

# Create the zip file excluding checkpoint-12000 directories
# -r: recursive
# -q: quiet mode (less verbose)
# -x: exclude pattern
zip -r "${ZIP_NAME}" runs \
    -x "*/checkpoint-12000/*" \
    -x "*/checkpoint-12000"

# Check if zip was successful
if [ $? -eq 0 ]; then
    # Get the size of the created zip file
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        ZIP_SIZE=$(ls -lh "${ZIP_NAME}" | awk '{print $5}')
    else
        # Linux
        ZIP_SIZE=$(ls -lh "${ZIP_NAME}" | awk '{print $5}')
    fi

    echo ""
    echo "=========================================="
    echo "✅ Zip created successfully!"
    echo "=========================================="
    echo "File: ${ZIP_NAME}"
    echo "Size: ${ZIP_SIZE}"
    echo ""
    echo "Excluded ${CHECKPOINT_DIRS} checkpoint-12000 directories"
    echo ""
    echo "To verify the contents:"
    echo "  unzip -l ${ZIP_NAME} | grep checkpoint-12000"
    echo "  (should return nothing if exclusion worked)"
    echo ""
    echo "To extract later:"
    echo "  unzip ${ZIP_NAME}"
    echo "=========================================="
else
    echo ""
    echo "❌ Error: Failed to create zip file"
    exit 1
fi