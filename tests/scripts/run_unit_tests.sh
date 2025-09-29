#!/bin/bash
# Run only unit tests
# These tests are fast and don't require external resources

echo "========================================="
echo "Running Unit Tests"
echo "========================================="
echo ""

# Configuration
PYTHON_CMD="/Users/mengzhao/miniconda3/envs/lancy/bin/python"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # tests directory
PROJECT_ROOT="$(cd "$BASE_DIR/.." && pwd)"    # project root

# Change to project root
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Run unit tests
echo "1. Testing modular structure..."
$PYTHON_CMD tests/unit/test_modular_structure.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Modular structure test passed${NC}"
else
    echo -e "${RED}✗ Modular structure test failed${NC}"
    exit 1
fi

echo ""
echo "2. Checking device availability..."
$PYTHON_CMD tests/unit/check_device.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Device check passed${NC}"
else
    echo -e "${RED}✗ Device check failed${NC}"
    exit 1
fi

echo ""
echo "========================================="
echo -e "${GREEN}All unit tests passed!${NC}"
echo "========================================="
echo ""
echo "Unit test outputs saved in: tests/outputs/unit/"