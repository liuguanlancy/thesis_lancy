#!/bin/bash
# Run all tests in the test suite
# This script runs unit, integration, and functional tests

echo "========================================="
echo "Running Complete Test Suite"
echo "========================================="
echo ""

# Configuration
PYTHON_CMD="/Users/mengzhao/miniconda3/envs/lancy/bin/python"
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # tests directory
PROJECT_ROOT="$(cd "$BASE_DIR/.." && pwd)"    # project root

# Change to project root for consistent relative paths
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    local test_type=$3
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -e "${YELLOW}[$test_type]${NC} Running: $test_name"
    echo "  File: $test_file"
    
    if [ ! -f "$test_file" ]; then
        echo -e "  ${YELLOW}⚠ SKIPPED${NC} - File not found"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        echo ""
        return
    fi
    
    # Run the test and capture exit code
    $PYTHON_CMD "$test_file" > /tmp/test_output_$$.log 2>&1
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "  ${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "  ${RED}✗ FAILED${NC} (exit code: $EXIT_CODE)"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        echo "  Last 10 lines of output:"
        tail -10 /tmp/test_output_$$.log | sed 's/^/    /'
    fi
    
    # Clean up temp file
    rm -f /tmp/test_output_$$.log
    echo ""
}

# =========================================
# UNIT TESTS
# =========================================
echo "========================================="
echo "UNIT TESTS"
echo "========================================="
echo ""

run_test "Modular Structure Test" "tests/unit/test_modular_structure.py" "UNIT"
run_test "Device Check" "tests/unit/check_device.py" "UNIT"

# =========================================
# INTEGRATION TESTS  
# =========================================
echo "========================================="
echo "INTEGRATION TESTS"
echo "========================================="
echo ""

run_test "Multi-GPU Test" "tests/integration/test_multi_gpu.py" "INTEGRATION"
run_test "MOE on M1 Test" "tests/integration/test_moe_m1.py" "INTEGRATION"
run_test "Reinforcement Learning Test" "tests/integration/test_rl.py" "INTEGRATION"
run_test "Multi-Dataset Evaluation Test" "tests/integration/test_multi_eval.py" "INTEGRATION"
run_test "Cosine Scheduler Test" "tests/integration/test_cosine_scheduler.py" "INTEGRATION"
run_test "Scheduler Info Test" "tests/integration/test_scheduler_info.py" "INTEGRATION"

# =========================================
# FUNCTIONAL TESTS (Model-specific)
# =========================================
echo "========================================="
echo "FUNCTIONAL TESTS"
echo "========================================="
echo ""

# These tests may require authentication or large downloads
echo "Note: Functional tests may require HuggingFace authentication or large model downloads"
echo ""

run_test "Llama 3.2 Model Test" "tests/functional/test_llama32.py" "FUNCTIONAL"
run_test "Qwen3 Model Test" "tests/functional/test_qwen3.py" "FUNCTIONAL"

# =========================================
# TEST SUMMARY
# =========================================
echo ""
echo "========================================="
echo "TEST SUMMARY"
echo "========================================="
echo "Total Tests:    $TOTAL_TESTS"
echo -e "Passed:         ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:         ${RED}$FAILED_TESTS${NC}"
echo -e "Skipped:        ${YELLOW}$SKIPPED_TESTS${NC}"
echo ""

# Calculate success rate
if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo "Success Rate:   ${SUCCESS_RATE}%"
fi

echo ""
echo "Test outputs are saved in: tests/outputs/"
echo "  - Unit test outputs:        tests/outputs/unit/"
echo "  - Integration test outputs: tests/outputs/integration/"
echo "  - Functional test outputs:  tests/outputs/functional/"
echo ""

# Exit with appropriate code
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "${RED}Some tests failed. Please check the output above.${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed successfully!${NC}"
    exit 0
fi