#!/bin/bash
# Run only integration tests
# These tests verify multi-component functionality

echo "========================================="
echo "Running Integration Tests"
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
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo -e "${YELLOW}Running in quick mode (skipping slow tests)${NC}"
    echo ""
fi

# Test counter
PASSED=0
FAILED=0
SKIPPED=0

# Function to run test
run_integration_test() {
    local test_name=$1
    local test_file=$2
    local is_slow=${3:-false}
    
    if [ "$QUICK_MODE" = true ] && [ "$is_slow" = true ]; then
        echo -e "${YELLOW}⚠ Skipping: $test_name (slow test)${NC}"
        SKIPPED=$((SKIPPED + 1))
        return
    fi
    
    echo "Running: $test_name"
    $PYTHON_CMD "$test_file"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# Run integration tests
run_integration_test "Multi-GPU Test" "tests/integration/test_multi_gpu.py" false
run_integration_test "MOE on M1 Test" "tests/integration/test_moe_m1.py" false
run_integration_test "Reinforcement Learning Test" "tests/integration/test_rl.py" false
run_integration_test "Multi-Dataset Evaluation Test" "tests/integration/test_multi_eval.py" true
run_integration_test "Cosine Scheduler Test" "tests/integration/test_cosine_scheduler.py" true
run_integration_test "Scheduler Info Test" "tests/integration/test_scheduler_info.py" false

# Summary
echo "========================================="
echo "INTEGRATION TEST SUMMARY"
echo "========================================="
echo -e "Passed:  ${GREEN}$PASSED${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Skipped: ${YELLOW}$SKIPPED${NC}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Some integration tests failed!${NC}"
    exit 1
else
    echo -e "${GREEN}All integration tests passed!${NC}"
fi

echo ""
echo "Integration test outputs saved in: tests/outputs/integration/"
echo ""
echo "To run all tests including slow ones:"
echo "  ./tests/scripts/run_integration_tests.sh"
echo ""
echo "To run only quick tests:"
echo "  ./tests/scripts/run_integration_tests.sh --quick"