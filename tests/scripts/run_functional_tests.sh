#!/bin/bash
# Run functional tests (model-specific tests)
# These tests may require model downloads and authentication

echo "========================================="
echo "Running Functional Tests (Model-Specific)"
echo "========================================="
echo ""
echo "Note: These tests may require:"
echo "  - HuggingFace authentication for gated models"
echo "  - Large model downloads (1-4GB per model)"
echo "  - GPU availability for optimal performance"
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
MODEL=""
if [ $# -gt 0 ]; then
    MODEL=$1
fi

# Function to run test
run_functional_test() {
    local model_name=$1
    local test_file=$2
    
    echo "========================================="
    echo "Testing: $model_name"
    echo "========================================="
    
    if [ ! -f "$test_file" ]; then
        echo -e "${YELLOW}⚠ Test file not found: $test_file${NC}"
        return 1
    fi
    
    echo "Running test..."
    $PYTHON_CMD "$test_file"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $model_name test completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ $model_name test failed${NC}"
        return 1
    fi
}

# Determine which tests to run
case "$MODEL" in
    "llama32"|"llama")
        echo "Running Llama 3.2 tests only..."
        run_functional_test "Llama 3.2" "tests/functional/test_llama32.py"
        ;;
    "qwen3"|"qwen")
        echo "Running Qwen3 tests only..."
        run_functional_test "Qwen3" "tests/functional/test_qwen3.py"
        ;;
    "")
        echo "Running all functional tests..."
        echo ""
        
        PASSED=0
        FAILED=0
        
        # Run Llama 3.2 test
        run_functional_test "Llama 3.2" "tests/functional/test_llama32.py"
        if [ $? -eq 0 ]; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
        
        echo ""
        
        # Run Qwen3 test
        run_functional_test "Qwen3" "tests/functional/test_qwen3.py"
        if [ $? -eq 0 ]; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
        fi
        
        # Summary
        echo ""
        echo "========================================="
        echo "FUNCTIONAL TEST SUMMARY"
        echo "========================================="
        echo -e "Passed: ${GREEN}$PASSED${NC}"
        echo -e "Failed: ${RED}$FAILED${NC}"
        
        if [ $FAILED -gt 0 ]; then
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Unknown model: $MODEL${NC}"
        echo ""
        echo "Usage: $0 [model]"
        echo ""
        echo "Available models:"
        echo "  llama32, llama  - Test Llama 3.2 models"
        echo "  qwen3, qwen     - Test Qwen3 models"
        echo ""
        echo "Run without arguments to test all models"
        exit 1
        ;;
esac

echo ""
echo "Functional test outputs saved in: tests/outputs/functional/"
echo ""
echo "To view TensorBoard for test runs:"
echo "  tensorboard --logdir tests/outputs/functional/"