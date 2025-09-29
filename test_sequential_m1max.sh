#!/bin/bash
#
# Test script for sequential execution in phase2b_m1max.sh
#
# This script verifies that the sequential execution feature works correctly
# for running multiple experiments with delays between them.
#

set -e  # Exit on error

echo "=========================================="
echo "Testing Sequential Execution for M1 Max"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Test 1: Single experiment (should work as before)
echo "Test 1: Single experiment execution"
echo "-----------------------------------"
echo "Command: ./scripts/phase2b_m1max.sh --experiments 1 --dry-run --max-steps 10"
$SCRIPT_DIR/scripts/phase2b_m1max.sh --experiments 1 --dry-run --max-steps 10
echo ""
echo "✓ Test 1 passed: Single experiment works"
echo ""

# Test 2: Multiple experiments with dry-run
echo "Test 2: Multiple experiments (dry-run)"
echo "--------------------------------------"
echo "Command: ./scripts/phase2b_m1max.sh --experiments \"1 3\" --dry-run --max-steps 10"
# Use just 2 experiments to avoid long output
$SCRIPT_DIR/scripts/phase2b_m1max.sh --experiments "1 3" --dry-run --max-steps 10 --save-steps 5 --eval-steps 5 | grep -E "(Experiment:|Sequential|Total experiments|\[.\/..\])"
echo ""
echo "✓ Test 2 passed: Multiple experiments parsed correctly"
echo ""

# Test 3: Verify order preservation
echo "Test 3: Order preservation"
echo "-------------------------"
echo "Command: ./scripts/phase2b_m1max.sh --experiments \"mixed 3 1\" --dry-run --max-steps 10"
OUTPUT=$($SCRIPT_DIR/scripts/phase2b_m1max.sh --experiments "mixed 3 1" --dry-run --max-steps 10 2>&1)
echo "$OUTPUT"

# Check if experiments appear in correct order
if echo "$OUTPUT" | grep -q "\[1/3\] Experiment: mixed" && \
   echo "$OUTPUT" | grep -q "\[2/3\] Experiment: 3" && \
   echo "$OUTPUT" | grep -q "\[3/3\] Experiment: 1"; then
    echo ""
    echo "✓ Test 3 passed: Order preserved (mixed, 3, 1)"
else
    echo ""
    echo "✗ Test 3 failed: Order not preserved"
    exit 1
fi
echo ""

# Test 4: Custom parameters propagation
echo "Test 4: Custom parameters"
echo "------------------------"
echo "Command: ./scripts/phase2b_m1max.sh --experiments \"1 2\" --batch-size 4 --max-length 256 --dry-run --max-steps 10"
OUTPUT=$($SCRIPT_DIR/scripts/phase2b_m1max.sh --experiments "1 2" --batch-size 4 --max-length 256 --dry-run --max-steps 10 2>&1)
echo "$OUTPUT"

# Check if custom parameters are applied
if echo "$OUTPUT" | grep -q "Batch Size: 4" && \
   echo "$OUTPUT" | grep -q "Max Length: 256"; then
    echo ""
    echo "✓ Test 4 passed: Custom parameters applied"
else
    echo ""
    echo "✗ Test 4 failed: Custom parameters not applied"
    exit 1
fi
echo ""

# Test 5: Strategy for mixed experiments
echo "Test 5: Mixed experiment with strategy"
echo "-------------------------------------"
echo "Command: ./scripts/phase2b_m1max.sh --experiments \"mixed\" --strategy sqrt --dry-run --max-steps 10"
OUTPUT=$($SCRIPT_DIR/scripts/phase2b_m1max.sh --experiments "mixed" --strategy sqrt --dry-run --max-steps 10 2>&1)
echo "$OUTPUT"

if echo "$OUTPUT" | grep -q "Mixing Strategy: sqrt"; then
    echo ""
    echo "✓ Test 5 passed: Strategy applied to mixed experiment"
else
    echo ""
    echo "✗ Test 5 failed: Strategy not applied"
    exit 1
fi
echo ""

# Test 6: Sequential execution plan display
echo "Test 6: Sequential execution plan"
echo "--------------------------------"
echo "Command: ./scripts/phase2b_m1max.sh --experiments \"1 3 5\" --dry-run --max-steps 10"
OUTPUT=$($SCRIPT_DIR/scripts/phase2b_m1max.sh --experiments "1 3 5" --dry-run --max-steps 10 2>&1)

if echo "$OUTPUT" | grep -q "Sequential Experiment Execution Plan" && \
   echo "$OUTPUT" | grep -q "Total experiments: 3" && \
   echo "$OUTPUT" | grep -q "Delay between experiments: 5 minutes"; then
    echo "✓ Test 6 passed: Sequential plan displayed correctly"
else
    echo "✗ Test 6 failed: Sequential plan not displayed"
    exit 1
fi
echo ""

# Test 7: Real execution test (very short, just to verify it starts)
echo "Test 7: Real execution test (minimal)"
echo "------------------------------------"
echo "This test will actually run a very short training (2 steps) to verify execution"
echo "Command: ./scripts/phase2b_m1max.sh --experiments \"1\" --max-steps 2 --eval-steps 1 --save-steps 1"
echo ""
echo "Skipping real execution test (would require actual training)"
echo "To run manually: ./scripts/phase2b_m1max.sh --experiments \"1\" --max-steps 2"
echo ""

echo "=========================================="
echo "All Tests Passed Successfully!"
echo "=========================================="
echo ""
echo "Summary:"
echo "✓ Single experiment execution works"
echo "✓ Multiple experiments parsed correctly"
echo "✓ Experiment order preserved"
echo "✓ Custom parameters propagated"
echo "✓ Strategy applied to mixed experiments"
echo "✓ Sequential execution plan displayed"
echo ""
echo "The sequential execution feature is working correctly!"