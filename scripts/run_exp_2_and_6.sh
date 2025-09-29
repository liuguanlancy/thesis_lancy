#!/bin/bash
#
# Run FinGPT Sentiment (Exp 2) and SEC Reports (Exp 6) experiments sequentially
# Each experiment runs 3 model sizes with 5-minute intervals
#

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=============================================="
echo "Running Experiments 2 and 6 Sequentially"
echo "=============================================="
echo "Schedule:"
echo "  1. FinGPT Sentiment (Exp 2) - 3 models"
echo "  2. SEC Reports (Exp 6) - 3 models"
echo ""
echo "Total estimated time: ~7 hours"
echo "=============================================="
echo ""

# Record start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Starting experiments at: $START_TIME"
echo ""

# Run Experiment 2: FinGPT Sentiment
echo "=============================================="
echo "[EXPERIMENT 2] Starting FinGPT Sentiment"
echo "=============================================="
"$SCRIPT_DIR/2_run_fingpt_sentiment_experiments.sh"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Experiment 2 (FinGPT Sentiment) completed successfully!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
else
    echo ""
    echo "❌ ERROR: Experiment 2 (FinGPT Sentiment) failed!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
    exit 1
fi

# Wait 10 minutes before starting next experiment
echo ""
echo "=============================================="
echo "Waiting 10 minutes before Experiment 6..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Next experiment will start at: $(date -v +10M +"%Y-%m-%d %H:%M:%S")"
else
    echo "Next experiment will start at: $(date -d "+10 minutes" +"%Y-%m-%d %H:%M:%S")"
fi
echo "(Press Ctrl+C to cancel)"
echo "=============================================="
sleep 600  # 10 minutes = 600 seconds

# Run Experiment 6: SEC Reports
echo ""
echo "=============================================="
echo "[EXPERIMENT 6] Starting SEC Reports"
echo "=============================================="
"$SCRIPT_DIR/6_run_sec_reports_experiments.sh"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Experiment 6 (SEC Reports) completed successfully!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
else
    echo ""
    echo "❌ ERROR: Experiment 6 (SEC Reports) failed!"
    echo "Time: $(date +"%Y-%m-%d %H:%M:%S")"
    exit 1
fi

# Final summary
echo ""
echo "=============================================="
echo "🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=============================================="
echo "Start time:      $START_TIME"
echo "Completion time: $(date +"%Y-%m-%d %H:%M:%S")"
echo ""
echo "Summary of completed experiments:"
echo "  ✅ Experiment 2: FinGPT Sentiment (3 models)"
echo "     - 0.6B model"
echo "     - 4B model"
echo "     - 1.7B model"
echo ""
echo "  ✅ Experiment 6: SEC Reports (3 models)"
echo "     - 0.6B model"
echo "     - 4B model"
echo "     - 1.7B model"
echo ""
echo "Check the following directories for results:"
echo "  - runs/*_qwen3_0.6b/fingpt_sentiment/"
echo "  - runs/*_qwen3_1.7b/fingpt_sentiment/"
echo "  - runs/*_qwen3_4b/fingpt_sentiment/"
echo "  - runs/*_qwen3_0.6b/sec_reports/"
echo "  - runs/*_qwen3_1.7b/sec_reports/"
echo "  - runs/*_qwen3_4b/sec_reports/"
echo "=============================================="