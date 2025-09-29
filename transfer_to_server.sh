#!/bin/bash
# Script to transfer the lancy_thesis project to remote SSH server
# Server details provided by user

# Configuration
LOCAL_DIR="/Users/mengzhao/PycharmProjects/lancy_thesis"
REMOTE_HOST="root@connect.bjb1.seetacloud.com"
REMOTE_PORT="22306"
REMOTE_PASS="zHrG2qiHK2TK"
REMOTE_DIR="/root/lancy_thesis"  # Destination on remote server

echo "=========================================="
echo "Transferring Lancy Thesis to Remote Server"
echo "=========================================="
echo "Source: $LOCAL_DIR"
echo "Destination: $REMOTE_HOST:$REMOTE_DIR"
echo "Port: $REMOTE_PORT"
echo ""
echo "EXCLUDING: runs/, debug_grpo/, test_runs/, tests/outputs/, .git/"
echo "=========================================="
echo ""

# Check actual size to transfer
echo "Calculating transfer size (excluding large directories)..."
cd "$LOCAL_DIR"
TRANSFER_FILES=$(find . -type f \
    -not -path "./runs/*" \
    -not -path "./debug_grpo/*" \
    -not -path "./test_runs/*" \
    -not -path "./tests/outputs/*" \
    -not -path "./.git/*" \
    -not -name "*.pyc" \
    -not -path "*/__pycache__/*" | wc -l)

echo "Files to transfer: $TRANSFER_FILES files"
echo "Estimated size: < 100MB (source code and configs only)"
echo ""
echo "=========================================="
echo ""

# RECOMMENDED METHOD: Using rsync with exclusions
echo "RECOMMENDED: Using rsync (supports resume and progress)"
echo "-------------------------------------------------------"
echo "Copy and run this command:"
echo ""
cat << 'EOF'
rsync -avzP \
    --exclude='runs/' \
    --exclude='debug_grpo/' \
    --exclude='test_runs/' \
    --exclude='tests/outputs/' \
    --exclude='.git/' \
    --exclude='*.pyc' \
    --exclude='__pycache__/' \
    --exclude='.DS_Store' \
    -e "ssh -p 22306" \
    /Users/mengzhao/PycharmProjects/lancy_thesis/ \
    root@connect.bjb1.seetacloud.com:/root/lancy_thesis/
EOF

echo ""
echo "Password when prompted: zHrG2qiHK2TK"
echo ""
echo "=========================================="
echo ""

# ALTERNATIVE: Using tar + ssh (fast for many small files)
echo "ALTERNATIVE: Using tar + ssh (one-shot transfer)"
echo "-------------------------------------------------"
echo "Copy and run this command:"
echo ""
cat << 'EOF'
cd /Users/mengzhao/PycharmProjects && \
tar czf - lancy_thesis \
    --exclude='lancy_thesis/runs' \
    --exclude='lancy_thesis/debug_grpo' \
    --exclude='lancy_thesis/test_runs' \
    --exclude='lancy_thesis/tests/outputs' \
    --exclude='lancy_thesis/.git' \
    --exclude='*.pyc' \
    --exclude='*/__pycache__' | \
ssh -p 22306 root@connect.bjb1.seetacloud.com \
    "mkdir -p /root && cd /root && tar xzf -"
EOF

echo ""
echo "Password when prompted: zHrG2qiHK2TK"
echo ""
echo "=========================================="
echo ""

# Setup commands for after transfer
echo "AFTER TRANSFER - Server Setup Commands:"
echo "---------------------------------------"
echo ""
echo "1. Connect to server:"
echo "   ssh -p 22306 root@connect.bjb1.seetacloud.com"
echo "   Password: zHrG2qiHK2TK"
echo ""
echo "2. Navigate to project:"
echo "   cd /root/lancy_thesis"
echo ""
echo "3. Create required directories:"
echo "   mkdir -p runs tests/outputs"
echo ""
echo "4. Set up Python environment (if conda available):"
echo "   conda env create -f lancy.yml"
echo "   conda activate lancy"
echo ""
echo "   Or with pip:"
echo "   pip install torch transformers datasets accelerate peft"
echo ""
echo "5. Test the installation:"
echo "   python train.py --help"
echo ""
echo "6. Make scripts executable:"
echo "   chmod +x scripts/*.sh"
echo "   chmod +x tests/scripts/*.sh"
echo ""
echo "=========================================="
echo "IMPORTANT NOTES:"
echo "=========================================="
echo "✓ Source code, configs, and scripts will be transferred"
echo "✓ Training outputs (runs/) will NOT be transferred - saves 11GB"
echo "✓ Test outputs will NOT be transferred"
echo "✓ Git history will NOT be transferred"
echo ""
echo "The transfer should be < 100MB and take only a few minutes"
echo "=========================================="