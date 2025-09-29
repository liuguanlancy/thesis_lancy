#!/bin/bash
#
# GitHub SSH Setup Script
# Quickly configure SSH access for GitHub on Linux/macOS machines
#
# Usage: ./setup_github_ssh.sh
#

set -e  # Exit on error

# Configuration
GITHUB_USERNAME="liuguanlancy"
GITHUB_EMAIL="guanlancy.liu@gmail.com"
SSH_KEY_TYPE="ed25519"  # More secure and compact than RSA
SSH_KEY_COMMENT="$GITHUB_EMAIL"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Header
echo ""
echo "=========================================="
echo "GitHub SSH Setup Script"
echo "=========================================="
echo "Username: $GITHUB_USERNAME"
echo "Email: $GITHUB_EMAIL"
echo "=========================================="
echo ""

# Step 1: Check if SSH key already exists
SSH_KEY_PATH="$HOME/.ssh/id_${SSH_KEY_TYPE}"
if [ -f "$SSH_KEY_PATH" ]; then
    print_warning "SSH key already exists at $SSH_KEY_PATH"
    read -p "Do you want to create a new key? This will overwrite the existing one (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Using existing SSH key"
        SSH_KEY_EXISTS=true
    else
        SSH_KEY_EXISTS=false
    fi
else
    SSH_KEY_EXISTS=false
fi

# Step 2: Generate SSH key if needed
if [ "$SSH_KEY_EXISTS" = false ]; then
    print_info "Generating new SSH key..."
    ssh-keygen -t $SSH_KEY_TYPE -C "$SSH_KEY_COMMENT" -f "$SSH_KEY_PATH" -N ""
    print_success "SSH key generated at $SSH_KEY_PATH"
fi

# Step 3: Start SSH agent if not running
print_info "Starting SSH agent..."
eval "$(ssh-agent -s)" > /dev/null 2>&1
print_success "SSH agent started"

# Step 4: Add SSH key to agent
print_info "Adding SSH key to SSH agent..."
ssh-add "$SSH_KEY_PATH" 2>/dev/null || {
    # For macOS, might need to add to keychain
    if [[ "$OSTYPE" == "darwin"* ]]; then
        ssh-add --apple-use-keychain "$SSH_KEY_PATH" 2>/dev/null || ssh-add "$SSH_KEY_PATH"
    else
        ssh-add "$SSH_KEY_PATH"
    fi
}
print_success "SSH key added to agent"

# Step 5: Configure Git
print_info "Configuring Git..."
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"
print_success "Git configured with username and email"

# Step 6: Create/update SSH config
print_info "Configuring SSH..."
SSH_CONFIG="$HOME/.ssh/config"
if [ ! -f "$SSH_CONFIG" ]; then
    touch "$SSH_CONFIG"
    chmod 600 "$SSH_CONFIG"
fi

# Check if GitHub host already exists in config
if ! grep -q "Host github.com" "$SSH_CONFIG" 2>/dev/null; then
    print_info "Adding GitHub to SSH config..."
    cat >> "$SSH_CONFIG" << EOF

# GitHub
Host github.com
    HostName github.com
    User git
    IdentityFile $SSH_KEY_PATH
    AddKeysToAgent yes
EOF
    # Add macOS-specific keychain option if on macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "    UseKeychain yes" >> "$SSH_CONFIG"
    fi
    print_success "GitHub added to SSH config"
else
    print_info "GitHub already configured in SSH config"
fi

# Step 7: Display public key
echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
print_info "Your SSH public key (copy this to GitHub):"
echo ""
echo "----------------------------------------"
cat "${SSH_KEY_PATH}.pub"
echo "----------------------------------------"
echo ""

# Step 8: Instructions for adding key to GitHub
echo "To add this key to GitHub:"
echo ""
echo "1. Copy the SSH key above"
echo "2. Go to: https://github.com/settings/keys"
echo "3. Click 'New SSH key'"
echo "4. Give it a title (e.g., '$(hostname) - $(date +%Y-%m-%d)')"
echo "5. Paste the key and click 'Add SSH key'"
echo ""

# Step 9: Offer to test connection
read -p "Would you like to test the GitHub SSH connection? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Testing GitHub SSH connection..."
    echo ""
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        print_success "GitHub SSH connection successful!"
    else
        # GitHub returns exit code 1 even on success, so check the output
        ssh -T git@github.com 2>&1 | grep -q "Hi $GITHUB_USERNAME" && {
            print_success "GitHub SSH connection successful!"
        } || {
            print_warning "Connection test inconclusive. This is normal if you haven't added the key to GitHub yet."
        }
    fi
fi

# Step 10: Optional - Open GitHub settings in browser
if command -v xdg-open &> /dev/null; then
    # Linux
    read -p "Would you like to open GitHub SSH settings in your browser? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        xdg-open "https://github.com/settings/keys" 2>/dev/null &
    fi
elif command -v open &> /dev/null; then
    # macOS
    read -p "Would you like to open GitHub SSH settings in your browser? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "https://github.com/settings/keys"
    fi
fi

echo ""
echo "=========================================="
echo "Quick Commands Reference:"
echo "=========================================="
echo "Test connection:     ssh -T git@github.com"
echo "Clone with SSH:      git clone git@github.com:$GITHUB_USERNAME/repo-name.git"
echo "Change remote:       git remote set-url origin git@github.com:$GITHUB_USERNAME/repo-name.git"
echo "Show public key:     cat ${SSH_KEY_PATH}.pub"
echo "=========================================="
echo ""
print_success "Setup complete! Don't forget to add the SSH key to GitHub if you haven't already."
