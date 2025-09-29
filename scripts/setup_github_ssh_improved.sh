#!/bin/bash
#
# GitHub SSH Setup Script (Improved Version)
# Securely configure SSH access for GitHub on Linux/macOS/WSL
#
# Usage: 
#   ./setup_github_ssh_improved.sh [options]
#   
# Options:
#   -u, --username <name>    GitHub username (required if not set)
#   -e, --email <email>      GitHub email (required if not set)
#   -k, --key-name <name>    Custom SSH key name (default: id_ed25519)
#   -p, --passphrase         Use passphrase for SSH key (recommended)
#   -b, --backup             Backup existing keys before overwriting
#   -c, --clipboard          Copy public key to clipboard
#   -s, --skip-git           Skip Git configuration
#   -q, --quiet              Quiet mode (minimal output)
#   -v, --verbose            Verbose mode (detailed output)
#   -h, --help               Show this help message
#
# Examples:
#   ./setup_github_ssh_improved.sh -u johndoe -e john@example.com
#   ./setup_github_ssh_improved.sh --username johndoe --email john@example.com --passphrase
#   ./setup_github_ssh_improved.sh -b -c  # Backup and copy to clipboard
#

set -euo pipefail  # Exit on error, undefined variables, and pipe failures
IFS=$'\n\t'       # Set Internal Field Separator for better security

# shellcheck disable=SC2034  # Some variables are used in sourced files
SCRIPT_VERSION="2.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default Configuration
DEFAULT_USERNAME="zhaopku"
DEFAULT_EMAIL="zhaomeng.pku@outlook.com"
DEFAULT_KEY_TYPE="ed25519"  # More secure and compact than RSA
DEFAULT_KEY_NAME="id_${DEFAULT_KEY_TYPE}"

# Runtime Configuration
GITHUB_USERNAME="${DEFAULT_USERNAME}"
GITHUB_EMAIL="${DEFAULT_EMAIL}"
SSH_KEY_TYPE="${DEFAULT_KEY_TYPE}"
SSH_KEY_NAME="${DEFAULT_KEY_NAME}"
USE_PASSPHRASE=false
BACKUP_EXISTING=false
COPY_TO_CLIPBOARD=false
SKIP_GIT_CONFIG=false
QUIET_MODE=false
VERBOSE_MODE=false

# Colors for output (disabled in quiet mode)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging functions
log() {
    if [[ "${QUIET_MODE}" != "true" ]]; then
        echo -e "$@"
    fi
}

log_verbose() {
    if [[ "${VERBOSE_MODE}" == "true" ]]; then
        echo -e "${CYAN}[VERBOSE]${NC} $*" >&2
    fi
}

print_info() {
    log "${BLUE}[INFO]${NC} $1"
}

print_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_step() {
    log "\n${BOLD}${MAGENTA}==> $1${NC}"
}

# Error handling
trap 'handle_error $? $LINENO' ERR

handle_error() {
    local exit_code=$1
    local line_number=$2
    print_error "Script failed with exit code $exit_code at line $line_number"
    cleanup
    exit "$exit_code"
}

cleanup() {
    log_verbose "Performing cleanup..."
    # Add any cleanup tasks here
}

# Help function
show_help() {
    head -n 30 "$0" | grep '^#' | sed 's/^# *//' | sed 's/^#//'
    exit 0
}

# Validate email format
validate_email() {
    local email=$1
    local regex="^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    if [[ ! $email =~ $regex ]]; then
        print_error "Invalid email format: $email"
        return 1
    fi
    return 0
}

# Check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required commands
    for cmd in git ssh-keygen ssh-agent ssh-add ssh; do
        if ! command -v "$cmd" &> /dev/null; then
            missing_tools+=("$cmd")
        else
            log_verbose "Found: $cmd ($(command -v "$cmd"))"
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing_tools[*]}"
        print_info "Please install the missing tools and try again."
        exit 1
    fi
    
    print_success "All prerequisites met"
}

# Detect platform
detect_platform() {
    local platform="unknown"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if grep -q Microsoft /proc/version 2>/dev/null; then
            platform="wsl"
        else
            platform="linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        platform="macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
        platform="windows"
    fi
    
    log_verbose "Detected platform: $platform"
    echo "$platform"
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--username)
                GITHUB_USERNAME="$2"
                shift 2
                ;;
            -e|--email)
                GITHUB_EMAIL="$2"
                shift 2
                ;;
            -k|--key-name)
                SSH_KEY_NAME="$2"
                shift 2
                ;;
            -p|--passphrase)
                USE_PASSPHRASE=true
                shift
                ;;
            -b|--backup)
                BACKUP_EXISTING=true
                shift
                ;;
            -c|--clipboard)
                COPY_TO_CLIPBOARD=true
                shift
                ;;
            -s|--skip-git)
                SKIP_GIT_CONFIG=true
                shift
                ;;
            -q|--quiet)
                QUIET_MODE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE_MODE=true
                shift
                ;;
            -h|--help)
                show_help
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                ;;
        esac
    done
}

# Ensure SSH directory exists with proper permissions
ensure_ssh_directory() {
    local ssh_dir="$HOME/.ssh"
    
    if [[ ! -d "$ssh_dir" ]]; then
        print_info "Creating SSH directory..."
        mkdir -p "$ssh_dir"
    fi
    
    # Set proper permissions
    chmod 700 "$ssh_dir"
    log_verbose "SSH directory permissions set to 700"
}

# Backup existing SSH keys
backup_ssh_keys() {
    local ssh_key_path="$1"
    local backup_dir="$HOME/.ssh/backups"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    if [[ -f "$ssh_key_path" ]] || [[ -f "${ssh_key_path}.pub" ]]; then
        print_info "Backing up existing SSH keys..."
        
        mkdir -p "$backup_dir"
        
        if [[ -f "$ssh_key_path" ]]; then
            cp "$ssh_key_path" "$backup_dir/$(basename "$ssh_key_path").${timestamp}"
            log_verbose "Backed up private key to $backup_dir"
        fi
        
        if [[ -f "${ssh_key_path}.pub" ]]; then
            cp "${ssh_key_path}.pub" "$backup_dir/$(basename "$ssh_key_path").pub.${timestamp}"
            log_verbose "Backed up public key to $backup_dir"
        fi
        
        print_success "Existing keys backed up to $backup_dir"
    fi
}

# Generate SSH key
generate_ssh_key() {
    local ssh_key_path="$HOME/.ssh/$SSH_KEY_NAME"
    local key_exists=false
    
    print_step "Setting up SSH key..."
    
    # Check if key exists
    if [[ -f "$ssh_key_path" ]]; then
        print_warning "SSH key already exists at $ssh_key_path"
        
        if [[ "${BACKUP_EXISTING}" == "true" ]]; then
            backup_ssh_keys "$ssh_key_path"
        fi
        
        read -p "Do you want to create a new key? This will overwrite the existing one (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Using existing SSH key"
            key_exists=true
        fi
    fi
    
    # Generate new key if needed
    if [[ "$key_exists" == "false" ]]; then
        print_info "Generating new SSH key..."
        
        local ssh_options=("-t" "$SSH_KEY_TYPE" "-C" "$GITHUB_EMAIL" "-f" "$ssh_key_path")
        
        if [[ "${USE_PASSPHRASE}" == "true" ]]; then
            print_info "You will be prompted to enter a passphrase (recommended for security)"
            ssh-keygen "${ssh_options[@]}"
        else
            ssh_options+=("-N" "")
            ssh-keygen "${ssh_options[@]}"
        fi
        
        print_success "SSH key generated at $ssh_key_path"
    fi
    
    # Set proper permissions
    chmod 600 "$ssh_key_path"
    chmod 644 "${ssh_key_path}.pub"
    
    echo "$ssh_key_path"
}

# Configure SSH agent
configure_ssh_agent() {
    local ssh_key_path="$1"
    local platform=$(detect_platform)
    
    print_step "Configuring SSH agent..."
    
    # Start SSH agent if not running
    if ! pgrep -x ssh-agent > /dev/null; then
        print_info "Starting SSH agent..."
        eval "$(ssh-agent -s)" > /dev/null 2>&1
        log_verbose "SSH agent started with PID $SSH_AGENT_PID"
    else
        log_verbose "SSH agent already running"
    fi
    
    # Add SSH key to agent
    print_info "Adding SSH key to SSH agent..."
    
    case "$platform" in
        macos)
            # macOS-specific: use keychain
            if ssh-add --apple-use-keychain "$ssh_key_path" 2>/dev/null; then
                print_success "SSH key added to agent and keychain"
            else
                ssh-add "$ssh_key_path"
                print_success "SSH key added to agent"
            fi
            ;;
        *)
            ssh-add "$ssh_key_path"
            print_success "SSH key added to agent"
            ;;
    esac
}

# Configure Git
configure_git() {
    if [[ "${SKIP_GIT_CONFIG}" == "true" ]]; then
        log_verbose "Skipping Git configuration"
        return
    fi
    
    print_step "Configuring Git..."
    
    # Check current Git config
    local current_name=$(git config --global user.name 2>/dev/null || echo "")
    local current_email=$(git config --global user.email 2>/dev/null || echo "")
    
    if [[ "$current_name" == "$GITHUB_USERNAME" ]] && [[ "$current_email" == "$GITHUB_EMAIL" ]]; then
        print_info "Git already configured correctly"
    else
        git config --global user.name "$GITHUB_USERNAME"
        git config --global user.email "$GITHUB_EMAIL"
        print_success "Git configured with username and email"
    fi
    
    # Additional useful Git configurations
    git config --global init.defaultBranch main
    log_verbose "Set default branch to 'main'"
}

# Configure SSH config file
configure_ssh_config() {
    local ssh_key_path="$1"
    local platform=$(detect_platform)
    
    print_step "Configuring SSH..."
    
    local ssh_config="$HOME/.ssh/config"
    
    # Create config file if it doesn't exist
    if [[ ! -f "$ssh_config" ]]; then
        touch "$ssh_config"
        chmod 600 "$ssh_config"
        log_verbose "Created SSH config file"
    fi
    
    # Check if GitHub host already exists in config
    if grep -q "Host github.com" "$ssh_config" 2>/dev/null; then
        print_info "GitHub already configured in SSH config"
        
        # Optionally update the existing configuration
        read -p "Do you want to update the existing GitHub SSH config? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # Remove existing GitHub configuration
            sed -i.bak '/^# GitHub$/,/^$/d' "$ssh_config" 2>/dev/null || \
            sed -i '' '/^# GitHub$/,/^$/d' "$ssh_config" 2>/dev/null || true
            
            sed -i.bak '/^Host github.com$/,/^$/d' "$ssh_config" 2>/dev/null || \
            sed -i '' '/^Host github.com$/,/^$/d' "$ssh_config" 2>/dev/null || true
        else
            return
        fi
    fi
    
    print_info "Adding GitHub to SSH config..."
    
    # Build SSH config entry
    local config_entry="
# GitHub
Host github.com
    HostName github.com
    User git
    IdentityFile $ssh_key_path
    AddKeysToAgent yes"
    
    # Add platform-specific options
    case "$platform" in
        macos)
            config_entry="${config_entry}
    UseKeychain yes"
            ;;
        wsl|linux)
            config_entry="${config_entry}
    IdentitiesOnly yes"
            ;;
    esac
    
    echo "$config_entry" >> "$ssh_config"
    print_success "GitHub added to SSH config"
}

# Copy to clipboard
copy_to_clipboard() {
    local ssh_key_path="$1"
    local platform=$(detect_platform)
    local public_key=$(cat "${ssh_key_path}.pub")
    
    case "$platform" in
        macos)
            echo "$public_key" | pbcopy
            print_success "SSH public key copied to clipboard"
            ;;
        linux|wsl)
            if command -v xclip &> /dev/null; then
                echo "$public_key" | xclip -selection clipboard
                print_success "SSH public key copied to clipboard"
            elif command -v xsel &> /dev/null; then
                echo "$public_key" | xsel --clipboard
                print_success "SSH public key copied to clipboard"
            else
                print_warning "No clipboard tool found. Install xclip or xsel to enable clipboard support"
            fi
            ;;
        *)
            print_warning "Clipboard copy not supported on this platform"
            ;;
    esac
}

# Test GitHub connection
test_github_connection() {
    print_step "Testing GitHub SSH connection..."
    
    local test_output
    test_output=$(ssh -T git@github.com 2>&1 || true)
    
    if echo "$test_output" | grep -q "successfully authenticated"; then
        print_success "GitHub SSH connection successful!"
        return 0
    elif echo "$test_output" | grep -q "Hi $GITHUB_USERNAME"; then
        print_success "GitHub SSH connection successful!"
        return 0
    else
        print_warning "Connection test inconclusive. This is normal if you haven't added the key to GitHub yet."
        log_verbose "Test output: $test_output"
        return 1
    fi
}

# Display completion information
display_completion() {
    local ssh_key_path="$1"
    local platform=$(detect_platform)
    
    log ""
    log "=========================================="
    log "${GREEN}${BOLD}SETUP COMPLETE!${NC}"
    log "=========================================="
    log ""
    
    print_info "Your SSH public key:"
    log ""
    log "${CYAN}----------------------------------------${NC}"
    cat "${ssh_key_path}.pub"
    log "${CYAN}----------------------------------------${NC}"
    log ""
    
    log "${BOLD}To add this key to GitHub:${NC}"
    log ""
    log "1. Copy the SSH key above"
    log "2. Go to: ${BLUE}https://github.com/settings/keys${NC}"
    log "3. Click '${BOLD}New SSH key${NC}'"
    log "4. Give it a title (e.g., '${YELLOW}$(hostname) - $(date +%Y-%m-%d)${NC}')"
    log "5. Paste the key and click '${BOLD}Add SSH key${NC}'"
    log ""
    
    # Offer to open browser
    if [[ "${QUIET_MODE}" != "true" ]]; then
        read -p "Would you like to open GitHub SSH settings in your browser? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            case "$platform" in
                macos)
                    open "https://github.com/settings/keys"
                    ;;
                linux|wsl)
                    if command -v xdg-open &> /dev/null; then
                        xdg-open "https://github.com/settings/keys" 2>/dev/null &
                    fi
                    ;;
            esac
        fi
    fi
    
    log ""
    log "=========================================="
    log "${BOLD}Quick Commands Reference:${NC}"
    log "=========================================="
    log "Test connection:     ${YELLOW}ssh -T git@github.com${NC}"
    log "Clone with SSH:      ${YELLOW}git clone git@github.com:$GITHUB_USERNAME/repo-name.git${NC}"
    log "Change remote:       ${YELLOW}git remote set-url origin git@github.com:$GITHUB_USERNAME/repo-name.git${NC}"
    log "Show public key:     ${YELLOW}cat ${ssh_key_path}.pub${NC}"
    log "List SSH keys:       ${YELLOW}ssh-add -l${NC}"
    log "=========================================="
    log ""
}

# Main function
main() {
    # Parse arguments
    parse_arguments "$@"
    
    # Validate inputs
    if ! validate_email "$GITHUB_EMAIL"; then
        exit 1
    fi
    
    # Show header
    if [[ "${QUIET_MODE}" != "true" ]]; then
        log ""
        log "=========================================="
        log "${BOLD}GitHub SSH Setup Script v${SCRIPT_VERSION}${NC}"
        log "=========================================="
        log "Username: ${YELLOW}$GITHUB_USERNAME${NC}"
        log "Email: ${YELLOW}$GITHUB_EMAIL${NC}"
        log "Key Type: ${YELLOW}$SSH_KEY_TYPE${NC}"
        log "Key Name: ${YELLOW}$SSH_KEY_NAME${NC}"
        log "=========================================="
        log ""
    fi
    
    # Run setup steps
    check_prerequisites
    ensure_ssh_directory
    
    local ssh_key_path
    ssh_key_path=$(generate_ssh_key)
    
    configure_ssh_agent "$ssh_key_path"
    configure_git
    configure_ssh_config "$ssh_key_path"
    
    # Copy to clipboard if requested
    if [[ "${COPY_TO_CLIPBOARD}" == "true" ]]; then
        copy_to_clipboard "$ssh_key_path"
    fi
    
    # Display completion information
    display_completion "$ssh_key_path"
    
    # Test connection
    if [[ "${QUIET_MODE}" != "true" ]]; then
        read -p "Would you like to test the GitHub SSH connection? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            test_github_connection
        fi
    fi
    
    print_success "Setup complete! Don't forget to add the SSH key to GitHub if you haven't already."
}

# Run main function
main "$@"