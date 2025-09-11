#!/bin/bash
# JJongal-AI Dependencies Installation Script
# Automated installation of system dependencies, Python packages, and development tools

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Script configuration
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="JJongal-AI Dependencies Installation Script"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Platform detection
detect_platform() {
    case "$(uname -s)" in
        Linux*)
            if [[ -f /etc/debian_version ]]; then
                PLATFORM="debian"
                PACKAGE_MANAGER="apt"
            elif [[ -f /etc/redhat-release ]]; then
                PLATFORM="redhat"
                PACKAGE_MANAGER="yum"
            elif [[ -f /etc/arch-release ]]; then
                PLATFORM="arch"
                PACKAGE_MANAGER="pacman"
            else
                PLATFORM="linux"
                PACKAGE_MANAGER="unknown"
            fi
            ;;
        Darwin*)
            PLATFORM="macos"
            PACKAGE_MANAGER="brew"
            ;;
        CYGWIN*|MINGW32*|MSYS*|MINGW*)
            PLATFORM="windows"
            PACKAGE_MANAGER="choco"
            ;;
        *)
            PLATFORM="unknown"
            PACKAGE_MANAGER="unknown"
            ;;
    esac
}

# Logging functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
${SCRIPT_NAME} v${SCRIPT_VERSION}

USAGE:
    $0 [OPTIONS] [COMPONENT]

COMPONENTS:
    all             Install all dependencies (default)
    system          System packages only
    python          Python and pip dependencies
    docker          Docker and Docker Compose
    audio           Audio processing dependencies
    ml              Machine learning dependencies
    dev             Development tools
    optional        Optional enhancement tools

OPTIONS:
    --platform PLATFORM     Force platform detection (macos, debian, redhat, arch)
    --skip-system           Skip system package installation
    --skip-python           Skip Python dependency installation
    --skip-docker           Skip Docker installation
    --skip-optional         Skip optional dependencies
    --upgrade               Upgrade existing packages
    --force                 Force installation without confirmation
    --dry-run               Show what would be installed
    --debug                 Enable debug output
    --user                  Install Python packages as user (--user flag)
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Install all dependencies
    $0 system              # Install system packages only
    $0 python --user       # Install Python packages for user
    $0 --skip-optional     # Skip optional dependencies
    $0 --dry-run           # Preview installation

SUPPORTED PLATFORMS:
    - macOS (Homebrew)
    - Ubuntu/Debian (APT)
    - CentOS/RHEL (YUM)
    - Arch Linux (Pacman)
    - Windows (Chocolatey)

EOF
}

# Parse command line arguments
parse_args() {
    COMPONENT="all"
    SKIP_SYSTEM=false
    SKIP_PYTHON=false
    SKIP_DOCKER=false
    SKIP_OPTIONAL=false
    UPGRADE=false
    FORCE=false
    DRY_RUN=false
    DEBUG=false
    USER_INSTALL=false
    FORCE_PLATFORM=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            all|system|python|docker|audio|ml|dev|optional)
                COMPONENT="$1"
                shift
                ;;
            --platform)
                FORCE_PLATFORM="$2"
                shift 2
                ;;
            --skip-system)
                SKIP_SYSTEM=true
                shift
                ;;
            --skip-python)
                SKIP_PYTHON=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER=true
                shift
                ;;
            --skip-optional)
                SKIP_OPTIONAL=true
                shift
                ;;
            --upgrade)
                UPGRADE=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            --user)
                USER_INSTALL=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system packages
install_system_packages() {
    if [[ "${SKIP_SYSTEM}" == "true" ]]; then
        log "Skipping system package installation"
        return
    fi
    
    step "Installing system dependencies..."
    
    # Common system packages needed by JJongal-AI
    case "${PLATFORM}" in
        "macos")
            install_macos_packages
            ;;
        "debian")
            install_debian_packages
            ;;
        "redhat")
            install_redhat_packages
            ;;
        "arch")
            install_arch_packages
            ;;
        "windows")
            install_windows_packages
            ;;
        *)
            warn "Unsupported platform: ${PLATFORM}"
            warn "Please install dependencies manually"
            ;;
    esac
    
    success "System packages installation completed"
}

# macOS package installation
install_macos_packages() {
    log "Installing macOS dependencies with Homebrew..."
    
    # Check if Homebrew is installed
    if ! command_exists brew; then
        log "Installing Homebrew..."
        if [[ "${DRY_RUN}" == "false" ]]; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
    fi
    
    local packages=(
        # Basic tools
        "curl"
        "wget" 
        "git"
        "jq"
        
        # Python
        "python@3.10"
        "python@3.11"
        
        # Audio processing
        "ffmpeg"
        "portaudio"
        "libsndfile"
        
        # Database
        "sqlite"
        "postgresql"
        "redis"
        
        # Development tools
        "node"
        "yarn"
        "nginx"
    )
    
    for package in "${packages[@]}"; do
        if [[ "${DRY_RUN}" == "true" ]]; then
            log "[DRY-RUN] Would install: ${package}"
        else
            if ! brew list "${package}" &>/dev/null; then
                log "Installing ${package}..."
                brew install "${package}" || warn "Failed to install ${package}"
            else
                debug "${package} already installed"
                if [[ "${UPGRADE}" == "true" ]]; then
                    brew upgrade "${package}" || debug "No upgrade available for ${package}"
                fi
            fi
        fi
    done
    
    # Install optional packages
    if [[ "${SKIP_OPTIONAL}" == "false" ]]; then
        local optional_packages=(
            "htop"
            "tree"
            "neofetch"
            "docker"
            "docker-compose"
        )
        
        for package in "${optional_packages[@]}"; do
            if [[ "${DRY_RUN}" == "true" ]]; then
                log "[DRY-RUN] Would install optional: ${package}"
            else
                brew install "${package}" 2>/dev/null || debug "Optional package ${package} installation skipped"
            fi
        done
    fi
}

# Debian/Ubuntu package installation
install_debian_packages() {
    log "Installing Debian/Ubuntu dependencies with APT..."
    
    # Update package list
    if [[ "${DRY_RUN}" == "false" ]]; then
        sudo apt update
    fi
    
    local packages=(
        # Basic tools
        "curl"
        "wget"
        "git"
        "jq"
        "build-essential"
        
        # Python
        "python3"
        "python3-pip"
        "python3-venv"
        "python3-dev"
        
        # Audio processing
        "ffmpeg"
        "libportaudio2"
        "portaudio19-dev"
        "libsndfile1"
        "libsndfile1-dev"
        
        # Database
        "sqlite3"
        "libsqlite3-dev"
        "postgresql-client"
        "redis-tools"
        
        # Development
        "nodejs"
        "npm"
        "nginx"
        
        # System utilities
        "htop"
        "tree"
        "neofetch"
    )
    
    for package in "${packages[@]}"; do
        if [[ "${DRY_RUN}" == "true" ]]; then
            log "[DRY-RUN] Would install: ${package}"
        else
            log "Installing ${package}..."
            sudo apt install -y "${package}" || warn "Failed to install ${package}"
        fi
    done
    
    # Install Docker if not skipped
    if [[ "${SKIP_DOCKER}" == "false" ]]; then
        install_docker_debian
    fi
}

# RedHat/CentOS package installation
install_redhat_packages() {
    log "Installing RedHat/CentOS dependencies with YUM..."
    
    local packages=(
        # Basic tools
        "curl"
        "wget"
        "git"
        "jq"
        "gcc"
        "gcc-c++"
        "make"
        
        # Python
        "python3"
        "python3-pip"
        "python3-devel"
        
        # Audio processing
        "ffmpeg"
        "portaudio-devel"
        
        # Database
        "sqlite"
        "sqlite-devel"
        "postgresql"
        "redis"
        
        # Development
        "nodejs"
        "npm"
        "nginx"
    )
    
    for package in "${packages[@]}"; do
        if [[ "${DRY_RUN}" == "true" ]]; then
            log "[DRY-RUN] Would install: ${package}"
        else
            log "Installing ${package}..."
            sudo yum install -y "${package}" || warn "Failed to install ${package}"
        fi
    done
}

# Arch Linux package installation  
install_arch_packages() {
    log "Installing Arch Linux dependencies with Pacman..."
    
    # Update package database
    if [[ "${DRY_RUN}" == "false" ]]; then
        sudo pacman -Sy
    fi
    
    local packages=(
        # Basic tools
        "curl"
        "wget" 
        "git"
        "jq"
        "base-devel"
        
        # Python
        "python"
        "python-pip"
        
        # Audio processing
        "ffmpeg"
        "portaudio"
        
        # Database
        "sqlite"
        "postgresql"
        "redis"
        
        # Development
        "nodejs"
        "npm"
        "nginx"
    )
    
    for package in "${packages[@]}"; do
        if [[ "${DRY_RUN}" == "true" ]]; then
            log "[DRY-RUN] Would install: ${package}"
        else
            log "Installing ${package}..."
            sudo pacman -S --noconfirm "${package}" || warn "Failed to install ${package}"
        fi
    done
}

# Windows package installation
install_windows_packages() {
    log "Installing Windows dependencies with Chocolatey..."
    
    # Check if Chocolatey is installed
    if ! command_exists choco; then
        error "Chocolatey is not installed. Please install Chocolatey first:"
        error "https://chocolatey.org/install"
        exit 1
    fi
    
    local packages=(
        "python"
        "git"
        "nodejs"
        "docker-desktop"
        "ffmpeg"
        "sqlite"
        "jq"
    )
    
    for package in "${packages[@]}"; do
        if [[ "${DRY_RUN}" == "true" ]]; then
            log "[DRY-RUN] Would install: ${package}"
        else
            log "Installing ${package}..."
            choco install -y "${package}" || warn "Failed to install ${package}"
        fi
    done
}

# Install Docker on Debian/Ubuntu
install_docker_debian() {
    if command_exists docker; then
        log "Docker already installed"
        return
    fi
    
    log "Installing Docker..."
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would install Docker"
        return
    fi
    
    # Add Docker GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Update and install Docker
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker "${USER}"
    
    success "Docker installed. Please log out and back in for group changes to take effect."
}

# Install Python dependencies
install_python_dependencies() {
    if [[ "${SKIP_PYTHON}" == "true" ]]; then
        log "Skipping Python dependency installation"
        return
    fi
    
    step "Installing Python dependencies..."
    
    # Detect Python command
    local python_cmd=""
    for cmd in python3.11 python3.10 python3 python; do
        if command_exists "${cmd}"; then
            python_cmd="${cmd}"
            break
        fi
    done
    
    if [[ -z "${python_cmd}" ]]; then
        error "Python not found. Please install Python first."
        exit 1
    fi
    
    log "Using Python: ${python_cmd} ($(${python_cmd} --version))"
    
    # Upgrade pip
    log "Upgrading pip..."
    if [[ "${DRY_RUN}" == "false" ]]; then
        local pip_flags=""
        if [[ "${USER_INSTALL}" == "true" ]]; then
            pip_flags="--user"
        fi
        
        ${python_cmd} -m pip install --upgrade pip ${pip_flags}
    fi
    
    # Install requirements
    local requirements_files=(
        "${PROJECT_ROOT}/requirements.txt"
        "${PROJECT_ROOT}/requirements-dev.txt"
        "${PROJECT_ROOT}/requirements-test.txt"
    )
    
    for req_file in "${requirements_files[@]}"; do
        if [[ -f "${req_file}" ]]; then
            log "Installing from $(basename "${req_file}")..."
            if [[ "${DRY_RUN}" == "false" ]]; then
                local pip_flags=""
                if [[ "${USER_INSTALL}" == "true" ]]; then
                    pip_flags="--user"
                fi
                
                ${python_cmd} -m pip install -r "${req_file}" ${pip_flags} || warn "Failed to install from ${req_file}"
            else
                log "[DRY-RUN] Would install from ${req_file}"
            fi
        fi
    done
    
    success "Python dependencies installation completed"
}

# Install audio processing dependencies
install_audio_dependencies() {
    step "Installing audio processing dependencies..."
    
    case "${PLATFORM}" in
        "macos")
            if [[ "${DRY_RUN}" == "false" ]]; then
                # Additional audio libraries for macOS
                brew install sox libmad libid3tag || warn "Some audio packages failed to install"
            fi
            ;;
        "debian")
            if [[ "${DRY_RUN}" == "false" ]]; then
                sudo apt install -y sox libmad0-dev libid3tag0-dev || warn "Some audio packages failed to install"
            fi
            ;;
        *)
            log "Audio dependencies for ${PLATFORM} not specifically handled"
            ;;
    esac
    
    success "Audio dependencies installation completed"
}

# Install ML/AI dependencies  
install_ml_dependencies() {
    step "Installing ML/AI dependencies..."
    
    # These are typically installed via pip in requirements.txt
    # But we can ensure some system-level dependencies are available
    
    case "${PLATFORM}" in
        "macos")
            if [[ "${DRY_RUN}" == "false" ]]; then
                brew install openblas lapack || warn "ML system packages failed to install"
            fi
            ;;
        "debian")
            if [[ "${DRY_RUN}" == "false" ]]; then
                sudo apt install -y libopenblas-dev liblapack-dev || warn "ML system packages failed to install" 
            fi
            ;;
        *)
            log "ML system dependencies for ${PLATFORM} not specifically handled"
            ;;
    esac
    
    success "ML dependencies installation completed"
}

# Install development tools
install_dev_dependencies() {
    step "Installing development tools..."
    
    # Development tools that are useful but not strictly required
    case "${PLATFORM}" in
        "macos")
            if [[ "${DRY_RUN}" == "false" ]]; then
                brew install --cask visual-studio-code || debug "VS Code installation skipped"
                brew install pre-commit || debug "pre-commit installation skipped"
            fi
            ;;
        "debian")
            if [[ "${DRY_RUN}" == "false" ]]; then
                # Install VS Code
                wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
                sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/
                sudo sh -c 'echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list'
                sudo apt update
                sudo apt install -y code || debug "VS Code installation skipped"
            fi
            ;;
    esac
    
    success "Development tools installation completed"
}

# Verify installation
verify_installation() {
    step "Verifying installation..."
    
    local verification_errors=0
    
    # Check critical commands
    local critical_commands=(
        "python3"
        "pip3"
        "git"
        "curl"
    )
    
    for cmd in "${critical_commands[@]}"; do
        if command_exists "${cmd}"; then
            success " ${cmd} is available"
        else
            error " ${cmd} is NOT available"
            ((verification_errors++))
        fi
    done
    
    # Check Python packages
    log "Checking Python packages..."
    if command_exists python3; then
        local python_packages=(
            "fastapi"
            "uvicorn"
            "openai"
            "langchain"
        )
        
        for package in "${python_packages[@]}"; do
            if python3 -c "import ${package}" 2>/dev/null; then
                success " Python package ${package} is available"
            else
                warn " Python package ${package} is NOT available"
            fi
        done
    fi
    
    # Check Docker
    if command_exists docker; then
        success " Docker is available"
        if docker info &>/dev/null; then
            success " Docker daemon is running"
        else
            warn " Docker daemon is not running"
        fi
    else
        warn " Docker is NOT available"
    fi
    
    if [[ ${verification_errors} -eq 0 ]]; then
        success "All critical dependencies verified successfully!"
    else
        warn "Some dependencies are missing. Please check the errors above."
    fi
}

# Show installation summary
show_installation_summary() {
    echo ""
    success "<‰ Dependencies installation completed!"
    echo ""
    log "Installation Summary:"
    log "  Platform: ${PLATFORM}"
    log "  Component: ${COMPONENT}"
    log "  Package Manager: ${PACKAGE_MANAGER}"
    log "  Python User Install: ${USER_INSTALL}"
    log "  Upgrade Mode: ${UPGRADE}"
    
    if [[ "${SKIP_SYSTEM}" == "true" ]]; then
        log "  System packages: SKIPPED"
    else
        log "  System packages: INSTALLED"
    fi
    
    if [[ "${SKIP_PYTHON}" == "true" ]]; then
        log "  Python packages: SKIPPED"
    else
        log "  Python packages: INSTALLED"
    fi
    
    if [[ "${SKIP_DOCKER}" == "true" ]]; then
        log "  Docker: SKIPPED"
    else
        log "  Docker: INSTALLED"
    fi
    
    echo ""
    log "Next steps:"
    log "  1. Restart your terminal/shell session"
    log "  2. Run: source ~/.bashrc (Linux) or source ~/.zshrc (macOS)"
    log "  3. Test the installation: python3 -c 'import fastapi; print(\"FastAPI is ready!\")'"
    log "  4. Run the setup script: ./scripts/setup/setup_env.sh"
    
    if [[ "${PLATFORM}" == "debian" ]] && command_exists docker; then
        echo ""
        warn "Docker was installed. You need to log out and back in for Docker permissions to take effect."
    fi
}

# Main execution
main() {
    echo -e "${CYAN}${SCRIPT_NAME} v${SCRIPT_VERSION}${NC}"
    echo ""
    
    # Parse arguments
    parse_args "$@"
    
    # Detect platform
    detect_platform
    if [[ -n "${FORCE_PLATFORM}" ]]; then
        PLATFORM="${FORCE_PLATFORM}"
        log "Platform forced to: ${PLATFORM}"
    fi
    
    log "Detected platform: ${PLATFORM} (${PACKAGE_MANAGER})"
    
    # Show configuration
    log "Installation Configuration:"
    log "  Component: ${COMPONENT}"
    log "  Platform: ${PLATFORM}"
    log "  Skip System: ${SKIP_SYSTEM}"
    log "  Skip Python: ${SKIP_PYTHON}"
    log "  Skip Docker: ${SKIP_DOCKER}"
    log "  Skip Optional: ${SKIP_OPTIONAL}"
    log "  Dry Run: ${DRY_RUN}"
    echo ""
    
    # Confirmation for non-force mode
    if [[ "${FORCE}" == "false" ]] && [[ "${DRY_RUN}" == "false" ]]; then
        echo -e "${YELLOW}Proceed with dependencies installation? (y/N)${NC}"
        read -r confirmation
        if [[ ! "${confirmation}" =~ ^[Yy]$ ]]; then
            log "Installation cancelled by user"
            exit 0
        fi
    fi
    
    # Execute installation based on component
    case "${COMPONENT}" in
        "all")
            install_system_packages
            install_python_dependencies
            install_audio_dependencies
            install_ml_dependencies
            if [[ "${SKIP_OPTIONAL}" == "false" ]]; then
                install_dev_dependencies
            fi
            ;;
        "system")
            install_system_packages
            ;;
        "python")
            install_python_dependencies
            ;;
        "docker")
            if [[ "${PLATFORM}" == "debian" ]]; then
                install_docker_debian
            else
                warn "Docker installation not specifically supported for ${PLATFORM}"
            fi
            ;;
        "audio")
            install_audio_dependencies
            ;;
        "ml")
            install_ml_dependencies
            ;;
        "dev")
            install_dev_dependencies
            ;;
        "optional")
            install_dev_dependencies
            ;;
        *)
            error "Unknown component: ${COMPONENT}"
            exit 1
            ;;
    esac
    
    # Verify installation unless it's a dry run
    if [[ "${DRY_RUN}" == "false" ]]; then
        verify_installation
        show_installation_summary
    else
        log "Dry run completed. No actual installation performed."
    fi
    
    success "Dependencies installation script completed! =€"
}

# Handle interruption
trap 'error "Installation interrupted by user"; exit 130' INT TERM

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi