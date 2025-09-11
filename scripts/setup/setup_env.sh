#!/bin/bash
# JJongal-AI Environment Setup Script
# Automates development environment setup, dependency installation, and configuration

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script information
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="JJongal-AI Setup Script"

# Default values
DEFAULT_PYTHON_VERSION="3.10"
DEFAULT_ENVIRONMENT="development"
REQUIREMENTS_FILE="requirements.txt"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

# Help function
show_help() {
    cat << EOF
${SCRIPT_NAME} v${SCRIPT_VERSION}

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Target environment (development, testing, production)
                            Default: ${DEFAULT_ENVIRONMENT}
    -p, --python-version VER Python version to use (3.9, 3.10, 3.11)
                            Default: ${DEFAULT_PYTHON_VERSION}
    -r, --requirements FILE  Requirements file to install
                            Default: ${REQUIREMENTS_FILE}
    --no-venv               Skip virtual environment creation
    --no-deps               Skip dependency installation  
    --no-db                 Skip database setup
    --no-vector-db          Skip vector database initialization
    --docker-setup          Setup for Docker environment
    --ci-setup              Setup for CI/CD environment
    --quick                 Quick setup (minimal checks)
    -h, --help              Show this help message

EXAMPLES:
    $0                                  # Default development setup
    $0 -e testing                      # Setup for testing
    $0 --docker-setup                  # Setup for Docker container
    $0 --ci-setup                      # Setup for CI/CD pipeline
    $0 -p 3.11 --no-venv              # Use system Python 3.11

EOF
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT="${DEFAULT_ENVIRONMENT}"
    PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
    SKIP_VENV=false
    SKIP_DEPS=false
    SKIP_DB=false
    SKIP_VECTOR_DB=false
    DOCKER_SETUP=false
    CI_SETUP=false
    QUICK_SETUP=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -p|--python-version)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            -r|--requirements)
                REQUIREMENTS_FILE="$2"
                shift 2
                ;;
            --no-venv)
                SKIP_VENV=true
                shift
                ;;
            --no-deps)
                SKIP_DEPS=true
                shift
                ;;
            --no-db)
                SKIP_DB=true
                shift
                ;;
            --no-vector-db)
                SKIP_VECTOR_DB=true
                shift
                ;;
            --docker-setup)
                DOCKER_SETUP=true
                shift
                ;;
            --ci-setup)
                CI_SETUP=true
                SKIP_VENV=true
                shift
                ;;
            --quick)
                QUICK_SETUP=true
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
    
    # Validate environment
    case $ENVIRONMENT in
        development|testing|production)
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT"
            error "Valid options: development, testing, production"
            exit 1
            ;;
    esac
}

# System requirements check
check_system_requirements() {
    step "Checking system requirements..."
    
    # Check operating system
    OS="$(uname -s)"
    case "${OS}" in
        Linux*)     MACHINE=Linux;;
        Darwin*)    MACHINE=Mac;;
        CYGWIN*)    MACHINE=Cygwin;;
        MINGW*)     MACHINE=MinGw;;
        *)          MACHINE="UNKNOWN:${OS}"
    esac
    log "Detected OS: ${MACHINE}"
    
    # Check Python version
    if command -v python${PYTHON_VERSION} &> /dev/null; then
        PYTHON_CMD="python${PYTHON_VERSION}"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        DETECTED_VERSION=$(python3 --version | sed 's/Python //')
        warn "Requested Python ${PYTHON_VERSION} not found, using Python ${DETECTED_VERSION}"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
        DETECTED_VERSION=$(python --version | sed 's/Python //')
        warn "Using system Python ${DETECTED_VERSION}"
    else
        error "Python not found. Please install Python ${PYTHON_VERSION} or later."
        exit 1
    fi
    
    # Check pip
    if ! ${PYTHON_CMD} -m pip --version &> /dev/null; then
        error "pip not found. Please install pip for Python ${PYTHON_VERSION}."
        exit 1
    fi
    
    # Check git
    if ! command -v git &> /dev/null; then
        error "git not found. Please install git."
        exit 1
    fi
    
    # Check additional tools based on environment
    if [[ "${DOCKER_SETUP}" == "false" ]]; then
        # Check for development tools
        if ! command -v curl &> /dev/null; then
            warn "curl not found. Some features may not work."
        fi
        
        if [[ "${MACHINE}" == "Mac" ]] && ! command -v brew &> /dev/null; then
            warn "Homebrew not found. Consider installing it for easier package management."
        fi
    fi
    
    success "System requirements check completed"
}

# Create virtual environment
setup_virtual_environment() {
    if [[ "${SKIP_VENV}" == "true" ]]; then
        log "Skipping virtual environment setup"
        return
    fi
    
    step "Setting up Python virtual environment..."
    
    VENV_PATH="${PROJECT_ROOT}/fairy_tail"
    
    # Remove existing virtual environment if it exists and is problematic
    if [[ -d "${VENV_PATH}" ]]; then
        if [[ "${QUICK_SETUP}" == "false" ]]; then
            log "Existing virtual environment found. Checking compatibility..."
            
            if source "${VENV_PATH}/bin/activate" 2>/dev/null && python --version | grep -q "Python ${PYTHON_VERSION}"; then
                log "Compatible virtual environment found, using existing one"
                return
            else
                warn "Incompatible virtual environment found, recreating..."
                rm -rf "${VENV_PATH}"
            fi
        else
            log "Using existing virtual environment"
            return
        fi
    fi
    
    # Create new virtual environment
    log "Creating virtual environment with Python ${PYTHON_VERSION}..."
    ${PYTHON_CMD} -m venv "${VENV_PATH}"
    
    # Activate virtual environment
    source "${VENV_PATH}/bin/activate"
    
    # Upgrade pip, setuptools, and wheel
    log "Upgrading pip, setuptools, and wheel..."
    pip install --upgrade pip setuptools wheel
    
    success "Virtual environment created and activated"
}

# Install Python dependencies
install_dependencies() {
    if [[ "${SKIP_DEPS}" == "true" ]]; then
        log "Skipping dependency installation"
        return
    fi
    
    step "Installing Python dependencies..."
    
    # Activate virtual environment if it exists
    if [[ "${SKIP_VENV}" == "false" ]] && [[ -f "${PROJECT_ROOT}/fairy_tail/bin/activate" ]]; then
        source "${PROJECT_ROOT}/fairy_tail/bin/activate"
    fi
    
    # Check if requirements file exists
    if [[ ! -f "${PROJECT_ROOT}/${REQUIREMENTS_FILE}" ]]; then
        error "Requirements file not found: ${PROJECT_ROOT}/${REQUIREMENTS_FILE}"
        exit 1
    fi
    
    log "Installing from ${REQUIREMENTS_FILE}..."
    pip install -r "${PROJECT_ROOT}/${REQUIREMENTS_FILE}"
    
    # Install development dependencies for development environment
    if [[ "${ENVIRONMENT}" == "development" ]] && [[ -f "${PROJECT_ROOT}/requirements-dev.txt" ]]; then
        log "Installing development dependencies..."
        pip install -r "${PROJECT_ROOT}/requirements-dev.txt"
    fi
    
    # Install testing dependencies for testing environment
    if [[ "${ENVIRONMENT}" == "testing" ]] && [[ -f "${PROJECT_ROOT}/requirements-test.txt" ]]; then
        log "Installing testing dependencies..."
        pip install -r "${PROJECT_ROOT}/requirements-test.txt"
    fi
    
    success "Dependencies installed successfully"
}

# Setup environment configuration
setup_environment_config() {
    step "Setting up environment configuration..."
    
    # Determine config file
    ENV_FILE="${PROJECT_ROOT}/config/environments/${ENVIRONMENT}.env"
    
    if [[ ! -f "${ENV_FILE}" ]]; then
        error "Environment config file not found: ${ENV_FILE}"
        exit 1
    fi
    
    # Create .env symlink for convenience
    ENV_SYMLINK="${PROJECT_ROOT}/.env"
    if [[ -L "${ENV_SYMLINK}" ]] || [[ -f "${ENV_SYMLINK}" ]]; then
        rm -f "${ENV_SYMLINK}"
    fi
    
    ln -s "config/environments/${ENVIRONMENT}.env" "${ENV_SYMLINK}"
    log "Created .env symlink to ${ENVIRONMENT}.env"
    
    # Create necessary directories
    log "Creating necessary directories..."
    
    directories=(
        "${PROJECT_ROOT}/data"
        "${PROJECT_ROOT}/logs/${ENVIRONMENT}"
        "${PROJECT_ROOT}/temp/${ENVIRONMENT}"
        "${PROJECT_ROOT}/uploads/${ENVIRONMENT}"
        "${PROJECT_ROOT}/data/vector_db/${ENVIRONMENT}"
    )
    
    # Additional directories for testing
    if [[ "${ENVIRONMENT}" == "testing" ]]; then
        directories+=(
            "${PROJECT_ROOT}/test_logs"
            "${PROJECT_ROOT}/test_temp"
            "${PROJECT_ROOT}/test_uploads"
            "${PROJECT_ROOT}/test_data"
        )
    fi
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "${dir}" ]]; then
            mkdir -p "${dir}"
            log "Created directory: ${dir}"
        fi
    done
    
    success "Environment configuration completed"
}

# Setup database
setup_database() {
    if [[ "${SKIP_DB}" == "true" ]]; then
        log "Skipping database setup"
        return
    fi
    
    step "Setting up database..."
    
    # Source environment variables
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        source "${PROJECT_ROOT}/.env"
    fi
    
    # Activate virtual environment
    if [[ "${SKIP_VENV}" == "false" ]] && [[ -f "${PROJECT_ROOT}/fairy_tail/bin/activate" ]]; then
        source "${PROJECT_ROOT}/fairy_tail/bin/activate"
    fi
    
    cd "${PROJECT_ROOT}"
    
    # Run database setup script
    if [[ -f "scripts/data/setup_database.py" ]]; then
        log "Running database setup script..."
        python scripts/data/setup_database.py
    else
        log "Creating database tables using application code..."
        python -c "
import sys
sys.path.append('${PROJECT_ROOT}')
from src.data.database.db_manager import DatabaseManager
db_manager = DatabaseManager()
db_manager.create_tables()
print('Database tables created')
" || warn "Database setup failed (may not be critical)"
    fi
    
    success "Database setup completed"
}

# Setup Vector Database
setup_vector_database() {
    if [[ "${SKIP_VECTOR_DB}" == "true" ]]; then
        log "Skipping vector database setup"
        return
    fi
    
    step "Setting up Vector Database..."
    
    # Activate virtual environment
    if [[ "${SKIP_VENV}" == "false" ]] && [[ -f "${PROJECT_ROOT}/fairy_tail/bin/activate" ]]; then
        source "${PROJECT_ROOT}/fairy_tail/bin/activate"
    fi
    
    cd "${PROJECT_ROOT}"
    
    # Initialize vector database with sample data for development
    if [[ "${ENVIRONMENT}" == "development" ]] || [[ "${ENVIRONMENT}" == "testing" ]]; then
        if [[ -f "scripts/data/populate_vector_db.py" ]]; then
            log "Populating vector database with sample data..."
            python scripts/data/populate_vector_db.py || warn "Vector DB population failed (non-critical)"
        else
            warn "Vector DB population script not found"
        fi
    fi
    
    success "Vector Database setup completed"
}

# Run post-setup validation
validate_setup() {
    step "Validating setup..."
    
    # Activate virtual environment
    if [[ "${SKIP_VENV}" == "false" ]] && [[ -f "${PROJECT_ROOT}/fairy_tail/bin/activate" ]]; then
        source "${PROJECT_ROOT}/fairy_tail/bin/activate"
    fi
    
    cd "${PROJECT_ROOT}"
    
    # Test import of main application
    log "Testing application imports..."
    python -c "
import sys
sys.path.append('${PROJECT_ROOT}')
try:
    import app
    print(' Main application import successful')
except ImportError as e:
    print(f' Application import failed: {e}')
    sys.exit(1)
"
    
    # Test critical dependencies
    log "Testing critical dependencies..."
    python -c "
import fastapi
import uvicorn
import openai
import langchain
import chromadb
import sqlalchemy
print(' All critical dependencies available')
"
    
    # Run basic health check if not in CI
    if [[ "${CI_SETUP}" == "false" ]] && [[ "${QUICK_SETUP}" == "false" ]]; then
        log "Running application health check..."
        timeout 10s python -c "
import sys
sys.path.append('${PROJECT_ROOT}')
from src.api.v1.endpoints.health import get_health_status
health = get_health_status()
print(f' Health check passed: {health}')
" || warn "Health check failed (non-critical)"
    fi
    
    success "Setup validation completed"
}

# Display post-setup information
show_completion_info() {
    echo ""
    success "<� JJongal-AI environment setup completed!"
    echo ""
    log "Setup Summary:"
    log "  Environment: ${ENVIRONMENT}"
    log "  Python: $(python --version 2>&1)"
    log "  Project Root: ${PROJECT_ROOT}"
    
    if [[ "${SKIP_VENV}" == "false" ]]; then
        log "  Virtual Environment: ${PROJECT_ROOT}/fairy_tail"
    fi
    
    echo ""
    log "Next steps:"
    
    if [[ "${SKIP_VENV}" == "false" ]]; then
        log "  1. Activate virtual environment:"
        log "     source fairy_tail/bin/activate"
    fi
    
    log "  2. Start the application:"
    log "     python app.py"
    log "     # or"
    log "     make dev"
    
    log "  3. Run tests:"
    log "     python -m pytest tests/"
    
    log "  4. Access API documentation:"
    log "     http://localhost:8000/docs"
    
    echo ""
    log "Configuration files:"
    log "  Environment: config/environments/${ENVIRONMENT}.env"
    log "  Logging: config/logging/${ENVIRONMENT}.yaml" 
    log "  Vector DB: config/database/vector_db_config.yaml"
    
    if [[ "${ENVIRONMENT}" == "development" ]]; then
        echo ""
        warn "Development Environment Notes:"
        warn "  • Update API keys in config/environments/development.env"
        warn "  • OPENAI_API_KEY and ELEVENLABS_API_KEY need real values"
        warn "  • Database is SQLite by default (./data/dev_jjongal.db)"
    fi
}

# Main execution
main() {
    echo -e "${CYAN}${SCRIPT_NAME} v${SCRIPT_VERSION}${NC}"
    echo ""
    
    cd "${PROJECT_ROOT}"
    
    # Parse arguments
    parse_args "$@"
    
    log "Starting setup with the following configuration:"
    log "  Environment: ${ENVIRONMENT}"
    log "  Python Version: ${PYTHON_VERSION}" 
    log "  Skip Virtual Env: ${SKIP_VENV}"
    log "  Skip Dependencies: ${SKIP_DEPS}"
    log "  Docker Setup: ${DOCKER_SETUP}"
    log "  CI Setup: ${CI_SETUP}"
    echo ""
    
    # Execute setup steps
    check_system_requirements
    setup_virtual_environment
    install_dependencies
    setup_environment_config
    setup_database
    setup_vector_database
    validate_setup
    show_completion_info
    
    success "Setup completed successfully! =�"
}

# Handle script interruption
trap 'error "Setup interrupted by user"; exit 130' INT TERM

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi