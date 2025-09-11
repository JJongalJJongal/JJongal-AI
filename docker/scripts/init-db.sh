#!/bin/bash
# JJongal-AI Database Initialization Script
# Initializes Vector Database and other data stores for the container

set -euo pipefail

# Color codes for logging
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INIT-DB:${NC} $1"
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

# Configuration
APP_DIR="/app"
VECTOR_DB_DIR="${VECTOR_DB_PATH:-/app/src/data/vector_db}"
CONFIG_DIR="/app/config"
OUTPUT_DIR="/app/output"

log "Starting JJongal-AI Database Initialization..."

# Function to initialize vector database directories
init_vector_db_structure() {
    log "Initializing Vector Database structure..."
    
    # Create vector DB subdirectories
    local vector_subdirs=(
        "${VECTOR_DB_DIR}/main"
        "${VECTOR_DB_DIR}/detailed"
        "${VECTOR_DB_DIR}/summary"
    )
    
    for dir in "${vector_subdirs[@]}"; do
        if [ ! -d "${dir}" ]; then
            mkdir -p "${dir}"
            log "Created directory: ${dir}"
        fi
    done
    
    # Set appropriate permissions
    chmod -R 755 "${VECTOR_DB_DIR}"
    success "Vector Database structure initialized"
}

# Function to initialize output directories
init_output_structure() {
    log "Initializing output directory structure..."
    
    local output_subdirs=(
        "${OUTPUT_DIR}/stories"
        "${OUTPUT_DIR}/metadata"
        "${OUTPUT_DIR}/temp/images"
        "${OUTPUT_DIR}/temp/audio"
        "${OUTPUT_DIR}/temp/voice_samples"
        "${OUTPUT_DIR}/conversations"
        "${OUTPUT_DIR}/workflow_states"
    )
    
    for dir in "${output_subdirs[@]}"; do
        if [ ! -d "${dir}" ]; then
            mkdir -p "${dir}"
            log "Created directory: ${dir}"
        fi
    done
    
    # Set appropriate permissions
    chmod -R 755 "${OUTPUT_DIR}"
    success "Output directory structure initialized"
}

# Function to populate vector database with initial data
populate_vector_db() {
    if [ "${LOAD_INITIAL_DATA:-false}" = "true" ]; then
        log "Populating Vector Database with initial data..."
        
        cd "${APP_DIR}"
        
        # Check if population script exists
        if [ -f "scripts/data/populate_vector_db.py" ]; then
            if python scripts/data/populate_vector_db.py; then
                # Mark as initialized
                touch "${VECTOR_DB_DIR}/.initialized"
                success "Vector Database populated with initial data"
            else
                warn "Failed to populate Vector Database (non-critical)"
            fi
        else
            warn "Vector DB population script not found, skipping initial data load"
        fi
    else
        log "Initial data loading disabled (LOAD_INITIAL_DATA=${LOAD_INITIAL_DATA:-false})"
    fi
}

# Function to validate configuration files
validate_config() {
    log "Validating configuration files..."
    
    # Check if essential config files exist
    local config_files=(
        "${CONFIG_DIR}/prompts/chatbot_a_prompts.json"
        "${CONFIG_DIR}/prompts/chatbot_b_prompts.json"
    )
    
    local missing_files=()
    
    for file in "${config_files[@]}"; do
        if [ ! -f "${file}" ]; then
            missing_files+=("${file}")
        fi
    done
    
    if [ ${#missing_files[@]} -gt 0 ]; then
        warn "Missing configuration files:"
        for file in "${missing_files[@]}"; do
            warn "  - ${file}"
        done
    else
        success "All essential configuration files found"
    fi
}

# Function to run health checks
health_check() {
    log "Running basic health checks..."
    
    # Check directory permissions
    if [ -w "${VECTOR_DB_DIR}" ] && [ -w "${OUTPUT_DIR}" ]; then
        success "Directory permissions OK"
    else
        error "Directory permission issues detected"
        return 1
    fi
    
    # Check Python environment
    if python -c "import sys; print('Python', sys.version)" >/dev/null 2>&1; then
        success "Python environment OK"
    else
        error "Python environment issues detected"
        return 1
    fi
    
    # Check essential Python packages
    if python -c "import chromadb, langchain, openai, fastapi" >/dev/null 2>&1; then
        success "Essential Python packages available"
    else
        warn "Some Python packages may be missing"
    fi
    
    success "Health checks completed"
}

# Main initialization sequence
main() {
    log "JJongal-AI Database Initialization Script v1.0"
    log "Environment: ${APP_ENV:-production}"
    log "Vector DB Path: ${VECTOR_DB_DIR}"
    
    # Execute initialization steps
    init_vector_db_structure
    init_output_structure
    validate_config
    populate_vector_db
    health_check
    
    success "Database initialization completed successfully!"
    
    # Log final status
    log "Initialization Summary:"
    log "  - Vector DB: ${VECTOR_DB_DIR}"
    log "  - Output Dir: ${OUTPUT_DIR}"
    log "  - Config Dir: ${CONFIG_DIR}"
    
    if [ -f "${VECTOR_DB_DIR}/.initialized" ]; then
        log "  - Initial Data: Loaded"
    else
        log "  - Initial Data: Not loaded"
    fi
}

# Handle script arguments
case "${1:-init}" in
    "init"|"initialize")
        main
        ;;
    "check"|"health")
        log "Running health check only..."
        health_check
        ;;
    "populate")
        log "Running data population only..."
        populate_vector_db
        ;;
    "help"|"-h"|"--help")
        echo "JJongal-AI Database Initialization Script"
        echo ""
        echo "Usage: $0 [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  init, initialize    Run full database initialization (default)"
        echo "  check, health       Run health checks only"
        echo "  populate            Populate vector database with initial data"
        echo "  help, -h, --help    Show this help message"
        exit 0
        ;;
    *)
        error "Unknown command: $1"
        error "Use 'help' for available commands"
        exit 1
        ;;
esac