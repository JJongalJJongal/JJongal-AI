#!/bin/bash
# JJongal-AI Docker Container Entrypoint Script
# Handles container initialization, environment setup, and application startup

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Environment variables with defaults
ENVIRONMENT=${ENVIRONMENT:-production}
APP_PORT=${PORT:-8000}
APP_HOST=${HOST:-0.0.0.0}
WORKERS=${WORKERS:-4}
LOG_LEVEL=${LOG_LEVEL:-INFO}

# Application paths
APP_DIR="/app"
LOG_DIR="/var/log/jjongal"
DATA_DIR="/data"
TEMP_DIR="/tmp/jjongal"
VECTOR_DB_DIR="${VECTOR_DB_PATH:-/app/src/data/vector_db}"

log "Starting JJongal-AI container..."
log "Environment: ${ENVIRONMENT}"
log "Port: ${APP_PORT}"
log "Workers: ${WORKERS}"

# Function to create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    directories=(
        "${LOG_DIR}"
        "${DATA_DIR}"
        "${TEMP_DIR}"
        "${VECTOR_DB_DIR}"
        "/data/uploads"
        "/data/backups"
        "/tmp/audio_samples"
        "/tmp/generated_images"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "${dir}" ]; then
            mkdir -p "${dir}"
            log "Created directory: ${dir}"
        fi
    done
    
    # Set appropriate permissions
    chmod 755 "${LOG_DIR}"
    chmod 755 "${DATA_DIR}"
    chmod 777 "${TEMP_DIR}"  # Temporary files need broader access
    
    success "Directories created successfully"
}

# Function to validate environment variables
validate_environment() {
    log "Validating environment configuration..."
    
    # Critical environment variables
    required_vars=()
    
    # Production-specific required variables
    if [ "${ENVIRONMENT}" = "production" ]; then
        required_vars+=(
            "DATABASE_URL"
            "JWT_SECRET_KEY"
            "OPENAI_API_KEY"
            "REDIS_URL"
        )
    fi
    
    # Check required variables
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            error "Required environment variable ${var} is not set"
            exit 1
        fi
    done
    
    # Validate critical configurations
    if [ "${ENVIRONMENT}" = "production" ]; then
        if [[ "${JWT_SECRET_KEY}" == *"dev"* ]] || [[ "${JWT_SECRET_KEY}" == *"test"* ]]; then
            error "Production environment detected with development JWT secret"
            exit 1
        fi
        
        if [[ "${OPENAI_API_KEY}" == *"test"* ]] || [[ "${OPENAI_API_KEY}" == *"mock"* ]]; then
            error "Production environment detected with test API keys"
            exit 1
        fi
    fi
    
    success "Environment validation completed"
}

# Function to wait for database connectivity
wait_for_database() {
    if [ -z "${DATABASE_URL:-}" ]; then
        log "No database URL provided, skipping database check"
        return
    fi
    
    log "Waiting for database connectivity..."
    
    # Extract host and port from DATABASE_URL
    if [[ "${DATABASE_URL}" == postgres* ]]; then
        # PostgreSQL URL parsing
        DB_HOST=$(echo "${DATABASE_URL}" | sed -n 's/.*@\([^:]*\):.*/\1/p')
        DB_PORT=$(echo "${DATABASE_URL}" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
        
        if [ -n "${DB_HOST}" ] && [ -n "${DB_PORT}" ]; then
            log "Checking PostgreSQL connection to ${DB_HOST}:${DB_PORT}"
            
            for i in {1..30}; do
                if nc -z "${DB_HOST}" "${DB_PORT}"; then
                    success "Database is ready"
                    return
                fi
                warn "Database not ready, waiting... (${i}/30)"
                sleep 2
            done
            
            error "Database connection timeout"
            exit 1
        fi
    elif [[ "${DATABASE_URL}" == sqlite* ]]; then
        log "SQLite database detected, no connection wait needed"
    else
        log "Unknown database type, skipping connection check"
    fi
}

# Function to wait for Redis (if enabled)
wait_for_redis() {
    if [ "${REDIS_ENABLED:-false}" != "true" ] || [ -z "${REDIS_URL:-}" ]; then
        log "Redis not enabled, skipping Redis check"
        return
    fi
    
    log "Waiting for Redis connectivity..."
    
    # Extract host and port from REDIS_URL
    REDIS_HOST=$(echo "${REDIS_URL}" | sed -n 's/.*@\([^:]*\):.*/\1/p' || echo "localhost")
    REDIS_PORT=$(echo "${REDIS_URL}" | sed -n 's/.*:\([0-9]*\).*/\1/p' || echo "6379")
    
    if [ -z "${REDIS_HOST}" ]; then
        REDIS_HOST="localhost"
    fi
    if [ -z "${REDIS_PORT}" ]; then
        REDIS_PORT="6379"
    fi
    
    log "Checking Redis connection to ${REDIS_HOST}:${REDIS_PORT}"
    
    for i in {1..15}; do
        if nc -z "${REDIS_HOST}" "${REDIS_PORT}"; then
            success "Redis is ready"
            return
        fi
        warn "Redis not ready, waiting... (${i}/15)"
        sleep 2
    done
    
    warn "Redis connection timeout (non-critical, continuing...)"
}

# Function to initialize Vector Database
init_vector_database() {
    log "Initializing Vector Database..."
    
    if [ ! -d "${VECTOR_DB_DIR}" ]; then
        mkdir -p "${VECTOR_DB_DIR}"
        log "Created Vector DB directory: ${VECTOR_DB_DIR}"
    fi
    
    # Check if vector DB is already populated
    if [ "${LOAD_TEST_DATA:-false}" = "true" ] && [ ! -f "${VECTOR_DB_DIR}/.initialized" ]; then
        log "Populating Vector Database with initial data..."
        
        if python -c "import sys; sys.path.append('${APP_DIR}'); from scripts.data.populate_vector_db import main; main()" 2>&1; then
            touch "${VECTOR_DB_DIR}/.initialized"
            success "Vector Database initialized"
        else
            warn "Vector Database initialization failed (non-critical)"
        fi
    else
        log "Vector Database already initialized or test data loading disabled"
    fi
}

# Function to run database migrations
run_migrations() {
    if [ "${RESET_DB_ON_STARTUP:-false}" = "true" ] && [ "${ENVIRONMENT}" != "production" ]; then
        log "Resetting database (non-production environment)"
        python -c "
import sys
sys.path.append('${APP_DIR}')
from src.data.database.db_manager import DatabaseManager
db_manager = DatabaseManager()
db_manager.reset_database()
print('Database reset completed')
" || warn "Database reset failed"
    fi
    
    log "Running database migrations..."
    python -c "
import sys
sys.path.append('${APP_DIR}')
from src.data.database.db_manager import DatabaseManager
db_manager = DatabaseManager()
db_manager.create_tables()
print('Database migrations completed')
" || error "Database migrations failed"
}

# Function to perform health checks
health_check() {
    log "Performing startup health checks..."
    
    # Check Python dependencies
    python -c "
import fastapi
import uvicorn
import openai
import langchain
import chromadb
print('All critical Python packages available')
" || {
        error "Critical Python packages missing"
        exit 1
    }
    
    # Check application can be imported
    python -c "
import sys
sys.path.append('${APP_DIR}')
import app
print('Application module imported successfully')
" || {
        error "Application import failed"
        exit 1
    }
    
    success "Health checks passed"
}

# Function to start the application
start_application() {
    log "Starting JJongal-AI application..."
    log "Configuration:"
    log "  Host: ${APP_HOST}"
    log "  Port: ${APP_PORT}"
    log "  Workers: ${WORKERS}"
    log "  Log Level: ${LOG_LEVEL}"
    log "  Environment: ${ENVIRONMENT}"
    
    cd "${APP_DIR}"
    
    # Start the application based on environment
    if [ "${ENVIRONMENT}" = "development" ]; then
        log "Starting in development mode with auto-reload"
        exec uvicorn app:app \
            --host "${APP_HOST}" \
            --port "${APP_PORT}" \
            --log-level "${LOG_LEVEL,,}" \
            --reload \
            --reload-dir src \
            --access-log
    elif [ "${ENVIRONMENT}" = "testing" ]; then
        log "Starting in testing mode"
        exec uvicorn app:app \
            --host "${APP_HOST}" \
            --port "${APP_PORT}" \
            --log-level "${LOG_LEVEL,,}" \
            --workers 1
    else
        log "Starting in production mode"
        exec uvicorn app:app \
            --host "${APP_HOST}" \
            --port "${APP_PORT}" \
            --log-level "${LOG_LEVEL,,}" \
            --workers "${WORKERS}" \
            --access-log \
            --proxy-headers \
            --forwarded-allow-ips "*"
    fi
}

# Function to handle graceful shutdown
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill -TERM 2>/dev/null || true
    
    # Clean temporary files
    if [ -d "${TEMP_DIR}" ]; then
        rm -rf "${TEMP_DIR}/*" 2>/dev/null || true
    fi
    
    log "Cleanup completed"
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup SIGTERM SIGINT

# Main execution flow
main() {
    log "JJongal-AI Entrypoint Script v1.0"
    
    # Validate command line arguments
    if [ $# -gt 0 ]; then
        case "$1" in
            "bash"|"sh")
                log "Starting interactive shell"
                exec /bin/bash
                ;;
            "python")
                shift
                log "Running Python command: $*"
                cd "${APP_DIR}"
                exec python "$@"
                ;;
            "migrate")
                log "Running migrations only"
                create_directories
                validate_environment
                wait_for_database
                run_migrations
                success "Migrations completed"
                exit 0
                ;;
            "test")
                log "Running tests"
                cd "${APP_DIR}"
                exec python -m pytest "$@"
                ;;
            "help")
                echo "JJongal-AI Docker Entrypoint"
                echo "Usage: $0 [COMMAND]"
                echo ""
                echo "Commands:"
                echo "  bash         Start interactive shell"
                echo "  python       Run Python command"
                echo "  migrate      Run database migrations only"
                echo "  test         Run test suite"
                echo "  help         Show this help"
                echo ""
                echo "Default: Start JJongal-AI application"
                exit 0
                ;;
            *)
                warn "Unknown command: $1"
                warn "Use 'help' for available commands"
                ;;
        esac
    fi
    
    # Standard startup sequence
    create_directories
    validate_environment
    wait_for_database
    wait_for_redis
    run_migrations
    init_vector_database
    health_check
    start_application
}

# Execute main function
main "$@"