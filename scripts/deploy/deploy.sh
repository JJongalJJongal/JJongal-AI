#!/bin/bash
# JJongal-AI Deployment Script
# Automated deployment with environment validation, health checks, and rollback capability

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
SCRIPT_NAME="JJongal-AI Deployment Script"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Default configuration
DEFAULT_ENVIRONMENT="production"
DEFAULT_TIMEOUT="300"
DEFAULT_HEALTH_CHECK_RETRIES="10"
DEFAULT_ROLLBACK_STRATEGY="previous"

# Deployment settings
DEPLOY_TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
DEPLOY_TAG="${DEPLOY_TIMESTAMP}"
BACKUP_DIR="/tmp/jjongal_deploy_backup_${DEPLOY_TIMESTAMP}"
LOG_FILE="/tmp/jjongal_deploy_${DEPLOY_TIMESTAMP}.log"

# Docker settings
DOCKER_REGISTRY="${DOCKER_REGISTRY:-}"
DOCKER_REPO="${DOCKER_REPO:-jjongal-ai}"
DOCKER_TAG="${DOCKER_TAG:-latest}"

# Logging functions
log() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "${LOG_FILE}"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "${LOG_FILE}"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" | tee -a "${LOG_FILE}"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "${LOG_FILE}" >&2
}

step() {
    echo -e "${CYAN}[STEP]${NC} $1" | tee -a "${LOG_FILE}"
}

debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1" | tee -a "${LOG_FILE}"
    fi
}

# Help function
show_help() {
    cat << EOF
${SCRIPT_NAME} v${SCRIPT_VERSION}

USAGE:
    $0 [OPTIONS] [COMMAND]

COMMANDS:
    deploy          Full deployment (default)
    rollback        Rollback to previous version
    health-check    Run health checks only
    backup          Create backup only
    validate        Validate deployment configuration

OPTIONS:
    -e, --environment ENV       Target environment (production, staging)
                               Default: ${DEFAULT_ENVIRONMENT}
    -t, --tag TAG              Docker tag to deploy
                               Default: ${DOCKER_TAG}
    -r, --registry URL         Docker registry URL
    --timeout SECONDS          Deployment timeout
                               Default: ${DEFAULT_TIMEOUT}
    --skip-backup              Skip backup creation
    --skip-health-check        Skip health checks
    --skip-rollback-prep       Skip rollback preparation  
    --force                    Force deployment without confirmations
    --dry-run                  Show what would be deployed without executing
    --debug                    Enable debug output
    --rollback-to TAG          Rollback to specific tag/version
    --rollback-strategy STRATEGY  Rollback strategy (previous, tag, manual)
                                 Default: ${DEFAULT_ROLLBACK_STRATEGY}
    -h, --help                 Show this help message

EXAMPLES:
    $0                                    # Deploy latest to production
    $0 -e staging                         # Deploy to staging
    $0 deploy -t v1.2.3                  # Deploy specific version
    $0 rollback                          # Rollback to previous version
    $0 rollback --rollback-to v1.2.2     # Rollback to specific version
    $0 health-check                      # Run health checks only
    $0 --dry-run                         # Preview deployment

ENVIRONMENT VARIABLES:
    DOCKER_REGISTRY          Docker registry URL
    DOCKER_REPO             Docker repository name
    DEPLOY_KEY              SSH deployment key
    SLACK_WEBHOOK           Slack notification webhook
    DEPLOY_ENV_FILE         Path to deployment environment file

EOF
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT="${DEFAULT_ENVIRONMENT}"
    TIMEOUT="${DEFAULT_TIMEOUT}"
    HEALTH_CHECK_RETRIES="${DEFAULT_HEALTH_CHECK_RETRIES}"
    ROLLBACK_STRATEGY="${DEFAULT_ROLLBACK_STRATEGY}"
    SKIP_BACKUP=false
    SKIP_HEALTH_CHECK=false
    SKIP_ROLLBACK_PREP=false
    FORCE=false
    DRY_RUN=false
    DEBUG=false
    ROLLBACK_TO=""
    COMMAND="deploy"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            deploy|rollback|health-check|backup|validate)
                COMMAND="$1"
                shift
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--tag)
                DOCKER_TAG="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-health-check)
                SKIP_HEALTH_CHECK=true
                shift
                ;;
            --skip-rollback-prep)
                SKIP_ROLLBACK_PREP=true
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
            --rollback-to)
                ROLLBACK_TO="$2"
                shift 2
                ;;
            --rollback-strategy)
                ROLLBACK_STRATEGY="$2"
                shift 2
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
        production|staging)
            ;;
        *)
            error "Invalid environment: $ENVIRONMENT"
            error "Valid options: production, staging"
            exit 1
            ;;
    esac
    
    # Set environment-specific configuration
    if [[ -n "${DOCKER_REGISTRY}" ]]; then
        FULL_IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_REPO}:${DOCKER_TAG}"
    else
        FULL_IMAGE_NAME="${DOCKER_REPO}:${DOCKER_TAG}"
    fi
    
    # Load deployment environment file if specified
    if [[ -n "${DEPLOY_ENV_FILE:-}" ]] && [[ -f "${DEPLOY_ENV_FILE}" ]]; then
        source "${DEPLOY_ENV_FILE}"
        log "Loaded deployment configuration from ${DEPLOY_ENV_FILE}"
    fi
}

# Send Slack notification
send_notification() {
    local status="$1"
    local message="$2"
    local color="#36a64f"  # Green
    
    if [[ "${status}" == "error" ]] || [[ "${status}" == "failed" ]]; then
        color="#d00000"  # Red
    elif [[ "${status}" == "warning" ]]; then
        color="#ffb000"  # Yellow
    fi
    
    if [[ -n "${SLACK_WEBHOOK:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{
                \"attachments\": [{
                    \"color\": \"${color}\",
                    \"title\": \"JJongal-AI Deployment ${status^}\",
                    \"text\": \"${message}\",
                    \"fields\": [
                        {\"title\": \"Environment\", \"value\": \"${ENVIRONMENT}\", \"short\": true},
                        {\"title\": \"Version\", \"value\": \"${DOCKER_TAG}\", \"short\": true},
                        {\"title\": \"Timestamp\", \"value\": \"$(date)\", \"short\": true}
                    ]
                }]
            }" \
            "${SLACK_WEBHOOK}" > /dev/null 2>&1 || warn "Failed to send Slack notification"
    fi
}

# Validate deployment prerequisites
validate_prerequisites() {
    step "Validating deployment prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check if Docker image exists (skip for dry-run)
    if [[ "${DRY_RUN}" == "false" ]]; then
        if [[ -n "${DOCKER_REGISTRY}" ]]; then
            log "Checking if Docker image exists in registry..."
            if ! docker manifest inspect "${FULL_IMAGE_NAME}" &> /dev/null; then
                error "Docker image not found: ${FULL_IMAGE_NAME}"
                error "Please build and push the image first"
                exit 1
            fi
        else
            if ! docker image inspect "${FULL_IMAGE_NAME}" &> /dev/null; then
                error "Local Docker image not found: ${FULL_IMAGE_NAME}"
                error "Please build the image first"
                exit 1
            fi
        fi
    fi
    
    # Check environment configuration
    local env_file="${PROJECT_ROOT}/config/environments/${ENVIRONMENT}.env"
    if [[ ! -f "${env_file}" ]]; then
        error "Environment configuration not found: ${env_file}"
        exit 1
    fi
    
    # Check deployment configuration
    local compose_file="${PROJECT_ROOT}/docker/docker-compose.yml"
    if [[ ! -f "${compose_file}" ]]; then
        error "Docker Compose file not found: ${compose_file}"
        exit 1
    fi
    
    success "Prerequisites validation completed"
}

# Create backup of current deployment
create_backup() {
    if [[ "${SKIP_BACKUP}" == "true" ]]; then
        log "Skipping backup creation"
        return
    fi
    
    step "Creating deployment backup..."
    
    # Create backup directory
    mkdir -p "${BACKUP_DIR}"
    
    # Backup current Docker Compose configuration
    if [[ -f "${PROJECT_ROOT}/docker/docker-compose.yml" ]]; then
        cp "${PROJECT_ROOT}/docker/docker-compose.yml" "${BACKUP_DIR}/docker-compose.yml.bak"
        debug "Backed up docker-compose.yml"
    fi
    
    # Backup environment configuration
    if [[ -f "${PROJECT_ROOT}/config/environments/${ENVIRONMENT}.env" ]]; then
        cp "${PROJECT_ROOT}/config/environments/${ENVIRONMENT}.env" "${BACKUP_DIR}/${ENVIRONMENT}.env.bak"
        debug "Backed up environment configuration"
    fi
    
    # Backup database (if not using external service)
    if [[ -f "${PROJECT_ROOT}/data/${ENVIRONMENT}_jjongal.db" ]]; then
        cp "${PROJECT_ROOT}/data/${ENVIRONMENT}_jjongal.db" "${BACKUP_DIR}/database.bak"
        debug "Backed up SQLite database"
    fi
    
    # Save current Docker image info
    if docker-compose -f "${PROJECT_ROOT}/docker/docker-compose.yml" images --quiet > /dev/null 2>&1; then
        docker-compose -f "${PROJECT_ROOT}/docker/docker-compose.yml" images > "${BACKUP_DIR}/current_images.txt"
        debug "Saved current image information"
    fi
    
    # Create rollback script
    cat > "${BACKUP_DIR}/rollback.sh" << EOF
#!/bin/bash
# Auto-generated rollback script
set -euo pipefail

echo "Rolling back JJongal-AI deployment..."

# Restore configurations
if [[ -f "${BACKUP_DIR}/docker-compose.yml.bak" ]]; then
    cp "${BACKUP_DIR}/docker-compose.yml.bak" "${PROJECT_ROOT}/docker/docker-compose.yml"
    echo "Restored docker-compose.yml"
fi

if [[ -f "${BACKUP_DIR}/${ENVIRONMENT}.env.bak" ]]; then
    cp "${BACKUP_DIR}/${ENVIRONMENT}.env.bak" "${PROJECT_ROOT}/config/environments/${ENVIRONMENT}.env"
    echo "Restored environment configuration"
fi

# Restart services
cd "${PROJECT_ROOT}"
docker-compose -f docker/docker-compose.yml down
docker-compose -f docker/docker-compose.yml up -d

echo "Rollback completed"
EOF
    
    chmod +x "${BACKUP_DIR}/rollback.sh"
    
    log "Backup created at: ${BACKUP_DIR}"
    success "Backup creation completed"
}

# Update Docker Compose configuration
update_compose_config() {
    step "Updating Docker Compose configuration..."
    
    local compose_file="${PROJECT_ROOT}/docker/docker-compose.yml"
    local temp_compose="${compose_file}.tmp"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would update image tag to: ${FULL_IMAGE_NAME}"
        return
    fi
    
    # Update image tag in docker-compose.yml
    if command -v yq &> /dev/null; then
        # Use yq if available for YAML manipulation
        yq eval ".services.app.image = \"${FULL_IMAGE_NAME}\"" "${compose_file}" > "${temp_compose}"
        mv "${temp_compose}" "${compose_file}"
    else
        # Fallback to sed for simple replacement
        sed "s|image: ${DOCKER_REPO}:.*|image: ${FULL_IMAGE_NAME}|g" "${compose_file}" > "${temp_compose}"
        mv "${temp_compose}" "${compose_file}"
    fi
    
    log "Updated image reference to: ${FULL_IMAGE_NAME}"
    success "Docker Compose configuration updated"
}

# Deploy the application
deploy_application() {
    step "Deploying JJongal-AI application..."
    
    cd "${PROJECT_ROOT}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would execute deployment commands:"
        log "[DRY-RUN]   docker-compose -f docker/docker-compose.yml down"
        log "[DRY-RUN]   docker-compose -f docker/docker-compose.yml pull"
        log "[DRY-RUN]   docker-compose -f docker/docker-compose.yml up -d"
        return
    fi
    
    # Stop current services
    log "Stopping current services..."
    docker-compose -f docker/docker-compose.yml down --timeout 30
    
    # Pull new images
    log "Pulling Docker images..."
    docker-compose -f docker/docker-compose.yml pull
    
    # Start services with new configuration
    log "Starting services..."
    docker-compose -f docker/docker-compose.yml up -d
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 10
    
    success "Application deployment completed"
}

# Run health checks
run_health_checks() {
    if [[ "${SKIP_HEALTH_CHECK}" == "true" ]]; then
        log "Skipping health checks"
        return
    fi
    
    step "Running health checks..."
    
    local health_url="http://localhost:8000/api/v1/health"
    local retry_count=0
    local max_retries="${HEALTH_CHECK_RETRIES}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would run health checks against: ${health_url}"
        return
    fi
    
    while [[ $retry_count -lt $max_retries ]]; do
        if curl -sf "${health_url}" > /dev/null 2>&1; then
            local health_response=$(curl -s "${health_url}")
            if echo "${health_response}" | jq -e '.status == "healthy"' > /dev/null 2>&1; then
                success "Health check passed"
                log "Health response: ${health_response}"
                return
            else
                warn "Health check returned unhealthy status: ${health_response}"
            fi
        else
            warn "Health check failed (attempt $((retry_count + 1))/${max_retries})"
        fi
        
        retry_count=$((retry_count + 1))
        if [[ $retry_count -lt $max_retries ]]; then
            log "Waiting 30 seconds before retry..."
            sleep 30
        fi
    done
    
    error "Health checks failed after ${max_retries} attempts"
    
    if [[ "${FORCE}" == "false" ]]; then
        error "Deployment failed health checks"
        send_notification "failed" "Deployment failed health checks"
        exit 1
    else
        warn "Continuing deployment despite health check failures (--force enabled)"
    fi
}

# Perform rollback
perform_rollback() {
    step "Performing rollback..."
    
    case "${ROLLBACK_STRATEGY}" in
        "previous")
            if [[ -f "${BACKUP_DIR}/rollback.sh" ]]; then
                log "Executing automatic rollback..."
                bash "${BACKUP_DIR}/rollback.sh"
            else
                error "No previous backup found for automatic rollback"
                exit 1
            fi
            ;;
        "tag")
            if [[ -z "${ROLLBACK_TO}" ]]; then
                error "Rollback tag not specified. Use --rollback-to TAG"
                exit 1
            fi
            log "Rolling back to specific tag: ${ROLLBACK_TO}"
            DOCKER_TAG="${ROLLBACK_TO}"
            FULL_IMAGE_NAME="${DOCKER_REPO}:${DOCKER_TAG}"
            update_compose_config
            deploy_application
            ;;
        "manual")
            log "Manual rollback requested. Please execute rollback manually."
            log "Backup location: ${BACKUP_DIR}"
            exit 0
            ;;
        *)
            error "Unknown rollback strategy: ${ROLLBACK_STRATEGY}"
            exit 1
            ;;
    esac
    
    # Run health checks after rollback
    run_health_checks
    
    success "Rollback completed successfully"
    send_notification "success" "Rollback completed successfully"
}

# Clean up old backups
cleanup_old_backups() {
    step "Cleaning up old backups..."
    
    # Keep last 5 backups
    local backup_parent="/tmp"
    local backup_pattern="jjongal_deploy_backup_*"
    
    # Find and remove old backups (keep latest 5)
    find "${backup_parent}" -maxdepth 1 -name "${backup_pattern}" -type d | \
        sort -r | \
        tail -n +6 | \
        xargs -r rm -rf
    
    debug "Old backups cleaned up"
}

# Display deployment summary
show_deployment_summary() {
    echo ""
    success "=€ JJongal-AI deployment completed!"
    echo ""
    log "Deployment Summary:"
    log "  Environment: ${ENVIRONMENT}"
    log "  Docker Image: ${FULL_IMAGE_NAME}"
    log "  Deployment Time: $(date)"
    log "  Backup Location: ${BACKUP_DIR}"
    log "  Log File: ${LOG_FILE}"
    
    if [[ "${SKIP_HEALTH_CHECK}" == "false" ]]; then
        log "  Health Status:  Healthy"
    fi
    
    echo ""
    log "Service URLs:"
    log "  API Documentation: http://localhost:8000/docs"
    log "  Health Check: http://localhost:8000/api/v1/health"
    log "  Application: http://localhost:8000"
    
    echo ""
    log "Post-deployment commands:"
    log "  View logs: docker-compose -f docker/docker-compose.yml logs -f"
    log "  Check status: docker-compose -f docker/docker-compose.yml ps"
    log "  Rollback: $0 rollback"
}

# Main execution function
main() {
    echo -e "${CYAN}${SCRIPT_NAME} v${SCRIPT_VERSION}${NC}"
    echo ""
    
    # Initialize log file
    touch "${LOG_FILE}"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Log deployment start
    log "Starting deployment with configuration:"
    log "  Command: ${COMMAND}"
    log "  Environment: ${ENVIRONMENT}"
    log "  Docker Image: ${FULL_IMAGE_NAME}"
    log "  Timeout: ${TIMEOUT}s"
    log "  Dry Run: ${DRY_RUN}"
    log "  Force: ${FORCE}"
    echo ""
    
    # Send start notification
    send_notification "started" "Deployment started for ${ENVIRONMENT} environment"
    
    # Execute command
    case "${COMMAND}" in
        "deploy")
            if [[ "${FORCE}" == "false" ]] && [[ "${DRY_RUN}" == "false" ]]; then
                echo -e "${YELLOW}Are you sure you want to deploy to ${ENVIRONMENT}? (y/N)${NC}"
                read -r confirmation
                if [[ ! "${confirmation}" =~ ^[Yy]$ ]]; then
                    log "Deployment cancelled by user"
                    exit 0
                fi
            fi
            
            validate_prerequisites
            create_backup
            update_compose_config
            deploy_application
            run_health_checks
            cleanup_old_backups
            show_deployment_summary
            send_notification "success" "Deployment completed successfully"
            ;;
        "rollback")
            perform_rollback
            ;;
        "health-check")
            run_health_checks
            success "Health check completed"
            ;;
        "backup")
            create_backup
            success "Backup completed"
            ;;
        "validate")
            validate_prerequisites
            success "Validation completed"
            ;;
        *)
            error "Unknown command: ${COMMAND}"
            exit 1
            ;;
    esac
    
    success "Operation completed successfully! <‰"
}

# Handle script interruption
trap 'error "Deployment interrupted by user"; send_notification "failed" "Deployment interrupted"; exit 130' INT TERM

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi