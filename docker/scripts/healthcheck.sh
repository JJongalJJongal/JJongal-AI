#!/bin/bash
# JJongal-AI Docker Health Check Script
# Comprehensive health monitoring for containerized application

set -euo pipefail

# Configuration
HEALTH_CHECK_TIMEOUT=${HEALTH_CHECK_TIMEOUT:-10}
API_HOST=${HOST:-localhost}
API_PORT=${PORT:-8000}
HEALTH_ENDPOINT="/api/v1/health"
READY_ENDPOINT="/api/v1/health/ready"
LIVE_ENDPOINT="/api/v1/health/live"

# Health check URLs
HEALTH_URL="http://${API_HOST}:${API_PORT}${HEALTH_ENDPOINT}"
READY_URL="http://${API_HOST}:${API_PORT}${READY_ENDPOINT}"
LIVE_URL="http://${API_HOST}:${API_PORT}${LIVE_ENDPOINT}"

# Exit codes
EXIT_SUCCESS=0
EXIT_HEALTH_FAIL=1
EXIT_TIMEOUT=2
EXIT_CONNECTION_FAIL=3

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] HEALTHCHECK: $1" >&2
}

error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] HEALTHCHECK ERROR: $1" >&2
}

# Function to check if service is listening on port
check_port_listening() {
    local port="$1"
    
    if command -v ss &> /dev/null; then
        ss -tuln | grep -q ":${port} " && return 0
    elif command -v netstat &> /dev/null; then
        netstat -tuln | grep -q ":${port} " && return 0
    elif command -v nc &> /dev/null; then
        nc -z localhost "${port}" 2>/dev/null && return 0
    else
        # Fallback: try to connect using curl with very short timeout
        curl -s --connect-timeout 1 "http://localhost:${port}" &>/dev/null && return 0
    fi
    
    return 1
}

# Function to check HTTP endpoint with timeout
check_http_endpoint() {
    local url="$1"
    local expected_status="${2:-200}"
    local timeout="${3:-5}"
    
    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        error "curl is not available for health checks"
        return 1
    fi
    
    # Make HTTP request with timeout
    local response
    if response=$(curl -s -f --max-time "${timeout}" --connect-timeout 2 "${url}" 2>/dev/null); then
        # Check if response is valid JSON and contains expected health data
        if command -v jq &> /dev/null; then
            # Use jq if available for JSON parsing
            if echo "${response}" | jq -e '.status' &>/dev/null; then
                local status
                status=$(echo "${response}" | jq -r '.status')
                if [[ "${status}" == "healthy" ]] || [[ "${status}" == "ok" ]]; then
                    return 0
                else
                    error "Health endpoint returned unhealthy status: ${status}"
                    return 1
                fi
            else
                error "Invalid JSON response from health endpoint"
                return 1
            fi
        else
            # Fallback: check for basic health indicators in response
            if [[ "${response}" == *"healthy"* ]] || [[ "${response}" == *"ok"* ]]; then
                return 0
            else
                error "Health endpoint response does not indicate healthy status"
                return 1
            fi
        fi
    else
        error "Failed to connect to health endpoint: ${url}"
        return 1
    fi
}

# Function to check database connectivity
check_database_health() {
    # This is a simplified check - in a real scenario, you might want to
    # make a more specific database health check through the API
    
    if [[ -n "${DATABASE_URL:-}" ]]; then
        log "Database health check via API endpoint"
        # Database health is checked through the main health endpoint
        return 0
    else
        log "No database configuration found, skipping database health check"
        return 0
    fi
}

# Function to check Redis connectivity (if enabled)
check_redis_health() {
    if [[ "${REDIS_ENABLED:-false}" == "true" ]] && [[ -n "${REDIS_URL:-}" ]]; then
        log "Redis health check via API endpoint"
        # Redis health is checked through the main health endpoint
        return 0
    else
        log "Redis not enabled, skipping Redis health check"
        return 0
    fi
}

# Function to check Vector Database health
check_vector_db_health() {
    if [[ -n "${VECTOR_DB_PATH:-}" ]]; then
        # Check if vector database directory exists and is accessible
        if [[ -d "${VECTOR_DB_PATH}" ]]; then
            log "Vector database directory accessible"
            return 0
        else
            error "Vector database directory not accessible: ${VECTOR_DB_PATH}"
            return 1
        fi
    else
        log "No vector database configuration, skipping vector DB health check"
        return 0
    fi
}

# Function to check application processes
check_application_processes() {
    # Check if Python processes are running
    if pgrep -f "python.*app" > /dev/null; then
        log "Application processes are running"
        return 0
    elif pgrep -f "uvicorn.*app" > /dev/null; then
        log "Uvicorn processes are running"
        return 0
    else
        error "No application processes found"
        return 1
    fi
}

# Function to check memory usage
check_memory_usage() {
    local max_memory_percent="${MAX_MEMORY_USAGE:-90}"
    
    if command -v free &> /dev/null; then
        local memory_usage
        memory_usage=$(free | awk '/^Mem:/ {printf "%.0f", ($3/$2)*100}')
        
        if [[ "${memory_usage}" -gt "${max_memory_percent}" ]]; then
            error "Memory usage too high: ${memory_usage}%"
            return 1
        else
            log "Memory usage acceptable: ${memory_usage}%"
            return 0
        fi
    else
        log "Memory check not available (free command not found)"
        return 0
    fi
}

# Function to check disk space
check_disk_space() {
    local max_disk_percent="${MAX_DISK_USAGE:-90}"
    
    if command -v df &> /dev/null; then
        # Check root filesystem
        local disk_usage
        disk_usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
        
        if [[ "${disk_usage}" -gt "${max_disk_percent}" ]]; then
            error "Disk usage too high: ${disk_usage}%"
            return 1
        else
            log "Disk usage acceptable: ${disk_usage}%"
            return 0
        fi
    else
        log "Disk check not available (df command not found)"
        return 0
    fi
}

# Main health check function
perform_health_check() {
    local check_type="${1:-full}"
    local failed_checks=0
    
    log "Starting ${check_type} health check"
    
    # Basic port check
    if ! check_port_listening "${API_PORT}"; then
        error "Application is not listening on port ${API_PORT}"
        return ${EXIT_CONNECTION_FAIL}
    fi
    log "Port ${API_PORT} is listening"
    
    # Application process check
    if ! check_application_processes; then
        ((failed_checks++))
    fi
    
    # HTTP endpoint checks
    case "${check_type}" in
        "live"|"liveness")
            # Minimal liveness check
            if ! check_http_endpoint "${LIVE_URL}" 200 3; then
                error "Liveness check failed"
                return ${EXIT_HEALTH_FAIL}
            fi
            log "Liveness check passed"
            ;;
        "ready"|"readiness")
            # Readiness check includes dependencies
            if ! check_http_endpoint "${READY_URL}" 200 5; then
                error "Readiness check failed"
                return ${EXIT_HEALTH_FAIL}
            fi
            log "Readiness check passed"
            ;;
        "full"|*)
            # Full health check
            if ! check_http_endpoint "${HEALTH_URL}" 200 "${HEALTH_CHECK_TIMEOUT}"; then
                error "Full health check failed"
                return ${EXIT_HEALTH_FAIL}
            fi
            log "Full health check passed"
            
            # Additional checks for full health check
            if ! check_database_health; then
                ((failed_checks++))
            fi
            
            if ! check_redis_health; then
                ((failed_checks++))
            fi
            
            if ! check_vector_db_health; then
                ((failed_checks++))
            fi
            
            if ! check_memory_usage; then
                ((failed_checks++))
            fi
            
            if ! check_disk_space; then
                ((failed_checks++))
            fi
            ;;
    esac
    
    # Evaluate overall health
    if [[ ${failed_checks} -eq 0 ]]; then
        log "All health checks passed"
        return ${EXIT_SUCCESS}
    elif [[ ${failed_checks} -le 2 ]]; then
        log "Health check passed with ${failed_checks} minor issues"
        return ${EXIT_SUCCESS}
    else
        error "Health check failed with ${failed_checks} issues"
        return ${EXIT_HEALTH_FAIL}
    fi
}

# Function to show health status
show_health_status() {
    echo "JJongal-AI Health Check Status"
    echo "=============================="
    echo "Timestamp: $(date)"
    echo "Host: ${API_HOST}:${API_PORT}"
    echo "Health URL: ${HEALTH_URL}"
    echo ""
    
    # Try to get detailed health information
    if command -v curl &> /dev/null; then
        echo "Detailed Health Information:"
        if response=$(curl -s --max-time 5 "${HEALTH_URL}" 2>/dev/null); then
            if command -v jq &> /dev/null; then
                echo "${response}" | jq . 2>/dev/null || echo "${response}"
            else
                echo "${response}"
            fi
        else
            echo "Failed to retrieve health information"
        fi
    fi
}

# Main execution
main() {
    local check_type="full"
    local show_status=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            live|liveness)
                check_type="live"
                shift
                ;;
            ready|readiness)
                check_type="ready"
                shift
                ;;
            full|health)
                check_type="full"
                shift
                ;;
            status)
                show_status=true
                shift
                ;;
            --timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            --help|-h)
                echo "JJongal-AI Docker Health Check"
                echo ""
                echo "Usage: $0 [TYPE] [OPTIONS]"
                echo ""
                echo "Types:"
                echo "  live, liveness    Basic liveness check"
                echo "  ready, readiness  Readiness check with dependencies"
                echo "  full, health      Full health check (default)"
                echo "  status            Show detailed health status"
                echo ""
                echo "Options:"
                echo "  --timeout SECONDS  Health check timeout (default: ${HEALTH_CHECK_TIMEOUT})"
                echo "  --help, -h         Show this help"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    if [[ "${show_status}" == "true" ]]; then
        show_health_status
        exit 0
    fi
    
    # Perform the health check
    if perform_health_check "${check_type}"; then
        log "Health check completed successfully"
        exit ${EXIT_SUCCESS}
    else
        error "Health check failed"
        exit ${EXIT_HEALTH_FAIL}
    fi
}

# Execute main function with all arguments
main "$@"