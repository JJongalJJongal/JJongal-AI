#!/bin/bash
# JJongal-AI Data Backup Script
# Comprehensive backup solution for databases, files, and configurations

set -euo pipefail

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Script configuration
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="JJongal-AI Data Backup Script"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Default configuration
DEFAULT_BACKUP_DIR="/data/backups"
DEFAULT_RETENTION_DAYS="30"
DEFAULT_COMPRESSION="gzip"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BACKUP_PREFIX="jjongal_backup"

# Backup configuration
BACKUP_DIR="${BACKUP_DIR:-$DEFAULT_BACKUP_DIR}"
RETENTION_DAYS="${RETENTION_DAYS:-$DEFAULT_RETENTION_DAYS}"
COMPRESSION="${COMPRESSION:-$DEFAULT_COMPRESSION}"
BACKUP_NAME="${BACKUP_PREFIX}_${TIMESTAMP}"
FULL_BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# AWS S3 configuration (optional)
S3_ENABLED="${S3_ENABLED:-false}"
S3_BUCKET="${BACKUP_S3_BUCKET:-}"
S3_PREFIX="${S3_BACKUP_PREFIX:-jjongal-backups}"

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
    $0 [OPTIONS] [COMMAND]

COMMANDS:
    full            Full backup of all data (default)
    database        Database backup only
    files           Files backup only  
    config          Configuration backup only
    vector-db       Vector database backup only
    restore         Restore from backup
    list            List available backups
    cleanup         Clean up old backups

OPTIONS:
    -d, --backup-dir DIR        Backup directory
                               Default: ${DEFAULT_BACKUP_DIR}
    -r, --retention-days DAYS   Retention period in days
                               Default: ${DEFAULT_RETENTION_DAYS}
    -c, --compression TYPE      Compression type (gzip, xz, none)
                               Default: ${DEFAULT_COMPRESSION}
    --s3-upload                Enable S3 upload
    --s3-bucket BUCKET         S3 bucket name
    --encrypt                  Enable encryption
    --exclude-uploads          Exclude user uploads from backup
    --exclude-logs             Exclude log files from backup
    --dry-run                  Show what would be backed up
    --force                    Force backup without confirmation
    --restore-from FILE        Restore from specific backup file
    -h, --help                 Show this help message

EXAMPLES:
    $0                                    # Full backup
    $0 database                          # Database only
    $0 --s3-upload                       # Full backup with S3 upload
    $0 restore --restore-from backup.tar.gz
    $0 cleanup                           # Clean old backups
    $0 list                             # List available backups

ENVIRONMENT VARIABLES:
    BACKUP_DIR              Default backup directory
    RETENTION_DAYS          Backup retention period
    S3_BUCKET              S3 bucket for backup uploads
    BACKUP_ENCRYPTION_KEY   Encryption key for backups

EOF
}

# Parse command line arguments
parse_args() {
    COMMAND="full"
    DRY_RUN=false
    FORCE=false
    ENCRYPT=false
    EXCLUDE_UPLOADS=false
    EXCLUDE_LOGS=false
    RESTORE_FROM=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            full|database|files|config|vector-db|restore|list|cleanup)
                COMMAND="$1"
                shift
                ;;
            -d|--backup-dir)
                BACKUP_DIR="$2"
                shift 2
                ;;
            -r|--retention-days)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -c|--compression)
                COMPRESSION="$2"
                shift 2
                ;;
            --s3-upload)
                S3_ENABLED=true
                shift
                ;;
            --s3-bucket)
                S3_BUCKET="$2"
                shift 2
                ;;
            --encrypt)
                ENCRYPT=true
                shift
                ;;
            --exclude-uploads)
                EXCLUDE_UPLOADS=true
                shift
                ;;
            --exclude-logs)
                EXCLUDE_LOGS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --restore-from)
                RESTORE_FROM="$2"
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
    
    # Validate compression type
    case $COMPRESSION in
        gzip|xz|none)
            ;;
        *)
            error "Invalid compression type: $COMPRESSION"
            error "Valid options: gzip, xz, none"
            exit 1
            ;;
    esac
    
    # Update backup path based on compression
    case $COMPRESSION in
        gzip)
            FULL_BACKUP_PATH="${FULL_BACKUP_PATH}.tar.gz"
            ;;
        xz)
            FULL_BACKUP_PATH="${FULL_BACKUP_PATH}.tar.xz"
            ;;
        none)
            FULL_BACKUP_PATH="${FULL_BACKUP_PATH}.tar"
            ;;
    esac
}

# Create backup directory
create_backup_directory() {
    if [[ ! -d "${BACKUP_DIR}" ]]; then
        step "Creating backup directory: ${BACKUP_DIR}"
        if [[ "${DRY_RUN}" == "false" ]]; then
            mkdir -p "${BACKUP_DIR}"
        fi
    fi
    
    # Check write permissions
    if [[ "${DRY_RUN}" == "false" ]] && [[ ! -w "${BACKUP_DIR}" ]]; then
        error "No write permission to backup directory: ${BACKUP_DIR}"
        exit 1
    fi
}

# Load environment configuration
load_environment() {
    # Try to load environment configuration
    local env_files=(
        "${PROJECT_ROOT}/.env"
        "${PROJECT_ROOT}/config/environments/production.env"
        "${PROJECT_ROOT}/config/environments/development.env"
    )
    
    for env_file in "${env_files[@]}"; do
        if [[ -f "${env_file}" ]]; then
            source "${env_file}"
            log "Loaded environment from: ${env_file}"
            break
        fi
    done
}

# Backup database
backup_database() {
    step "Backing up database..."
    
    local db_backup_dir="${PROJECT_ROOT}/temp/db_backup_${TIMESTAMP}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would backup database to: ${db_backup_dir}"
        return
    fi
    
    mkdir -p "${db_backup_dir}"
    
    # Handle different database types
    if [[ -n "${DATABASE_URL:-}" ]]; then
        if [[ "${DATABASE_URL}" == sqlite* ]]; then
            # SQLite backup
            local sqlite_file=$(echo "${DATABASE_URL}" | sed 's/sqlite:\/\/\///g')
            if [[ -f "${sqlite_file}" ]]; then
                log "Backing up SQLite database: ${sqlite_file}"
                cp "${sqlite_file}" "${db_backup_dir}/database.sqlite"
                
                # Create SQL dump for portability
                if command -v sqlite3 &> /dev/null; then
                    sqlite3 "${sqlite_file}" .dump > "${db_backup_dir}/database.sql"
                    log "Created SQL dump: database.sql"
                fi
            else
                warn "SQLite database file not found: ${sqlite_file}"
            fi
        elif [[ "${DATABASE_URL}" == postgres* ]]; then
            # PostgreSQL backup
            log "Backing up PostgreSQL database..."
            if command -v pg_dump &> /dev/null; then
                pg_dump "${DATABASE_URL}" > "${db_backup_dir}/database.sql" || {
                    warn "PostgreSQL backup failed"
                    return 1
                }
                log "PostgreSQL backup completed"
            else
                warn "pg_dump not available, skipping PostgreSQL backup"
            fi
        elif [[ "${DATABASE_URL}" == mysql* ]]; then
            # MySQL backup
            log "Backing up MySQL database..."
            if command -v mysqldump &> /dev/null; then
                # Extract connection details from URL for mysqldump
                mysqldump --single-transaction --routines --triggers "${DATABASE_URL}" > "${db_backup_dir}/database.sql" || {
                    warn "MySQL backup failed"
                    return 1
                }
                log "MySQL backup completed"
            else
                warn "mysqldump not available, skipping MySQL backup"
            fi
        else
            warn "Unknown database type: ${DATABASE_URL}"
        fi
    else
        warn "No DATABASE_URL configured, skipping database backup"
    fi
    
    success "Database backup completed"
    echo "${db_backup_dir}"
}

# Backup vector database
backup_vector_database() {
    step "Backing up vector database..."
    
    local vector_db_backup_dir="${PROJECT_ROOT}/temp/vector_db_backup_${TIMESTAMP}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would backup vector database to: ${vector_db_backup_dir}"
        return
    fi
    
    # Check for vector database directory
    local vector_db_paths=(
        "${VECTOR_DB_PATH:-}"
        "${CHROMA_PERSIST_DIRECTORY:-}"
        "${PROJECT_ROOT}/data/vector_db"
        "${PROJECT_ROOT}/data/vector_db/dev"
        "${PROJECT_ROOT}/data/vector_db/prod"
    )
    
    mkdir -p "${vector_db_backup_dir}"
    
    for vector_path in "${vector_db_paths[@]}"; do
        if [[ -n "${vector_path}" ]] && [[ -d "${vector_path}" ]]; then
            log "Backing up vector database from: ${vector_path}"
            cp -r "${vector_path}" "${vector_db_backup_dir}/vector_db"
            success "Vector database backup completed"
            echo "${vector_db_backup_dir}"
            return
        fi
    done
    
    warn "No vector database directory found, skipping vector DB backup"
    rmdir "${vector_db_backup_dir}" 2>/dev/null || true
}

# Backup configuration files
backup_configuration() {
    step "Backing up configuration files..."
    
    local config_backup_dir="${PROJECT_ROOT}/temp/config_backup_${TIMESTAMP}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would backup configuration to: ${config_backup_dir}"
        return
    fi
    
    mkdir -p "${config_backup_dir}"
    
    # Backup configuration directories and files
    local config_paths=(
        "config/"
        ".env"
        "docker/docker-compose.yml"
        "docker/Dockerfile"
        "requirements.txt"
        "Makefile"
    )
    
    for config_path in "${config_paths[@]}"; do
        local full_path="${PROJECT_ROOT}/${config_path}"
        if [[ -e "${full_path}" ]]; then
            if [[ -d "${full_path}" ]]; then
                cp -r "${full_path}" "${config_backup_dir}/"
                log "Backed up directory: ${config_path}"
            else
                cp "${full_path}" "${config_backup_dir}/"
                log "Backed up file: ${config_path}"
            fi
        fi
    done
    
    success "Configuration backup completed"
    echo "${config_backup_dir}"
}

# Backup application files
backup_files() {
    step "Backing up application files..."
    
    local files_backup_dir="${PROJECT_ROOT}/temp/files_backup_${TIMESTAMP}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would backup files to: ${files_backup_dir}"
        return
    fi
    
    mkdir -p "${files_backup_dir}"
    
    # Define what to backup
    local backup_paths=(
        "data/"
        "uploads/"
        "temp/"
    )
    
    # Add uploads if not excluded
    if [[ "${EXCLUDE_UPLOADS}" == "false" ]]; then
        backup_paths+=("uploads/")
    fi
    
    # Add logs if not excluded
    if [[ "${EXCLUDE_LOGS}" == "false" ]]; then
        backup_paths+=("logs/")
    fi
    
    for backup_path in "${backup_paths[@]}"; do
        local full_path="${PROJECT_ROOT}/${backup_path}"
        if [[ -d "${full_path}" ]]; then
            # Skip empty directories
            if [[ -n "$(find "${full_path}" -type f 2>/dev/null | head -1)" ]]; then
                cp -r "${full_path}" "${files_backup_dir}/"
                log "Backed up directory: ${backup_path}"
            else
                log "Skipped empty directory: ${backup_path}"
            fi
        fi
    done
    
    success "Files backup completed"
    echo "${files_backup_dir}"
}

# Create compressed archive
create_archive() {
    local source_dirs=("$@")
    
    if [[ ${#source_dirs[@]} -eq 0 ]]; then
        error "No source directories provided for archive creation"
        return 1
    fi
    
    step "Creating compressed archive: ${FULL_BACKUP_PATH}"
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would create archive with compression: ${COMPRESSION}"
        log "[DRY-RUN] Source directories: ${source_dirs[*]}"
        return
    fi
    
    # Create temporary directory for organizing backup content
    local temp_backup_dir="${PROJECT_ROOT}/temp/backup_staging_${TIMESTAMP}"
    mkdir -p "${temp_backup_dir}"
    
    # Copy all backup directories to staging area
    for dir in "${source_dirs[@]}"; do
        if [[ -d "${dir}" ]]; then
            cp -r "${dir}"/* "${temp_backup_dir}/" 2>/dev/null || true
        fi
    done
    
    # Create metadata file
    cat > "${temp_backup_dir}/backup_metadata.json" << EOF
{
    "backup_timestamp": "${TIMESTAMP}",
    "backup_version": "${SCRIPT_VERSION}",
    "backup_type": "${COMMAND}",
    "source_host": "$(hostname)",
    "source_path": "${PROJECT_ROOT}",
    "compression": "${COMPRESSION}",
    "encrypted": ${ENCRYPT},
    "excluded_uploads": ${EXCLUDE_UPLOADS},
    "excluded_logs": ${EXCLUDE_LOGS}
}
EOF
    
    # Create archive with appropriate compression
    cd "${PROJECT_ROOT}/temp"
    
    case $COMPRESSION in
        gzip)
            tar -czf "${FULL_BACKUP_PATH}" -C "backup_staging_${TIMESTAMP}" .
            ;;
        xz)
            tar -cJf "${FULL_BACKUP_PATH}" -C "backup_staging_${TIMESTAMP}" .
            ;;
        none)
            tar -cf "${FULL_BACKUP_PATH}" -C "backup_staging_${TIMESTAMP}" .
            ;;
    esac
    
    # Clean up temporary directories
    rm -rf "${temp_backup_dir}"
    for dir in "${source_dirs[@]}"; do
        if [[ -d "${dir}" ]]; then
            rm -rf "${dir}"
        fi
    done
    
    # Get archive size
    local archive_size
    if command -v du &> /dev/null; then
        archive_size=$(du -h "${FULL_BACKUP_PATH}" | cut -f1)
        log "Archive size: ${archive_size}"
    fi
    
    success "Archive created: ${FULL_BACKUP_PATH}"
}

# Encrypt backup if requested
encrypt_backup() {
    if [[ "${ENCRYPT}" == "false" ]]; then
        return
    fi
    
    step "Encrypting backup..."
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would encrypt backup file"
        return
    fi
    
    if [[ -z "${BACKUP_ENCRYPTION_KEY:-}" ]]; then
        error "Encryption requested but BACKUP_ENCRYPTION_KEY not set"
        exit 1
    fi
    
    # Use OpenSSL for encryption
    if command -v openssl &> /dev/null; then
        local encrypted_file="${FULL_BACKUP_PATH}.enc"
        openssl enc -aes-256-cbc -salt -in "${FULL_BACKUP_PATH}" -out "${encrypted_file}" -k "${BACKUP_ENCRYPTION_KEY}"
        rm "${FULL_BACKUP_PATH}"
        FULL_BACKUP_PATH="${encrypted_file}"
        success "Backup encrypted: ${FULL_BACKUP_PATH}"
    else
        error "OpenSSL not available for encryption"
        exit 1
    fi
}

# Upload to S3 if enabled
upload_to_s3() {
    if [[ "${S3_ENABLED}" == "false" ]]; then
        return
    fi
    
    step "Uploading backup to S3..."
    
    if [[ -z "${S3_BUCKET}" ]]; then
        error "S3 upload enabled but S3_BUCKET not set"
        exit 1
    fi
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would upload to S3: s3://${S3_BUCKET}/${S3_PREFIX}/"
        return
    fi
    
    # Use AWS CLI if available
    if command -v aws &> /dev/null; then
        local s3_path="s3://${S3_BUCKET}/${S3_PREFIX}/$(basename "${FULL_BACKUP_PATH}")"
        aws s3 cp "${FULL_BACKUP_PATH}" "${s3_path}"
        success "Backup uploaded to: ${s3_path}"
    else
        error "AWS CLI not available for S3 upload"
        exit 1
    fi
}

# Clean up old backups
cleanup_old_backups() {
    step "Cleaning up old backups..."
    
    if [[ "${DRY_RUN}" == "true" ]]; then
        log "[DRY-RUN] Would clean up backups older than ${RETENTION_DAYS} days"
        return
    fi
    
    # Clean local backups
    if [[ -d "${BACKUP_DIR}" ]]; then
        find "${BACKUP_DIR}" -name "${BACKUP_PREFIX}_*" -type f -mtime +${RETENTION_DAYS} -delete
        log "Cleaned up local backups older than ${RETENTION_DAYS} days"
    fi
    
    # Clean S3 backups if enabled
    if [[ "${S3_ENABLED}" == "true" ]] && command -v aws &> /dev/null; then
        local cutoff_date=$(date -d "-${RETENTION_DAYS} days" +%Y-%m-%d)
        aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" | while read -r line; do
            local file_date=$(echo "${line}" | awk '{print $1}')
            if [[ "${file_date}" < "${cutoff_date}" ]]; then
                local file_name=$(echo "${line}" | awk '{print $4}')
                aws s3 rm "s3://${S3_BUCKET}/${S3_PREFIX}/${file_name}"
                log "Deleted old S3 backup: ${file_name}"
            fi
        done
    fi
    
    success "Old backups cleanup completed"
}

# List available backups
list_backups() {
    step "Available backups:"
    
    # List local backups
    if [[ -d "${BACKUP_DIR}" ]]; then
        log "Local backups in ${BACKUP_DIR}:"
        find "${BACKUP_DIR}" -name "${BACKUP_PREFIX}_*" -type f -exec ls -lh {} \; | \
            awk '{print "  " $9 " (" $5 ", " $6 " " $7 " " $8 ")"}'
    fi
    
    # List S3 backups if enabled
    if [[ "${S3_ENABLED}" == "true" ]] && command -v aws &> /dev/null; then
        log "S3 backups in s3://${S3_BUCKET}/${S3_PREFIX}/:"
        aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" | \
            awk '{print "  s3://'${S3_BUCKET}'/'${S3_PREFIX}'/" $4 " (" $3 ", " $1 " " $2 ")"}'
    fi
}

# Perform backup based on command
perform_backup() {
    local backup_dirs=()
    
    case "${COMMAND}" in
        "full")
            log "Performing full backup..."
            backup_dirs+=($(backup_database))
            backup_dirs+=($(backup_vector_database))  
            backup_dirs+=($(backup_configuration))
            backup_dirs+=($(backup_files))
            ;;
        "database")
            log "Performing database backup..."
            backup_dirs+=($(backup_database))
            ;;
        "vector-db")
            log "Performing vector database backup..."
            backup_dirs+=($(backup_vector_database))
            ;;
        "config")
            log "Performing configuration backup..."
            backup_dirs+=($(backup_configuration))
            ;;
        "files")
            log "Performing files backup..."
            backup_dirs+=($(backup_files))
            ;;
        *)
            error "Unknown backup command: ${COMMAND}"
            exit 1
            ;;
    esac
    
    # Filter out empty directories
    local valid_dirs=()
    for dir in "${backup_dirs[@]}"; do
        if [[ -n "${dir}" ]] && [[ -d "${dir}" ]]; then
            valid_dirs+=("${dir}")
        fi
    done
    
    if [[ ${#valid_dirs[@]} -eq 0 ]]; then
        warn "No data to backup"
        return
    fi
    
    # Create archive
    create_archive "${valid_dirs[@]}"
    
    # Encrypt if requested
    encrypt_backup
    
    # Upload to S3 if enabled
    upload_to_s3
}

# Show backup summary
show_backup_summary() {
    echo ""
    success "<‰ Backup completed successfully!"
    echo ""
    log "Backup Summary:"
    log "  Type: ${COMMAND}"
    log "  File: ${FULL_BACKUP_PATH}"
    
    if [[ -f "${FULL_BACKUP_PATH}" ]]; then
        local file_size
        file_size=$(du -h "${FULL_BACKUP_PATH}" | cut -f1)
        log "  Size: ${file_size}"
    fi
    
    log "  Timestamp: ${TIMESTAMP}"
    log "  Compression: ${COMPRESSION}"
    log "  Encrypted: ${ENCRYPT}"
    
    if [[ "${S3_ENABLED}" == "true" ]]; then
        log "  S3 Upload:  Completed"
    fi
    
    echo ""
    log "Backup retention: ${RETENTION_DAYS} days"
    log "Next cleanup will remove backups older than $(date -d "+${RETENTION_DAYS} days" '+%Y-%m-%d')"
}

# Main execution
main() {
    echo -e "${CYAN}${SCRIPT_NAME} v${SCRIPT_VERSION}${NC}"
    echo ""
    
    # Parse arguments
    parse_args "$@"
    
    # Load environment
    load_environment
    
    # Create backup directory
    create_backup_directory
    
    # Show configuration
    log "Backup Configuration:"
    log "  Command: ${COMMAND}"
    log "  Backup Directory: ${BACKUP_DIR}"
    log "  Retention: ${RETENTION_DAYS} days"
    log "  Compression: ${COMPRESSION}"
    log "  S3 Upload: ${S3_ENABLED}"
    log "  Encryption: ${ENCRYPT}"
    log "  Dry Run: ${DRY_RUN}"
    echo ""
    
    # Execute command
    case "${COMMAND}" in
        "full"|"database"|"vector-db"|"config"|"files")
            # Confirmation for non-force mode
            if [[ "${FORCE}" == "false" ]] && [[ "${DRY_RUN}" == "false" ]]; then
                echo -e "${YELLOW}Proceed with ${COMMAND} backup? (y/N)${NC}"
                read -r confirmation
                if [[ ! "${confirmation}" =~ ^[Yy]$ ]]; then
                    log "Backup cancelled by user"
                    exit 0
                fi
            fi
            
            perform_backup
            show_backup_summary
            ;;
        "cleanup")
            cleanup_old_backups
            ;;
        "list")
            list_backups
            ;;
        "restore")
            if [[ -z "${RESTORE_FROM}" ]]; then
                error "Restore file not specified. Use --restore-from FILE"
                exit 1
            fi
            # Restore functionality would be implemented here
            error "Restore functionality not yet implemented"
            exit 1
            ;;
        *)
            error "Unknown command: ${COMMAND}"
            exit 1
            ;;
    esac
    
    success "Operation completed successfully! =€"
}

# Handle interruption
trap 'error "Backup interrupted by user"; exit 130' INT TERM

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi