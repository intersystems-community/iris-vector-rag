#!/bin/bash
# =============================================================================
# RAG Templates Framework - One-Click Startup Script
# =============================================================================
# This script provides one-click deployment of the complete RAG framework
# with all services, data loading, and health monitoring
# =============================================================================

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.full.yml"
ENV_FILE="${PROJECT_ROOT}/.env"
LOG_FILE="${PROJECT_ROOT}/logs/docker-startup.log"

# Default options
PROFILE="core"
LOAD_DATA=false
DETACHED=true
BUILD=false
CLEAN=false

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[RAG-DEPLOY]${NC} ${message}"
}

# Function to log messages
log_message() {
    local message=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" >> "$LOG_FILE"
    print_message "$BLUE" "$message"
}

# Function to check prerequisites
check_prerequisites() {
    print_message "$BLUE" "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_message "$RED" "ERROR: Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_message "$RED" "ERROR: Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_message "$RED" "ERROR: Docker daemon is not running"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        print_message "$YELLOW" "WARNING: .env file not found, copying from .env.example"
        if [[ -f "${PROJECT_ROOT}/.env.example" ]]; then
            cp "${PROJECT_ROOT}/.env.example" "$ENV_FILE"
            print_message "$YELLOW" "Please edit .env file with your configuration before continuing"
            read -p "Press Enter to continue after editing .env file..."
        else
            print_message "$RED" "ERROR: Neither .env nor .env.example found"
            exit 1
        fi
    fi
    
    print_message "$GREEN" "Prerequisites check passed"
}

# Function to create necessary directories
create_directories() {
    print_message "$BLUE" "Creating necessary directories..."
    
    local dirs=(
        "${PROJECT_ROOT}/logs"
        "${PROJECT_ROOT}/data/cache"
        "${PROJECT_ROOT}/data/uploads"
        "${PROJECT_ROOT}/docker/ssl"
        "${PROJECT_ROOT}/monitoring/data"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log_message "Created directory: $dir"
    done
}

# Function to clean up previous deployment
cleanup_previous() {
    if [[ "$CLEAN" == true ]]; then
        print_message "$YELLOW" "Cleaning up previous deployment..."
        
        # Stop and remove containers
        docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
        
        # Remove volumes if requested
        read -p "Remove data volumes? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true
            print_message "$YELLOW" "Volumes removed"
        fi
        
        # Clean up unused images
        docker system prune -f
        
        print_message "$GREEN" "Cleanup completed"
    fi
}

# Function to build images
build_images() {
    if [[ "$BUILD" == true ]]; then
        print_message "$BLUE" "Building Docker images..."
        
        local build_args=(
            "--build-arg" "BUILD_ENV=${BUILD_ENV:-production}"
            "--build-arg" "PYTHON_VERSION=${PYTHON_VERSION:-3.11}"
        )
        
        docker-compose -f "$COMPOSE_FILE" build "${build_args[@]}" --parallel
        
        print_message "$GREEN" "Images built successfully"
    fi
}

# Function to start services
start_services() {
    print_message "$BLUE" "Starting RAG Templates services with profile: $PROFILE"
    
    local compose_args=()
    
    # Add profile
    compose_args+=("--profile" "$PROFILE")
    
    # Add detached mode
    if [[ "$DETACHED" == true ]]; then
        compose_args+=("-d")
    fi
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" "${compose_args[@]}" up
    
    if [[ "$DETACHED" == true ]]; then
        print_message "$GREEN" "Services started in detached mode"
    fi
}

# Function to wait for services to be healthy
wait_for_services() {
    print_message "$BLUE" "Waiting for services to become healthy..."
    
    local max_wait=300  # 5 minutes
    local wait_time=0
    local check_interval=10
    
    while [[ $wait_time -lt $max_wait ]]; do
        local all_healthy=true
        
        # Check each service
        for service in iris_db redis rag_api; do
            if ! docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up (healthy)" 2>/dev/null; then
                all_healthy=false
                break
            fi
        done
        
        if [[ "$all_healthy" == true ]]; then
            print_message "$GREEN" "All core services are healthy!"
            return 0
        fi
        
        print_message "$YELLOW" "Waiting for services... (${wait_time}s/${max_wait}s)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done
    
    print_message "$RED" "Timeout waiting for services to become healthy"
    return 1
}

# Function to load sample data
load_sample_data() {
    if [[ "$LOAD_DATA" == true ]]; then
        print_message "$BLUE" "Loading sample data..."
        
        # Start data loader
        docker-compose -f "$COMPOSE_FILE" --profile with-data up data_loader
        
        print_message "$GREEN" "Sample data loaded successfully"
    fi
}

# Function to display service URLs
display_urls() {
    print_message "$GREEN" "RAG Templates Framework is now running!"
    echo
    print_message "$BLUE" "Service URLs:"
    echo -e "  ${GREEN}Streamlit App:${NC}      http://localhost:8501"
    echo -e "  ${GREEN}RAG API:${NC}           http://localhost:8000"
    echo -e "  ${GREEN}API Docs:${NC}          http://localhost:8000/docs"
    echo -e "  ${GREEN}Jupyter Notebook:${NC}  http://localhost:8888 (token: from .env)"
    echo -e "  ${GREEN}IRIS Portal:${NC}       http://localhost:52773/csp/sys/UtilHome.csp"
    echo -e "  ${GREEN}Monitoring:${NC}        http://localhost:9090"
    echo
    print_message "$BLUE" "Useful commands:"
    echo -e "  ${YELLOW}View logs:${NC}          docker-compose -f docker-compose.full.yml logs -f"
    echo -e "  ${YELLOW}Stop services:${NC}      docker-compose -f docker-compose.full.yml down"
    echo -e "  ${YELLOW}Restart service:${NC}    docker-compose -f docker-compose.full.yml restart <service>"
    echo
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

One-click deployment script for RAG Templates Framework

OPTIONS:
    -p, --profile PROFILE     Docker Compose profile (core, dev, prod, with-data)
    -d, --with-data          Load sample data after startup
    -f, --foreground         Run in foreground (not detached)
    -b, --build              Build images before starting
    -c, --clean              Clean up previous deployment
    -h, --help               Show this help message

PROFILES:
    core        Core services only (default)
    dev         Development mode with Jupyter
    prod        Production mode with Nginx and monitoring
    with-data   Core services + data loading

EXAMPLES:
    $0                                    # Start core services
    $0 --profile dev --with-data         # Development mode with sample data
    $0 --profile prod --build            # Production mode, build images
    $0 --clean --profile core            # Clean and restart core services

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -d|--with-data)
            LOAD_DATA=true
            shift
            ;;
        -f|--foreground)
            DETACHED=false
            shift
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_message "$RED" "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_message "Starting RAG Templates deployment with profile: $PROFILE"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Execute deployment steps
    check_prerequisites
    create_directories
    cleanup_previous
    build_images
    start_services
    
    if [[ "$DETACHED" == true ]]; then
        wait_for_services
        load_sample_data
        display_urls
    fi
    
    log_message "Deployment completed successfully"
}

# Error handling
trap 'print_message "$RED" "Script failed on line $LINENO"' ERR

# Run main function
main "$@"