#!/bin/bash

# Comprehensive DBAPI RAG System Test Runner
# This script provides an easy way to run the comprehensive DBAPI test with various configurations

set -e

# Default values
DOCUMENT_COUNT=1000
VERBOSE=false
CLEANUP_ONLY=false
HELP=false
REUSE_IRIS=false
CLEAN_IRIS=true
RESET_DATA=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Function to show help
show_help() {
    cat << EOF
Comprehensive DBAPI RAG System Test Runner

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -d, --documents COUNT    Number of documents to load (default: 1000)
    -v, --verbose           Enable verbose logging
    -c, --cleanup-only      Only cleanup existing containers and exit
    --reuse-iris            Reuse existing IRIS container if available
    --clean-iris            Force fresh container setup (default)
    --reset-data            Clear data but keep schema when reusing
    -h, --help              Show this help message

EXAMPLES:
    # Run test with default 1000 documents (fresh container)
    $0

    # Run test with 5000 documents
    $0 --documents 5000

    # Run test with verbose logging
    $0 --verbose

    # Reuse existing IRIS container if available
    $0 --reuse-iris

    # Reuse container but reset data
    $0 --reuse-iris --reset-data

    # Force fresh container (explicit)
    $0 --clean-iris

    # Cleanup existing containers only
    $0 --cleanup-only

ENVIRONMENT VARIABLES:
    TEST_DOCUMENT_COUNT     Override document count (same as --documents)
    IRIS_HOST              IRIS host (default: localhost)
    IRIS_PORT              IRIS port (default: 1972)
    IRIS_NAMESPACE         IRIS namespace (default: USER)
    IRIS_USER              IRIS user (default: _SYSTEM)
    IRIS_PASSWORD          IRIS password (default: SYS)

REQUIREMENTS:
    - Docker and docker-compose installed
    - Python 3.8+ with required packages
    - intersystems-irispython package installed
    - At least 4GB free disk space
    - At least 8GB RAM recommended for large document counts

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--documents)
            DOCUMENT_COUNT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--cleanup-only)
            CLEANUP_ONLY=true
            shift
            ;;
        --reuse-iris)
            REUSE_IRIS=true
            CLEAN_IRIS=false
            shift
            ;;
        --clean-iris)
            CLEAN_IRIS=true
            REUSE_IRIS=false
            shift
            ;;
        --reset-data)
            RESET_DATA=true
            shift
            ;;
        -h|--help)
            HELP=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    show_help
    exit 0
fi

# Function to check if IRIS container is running
check_iris_container() {
    local container_status=$(docker-compose ps iris_db --format json 2>/dev/null | grep -o '"State":"[^"]*"' | cut -d'"' -f4)
    if [ "$container_status" = "running" ]; then
        return 0  # Container is running
    else
        return 1  # Container is not running
    fi
}

# Function to check if IRIS container is healthy
check_iris_health() {
    local health_status=$(docker-compose ps iris_db --format json 2>/dev/null | grep -o '"Health":"[^"]*"' | cut -d'"' -f4)
    if [ "$health_status" = "healthy" ]; then
        return 0  # Container is healthy
    else
        return 1  # Container is not healthy
    fi
}

# Function to test IRIS connection
test_iris_connection() {
    print_status "Testing IRIS connection..."
    # Simple connection test using docker exec
    if docker-compose exec -T iris_db iris session iris -U%SYS <<< 'write "Connection test successful",!' >/dev/null 2>&1; then
        return 0  # Connection successful
    else
        return 1  # Connection failed
    fi
}

# Function to reset data in existing container
reset_iris_data() {
    print_status "Resetting IRIS data while preserving schema..."
    
    # Create a temporary SQL script to clear data but preserve schema
    cat > /tmp/reset_data.sql << 'EOF'
-- Clear data from all RAG tables while preserving schema
DELETE FROM RAG.SourceDocuments;
DELETE FROM RAG.Entities;
DELETE FROM RAG.Relationships;
DELETE FROM RAG.KnowledgeGraphNodes;
DELETE FROM RAG.KnowledgeGraphEdges;
DELETE FROM RAG.ChunkedDocuments;
-- Add other tables as needed
WRITE "Data reset completed",!
EOF

    # Execute the reset script
    if docker cp /tmp/reset_data.sql $(docker-compose ps -q iris_db):/tmp/reset_data.sql 2>/dev/null && \
       docker-compose exec -T iris_db iris session iris -U%SYS < /tmp/reset_data.sql >/dev/null 2>&1; then
        rm -f /tmp/reset_data.sql
        print_success "Data reset completed"
        return 0
    else
        rm -f /tmp/reset_data.sql
        print_error "Failed to reset data"
        return 1
    fi
}

# Cleanup function
cleanup_containers() {
    print_status "Cleaning up existing containers..."
    docker-compose down -v 2>/dev/null || true
    docker container prune -f 2>/dev/null || true
    docker volume prune -f 2>/dev/null || true
    print_success "Cleanup completed"
}

# If cleanup-only, do cleanup and exit
if [ "$CLEANUP_ONLY" = true ]; then
    cleanup_containers
    exit 0
fi

# Validate document count
if ! [[ "$DOCUMENT_COUNT" =~ ^[0-9]+$ ]] || [ "$DOCUMENT_COUNT" -lt 100 ]; then
    print_error "Document count must be a number >= 100"
    exit 1
fi

# Check prerequisites
print_status "Checking prerequisites..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose >/dev/null 2>&1; then
    print_error "docker-compose is not installed. Please install docker-compose and try again."
    exit 1
fi

# Check if Python is available
if ! command -v python3 >/dev/null 2>&1; then
    print_error "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if we're in the project root
if [ ! -f "docker-compose.yml" ] || [ ! -d "tests" ]; then
    print_error "This script must be run from the project root directory."
    exit 1
fi

# Check available disk space (at least 4GB)
AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
REQUIRED_SPACE=4194304  # 4GB in KB
if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    print_warning "Available disk space is less than 4GB. Test may fail due to insufficient space."
fi

print_success "Prerequisites check passed"

# Set environment variables
export TEST_DOCUMENT_COUNT="$DOCUMENT_COUNT"
export IRIS_HOST="${IRIS_HOST:-localhost}"
export IRIS_PORT="${IRIS_PORT:-1972}"
export IRIS_NAMESPACE="${IRIS_NAMESPACE:-USER}"
export IRIS_USER="${IRIS_USER:-_SYSTEM}"
export IRIS_PASSWORD="${IRIS_PASSWORD:-SYS}"
export RAG_CONNECTION_TYPE="dbapi"
export IRIS_REUSE_MODE="$REUSE_IRIS"
export IRIS_CLEAN_MODE="$CLEAN_IRIS"
export IRIS_RESET_DATA="$RESET_DATA"

# Set logging level
if [ "$VERBOSE" = true ]; then
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    LOG_LEVEL="DEBUG"
else
    LOG_LEVEL="INFO"
fi

# Create logs directory
mkdir -p logs

# Print test configuration
print_status "Test Configuration:"
echo "  Document Count: $DOCUMENT_COUNT"
echo "  IRIS Host: $IRIS_HOST"
echo "  IRIS Port: $IRIS_PORT"
echo "  IRIS Namespace: $IRIS_NAMESPACE"
echo "  IRIS User: $IRIS_USER"
echo "  Connection Type: dbapi"
echo "  Verbose Logging: $VERBOSE"
echo "  Container Mode: $([ "$REUSE_IRIS" = true ] && echo "REUSE" || echo "CLEAN")"
echo "  Reset Data: $RESET_DATA"
echo ""

# Container setup logic based on mode
CONTAINER_READY=false

if [ "$REUSE_IRIS" = true ]; then
    print_status "Checking for existing IRIS container..."
    
    if check_iris_container; then
        print_success "Found running IRIS container"
        
        if check_iris_health; then
            print_success "IRIS container is healthy"
            
            if test_iris_connection; then
                print_success "IRIS connection test passed"
                CONTAINER_READY=true
                
                # Reset data if requested
                if [ "$RESET_DATA" = true ]; then
                    if reset_iris_data; then
                        print_success "Data reset completed"
                    else
                        print_warning "Data reset failed, will continue with existing data"
                    fi
                fi
            else
                print_warning "IRIS connection test failed, will restart container"
            fi
        else
            print_warning "IRIS container is not healthy, will restart container"
        fi
    else
        print_status "No running IRIS container found, will start fresh container"
    fi
fi

# If container is not ready or clean mode is requested, setup fresh container
if [ "$CONTAINER_READY" = false ] || [ "$CLEAN_IRIS" = true ]; then
    print_status "Setting up fresh IRIS container..."
    cleanup_containers
    
    # Start fresh IRIS container
    print_status "Starting fresh IRIS container..."
    if ! docker-compose up -d iris_db; then # --wait REMOVED AGAIN
        print_error "Failed to dispatch IRIS container start (docker-compose up -d)"
        exit 1
    fi
    
    print_status "IRIS container start dispatched. Waiting 45 seconds for initialization..."
    sleep 45 # Increased initial sleep significantly
    
    # Health check loop removed as it's not reliable without a healthcheck in docker-compose.yml
    # The script will now rely on the 'test_iris_connection' below after this initial sleep.
    
    # Test connection
    if ! test_iris_connection; then
        print_error "IRIS connection test failed after container startup"
        exit 1
    fi
    
    print_status "Un-expiring user passwords in IRIS container..."
    if ! docker-compose exec -T iris_db iris session iris -U%SYS '##class(Security.Users).UnExpireUserPasswords("*")' >/dev/null 2>&1; then
        print_warning "Failed to un-expire user passwords. This might cause issues later."
    else
        print_success "User passwords un-expired."
    fi
    
    print_success "Fresh IRIS container is ready"
else
    print_success "Using existing IRIS container"
fi

# Estimate test duration
if [ "$DOCUMENT_COUNT" -ge 5000 ]; then
    ESTIMATED_DURATION="60-90 minutes"
elif [ "$DOCUMENT_COUNT" -ge 2000 ]; then
    ESTIMATED_DURATION="30-45 minutes"
elif [ "$DOCUMENT_COUNT" -ge 1000 ]; then
    ESTIMATED_DURATION="15-30 minutes"
else
    ESTIMATED_DURATION="10-15 minutes"
fi

print_status "Estimated test duration: $ESTIMATED_DURATION"
print_status "Starting comprehensive DBAPI RAG system test..."

# Run the test
START_TIME=$(date +%s)

if [ "$VERBOSE" = true ]; then
    python3 tests/test_comprehensive_dbapi_rag_system.py
else
    python3 tests/test_comprehensive_dbapi_rag_system.py 2>&1 | tee "logs/test_run_$(date +%s).log"
fi

TEST_EXIT_CODE=$?
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Print results
echo ""
echo "=" * 80
if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "Comprehensive DBAPI RAG system test completed successfully!"
else
    print_error "Comprehensive DBAPI RAG system test failed!"
fi
echo "=" * 80
print_status "Test duration: $DURATION seconds"

# Show log files
LOG_FILES=$(find logs -name "*$(date +%Y%m%d)*" -type f | head -5)
if [ -n "$LOG_FILES" ]; then
    print_status "Generated log files:"
    echo "$LOG_FILES" | while read -r file; do
        echo "  - $file"
    done
fi

# Show report files
REPORT_FILES=$(find logs -name "*comprehensive_dbapi_test_report*" -type f | head -3)
if [ -n "$REPORT_FILES" ]; then
    print_status "Generated report files:"
    echo "$REPORT_FILES" | while read -r file; do
        echo "  - $file"
    done
fi

# Cleanup on exit
cleanup_containers

exit $TEST_EXIT_CODE