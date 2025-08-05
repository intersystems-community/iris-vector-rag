#!/bin/bash

# Remote Server Setup Script for RAG Templates with Native VECTOR Types
# This script sets up a fresh RAG system with optimal performance

set -e  # Exit on any error

echo "🚀 Starting RAG Templates Remote Setup..."

# Check if we're in a git repository and get current branch info
if [ -d ".git" ]; then
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo "📋 Current branch: $CURRENT_BRANCH"
    
    # If this is not the main/master branch, remind user about branch-specific deployment
    if [ "$CURRENT_BRANCH" != "main" ] && [ "$CURRENT_BRANCH" != "master" ]; then
        echo "⚠️  Note: You are on branch '$CURRENT_BRANCH'"
        echo "   Make sure this branch contains the native VECTOR implementation"
        echo "   If deploying to remote server, use: git checkout $CURRENT_BRANCH"
    fi
else
    echo "⚠️  Not in a git repository - assuming manual file transfer"
fi

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

# Check prerequisites
print_status "Checking prerequisites..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

print_success "Prerequisites check passed"

# Check system resources
print_status "Checking system resources..."
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [ "$TOTAL_MEM" -lt 8 ]; then
    print_warning "System has less than 8GB RAM. Performance may be limited."
else
    print_success "System has ${TOTAL_MEM}GB RAM - sufficient for RAG operations"
fi

# Create necessary directories
print_status "Creating directory structure..."
mkdir -p logs
mkdir -p data/pmc_articles
mkdir -p backups
mkdir -p config/local
print_success "Directory structure created"

# Install Python dependencies
print_status "Installing Python dependencies..."
if command -v poetry &> /dev/null; then
    print_status "Using Poetry for dependency management..."
    poetry install
else
    print_status "Using pip for dependency management..."
    pip3 install -r requirements.txt
fi
print_success "Python dependencies installed"

# Stop any existing containers
print_status "Stopping any existing containers..."
docker-compose -f docker-compose.iris-only.yml down || true

# Pull latest IRIS image
print_status "Pulling latest IRIS image..."
docker-compose -f docker-compose.iris-only.yml pull

# Start IRIS container
print_status "Starting IRIS container with native VECTOR support..."
docker-compose -f docker-compose.iris-only.yml up -d

# Wait for IRIS to be ready
print_status "Waiting for IRIS to be ready..."
sleep 30

# Check if IRIS is responding
MAX_RETRIES=12
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker exec iris_db_rag_licensed iris terminal IRIS -U USER -c "write \"IRIS Ready\"" &> /dev/null; then
        print_success "IRIS is ready!"
        break
    else
        print_status "IRIS not ready yet, waiting... (attempt $((RETRY_COUNT + 1))/$MAX_RETRIES)"
        sleep 10
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    print_error "IRIS failed to start properly. Check logs with: docker logs iris_db_rag_licensed"
    exit 1
fi

# Initialize database with native VECTOR schema
print_status "Initializing database with native VECTOR schema..."
python3 common/db_init_with_indexes.py

# Verify schema creation
print_status "Verifying schema creation..."
python3 scripts/verify_native_vector_schema.py

# Create initial performance baseline
print_status "Creating performance baseline..."
python3 scripts/create_performance_baseline.py

# Set up monitoring
print_status "Setting up monitoring..."
python3 scripts/setup_monitoring.py

print_success "🎉 RAG Templates setup completed successfully!"
print_status ""
print_status "Next steps:"
print_status "1. Verify installation: python3 scripts/system_health_check.py"
print_status "2. Start data ingestion: python3 scripts/ingest_100k_documents.py"
print_status "3. Run benchmarks: python3 eval/enterprise_rag_benchmark_final.py"
print_status ""
print_status "IRIS Management Portal: http://localhost:52773/csp/sys/UtilHome.csp"
print_status "Default credentials: _SYSTEM / SYS"
print_status ""
print_status "For remote access, create SSH tunnel:"
print_status "ssh -L 52773:localhost:52773 user@$(hostname)"