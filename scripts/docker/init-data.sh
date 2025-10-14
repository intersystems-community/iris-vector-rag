#!/bin/bash
# =============================================================================
# RAG Templates Framework - Data Initialization Script
# =============================================================================
# This script initializes the RAG system with sample data, creates necessary
# database tables, and loads PMC sample documents for demonstration
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
DATA_DIR="${PROJECT_ROOT}/data"
LOG_FILE="${PROJECT_ROOT}/logs/data-init.log"

# Default options
FORCE_RELOAD=false
SAMPLE_SIZE="small"  # small, medium, large
DRY_RUN=false

# Function to print colored output
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[DATA-INIT]${NC} ${message}"
}

# Function to log messages
log_message() {
    local message=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" >> "$LOG_FILE"
    print_message "$BLUE" "$message"
}

# Function to check if services are running
check_services() {
    print_message "$BLUE" "Checking if required services are running..."
    
    local required_services=("iris_db" "redis" "rag_api")
    local all_running=true
    
    for service in "${required_services[@]}"; do
        if ! docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up" 2>/dev/null; then
            print_message "$RED" "Service $service is not running"
            all_running=false
        else
            print_message "$GREEN" "Service $service is running"
        fi
    done
    
    if [[ "$all_running" != true ]]; then
        print_message "$RED" "Not all required services are running. Please start them first:"
        print_message "$BLUE" "  ./scripts/docker/start.sh --profile core"
        exit 1
    fi
    
    # Wait for services to be fully healthy
    print_message "$BLUE" "Waiting for services to be healthy..."
    sleep 10
}

# Function to create database schema
create_database_schema() {
    print_message "$BLUE" "Creating database schema..."
    
    if [[ "$DRY_RUN" == true ]]; then
        print_message "$YELLOW" "[DRY RUN] Would create database schema"
        return 0
    fi
    
    # Run schema creation script in IRIS container
    docker-compose -f "$COMPOSE_FILE" exec -T iris_db iris session iris -U%SYS << 'EOF'
do ##class(%SQL.Statement).%ExecDirect(,"CREATE TABLE IF NOT EXISTS rag_documents (
    id INTEGER IDENTITY PRIMARY KEY,
    doc_id VARCHAR(255) UNIQUE,
    title VARCHAR(500),
    content TEXT,
    embedding VECTOR(DOUBLE, 1536),
    metadata JSON,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)")

do ##class(%SQL.Statement).%ExecDirect(,"CREATE TABLE IF NOT EXISTS rag_chunks (
    id INTEGER IDENTITY PRIMARY KEY,
    doc_id VARCHAR(255),
    chunk_id VARCHAR(255) UNIQUE,
    content TEXT,
    embedding VECTOR(DOUBLE, 1536),
    chunk_index INTEGER,
    metadata JSON,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)")

do ##class(%SQL.Statement).%ExecDirect(,"CREATE TABLE IF NOT EXISTS rag_queries (
    id INTEGER IDENTITY PRIMARY KEY,
    query_id VARCHAR(255) UNIQUE,
    query_text TEXT,
    query_embedding VECTOR(DOUBLE, 1536),
    response TEXT,
    retrieved_docs JSON,
    metadata JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)")

do ##class(%SQL.Statement).%ExecDirect(,"CREATE INDEX IF NOT EXISTS idx_docs_embedding ON rag_documents USING VECTOR(embedding)")
do ##class(%SQL.Statement).%ExecDirect(,"CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON rag_chunks USING VECTOR(embedding)")
do ##class(%SQL.Statement).%ExecDirect(,"CREATE INDEX IF NOT EXISTS idx_queries_embedding ON rag_queries USING VECTOR(embedding)")

write "Database schema created successfully",!
halt
EOF
    
    if [[ $? -eq 0 ]]; then
        print_message "$GREEN" "Database schema created successfully"
    else
        print_message "$RED" "Failed to create database schema"
        exit 1
    fi
}

# Function to check if data already exists
check_existing_data() {
    print_message "$BLUE" "Checking for existing data..."
    
    local doc_count=$(docker-compose -f "$COMPOSE_FILE" exec -T iris_db iris session iris -U%SYS << 'EOF' | tail -n 1
set stmt = ##class(%SQL.Statement).%New()
set result = stmt.%ExecDirect("SELECT COUNT(*) FROM rag_documents")
if result.%Next() {
    write result.%GetData(1)
} else {
    write "0"
}
halt
EOF
)
    
    doc_count=$(echo "$doc_count" | tr -d '\r\n' | sed 's/[^0-9]//g')
    
    if [[ -n "$doc_count" && "$doc_count" -gt 0 ]]; then
        print_message "$YELLOW" "Found $doc_count existing documents in database"
        
        if [[ "$FORCE_RELOAD" != true ]]; then
            read -p "Data already exists. Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_message "$BLUE" "Data initialization cancelled"
                exit 0
            fi
        else
            print_message "$YELLOW" "Force reload enabled, will overwrite existing data"
        fi
    else
        print_message "$GREEN" "No existing data found, proceeding with initialization"
    fi
}

# Function to load sample documents
load_sample_documents() {
    print_message "$BLUE" "Loading sample documents (size: $SAMPLE_SIZE)..."
    
    if [[ "$DRY_RUN" == true ]]; then
        print_message "$YELLOW" "[DRY RUN] Would load sample documents"
        return 0
    fi
    
    # Determine sample data based on size
    local data_path=""
    case "$SAMPLE_SIZE" in
        small)
            data_path="$DATA_DIR/sample_10_docs"
            ;;
        medium)
            data_path="$DATA_DIR/downloaded_pmc_docs"
            ;;
        large)
            data_path="$DATA_DIR/all_samples"
            ;;
        *)
            print_message "$RED" "Unknown sample size: $SAMPLE_SIZE"
            exit 1
            ;;
    esac
    
    if [[ ! -d "$data_path" ]]; then
        print_message "$RED" "Sample data directory not found: $data_path"
        print_message "$BLUE" "Available data directories:"
        ls -la "$DATA_DIR/"
        exit 1
    fi
    
    # Run data loader using API service
    print_message "$BLUE" "Executing data loading via API service..."
    
    docker-compose -f "$COMPOSE_FILE" exec -T rag_api python -c "
import sys
import os
sys.path.append('/app')

from rag_templates.ingestion.document_processor import DocumentProcessor
from rag_templates.core.vector_store import VectorStore
import logging

logging.basicConfig(level=logging.INFO)

# Initialize components
processor = DocumentProcessor()
vector_store = VectorStore()

# Load documents from data directory
data_path = '/app/data/sample_10_docs'
if os.path.exists(data_path):
    print(f'Loading documents from {data_path}')
    documents = processor.load_directory(data_path)
    print(f'Loaded {len(documents)} documents')
    
    # Process and store
    for doc in documents:
        try:
            chunks = processor.chunk_document(doc)
            vector_store.add_documents(chunks)
            print(f'Processed document: {doc.metadata.get(\"source\", \"unknown\")}')
        except Exception as e:
            print(f'Error processing document: {e}')
    
    print('Sample data loading completed')
else:
    print(f'Data path not found: {data_path}')
    sys.exit(1)
"
    
    if [[ $? -eq 0 ]]; then
        print_message "$GREEN" "Sample documents loaded successfully"
    else
        print_message "$RED" "Failed to load sample documents"
        exit 1
    fi
}

# Function to verify data loading
verify_data_loading() {
    print_message "$BLUE" "Verifying data loading..."
    
    # Check document count
    local doc_count=$(docker-compose -f "$COMPOSE_FILE" exec -T iris_db iris session iris -U%SYS << 'EOF' | tail -n 1
set stmt = ##class(%SQL.Statement).%New()
set result = stmt.%ExecDirect("SELECT COUNT(*) FROM rag_documents")
if result.%Next() {
    write result.%GetData(1)
} else {
    write "0"
}
halt
EOF
)
    
    doc_count=$(echo "$doc_count" | tr -d '\r\n' | sed 's/[^0-9]//g')
    
    # Check chunk count
    local chunk_count=$(docker-compose -f "$COMPOSE_FILE" exec -T iris_db iris session iris -U%SYS << 'EOF' | tail -n 1
set stmt = ##class(%SQL.Statement).%New()
set result = stmt.%ExecDirect("SELECT COUNT(*) FROM rag_chunks")
if result.%Next() {
    write result.%GetData(1)
} else {
    write "0"
}
halt
EOF
)
    
    chunk_count=$(echo "$chunk_count" | tr -d '\r\n' | sed 's/[^0-9]//g')
    
    print_message "$GREEN" "Data verification results:"
    echo -e "  ${GREEN}Documents:${NC} $doc_count"
    echo -e "  ${GREEN}Chunks:${NC}    $chunk_count"
    
    if [[ "$doc_count" -gt 0 && "$chunk_count" -gt 0 ]]; then
        print_message "$GREEN" "Data loading verification passed!"
        return 0
    else
        print_message "$RED" "Data loading verification failed!"
        return 1
    fi
}

# Function to test RAG pipeline
test_rag_pipeline() {
    print_message "$BLUE" "Testing RAG pipeline with sample query..."
    
    if [[ "$DRY_RUN" == true ]]; then
        print_message "$YELLOW" "[DRY RUN] Would test RAG pipeline"
        return 0
    fi
    
    # Test query via API
    local test_query="What are the symptoms of diabetes?"
    local api_url="http://localhost:8000"
    
    print_message "$BLUE" "Sending test query: '$test_query'"
    
    local response=$(curl -s -X POST "$api_url/api/v1/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$test_query\", \"pipeline\": \"basic\"}" \
        2>/dev/null || echo "ERROR")
    
    if [[ "$response" == "ERROR" ]]; then
        print_message "$RED" "Failed to test RAG pipeline - API not accessible"
        return 1
    elif echo "$response" | grep -q "error"; then
        print_message "$RED" "RAG pipeline test failed with error"
        echo "$response"
        return 1
    else
        print_message "$GREEN" "RAG pipeline test successful!"
        echo "Response preview: $(echo "$response" | head -c 200)..."
        return 0
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Initialize RAG Templates framework with sample data

OPTIONS:
    -f, --force              Force reload even if data exists
    -s, --size SIZE          Sample data size (small, medium, large) [default: small]
    -n, --dry-run           Show what would be done without executing
    -h, --help              Show this help message

SIZES:
    small       ~10 sample documents (fastest)
    medium      ~50 PMC documents
    large       Full dataset (slowest)

EXAMPLES:
    $0                      # Load small sample dataset
    $0 --size medium        # Load medium dataset
    $0 --force --size large # Force reload with large dataset

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--force)
            FORCE_RELOAD=true
            shift
            ;;
        -s|--size)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
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
    log_message "Starting data initialization with size: $SAMPLE_SIZE"
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Execute initialization steps
    check_services
    create_database_schema
    check_existing_data
    load_sample_documents
    verify_data_loading
    test_rag_pipeline
    
    print_message "$GREEN" "Data initialization completed successfully!"
    print_message "$BLUE" "You can now use the RAG system with sample data"
    
    log_message "Data initialization completed successfully"
}

# Error handling
trap 'print_message "$RED" "Script failed on line $LINENO"' ERR

# Run main function
main "$@"