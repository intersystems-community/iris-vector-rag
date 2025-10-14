#!/bin/bash

# RAGAS Evaluation Runner Script
# 
# This script sets up the environment and runs comprehensive RAGAS evaluation
# on all 4 pipelines using the PMC dataset to demonstrate >80% accuracy.
#
# Usage:
#   ./scripts/run_ragas_evaluation.sh [options]
#
# Environment Variables (with defaults):
#   RAGAS_NUM_QUERIES=15        - Number of queries to evaluate
#   RAGAS_PIPELINES=all         - Pipelines to test (comma-separated or 'all')
#   RAGAS_OUTPUT_DIR=outputs/reports/ragas_evaluations - Output directory
#   RAGAS_USE_CACHE=true        - Whether to use query result caching
#   RAGAS_LOG_LEVEL=INFO        - Logging level (DEBUG, INFO, WARNING, ERROR)

set -e  # Exit on any error

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default configuration
DEFAULT_NUM_QUERIES=15
DEFAULT_PIPELINES="BasicRAG,BasicRerank,CRAG,GraphRAG"
DEFAULT_OUTPUT_DIR="outputs/reports/ragas_evaluations"
DEFAULT_USE_CACHE="true"
DEFAULT_LOG_LEVEL="INFO"

# Help function
show_help() {
    cat << EOF
RAGAS Evaluation Runner

This script runs comprehensive RAGAS evaluation on all RAG pipelines using the PMC dataset.

Usage: $0 [OPTIONS]

Options:
    -q, --queries NUM        Number of queries to evaluate (default: $DEFAULT_NUM_QUERIES)
    -p, --pipelines LIST     Comma-separated list of pipelines or 'all' (default: all)
    -o, --output DIR         Output directory (default: $DEFAULT_OUTPUT_DIR)
    -c, --cache BOOL         Use caching (true/false, default: $DEFAULT_USE_CACHE)
    -l, --log-level LEVEL    Log level (DEBUG/INFO/WARNING/ERROR, default: $DEFAULT_LOG_LEVEL)
    --no-cache              Disable caching (same as --cache false)
    --test-mode             Run with minimal queries for testing (5 queries)
    --production-mode       Run full evaluation with all queries
    -h, --help              Show this help message

Examples:
    $0                                    # Run with defaults
    $0 --test-mode                        # Quick test run
    $0 --queries 10 --pipelines BasicRAG,CRAG
    $0 --output /tmp/ragas_results --no-cache

Environment Variables:
    RAGAS_NUM_QUERIES       Override default number of queries
    RAGAS_PIPELINES         Override default pipelines list
    RAGAS_OUTPUT_DIR        Override default output directory
    RAGAS_USE_CACHE         Override default caching behavior
    RAGAS_LOG_LEVEL         Override default log level

Pipeline Options:
    BasicRAG        - Basic RAG implementation
    BasicRerank     - Basic RAG with reranking
    CRAG            - Corrective RAG implementation
    GraphRAG        - Graph-based RAG implementation
    all             - All available pipelines (default)

The script will:
1. Validate environment and dependencies
2. Set up output directories
3. Run RAGAS evaluation on specified pipelines
4. Generate JSON and HTML reports
5. Display summary results
6. Exit with code 0 if target accuracy (>80%) is achieved

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -q|--queries)
                RAGAS_NUM_QUERIES="$2"
                shift 2
                ;;
            -p|--pipelines)
                RAGAS_PIPELINES="$2"
                shift 2
                ;;
            -o|--output)
                RAGAS_OUTPUT_DIR="$2"
                shift 2
                ;;
            -c|--cache)
                RAGAS_USE_CACHE="$2"
                shift 2
                ;;
            --no-cache)
                RAGAS_USE_CACHE="false"
                shift
                ;;
            -l|--log-level)
                RAGAS_LOG_LEVEL="$2"
                shift 2
                ;;
            --test-mode)
                RAGAS_NUM_QUERIES=5
                RAGAS_PIPELINES="BasicRAG,CRAG"
                log_info "Test mode enabled: 5 queries, BasicRAG and CRAG only"
                shift
                ;;
            --production-mode)
                RAGAS_NUM_QUERIES=15
                RAGAS_PIPELINES="$DEFAULT_PIPELINES"
                log_info "Production mode enabled: Full evaluation"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Set environment variables with defaults
setup_environment() {
    log_info "Setting up environment variables..."
    
    # Use command line args, then environment variables, then defaults
    export RAGAS_NUM_QUERIES="${RAGAS_NUM_QUERIES:-$DEFAULT_NUM_QUERIES}"
    export RAGAS_PIPELINES="${RAGAS_PIPELINES:-$DEFAULT_PIPELINES}"
    export RAGAS_OUTPUT_DIR="${RAGAS_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
    export RAGAS_USE_CACHE="${RAGAS_USE_CACHE:-$DEFAULT_USE_CACHE}"
    export RAGAS_LOG_LEVEL="${RAGAS_LOG_LEVEL:-$DEFAULT_LOG_LEVEL}"
    
    # Convert 'all' to full pipeline list
    if [[ "$RAGAS_PIPELINES" == "all" ]]; then
        export RAGAS_PIPELINES="$DEFAULT_PIPELINES"
    fi
    
    # Convert relative output path to absolute
    if [[ ! "$RAGAS_OUTPUT_DIR" = /* ]]; then
        export RAGAS_OUTPUT_DIR="$PROJECT_ROOT/$RAGAS_OUTPUT_DIR"
    fi
    
    log_info "Configuration:"
    log_info "  Queries: $RAGAS_NUM_QUERIES"
    log_info "  Pipelines: $RAGAS_PIPELINES"
    log_info "  Output Dir: $RAGAS_OUTPUT_DIR"
    log_info "  Use Cache: $RAGAS_USE_CACHE"
    log_info "  Log Level: $RAGAS_LOG_LEVEL"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/scripts/generate_ragas_evaluation.py" ]]; then
        log_error "RAGAS evaluation script not found at $PROJECT_ROOT/scripts/generate_ragas_evaluation.py"
        log_error "Please run this script from the project root directory"
        exit 1
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    # Check if virtual environment is activated or available
    if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        if [[ -f "$PROJECT_ROOT/venv/bin/activate" ]]; then
            log_info "Activating virtual environment..."
            source "$PROJECT_ROOT/venv/bin/activate"
        elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
            log_info "Activating virtual environment..."
            source "$PROJECT_ROOT/.venv/bin/activate"
        elif [[ -f "$PROJECT_ROOT/activate_env.sh" ]]; then
            log_info "Using project activation script..."
            source "$PROJECT_ROOT/activate_env.sh"
        else
            log_warning "No virtual environment detected. Ensure dependencies are installed globally."
        fi
    fi
    
    # Try to import key dependencies
    python3 -c "import numpy, pandas, scipy, sentence_transformers, transformers, torch" 2>/dev/null || {
        log_error "Required Python dependencies not found"
        log_error "Please install dependencies: pip install -r requirements.txt"
        exit 1
    }
    
    log_success "Dependencies check passed"
}

# Create output directory
setup_output_directory() {
    log_info "Setting up output directory: $RAGAS_OUTPUT_DIR"
    mkdir -p "$RAGAS_OUTPUT_DIR"
    
    if [[ ! -w "$RAGAS_OUTPUT_DIR" ]]; then
        log_error "Output directory is not writable: $RAGAS_OUTPUT_DIR"
        exit 1
    fi
    
    log_success "Output directory ready"
}

# Validate pipeline configuration
validate_pipelines() {
    log_info "Validating pipeline configuration..."
    
    # Define available pipelines
    local available_pipelines=("BasicRAG" "BasicRerank" "CRAG" "GraphRAG")
    local requested_pipelines
    IFS=',' read -ra requested_pipelines <<< "$RAGAS_PIPELINES"
    
    for pipeline in "${requested_pipelines[@]}"; do
        pipeline=$(echo "$pipeline" | xargs)  # Trim whitespace
        if [[ ! " ${available_pipelines[@]} " =~ " ${pipeline} " ]]; then
            log_error "Unknown pipeline: $pipeline"
            log_error "Available pipelines: ${available_pipelines[*]}"
            exit 1
        fi
    done
    
    log_success "Pipeline configuration validated"
}

# Run the evaluation
run_evaluation() {
    log_info "Starting RAGAS evaluation..."
    log_info "This may take several minutes depending on the number of queries and pipelines..."
    
    cd "$PROJECT_ROOT"
    
    # Set Python path
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Run the evaluation script
    local start_time=$(date +%s)
    
    if python3 scripts/generate_ragas_evaluation.py; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        log_success "RAGAS evaluation completed successfully in ${duration}s"
        return 0
    else
        local exit_code=$?
        log_error "RAGAS evaluation failed with exit code $exit_code"
        return $exit_code
    fi
}

# Display results summary
show_results_summary() {
    log_info "Looking for generated reports..."
    
    # Find the most recent report files
    local json_report=$(find "$RAGAS_OUTPUT_DIR" -name "ragas_report_*.json" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    local html_report=$(find "$RAGAS_OUTPUT_DIR" -name "ragas_report_*.html" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -f "$json_report" ]]; then
        log_success "JSON Report: $json_report"
        
        # Try to extract summary from JSON
        if command -v jq &> /dev/null; then
            echo ""
            log_info "Quick Summary:"
            
            # Extract best pipeline
            local best_pipeline=$(jq -r '.comparative_analysis.best_overall // "Unknown"' "$json_report" 2>/dev/null)
            log_info "  Best Overall Pipeline: $best_pipeline"
            
            # Extract target achievement
            local target_achieved=$(jq -r '
                .pipeline_metrics | 
                to_entries | 
                map(.value.answer_correctness.mean >= 0.8) | 
                any
            ' "$json_report" 2>/dev/null)
            
            if [[ "$target_achieved" == "true" ]]; then
                log_success "  Target Accuracy (>80%): ✅ ACHIEVED"
            else
                log_warning "  Target Accuracy (>80%): ❌ NOT ACHIEVED"
            fi
        fi
    else
        log_warning "No JSON report found in $RAGAS_OUTPUT_DIR"
    fi
    
    if [[ -f "$html_report" ]]; then
        log_success "HTML Report: $html_report"
        log_info "Open the HTML report in a web browser for detailed results"
    else
        log_warning "No HTML report found in $RAGAS_OUTPUT_DIR"
    fi
}

# Cleanup function
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "Evaluation failed. Check the logs for details."
        log_info "Log files may be available in: $RAGAS_OUTPUT_DIR"
    fi
}

# Main execution
main() {
    # Set up trap for cleanup
    trap cleanup EXIT
    
    log_info "🚀 RAGAS Evaluation Runner Starting"
    log_info "========================================"
    
    # Parse command line arguments
    parse_args "$@"
    
    # Setup environment
    setup_environment
    
    # Run pre-flight checks
    check_dependencies
    setup_output_directory
    validate_pipelines
    
    # Run the evaluation
    if run_evaluation; then
        echo ""
        log_success "🎉 RAGAS Evaluation completed successfully!"
        show_results_summary
        
        echo ""
        log_info "Next steps:"
        log_info "1. Review the generated reports"
        log_info "2. Analyze pipeline performance metrics"
        log_info "3. Use results for Phase 1 deliverables"
        
        exit 0
    else
        exit_code=$?
        log_error "💥 RAGAS Evaluation failed"
        show_results_summary
        exit $exit_code
    fi
}

# Run main function with all arguments
main "$@"