#!/bin/bash
#
# CI/CD Integration Script for Example Testing Framework.
#
# This script integrates the example testing framework with the existing CI/CD pipeline,
# providing standardized execution, reporting, and failure handling for all example scripts.
#

set -e

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TESTING_DIR="$PROJECT_ROOT/scripts/testing"

# Default configuration
DEFAULT_MODE="real"
DEFAULT_TIMEOUT=300
DEFAULT_CATEGORY=""
DEFAULT_PATTERN=""
VERBOSE=false
CONTINUE_ON_FAILURE=false
GENERATE_REPORTS=true
UPLOAD_ARTIFACTS=false

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

# Help function
show_help() {
    cat << EOF
Example Testing Framework - CI/CD Integration

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -m, --mode MODE          Execution mode: mock (default), real
    -c, --category CATEGORY  Test category: basic, advanced, demo, visualization
    -p, --pattern PATTERN    Pattern to match example names
    -t, --timeout TIMEOUT    Timeout per example in seconds (default: 300)
    -v, --verbose            Enable verbose output
    --continue-on-failure    Continue testing after failures
    --no-reports             Skip report generation
    --upload-artifacts       Upload test artifacts to CI system
    --dry-run               Show what would be tested without execution
    -h, --help              Show this help message

EXAMPLES:
    # Run all examples in mock mode (CI default)
    $0

    # Run only basic examples with verbose output
    $0 --category basic --verbose

    # Run examples matching pattern with real LLM calls
    $0 --pattern "basic" --mode real --timeout 600

    # Continue testing after failures for comprehensive reporting
    $0 --continue-on-failure --upload-artifacts

ENVIRONMENT VARIABLES:
    EXAMPLE_TEST_MODE       Override default execution mode
    EXAMPLE_TEST_TIMEOUT    Override default timeout
    CI_UPLOAD_ARTIFACTS     Enable artifact upload in CI
    DISABLE_EXAMPLE_TESTS   Skip example testing entirely

EXIT CODES:
    0   All tests passed
    1   Some tests failed
    2   Configuration error
    3   Environment setup failed
EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--mode)
                DEFAULT_MODE="$2"
                shift 2
                ;;
            -c|--category)
                DEFAULT_CATEGORY="$2"
                shift 2
                ;;
            -p|--pattern)
                DEFAULT_PATTERN="$2"
                shift 2
                ;;
            -t|--timeout)
                DEFAULT_TIMEOUT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --continue-on-failure)
                CONTINUE_ON_FAILURE=true
                shift
                ;;
            --no-reports)
                GENERATE_REPORTS=false
                shift
                ;;
            --upload-artifacts)
                UPLOAD_ARTIFACTS=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 2
                ;;
        esac
    done
}

# Check environment and prerequisites
check_environment() {
    log_info "Checking environment and prerequisites..."

    # Check if example testing is disabled
    if [[ "${DISABLE_EXAMPLE_TESTS:-false}" == "true" ]]; then
        log_warning "Example testing disabled by DISABLE_EXAMPLE_TESTS environment variable"
        exit 0
    fi

    # Validate IRIS database availability (constitutional requirement)
    log_info "Validating IRIS database connectivity per constitutional requirements..."
    if ! command -v python &> /dev/null; then
        log_error "Python not available for IRIS connectivity check"
        exit 3
    fi

    # Check for IRIS connectivity using framework's validation
    if [[ -f "$PROJECT_ROOT/evaluation_framework/test_iris_connectivity.py" ]]; then
        if ! python "$PROJECT_ROOT/evaluation_framework/test_iris_connectivity.py" &> /dev/null; then
            log_error "IRIS database connectivity check failed"
            log_error "Constitutional requirement: All tests must execute against live IRIS database"
            log_error "Please ensure IRIS is running: docker-compose -f docker-compose.licensed.yml up -d iris"
            exit 3
        fi
        log_success "IRIS database connectivity validated"
    else
        log_warning "IRIS connectivity validation script not found - proceeding with assumptions"
    fi

    # Override defaults with environment variables
    DEFAULT_MODE="${EXAMPLE_TEST_MODE:-$DEFAULT_MODE}"
    DEFAULT_TIMEOUT="${EXAMPLE_TEST_TIMEOUT:-$DEFAULT_TIMEOUT}"
    UPLOAD_ARTIFACTS="${CI_UPLOAD_ARTIFACTS:-$UPLOAD_ARTIFACTS}"

    # Check Python availability
    if ! command -v python &> /dev/null; then
        log_error "Python not found in PATH"
        exit 3
    fi

    # Check if testing framework exists
    if [[ ! -f "$TESTING_DIR/run_example_tests.py" ]]; then
        log_error "Example testing framework not found at $TESTING_DIR"
        log_error "Please ensure the testing framework is properly installed"
        exit 3
    fi

    # Check virtual environment activation
    if [[ -z "$VIRTUAL_ENV" ]] && [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
        log_info "Activating virtual environment..."
        source "$PROJECT_ROOT/.venv/bin/activate"
    fi

    # Verify required Python packages
    if ! python -c "import iris_rag" &> /dev/null; then
        log_error "iris_rag package not found. Please install project dependencies."
        exit 3
    fi

    log_success "Environment check completed"
}

# Build test command arguments
build_test_command() {
    local cmd_args=()

    cmd_args+=("--mode" "$DEFAULT_MODE")
    cmd_args+=("--timeout" "$DEFAULT_TIMEOUT")

    if [[ -n "$DEFAULT_CATEGORY" ]]; then
        cmd_args+=("--category" "$DEFAULT_CATEGORY")
    fi

    if [[ -n "$DEFAULT_PATTERN" ]]; then
        cmd_args+=("--pattern" "$DEFAULT_PATTERN")
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        cmd_args+=("--verbose")
    fi

    if [[ "$CONTINUE_ON_FAILURE" == "true" ]]; then
        cmd_args+=("--continue-on-failure")
    fi

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        cmd_args+=("--dry-run")
    fi

    echo "${cmd_args[@]}"
}

# Execute example tests
run_example_tests() {
    log_info "Starting example test execution..."

    local cmd_args
    cmd_args=($(build_test_command))

    log_info "Test configuration:"
    log_info "  Mode: $DEFAULT_MODE"
    log_info "  Timeout: ${DEFAULT_TIMEOUT}s"
    log_info "  Category: ${DEFAULT_CATEGORY:-all}"
    log_info "  Pattern: ${DEFAULT_PATTERN:-all}"
    log_info "  Verbose: $VERBOSE"
    log_info "  Continue on failure: $CONTINUE_ON_FAILURE"

    # Change to testing directory
    cd "$TESTING_DIR"

    # Execute the test runner
    local exit_code=0
    if ! python run_example_tests.py "${cmd_args[@]}"; then
        exit_code=$?
        log_error "Example testing failed with exit code $exit_code"
    else
        log_success "All example tests completed successfully"
    fi

    return $exit_code
}

# Handle test artifacts and reporting
handle_artifacts() {
    if [[ "$GENERATE_REPORTS" == "false" ]]; then
        log_info "Report generation disabled"
        return 0
    fi

    log_info "Processing test artifacts..."

    # Create artifacts directory
    local artifacts_dir="$PROJECT_ROOT/test-results/examples"
    mkdir -p "$artifacts_dir"

    # Copy generated reports
    if [[ -d "$TESTING_DIR/reports" ]]; then
        cp -r "$TESTING_DIR/reports"/* "$artifacts_dir/" 2>/dev/null || true
        log_info "Reports copied to $artifacts_dir"
    fi

    # Create summary for CI systems
    create_ci_summary "$artifacts_dir"

    # Upload artifacts if requested
    if [[ "$UPLOAD_ARTIFACTS" == "true" ]]; then
        upload_artifacts "$artifacts_dir"
    fi
}

# Create CI summary
create_ci_summary() {
    local artifacts_dir="$1"
    local summary_file="$artifacts_dir/ci_summary.txt"

    # Find the most recent JSON report
    local latest_json
    latest_json=$(find "$artifacts_dir" -name "example_test_report_*.json" -type f | sort | tail -n 1)

    if [[ -n "$latest_json" && -f "$latest_json" ]]; then
        log_info "Creating CI summary from $latest_json"

        # Extract key metrics using Python
        python << EOF > "$summary_file"
import json
import sys

try:
    with open("$latest_json") as f:
        data = json.load(f)

    summary = data.get("summary", {})

    print(f"=== Example Test Results ===")
    print(f"Total Examples: {summary.get('total', 0)}")
    print(f"Passed: {summary.get('passed', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Success Rate: {summary.get('success_rate', 0):.1%}")
    print(f"Average Execution Time: {summary.get('avg_execution_time', 0):.2f}s")
    print(f"Average Memory Usage: {summary.get('avg_memory_usage', 0):.1f}MB")

    # Show failed examples if any
    failed_examples = [r for r in data.get("results", []) if not r.get("success", False)]
    if failed_examples:
        print(f"\nFailed Examples:")
        for result in failed_examples:
            print(f"  - {result.get('script_path', 'unknown')}: {result.get('error_message', 'no error message')}")

except Exception as e:
    print(f"Error creating summary: {e}")
    sys.exit(1)
EOF

        if [[ -f "$summary_file" ]]; then
            log_info "CI summary created at $summary_file"
            if [[ "$VERBOSE" == "true" ]]; then
                cat "$summary_file"
            fi
        fi
    else
        log_warning "No JSON report found for CI summary"
    fi
}

# Upload artifacts (placeholder for CI-specific implementations)
upload_artifacts() {
    local artifacts_dir="$1"

    log_info "Uploading test artifacts..."

    # GitHub Actions
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
        echo "::set-output name=artifacts-path::$artifacts_dir"
        echo "::set-output name=example-test-results::$(cat "$artifacts_dir/ci_summary.txt" 2>/dev/null || echo 'No summary available')"
    fi

    # GitLab CI
    if [[ -n "${GITLAB_CI:-}" ]]; then
        # Artifacts are automatically collected from test-results/ directory
        log_info "GitLab CI will collect artifacts from test-results/"
    fi

    # Jenkins
    if [[ -n "${JENKINS_URL:-}" ]]; then
        # Archive artifacts using Jenkins workspace
        log_info "Jenkins artifacts available in workspace"
    fi

    log_success "Artifact upload configuration completed"
}

# Main execution function
main() {
    log_info "Example Testing Framework - CI/CD Integration"
    log_info "============================================="

    # Parse arguments
    parse_arguments "$@"

    # Environment setup
    check_environment

    # Show configuration for dry run
    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        log_info "DRY RUN - No tests will be executed"
        local cmd_args
        cmd_args=($(build_test_command))
        log_info "Would execute: python run_example_tests.py ${cmd_args[*]}"
        exit 0
    fi

    # Execute tests
    local test_exit_code=0
    if ! run_example_tests; then
        test_exit_code=$?
    fi

    # Handle artifacts regardless of test outcome
    handle_artifacts

    # Final status
    if [[ $test_exit_code -eq 0 ]]; then
        log_success "Example testing completed successfully"
    else
        log_error "Example testing failed - see reports for details"
    fi

    exit $test_exit_code
}

# Execute main function with all arguments
main "$@"