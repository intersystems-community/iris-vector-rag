#!/bin/bash
# Local test runner script for RAG templates framework

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

# Default values
TEST_TYPE="all"
VERBOSE=false
COVERAGE=true
PARALLEL=false
MARKERS=""
OUTPUT_DIR="${PROJECT_ROOT}/test-results"

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Local test runner for RAG templates framework

OPTIONS:
    -t, --type TYPE         Test type: all, unit, integration, e2e (default: all)
    -v, --verbose          Verbose output
    -c, --no-coverage      Disable coverage reporting
    -p, --parallel         Run tests in parallel
    -m, --markers MARKERS  Pytest markers to run
    -o, --output DIR       Output directory for results (default: test-results)
    -h, --help             Show this help message

EXAMPLES:
    $0                          # Run all tests
    $0 -t unit                  # Run only unit tests
    $0 -t integration -v        # Run integration tests with verbose output
    $0 -m "slow"               # Run only tests marked as slow
    $0 -p -c                   # Run tests in parallel without coverage

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -c|--no-coverage)
            COVERAGE=false
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -m|--markers)
            MARKERS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Change to project root
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

echo -e "${BLUE}RAG Templates Framework Test Runner${NC}"
echo -e "${BLUE}===================================${NC}"
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ ! -f ".venv/bin/activate" ]]; then
    echo -e "${YELLOW}Warning: No virtual environment detected${NC}"
    echo "Consider activating a virtual environment before running tests"
    echo ""
fi

# Install dependencies if needed
if [[ ! -f ".venv/pyvenv.cfg" ]] && command -v poetry &> /dev/null; then
    echo -e "${BLUE}Installing dependencies with Poetry...${NC}"
    poetry install --with dev,test
elif [[ -f "requirements-dev.txt" ]]; then
    echo -e "${BLUE}Installing dependencies with pip...${NC}"
    pip install -r requirements-dev.txt
fi

# Build pytest command
PYTEST_CMD="pytest"

# Add test directories based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD+=" tests/unit/"
        ;;
    integration)
        PYTEST_CMD+=" tests/integration/"
        ;;
    e2e)
        PYTEST_CMD+=" tests/e2e/"
        ;;
    all)
        PYTEST_CMD+=" tests/"
        ;;
    *)
        echo -e "${RED}Invalid test type: $TEST_TYPE${NC}"
        echo "Valid types: all, unit, integration, e2e"
        exit 1
        ;;
esac

# Add coverage options
if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD+=" --cov=rag_templates --cov=iris_rag --cov=mem0_integration"
    PYTEST_CMD+=" --cov-report=term-missing"
    PYTEST_CMD+=" --cov-report=html:${OUTPUT_DIR}/htmlcov"
    PYTEST_CMD+=" --cov-report=xml:${OUTPUT_DIR}/coverage.xml"
fi

# Add parallel execution
if [[ "$PARALLEL" == true ]]; then
    PYTEST_CMD+=" -n auto"
fi

# Add markers
if [[ -n "$MARKERS" ]]; then
    PYTEST_CMD+=" -m '$MARKERS'"
fi

# Add verbose output
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD+=" -v"
fi

# Add JUnit XML output
PYTEST_CMD+=" --junit-xml=${OUTPUT_DIR}/junit.xml"

# Add additional options
PYTEST_CMD+=" --tb=short --durations=10"

echo -e "${BLUE}Running tests...${NC}"
echo "Command: $PYTEST_CMD"
echo ""

# Run tests
if eval $PYTEST_CMD; then
    echo ""
    echo -e "${GREEN}✓ Tests completed successfully${NC}"
    
    if [[ "$COVERAGE" == true ]]; then
        echo -e "${BLUE}Coverage report generated at: ${OUTPUT_DIR}/htmlcov/index.html${NC}"
    fi
    
    echo -e "${BLUE}Test results saved to: ${OUTPUT_DIR}/${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi