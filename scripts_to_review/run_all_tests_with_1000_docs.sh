#!/bin/bash
# Script to run all tests with 1000 documents

set -e  # Exit immediately if a command exits with a non-zero status

echo "=== Running All Tests with 1000+ Documents ==="
echo "This script ensures all tests run with at least 1000 real PMC documents"
echo

# Ensure we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed. Please install it first."
    exit 1
fi

# Create a directory for logs
LOG_DIR="test_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/all_tests_1000docs_$(date +%Y%m%d_%H%M%S).log"

# Use the special conftest_1000docs.py for testing
export PYTEST_CONFTEST_PATH="tests/conftest_1000docs.py"

echo "Starting test run at $(date)"
echo "Log file: $LOG_FILE"
echo

# Run all tests with the 1000 document fixture
echo "Running tests with 1000+ documents..."

pytest -v tests/test_all_with_1000_docs.py \
    tests/test_basic_1000.py \
    tests/test_colbert_1000.py \
    tests/test_noderag_1000.py \
    tests/test_minimal_1000.py \
    | tee "$LOG_FILE"

echo
echo "Testing with real PMC documents (requires test container)..."
echo "Setting up test environment for real PMC documents..."

# Check for at least 1000 documents in each test
python verify_1000_docs_testing.py | tee -a "$LOG_FILE"

echo
echo "=== All tests completed $(date) ==="
echo "See $LOG_FILE for full log"
echo
echo "To run a specific test with 1000+ docs:"
echo "  PYTEST_CONFTEST_PATH=tests/conftest_1000docs.py pytest -v tests/test_basic_1000.py"
