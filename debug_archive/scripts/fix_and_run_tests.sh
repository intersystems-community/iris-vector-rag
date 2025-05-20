#!/bin/bash
# A focused script to fix common issues and run GraphRAG tests with 1000 documents.
# This script:
# 1. Sets env variables correctly
# 2. Runs a simplified test sequence that avoids common issues
# 3. Properly uses Poetry for dependency management

echo "=== Starting GraphRAG 1000-document tests ==="
echo "Current directory: $(pwd)"

# Set environment variables for test
export TEST_IRIS=true
export TEST_DOCUMENT_COUNT=1000
export COLLECT_PERFORMANCE_METRICS=true
export USE_MOCK_EMBEDDINGS=false

# Ensure we're using the right Python environment
if [ -x "$(command -v poetry)" ]; then
  PYTHON_CMD="poetry run python"
  echo "Using Poetry for Python environment"
else
  PYTHON_CMD="python"
  echo "Poetry not found, using system Python"
fi

# First run a simple validation to ensure the testcontainer works
echo "Validating testcontainer setup..."
$PYTHON_CMD debug_testcontainer.py 

# Now run the targeted tests using pytest directly through poetry
echo "Running GraphRAG tests with 1000 documents..."

# First the basic testcontainer test to verify it's set up
echo "Verifying testcontainer connection with a single test..."
$PYTHON_CMD -m pytest tests/test_graphrag_with_testcontainer.py::test_iris_testcontainer_setup -v

# If that works, run the more comprehensive suite
echo "Running full GraphRAG testcontainer tests..."
$PYTHON_CMD -m pytest tests/test_graphrag_with_testcontainer.py -v

# Summarize results
echo "Test execution completed."
echo "Check output above for test results."
echo ""
echo "If tests still skip, check database connection errors."
echo "You might need to restart Docker or free up more memory."
