#!/bin/bash
# Script to run tests with 1000 documents
# This script sets all necessary environment variables and runs pytest

# Set environment variables
export TEST_IRIS=true
export TEST_DOCUMENT_COUNT=1000
export COLLECT_PERFORMANCE_METRICS=true

# By default, use real embeddings unless specified
if [ "$1" == "--mock-embeddings" ]; then
    export USE_MOCK_EMBEDDINGS=true
    echo "Using mock embeddings (faster but less accurate)"
    shift
else
    export USE_MOCK_EMBEDDINGS=false
    echo "Using real embeddings (more accurate but slower)"
fi

# Determine which technique to test
TECHNIQUE=${1:-"all"}
echo "Testing technique: $TECHNIQUE"

# Build the test pattern
case "$TECHNIQUE" in
    "all")
        TEST_PATTERN="tests/test_graphrag_with_testcontainer.py tests/test_graphrag_large_scale.py"
        echo "Running all tests with testcontainer"
        ;;
    "graphrag")
        TEST_PATTERN="tests/test_graphrag_with_testcontainer.py tests/test_graphrag_large_scale.py"
        echo "Running GraphRAG tests"
        ;;
    "colbert")
        TEST_PATTERN="tests/test_colbert.py"
        echo "Running ColBERT tests"
        ;;
    "noderag")
        TEST_PATTERN="tests/test_noderag.py"
        echo "Running NodeRAG tests"
        ;;
    "hyde")
        TEST_PATTERN="tests/test_hyde.py"
        echo "Running HyDE tests"
        ;;
    "crag")
        TEST_PATTERN="tests/test_crag.py"
        echo "Running CRAG tests"
        ;;
    "basic")
        TEST_PATTERN="tests/test_basic_rag.py"
        echo "Running Basic RAG tests"
        ;;
    *)
        echo "Unknown technique: $TECHNIQUE"
        echo "Available options: all, graphrag, colbert, noderag, hyde, crag, basic"
        exit 1
        ;;
esac

# Robust pytest options
PYTEST_OPTS="-xvs --log-cli-level=INFO --tb=short"

# Always use 'poetry run' to ensure tests run in the correct environment
CMD="poetry run pytest $TEST_PATTERN $PYTEST_OPTS"
echo "Running command: $CMD"

# Execute with retries
MAX_RETRIES=3
SUCCESS=false

for ((i=1; i<=MAX_RETRIES; i++)); do
    if [ $i -gt 1 ]; then
        echo "Retry attempt $i/$MAX_RETRIES..."
        sleep 2
    fi
    
    # Run the command
    $CMD
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "Tests completed successfully!"
        SUCCESS=true
        break
    else
        echo "Attempt $i failed with errors"
    fi
done

if [ "$SUCCESS" = false ]; then
    echo "All $MAX_RETRIES attempts failed."
    exit 1
fi

exit 0
