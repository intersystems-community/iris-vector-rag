#!/bin/bash
# A very simple script that directly runs a GraphRAG test with 1000 documents
# This script focuses only on running the test and showing its output

# Set environment variables for testing
export TEST_IRIS=true
export TEST_DOCUMENT_COUNT=1000
export USE_MOCK_EMBEDDINGS=false
export COLLECT_PERFORMANCE_METRICS=true

echo "=== Running GraphRAG test with 1000 documents ==="
echo "Current directory: $(pwd)"
echo "Environment variables:"
echo "  TEST_IRIS=$TEST_IRIS"
echo "  TEST_DOCUMENT_COUNT=$TEST_DOCUMENT_COUNT"
echo "  USE_MOCK_EMBEDDINGS=$USE_MOCK_EMBEDDINGS"
echo "Testing started at: $(date)"
echo ""

# First run just the testcontainer setup test to verify that's working
echo "Running testcontainer setup test..."
poetry run pytest tests/test_graphrag_with_testcontainer.py::test_iris_testcontainer_setup -v

# Check if setup test succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "Testcontainer setup successful. Now running document loading test..."
    
    # Run the document loading test
    poetry run pytest tests/test_graphrag_with_testcontainer.py::test_load_pmc_documents -v
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Document loading successful. Now running full GraphRAG test suite..."
        
        # Run the full GraphRAG test suite
        poetry run pytest tests/test_graphrag_with_testcontainer.py -v
    else
        echo "Document loading test failed."
    fi
else
    echo "Testcontainer setup test failed. Cannot proceed."
fi

echo ""
echo "Testing completed at: $(date)"
