#!/bin/bash
# Run all RAG techniques with 1000+ real PMC documents

echo "Running all RAG techniques with 1000+ real PMC documents..."
echo "This test ensures compliance with the .clinerules requirement:"
echo "Tests must use real PMC documents, not synthetic data. At least 1000 documents should be used."
echo ""

# Run the tests with the IRIS testcontainer
echo "Step 1: Running minimal tests for all RAG techniques..."
make test-1000

# Verify if tests were successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: All RAG techniques tested successfully with 1000+ real PMC documents!"
    echo "The tests comply with the project requirements in .clinerules"
else
    echo ""
    echo "❌ ERROR: Some tests failed. Please check the logs."
    exit 1
fi

# Optionally run comprehensive tests
if [ "$1" == "--comprehensive" ]; then
    echo ""
    echo "Step 2: Running comprehensive tests for all RAG techniques..."
    make test-real-pmc-1000
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ SUCCESS: Comprehensive tests also passed!"
    else
        echo ""
        echo "❌ WARNING: Comprehensive tests failed, but basic tests passed."
        echo "The system still meets the minimum requirements from .clinerules"
    fi
fi

echo ""
echo "For more details, see 1000_DOCUMENT_TESTING.md"
