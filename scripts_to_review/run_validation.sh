#!/bin/bash
# run_validation.sh - Script to run SQL validation and E2E tests using Poetry

# Parse command line arguments
E2E_ONLY=false
ALL_TESTS=false

# Handle command line arguments
for arg in "$@"
do
    case $arg in
        --e2e)
        E2E_ONLY=true
        shift
        ;;
        --all)
        ALL_TESTS=true
        shift
        ;;
        -h|--help)
        echo "Usage: ./run_validation.sh [options]"
        echo "Options:"
        echo "  --e2e       Run only E2E tests with real dependencies"
        echo "  --all       Run all tests (unit, SQL validation, and E2E)"
        echo "  -h, --help  Show this help message"
        exit 0
        ;;
    esac
done

# Make sure required scripts are executable
chmod +x run_validation_tests.py

# Run the appropriate tests
if [ "$E2E_ONLY" = true ]; then
    echo "Running E2E tests with Poetry..."
    python run_validation_tests.py # Start the container first to make the connection available
    # Allow time for the container to initialize
    sleep 3
    # Run the E2E tests
    poetry run pytest tests/test_colbert.py::test_colbert_pipeline_e2e_metrics tests/test_noderag.py::test_noderag_pipeline_e2e_metrics tests/test_graphrag.py::test_graphrag_pipeline_e2e_metrics -v
elif [ "$ALL_TESTS" = true ]; then
    echo "Running all tests with Poetry..."
    # Start the container first to make the connection available
    python run_validation_tests.py
    # Allow time for the container to initialize
    sleep 3
    # Run all tests
    poetry run pytest
else
    # Default: Run SQL validation tests
    echo "Running SQL validation tests with Poetry..."
    poetry run python run_validation_tests.py
fi
