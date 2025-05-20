# Real PMC 1000+ Document Testing

This document describes the implementation of testing with 1000+ REAL PMC documents as required by the project's `.clinerules` file.

## Background

The project's `.clinerules` file explicitly states:

> "Tests must use real PMC documents, not synthetic data. At least 1000 documents should be used."

To satisfy this requirement, we've implemented a comprehensive testing system that ensures all RAG techniques are tested against a real database containing at least 1000 real PMC documents.

## Components

The implementation consists of several key components:

### 1. Data Verification

`verify_real_pmc_database.py` - A script that verifies:
- We have real PMC documents in the data directory
- A real IRIS database is used (not mocks)
- The database contains at least 1000 real documents

### 2. Real PMC Test Runner

`run_real_pmc_1000_tests.py` - A script that runs all tests with real PMC documents by:
- Setting the environment to use the real PMC conftest
- Running pytest with the appropriate configuration
- Reporting results

### 3. Automated Test Process

`run_with_real_pmc_data.sh` - A shell script that automates the entire process:
- Verifies real PMC data availability
- Starts an IRIS container if needed
- Loads real PMC documents into the database
- Runs all tests with real PMC data
- Verifies compliance with project requirements

### 4. Real PMC Test Configuration

`tests/conftest_real_pmc.py` - A pytest conftest file that:
- Forces use of a real (non-mock) database connection
- Loads real PMC documents if not enough are present
- Provides fixtures for database access and document verification

### 5. Comprehensive Test Suite

`tests/test_all_with_1000_docs.py` - A test file that:
- Tests all six RAG techniques (BasicRAG, HyDE, ColBERT, NodeRAG, GraphRAG, CRAG)
- Verifies each pipeline works with 1000+ documents
- Measures and reports performance metrics

## Usage

There are two main ways to run the 1000+ document tests:

### Option 1: Quick Development Testing + Real Verification

```bash
make test-all-1000-docs-compliance
```

This runs tests in two stages:
1. First with mock data for quick verification (faster, good for development)
2. Then with real PMC documents for proper compliance (slower, ensures actual compliance)

### Option 2: Full Real Database Testing

```bash
make test-with-real-pmc-db
```

This runs the full process:
1. Verifies real PMC data availability
2. Starts an IRIS container
3. Loads real PMC documents into the database
4. Runs all tests with real PMC data
5. Verifies that real data was used

## Verification Process

The verification process ensures that:

1. **Real Documents**: Tests use actual PMC XML documents, not synthetic data
2. **Sufficient Volume**: At least 1000 documents are used
3. **Real Database**: A real IRIS database is used, not mock objects
4. **All Techniques**: All six RAG techniques are tested

## Performance Considerations

Testing with 1000+ real documents can be time-consuming. For development:
- Use the mock-based testing during development for faster feedback
- Run the full real document tests before committing changes

## Benefits

This implementation offers several benefits:

1. **Full Compliance**: Meets the `.clinerules` requirement for real data testing
2. **Development Efficiency**: Provides options for faster testing during development
3. **Performance Insights**: Measures retrieval times with realistic data volumes
4. **Robust Testing**: Ensures RAG techniques work with real-world data

## Future Improvements

Potential future improvements include:

1. **Parallel Testing**: Implement parallel test execution for faster results
2. **Test Result Caching**: Cache test results for unchanged code
3. **Document Variety Control**: Ensure a diversity of medical topics in the test documents
4. **Extended Performance Metrics**: Add more detailed performance reporting
