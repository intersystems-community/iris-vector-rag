# Real PMC 1000+ Document Testing (Supplementary Guide)

**Note:** The primary and most up-to-date guide for all testing is [`docs/TESTING.md`](docs/TESTING.md). This document provides supplementary details on specific scripts and `make` targets related to 1000+ real PMC document testing, which may be alternative or older methods.

## Current Testing Status & Critical Blocker

**IMPORTANT:** As of May 21, 2025, full end-to-end testing with newly loaded real PMC data (especially data requiring vector embeddings) is **BLOCKED**.

This is due to a critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function, which prevents the successful loading of documents with their vector embeddings into the database. While text data can be loaded, operations requiring these embeddings cannot be performed on newly ingested real data.

Consequently, while the testing framework and some mechanisms described here are in place, their execution with real embeddings is impacted. Tests on text-based components or with mock/pre-existing embeddings may still function. For more details on this blocker, refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md).

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
- Loads real PMC documents (text content) into the database (Note: Loading of new embeddings is currently BLOCKED).
- Runs all tests with real PMC data (Note: Tests requiring new embeddings will be impacted by the blocker).
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
