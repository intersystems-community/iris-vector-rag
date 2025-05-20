# Real PMC Data Testing with 1000 Documents

This document describes how to test all RAG techniques with 1000 real PMC (PubMed Central) documents.

## Overview

The RAG techniques implemented in this project (BasicRAG, ColBERT, NodeRAG, GraphRAG, and Context Reduction) can be tested with real PMC documents to verify that they work correctly with actual medical research data. This provides a more realistic test scenario than using synthetic data.

## Requirements

1. **PMC Documents**: You need to have PMC XML files downloaded and stored in the `data/pmc_oas_downloaded` directory (or specify a different directory with the `PMC_DATA_DIR` environment variable).

2. **IRIS Database**: The tests use the IRIS testcontainer to create an isolated database environment.

3. **Python Environment**: Make sure you have the poetry environment set up with all dependencies installed.

## PMC Data Structure

The PMC data should be organized in a directory structure like:

```
data/pmc_oas_downloaded/
  ├── PMC006xxxxxx/
  │   └── PMC6627345.xml
  ├── PMC007xxxxxx/
  │   ├── PMC7524946.xml
  │   └── ...
  └── PMC010xxxxxx/
      ├── PMC10070455.xml
      └── ...
```

You need at least 10 valid PMC XML files for the tests to run, but ideally, you should have 1000 for a full-scale test.

## Running the Tests

### Using the Script

The simplest way to run the tests is to use the provided script:

```bash
./run_real_pmc_tests.sh
```

This script:
1. Checks if PMC data is available
2. Sets up the necessary environment variables
3. Runs the tests with pytest
4. Saves detailed logs to a file in the `test_results` directory

### Manual Execution

If you prefer, you can run the tests manually:

```bash
# Set the environment variables
export TEST_IRIS=true
export TEST_DOCUMENT_COUNT=1000
export USE_REAL_PMC_DATA=true
export PMC_DATA_DIR="data/pmc_oas_downloaded"  # Adjust as needed

# Run the test
poetry run python -m pytest tests/test_real_pmc_with_1000_docs.py -v --log-cli-level=INFO
```

## Test Structure

The `test_real_pmc_with_1000_docs.py` file contains tests for each RAG technique:

1. **Database Setup**: The test loads real PMC documents into the IRIS testcontainer database.

2. **Basic RAG**: Tests the standard vector-based retrieval.

3. **ColBERT**: Tests token-level retrieval using the ColBERT approach.

4. **NodeRAG**: Tests knowledge graph node-based retrieval, creating nodes from the PMC documents.

5. **GraphRAG**: Tests knowledge graph traversal, creating both nodes and edges from the PMC documents.

6. **Context Reduction**: Tests the context reduction technique with the real documents.

7. **All Techniques**: A combined test that runs all techniques in sequence to ensure they work together.

## Test Fixtures

The tests use several fixtures:

- `iris_with_real_pmc_data`: Sets up the IRIS testcontainer and loads real PMC documents.
- `mock_embedding_func`: Provides a consistent mock embedding function for testing.
- `mock_llm_func`: Provides a mock LLM function that returns deterministic responses.
- `sample_queries`: Provides a set of medical queries for testing.

## Scaling to More Documents

Once the tests pass with 1000 documents, you can scale up to even larger datasets:

1. Set the `TEST_DOCUMENT_COUNT` environment variable to a higher number.
2. Make sure you have enough PMC documents available.
3. Be aware that loading more documents will increase the test runtime.

## Performance Considerations

- Loading 1000 real PMC documents into the database takes time. The test includes logging of performance metrics.
- The tests are designed to work even with fewer documents (minimum 10) if 1000 aren't available.
- Mock embedding and LLM functions are used for testing to avoid network dependencies.

## Troubleshooting

If the tests fail:

1. **Not enough PMC documents**: Make sure you have enough PMC XML files in the specified directory.
2. **Memory issues**: The testcontainer might need more memory for large document counts.
3. **Database connection**: Check that the IRIS testcontainer is running correctly.
4. **Logs**: Examine the log file for detailed error information.

## Extending the Tests

If you want to add more tests or test with different data:

1. Add new test functions to `test_real_pmc_with_1000_docs.py`.
2. Use the `@pytest.mark.real_pmc` marker for new tests that require real PMC data.
3. Consider adding specific test cases for edge cases or specific document types.
