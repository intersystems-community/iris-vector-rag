# RAG Templates Testing Guide

This document provides an overview of the testing strategies, types of tests, and instructions for running tests for the RAG Templates project.

## Current Testing Status & Critical Blocker

**IMPORTANT:** As of May 21, 2025, full end-to-end testing and benchmarking with newly loaded real PMC data (especially data requiring vector embeddings) is **BLOCKED**.

This is due to a critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function, which prevents the successful loading of documents with their vector embeddings into the database. While text data can be loaded, operations requiring these embeddings (i.e., most RAG pipeline functionalities) cannot be performed on newly ingested real data.

Workarounds for *querying* existing vector data using client-side SQL (see [`common/vector_sql_utils.py`](common/vector_sql_utils.py:1)) have been implemented, but the data *loading* issue for embeddings remains unresolved.

Consequently, while the testing framework is in place:
- Unit and component tests are generally runnable.
- End-to-end tests relying on mock data or pre-existing data (if any) may run.
- End-to-end tests requiring fresh loading and use of real PMC document embeddings **cannot be fully executed as intended.**

For more details on this blocker, please refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md:1) and [`docs/MANAGEMENT_SUMMARY.md`](docs/MANAGEMENT_SUMMARY.md:1).

## Testing Requirements (.clinerules)

According to our [`.clinerules`](.clinerules:1) file, all tests must:

1. Use `pytest` for testing (not shell scripts).
2. Verify RAG techniques work with real data. (Partially BLOCKED for embedding-dependent aspects)
3. Use real PMC documents (not synthetic data). (Partially BLOCKED for embedding-dependent aspects)
4. Run with at least 1000 documents. (Partially BLOCKED for embedding-dependent aspects)
5. Test the complete pipeline from ingestion to answer generation. (Partially BLOCKED for embedding-dependent aspects)
6. Include assertions on actual results (not just logging).

## Test Environment Setup

The primary testing environment consists of:
- **Python 3.11+** managed on the host machine.
- **`uv`** for virtual environment creation and package installation (from `pyproject.toml` or `requirements.txt`).
- **InterSystems IRIS Database** running in a dedicated Docker container, managed by [`docker-compose.iris-only.yml`](docker-compose.iris-only.yml:1).

Refer to the main [`README.md`](README.md:1) for detailed setup instructions.
The IRIS database connection is typically managed by [`common/iris_connector.py`](common/iris_connector.py:1), which uses environment variables for configuration.

## Types of Tests and How to Run Them

### 1. Unit and Component Tests

These tests verify individual modules, functions, and classes in isolation, often using mocks.

```bash
# Activate your virtual environment (e.g., source .venv/bin/activate)

# Run all tests in the tests/ directory
pytest tests/

# Run tests for a specific component (example)
pytest tests/test_basic_rag.py
pytest tests/test_vector_sql_utils.py
```
Key component test files include:
- [`tests/test_basic_rag.py`](tests/test_basic_rag.py:1)
- [`tests/test_hyde.py`](tests/test_hyde.py:1)
- [`tests/test_crag.py`](tests/test_crag.py:1)
- [`tests/test_colbert.py`](tests/test_colbert.py:1)
- [`tests/test_noderag.py`](tests/test_noderag.py:1)
- [`tests/test_graphrag.py`](tests/test_graphrag.py:1)
- [`tests/test_iris_connector.py`](tests/test_iris_connector.py:1)
- [`tests/test_vector_sql_utils.py`](tests/test_vector_sql_utils.py:1)
- [`tests/test_context_reduction.py`](tests/test_context_reduction.py:1)
- [`tests/test_data_loader.py`](tests/test_data_loader.py:1) (May require specific data setup or be partially blocked if loading embeddings)

### 2. End-to-End (E2E) Tests

These tests verify the complete RAG pipelines. The primary script for running E2E tests is [`scripts/run_e2e_tests.py`](scripts/run_e2e_tests.py:1).

**E2E Tests with Mock Data:**
Many E2E tests can run using mocked database interactions or pre-defined small datasets if the test design allows. Consult individual test configurations.

**E2E Tests with Real Data (Text Processing):**
It might be possible to test parts of the pipeline that involve loading and processing the text content of PMC documents, provided these tests do not depend on vector embeddings.

**E2E Tests with Real Data (Embeddings & Vector Search):**
This is the target for full validation (e.g., using [`tests/test_all_with_1000_docs.py`](tests/test_all_with_1000_docs.py:1) or tests run by [`scripts/run_e2e_tests.py`](scripts/run_e2e_tests.py:1) that require embeddings).
- **Status:** Currently **BLOCKED** due to the inability to load new document embeddings.
- **Goal:** Test with at least 1000 real PMC documents.
- **Execution (once blocker is resolved):**
  ```bash
  # Example: Run all E2E tests, which should include 1000+ doc tests
  python scripts/run_e2e_tests.py --min-docs 1000 --output-dir test_results/e2e
  ```
The script [`scripts/verify_real_data_testing.py`](scripts/verify_real_data_testing.py:1) can be used to check the database state before running real-data tests.

### 3. Benchmarking

Benchmarking aims to compare the performance and quality of different RAG techniques.
- **Script:** [`scripts/run_rag_benchmarks.py`](scripts/run_rag_benchmarks.py:1)
- **Status:** Benchmarking with real data (requiring embeddings) is currently **BLOCKED**.
- **Goal:** Run against 1000+ real PMC documents with a real LLM.
- For more details, see [`docs/BENCHMARKING_README.md`](docs/BENCHMARKING_README.md:1) (Note: This file might need to be created or renamed from existing benchmark docs like `BENCHMARK_SETUP.md` or `BENCHMARK_EXECUTION_PLAN.md`).

## Vector Operations

Vector similarity searches (e.g., using `VECTOR_COSINE`) are performed using client-side SQL constructed via utilities in [`common/vector_sql_utils.py`](common/vector_sql_utils.py:1). This approach works around limitations of the built-in `TO_VECTOR()` SQL function when used with parameters in `SELECT` queries. The primary remaining issue is loading new embeddings, which also involves `TO_VECTOR()`.

## Test Coverage

To generate a test coverage report:

```bash
# Ensure .venv is active
uv pip install coverage # If not already installed
coverage run -m pytest tests/
coverage report
coverage html # For an HTML report in htmlcov/
```

## Alternative Testing Environments (e.g., Testcontainers)

Some specialized test scenarios or documents (e.g., [`docs/CONTEXT_REDUCTION_TESTING.md`](docs/CONTEXT_REDUCTION_TESTING.md:1), [`docs/REAL_DATA_TESTING.md`](docs/REAL_DATA_TESTING.md:1)) describe using Testcontainers for managing ephemeral IRIS instances. This can be an alternative for specific CI/CD needs or isolated tests but is not the primary local development and testing strategy. Refer to those documents for Testcontainer-specific setup and execution, keeping in mind the overarching data loading blocker.
