# Real Data Testing Plan

## Current Project Status & Critical Blocker

**IMPORTANT:** As of May 21, 2025, the execution of this Real Data Testing Plan is **BLOCKED**.

A critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function prevents the successful loading of documents with their vector embeddings into the database. While text data can be loaded, operations requiring these embeddings (i.e., most RAG pipeline functionalities and meaningful real-data testing/benchmarking) cannot be performed on newly ingested real data.

**This entire plan is contingent on the resolution of this blocker.** Steps involving embedding generation, embedding-dependent tests, and benchmarking with real embeddings cannot proceed until this issue is fixed.

For more details on this blocker, please refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md) and [`docs/MANAGEMENT_SUMMARY.md`](docs/MANAGEMENT_SUMMARY.md).

## Overview

This document outlines a detailed plan for testing all RAG techniques with real PMC data and a real LLM, once the aforementioned blocker is resolved. This testing is a **critical requirement** specified in our [`.clinerules`](.clinerules:1) file:

> "Tests must use real PMC documents, not synthetic data. At least 1000 documents should be used."

While we have implemented the infrastructure for testing with real data, the actual execution of these tests is pending the blocker resolution. This document provides a comprehensive plan for completing this essential task thereafter.

## Current Infrastructure Status (Pre-Blocker Resolution)

- **Infrastructure**: Testing infrastructure is largely in place, including scripts for verification and execution (though some scripts may need review/consolidation from `scripts_to_review/`).
- **Data**: Sample PMC documents are available for text processing. Capability to load their embeddings is BLOCKED.
- **Testing**: End-to-end test structures have been implemented but not executed with real data embeddings.
- **LLM Integration**: Real LLM integration for answer generation has been designed but not tested in a full real-data pipeline.

## Requirements

### 1. Real Data Requirements

- **Minimum Document Count**: At least 1000 real PMC documents must be used
- **Document Verification**: Documents must be verified as genuine PMC XML files
- **Database Storage**: Documents must be properly stored in the IRIS database
- **Embedding Generation**: Vector embeddings must be generated for all documents (Currently BLOCKED).

### 2. Real LLM Requirements

- **LLM Selection**: A real LLM must be used for answer generation (not mocks)
- **API Integration**: Proper API keys and configuration for the selected LLM
- **Error Handling**: Robust error handling for LLM API calls
- **Response Processing**: Proper processing of LLM responses

## Step-by-Step Testing Plan

### Phase 1: Preparation

1. **Verify PMC Document Availability**
   ```bash
   # Check if we have enough PMC documents (text content)
   # Note: Current verification scripts might need adjustment to distinguish between text-only and embedding presence.
   python scripts_to_review/verify_real_pmc_database.py # (Path and script functionality to be confirmed post Phase 0 review)
   ```

2. **Download Additional PMC Documents (if needed)**
   ```bash
   # If we don't have enough documents, download more (text content)
   python scripts_to_review/load_pmc_data.py --min-docs 1000 # (Path to be confirmed post Phase 0 review)
   ```

3. **Verify IRIS Docker Container**
   ```bash
   # Check if IRIS container is running
   docker ps | grep iris
   
   # If not running, start it
   docker-compose -f docker-compose.iris-only.yml up -d
   ```

4. **Configure LLM API Access**
   - Obtain API keys for the selected LLM (e.g., OpenAI, Anthropic, etc.)
   - Configure the API keys in the appropriate environment variables or configuration files
   - Test the LLM API connection:
   ```bash
   # Test LLM API connection
   python -c "from common.embedding_utils import test_llm_connection; test_llm_connection()"
   ```

### Phase 2: Database Setup

1. **Initialize Database Schema**
   ```bash
   # Initialize the database schema
   python run_db_init_local.py --force-recreate
   ```

2. **Load PMC Documents into Database**
   ```bash
   # Load document text content into the database
   python scripts_to_review/load_pmc_data.py --load-to-db # (Path to be confirmed post Phase 0 review)
   ```

3. **Generate and Load Embeddings for Documents**
   ```bash
   # Generate embeddings for all documents and attempt to load them.
   # THIS STEP IS CURRENTLY BLOCKED by the TO_VECTOR/ODBC issue.
   python scripts_to_review/generate_embeddings.py # (Path to be confirmed post Phase 0 review)
   ```

4. **Verify Database Content (Post Blocker Resolution)**
   ```bash
   # Verify that the database contains at least 1000 documents WITH embeddings
   python scripts_to_review/verify_real_pmc_database.py # (Path and script functionality to be confirmed post Phase 0 review)
   ```

### Phase 3: Test Execution

1. **Run End-to-End Tests (Post Blocker Resolution)**
   The primary script for E2E testing is [`scripts/run_e2e_tests.py`](scripts/run_e2e_tests.py:1). This script should be configured to run tests for all RAG techniques against the real data (1000+ documents with embeddings).
   ```bash
   # Ensure .venv is active
   # This command should cover all RAG techniques for E2E testing with real data.
   python scripts/run_e2e_tests.py --min-docs 1000 --output-dir test_results/e2e_real_data --llm-provider <your_llm_provider>
   ```
   Individual `pytest tests/test_*.py` files can still be run for debugging specific pipelines, but the E2E script provides a consolidated approach.
   (Note: The older `scripts_to_review/run_real_pmc_1000_tests.py` might be superseded by or integrated into `scripts/run_e2e_tests.py`.)
   ```

### Phase 4: Benchmarking

1. **Run Benchmarks for All Techniques**
   ```bash
   # Run benchmarks for all techniques
   python scripts/run_rag_benchmarks.py
   ```

2. **Generate Benchmark Visualizations**
   ```bash
   # Generate visualizations from benchmark results (script path to be confirmed)
   python scripts_to_review/demo_benchmark_analysis.py # (Path to be confirmed post Phase 0 review)
   ```

3. **Compare with Published Benchmarks**
   ```bash
   # Compare our results with published benchmarks
   python eval/comparative/analysis.py
   ```

### Phase 5: Documentation and Reporting

1. **Update Test Results Documentation**
   - Document the results of all tests in `docs/E2E_TEST_RESULTS.md`
   - Include any issues encountered and their resolutions

2. **Update Benchmark Results Documentation**
   - Document the benchmark results in `docs/BENCHMARK_RESULTS.md`
   - Include visualizations and comparative analysis

3. **Update Project Status Documentation**
   - Update [`PLAN_STATUS.md`](PLAN_STATUS.md) to reflect the completion of testing with real data (post-blocker).
   - Update [`docs/PROJECT_COMPLETION_REPORT.md`](docs/PROJECT_COMPLETION_REPORT.md) with actual results.

## Debugging Common Issues

### Database Connection Issues

If you encounter database connection issues:

1. **Check IRIS Container Status**
   ```bash
   docker ps | grep iris
   ```

2. **Check IRIS Logs**
   ```bash
   docker logs iris-container
   ```

3. **Verify Connection Settings**
   - Check `common/iris_connector.py` for correct connection parameters
   - Ensure the host, port, username, and password are correct

### Document Loading Issues

If you encounter issues loading documents:

1. **Check Document Format**
   - Ensure the PMC documents are valid XML files
   - Verify the document structure matches the expected format

2. **Check Storage Path**
   - Ensure the documents are stored in the expected directory
   - Verify the path in data loading scripts (e.g., `scripts_to_review/load_pmc_data.py`) is correct.

3. **Check Database Schema**
   - Verify the database schema matches the expected structure
   - Check for any SQL errors during document loading

### Embedding Generation Issues

If you encounter issues generating embeddings:

1. **Check Embedding API Access**
   - Verify the API keys for the embedding service are correct
   - Check for any rate limiting or quota issues

2. **Check Embedding Dimensions**
   - Ensure the embedding dimensions match the expected values
   - Verify the embedding model is correctly configured

3. **Check Batch Processing**
   - If processing large batches, check for memory issues
   - Consider reducing batch size if necessary

### LLM Integration Issues

If you encounter issues with LLM integration:

1. **Check API Keys**
   - Verify the API keys for the LLM service are correct
   - Check for any rate limiting or quota issues

2. **Check Request Format**
   - Ensure the requests to the LLM API are correctly formatted
   - Verify the prompt templates are properly structured

3. **Check Response Handling**
   - Ensure the LLM responses are correctly parsed
   - Verify error handling for failed API calls

## Conclusion

Completing the testing with real data (once the blocker is resolved) is a critical requirement for this project. This plan provides a comprehensive approach to fulfilling this requirement, ensuring that all RAG techniques are thoroughly tested with real PMC documents and a real LLM.

By following this plan (post-blocker resolution), we will:

1. Verify that all RAG techniques work correctly with real data, including embeddings.
2. Generate actual benchmark results for comparison.
3. Identify any issues that only appear with real data.
4. Provide empirical evidence for the effectiveness of different RAG techniques.

Only once this testing is complete can we confidently state that the project has met all requirements specified in the [`.clinerules`](.clinerules:1) file.