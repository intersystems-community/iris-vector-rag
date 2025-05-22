# Large-Scale Testing (Potentially Outdated - Refer to Primary Testing Guide)

**IMPORTANT NOTICE:** This document may contain outdated information, particularly regarding specific scripts, environment setup, and database schema details for embeddings. The primary and most up-to-date guide for all testing is [`docs/TESTING.md`](docs/TESTING.md). Refer to that document for canonical instructions.

This document describes an approach to running large-scale tests (1000+ documents, potentially up to 92,000+) for the RAG templates project.

## Current Testing Status & Critical Blocker

**IMPORTANT:** As of May 21, 2025, full end-to-end testing and benchmarking with newly loaded real PMC data (especially data requiring vector embeddings) is **BLOCKED**.

This is due to a critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function, which prevents the successful loading of documents with their vector embeddings into the database. While text data can be loaded, operations requiring these embeddings cannot be performed on newly ingested real data.

Consequently, any large-scale testing reliant on new embeddings is impacted. For more details on this blocker, refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md).

## Prerequisites

- Python 3.11+ installed.
- `uv` (Python package installer and virtual environment manager). See [`README.md`](README.md) for installation.
- Docker installed (if using Testcontainers or the primary dedicated IRIS Docker setup).
- Sufficient RAM (e.g., 8GB+) and disk space (1GB+ for data, more for larger datasets).

## Environment Setup

Ensure your Python virtual environment is set up using `uv` and dependencies are installed as per [`README.md`](README.md).
For Testcontainer-specific tests mentioned in older guides, you might need `testcontainers-iris`:
```bash
# Ensure .venv is active
uv pip install testcontainers-iris testcontainers # If using Testcontainer-based approaches
```

## Running Large-Scale Tests

We've provided several scripts for running tests with 1000 documents (Note: these scripts, `./run_tests_with_1000_docs.sh` and `./run_1000_docs_tests.py`, are located in `scripts_to_review/` and their canonical status should be verified against primary testing scripts like `scripts/run_e2e_tests.py`):

### Shell Script (Example from `scripts_to_review/`)

```bash
# Example: Run all tests with 1000 documents
./scripts_to_review/run_tests_with_1000_docs.sh all

# Example: Run only GraphRAG tests with 1000 documents
./scripts_to_review/run_tests_with_1000_docs.sh graphrag

# Example: Use mock embeddings for faster testing (less accurate)
./scripts_to_review/run_tests_with_1000_docs.sh --mock-embeddings all
```

### Python Script (Example from `scripts_to_review/`)

```bash
# Example: Run all tests with 1000 documents
python scripts_to_review/run_1000_docs_tests.py

# Example: Run only GraphRAG tests
python scripts_to_review/run_1000_docs_tests.py graphrag

# Example: Use mock embeddings
python scripts_to_review/run_1000_docs_tests.py --mock-embeddings
```

## Environment Variables

The tests use several environment variables to control behavior:

- `TEST_IRIS=true` - Use testcontainer for isolated testing
- `TEST_DOCUMENT_COUNT=1000` - Set the number of documents to use
- `USE_MOCK_EMBEDDINGS=false` - Use real embeddings (default)
- `COLLECT_PERFORMANCE_METRICS=true` - Collect detailed performance metrics

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check Docker is running
   - Ensure ports aren't conflicting 
   - Try running with fewer documents first

2. **Memory Issues**
   - Reduce document count
   - Use `--mock-embeddings` flag
   - Close other memory-intensive applications

3. **Slow Tests**
   - Use mock embeddings for faster testing
   - Run specific test suites instead of all tests
   - Consider using smaller document counts for development

### Database Schema Compatibility (Corrected)

The tests require database tables with specific column names and types. **Crucially, embeddings should be stored using the native `VECTOR` type available in IRIS 2025.1+ for efficient vector operations.**

- `SourceDocuments` table requires:
  - `doc_id VARCHAR(255) PRIMARY KEY`
  - `title VARCHAR(1000)`
  - `content CLOB` (or `LONGVARCHAR`)
  - `embedding VECTOR` (Note: Operations involving `TO_VECTOR()` on this column must use workarounds from [`common/vector_sql_utils.py`](common/vector_sql_utils.py:1) for querying, and loading new embeddings is currently BLOCKED.)

- `KnowledgeGraphNodes` table requires:
  - `node_id VARCHAR(255) PRIMARY KEY`
  - `node_type VARCHAR(100)`
  - `node_name VARCHAR(1000)`
  - `content CLOB` (or `description_text LONGVARCHAR`)
  - `embedding VECTOR` (Subject to the same `TO_VECTOR()` considerations as above)

## Performance Considerations

- Large-scale tests may take 30+ minutes to complete
- Consider using specific test suites during development
- Use `--mock-embeddings` flag when testing non-embedding functionality
- Set lower document counts (100-200) for quicker feedback during development
