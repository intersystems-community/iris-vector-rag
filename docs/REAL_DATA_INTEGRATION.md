# Real Data Integration Guide (Supplementary)

**Note:** For general project setup, see [`README.md`](README.md:1). For an overview of all testing procedures, see [`docs/TESTING.md`](docs/TESTING.md:1). This document provides specific details on integrating and testing with real PMC data, particularly concerning the `load_pmc_data.py` script.

## Current Testing Status & Critical Blocker

**IMPORTANT:** As of May 21, 2025, full end-to-end testing and use of newly loaded real PMC data (especially data requiring vector embeddings) is **BLOCKED**.

This is due to a critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function, which prevents the successful loading of documents with their vector embeddings into the database. While text data can be loaded using scripts like `load_pmc_data.py`, operations requiring these embeddings on newly ingested real data cannot be performed.

For more details on this blocker, refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md:1).

## Overview

This guide explains how to use a real IRIS database connection and process real PMC data files, focusing on the `scripts_to_review/load_pmc_data.py` script. The system supports:

- Real IRIS database connections
- Mock database connections for testing
- XML processing with metadata extraction
- Batched data loading for performance
- Conditional test execution based on data availability

## Prerequisites

1. InterSystems IRIS instance running and accessible (typically via `docker-compose.iris-only.yml up -d`).
2. Python 3.11+ environment set up with `uv` as per [`README.md`](README.md:1). Ensure your virtual environment is active.
3. Environment variables configured for database connection (see [`common/iris_connector.py`](common/iris_connector.py:1) or [`README.md`](README.md:1)):
   - `IRIS_HOST`, `IRIS_PORT`, `IRIS_NAMESPACE`, `IRIS_USERNAME`, `IRIS_PASSWORD`
4. PMC XML files stored in `data/pmc_oas_downloaded` directory (or custom location).

## Processing PMC Data (Text Content)

Use the `scripts_to_review/load_pmc_data.py` script to process PMC XML files and load their text content into IRIS.
**Note:** Loading of vector embeddings with this script is currently **BLOCKED**.

```bash
# Ensure your .venv is active, e.g., source .venv/bin/activate

# Process 1000 files (default) from the standard directory
python scripts_to_review/load_pmc_data.py

# Process a specific number of files
python scripts_to_review/load_pmc_data.py --limit 500

# Process files from a custom directory
python scripts_to_review/load_pmc_data.py --dir path/to/pmc/files

# Initialize the database schema before loading
python scripts_to_review/load_pmc_data.py --init-db

# Use a mock connection for testing the script's logic
python scripts_to_review/load_pmc_data.py --mock
```

Full options:

```
Options:
  --dir TEXT                   Directory containing PMC XML files
  --limit INTEGER              Maximum number of documents to process
  --batch INTEGER              Batch size for database inserts
  --mock                       Use mock database connection
  --init-db                    Initialize database schema before loading
  --force-recreate             Force recreate database tables (use with --init-db)
  --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]
                               Set logging level
```

## Testing with Real Data

We've implemented a flexible testing framework that can conditionally use real or mock data.

### Test Markers

- `@pytest.mark.force_real`: Test will only run with real data (skipped if unavailable)
- `@pytest.mark.force_mock`: Test will always use mock data, even if real data is available
- `@pytest.mark.real_data`: Test can adapt to use either real or mock data

### Test Fixtures

- `real_iris_available`: Boolean indicating if real IRIS connection is available
- `real_data_available`: Boolean indicating if real data is loaded in IRIS
- `use_real_data`: Boolean indicating whether to use real data for this test
- `iris_connection`: IRIS connection (real or mock) based on availability

### Sample Test

```python
@pytest.mark.real_data
def test_with_adaptive_behavior(iris_connection, use_real_data):
    """Test adapts its behavior based on whether real data is available."""
    cursor = iris_connection.cursor()
    
    if use_real_data:
        # Test with real data
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        result = cursor.fetchone()
        assert result is not None
    else:
        # Test with mock data
        mock_docs = [
            ("doc1", "Test Title 1", "Test Content 1", "[]", "[]"),
            ("doc2", "Test Title 2", "Test Content 2", "[]", "[]")
        ]
        cursor.executemany(
            "INSERT INTO SourceDocuments (doc_id, title, content, authors, keywords) VALUES (?, ?, ?, ?, ?)",
            mock_docs
        )
        # Verify insertion
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        result = cursor.fetchone()
        assert result[0] >= 2
    
    cursor.close()
```

See `tests/test_e2e_rag_pipelines.py` or other relevant files in `tests/` for more examples of test structure. (Note: `tests/test_real_data_sample.py` reference needs verification).

## Running Tests (Refer to `docs/TESTING.md`)

For general instructions on running tests, including unit tests and E2E tests (and their current limitations), please refer to the main [`docs/TESTING.md`](docs/TESTING.md:1) guide.

Example `pytest` commands (ensure virtual environment is active):
```bash
# Run all tests (respecting markers and current blockers)
pytest tests/

# Run tests marked for real data (will be skipped or fail if blocker prevents setup)
pytest -m force_real tests/

# Run tests specifically with mock data
pytest -m force_mock tests/

# Run a specific test file (example)
pytest tests/test_data_loader.py
# (Note: test_iris-connector, test-pmc-processor, test-real-data are not standard pytest commands; use direct file paths or -k for specific tests)
```


## Design Principles for Real Data Testing Framework

This implementation follows our context reduction strategy by:

1. **Modular Design**: Separating concerns into distinct modules
2. **Conditional Testing**: Only loading resources when needed
3. **Flexible Fixtures**: Adapting tests to available environment
4. **Clear Interfaces**: Well-defined boundaries between components

When adding new tests:

1. Use the `iris_connection` and `use_real_data` fixtures
2. Add the appropriate marker (`force_real`, `force_mock`, or `real_data`)
3. Implement adaptive behavior based on the `use_real_data` flag
4. Minimize resource usage by only loading what's needed

## Troubleshooting

### Cannot Connect to IRIS

- Verify environment variables are set correctly
- Ensure IRIS instance is running and accessible
- Check network connectivity
- Try running `python scripts_to_review/check_iris_module.py` (if this script is confirmed current) to verify connection.

### XML Processing Errors

- Ensure XML files are well-formed
- Check for adequate permissions to read files
- Look for detailed error messages in the logs

### Test Failures

- Check if real data (text) is available. Note that real embedding data loading is blocked.
- Verify database schema using `python scripts_to_review/load_pmc_data.py --init-db`
- Ensure tests are using the correct fixtures and respecting the current data loading limitations.
