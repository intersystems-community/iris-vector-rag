# Real Data Integration Guide

This document explains how to use the real IRIS database connection and process real PMC data files.

## Overview

We've implemented a complete pipeline for processing PMC XML files and loading them into IRIS, with support for:

- Real IRIS database connections
- Mock database connections for testing
- XML processing with metadata extraction
- Batched data loading for performance
- Conditional test execution based on data availability
- Poetry commands for running all components

## Prerequisites

1. InterSystems IRIS instance running and accessible
2. Environment variables configured for database connection:
   - `IRIS_HOST`: Hostname of IRIS instance
   - `IRIS_PORT`: Port number (default: 1972)
   - `IRIS_NAMESPACE`: Namespace (default: USER)
   - `IRIS_USERNAME`: Username
   - `IRIS_PASSWORD`: Password
3. PMC XML files stored in `data/pmc_oas_downloaded` directory (or custom location)

## Processing PMC Data

Use the `load_pmc_data.py` script to process PMC XML files and load them into IRIS.

```bash
# Process 1000 files (default) from the standard directory
poetry run load-pmc-data

# Process a specific number of files
poetry run load-pmc-data --limit 500

# Process files from a custom directory
poetry run load-pmc-data --dir path/to/pmc/files

# Initialize the database schema before loading
poetry run load-pmc-data --init-db

# Use a mock connection for testing
poetry run load-pmc-data --mock
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

See `tests/test_real_data_sample.py` for more examples.

## Running Tests

```bash
# Run all tests (using real data if available)
poetry run test

# Run tests that require real data (skipped if unavailable)
poetry run pytest -m force_real

# Run tests with mock data only
poetry run pytest -m force_mock

# Run a specific test file
poetry run test-iris-connector
poetry run test-pmc-processor
poetry run test-data-loader

# Run integration tests (can use real data if available)
poetry run test-real-data
```

## Context Reduction Strategy

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
- Try running `poetry run check-iris` to verify connection

### XML Processing Errors

- Ensure XML files are well-formed
- Check for adequate permissions to read files
- Look for detailed error messages in the logs

### Test Failures

- Check if real data is available using `poetry run test-real-data`
- Verify database schema using `poetry run load-pmc-data --init-db`
- Ensure tests are using the correct fixtures
