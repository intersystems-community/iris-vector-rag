# Testing with 1000+ Documents

This document describes how to run large-scale tests with 1000+ documents for the RAG templates project.

## Prerequisites

- Make sure you have Python 3.8+ installed
- Install Poetry for dependency management
- Docker installed (for testcontainer-based tests)
- At least 8GB of RAM available
- At least 1GB of free disk space

## Environment Setup

All tests should be run using Poetry to ensure consistent dependency management:

```bash
# Install dependencies
poetry install

# Install testcontainer requirements 
pip install -r testcontainer-requirements.txt
```

## Running Large-Scale Tests

We've provided several scripts for running tests with 1000 documents:

### Shell Script (Recommended)

```bash
# Run all tests with 1000 documents
./run_tests_with_1000_docs.sh all

# Run only GraphRAG tests with 1000 documents
./run_tests_with_1000_docs.sh graphrag

# Use mock embeddings for faster testing (less accurate)
./run_tests_with_1000_docs.sh --mock-embeddings all
```

### Python Script

```bash
# Run all tests with 1000 documents
./run_1000_docs_tests.py

# Run only GraphRAG tests
./run_1000_docs_tests.py graphrag

# Use mock embeddings
./run_1000_docs_tests.py --mock-embeddings
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

### Database Schema Compatibility

The tests require database tables with specific column names and types:

- `SourceDocuments` table requires:
  - `doc_id VARCHAR(255) PRIMARY KEY`
  - `title VARCHAR(1000)`
  - `content LONGVARCHAR`
  - `embedding VARCHAR(8000)`

- `KnowledgeGraphNodes` table requires:
  - `node_id VARCHAR(255) PRIMARY KEY`
  - `node_type VARCHAR(100)`
  - `node_name VARCHAR(1000)`
  - `description_text LONGVARCHAR` 
  - `embedding VARCHAR(8000)`

## Performance Considerations

- Large-scale tests may take 30+ minutes to complete
- Consider using specific test suites during development
- Use `--mock-embeddings` flag when testing non-embedding functionality
- Set lower document counts (100-200) for quicker feedback during development
