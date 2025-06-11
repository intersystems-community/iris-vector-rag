# Comprehensive DBAPI RAG System Test

This document describes the comprehensive DBAPI RAG system test that validates all RAG techniques using DBAPI connections with a fresh IRIS Docker container.

## Overview

The comprehensive DBAPI test (`tests/test_comprehensive_dbapi_rag_system.py`) is designed to:

1. **Set up a fresh IRIS Docker container** using docker-compose
2. **Configure the connection manager** to use DBAPI globally
3. **Initialize the database schema** with all required tables and indexes
4. **Load test data** (configurable, default 1000 PMC documents)
5. **Run all 7 RAG techniques** using DBAPI connections:
   - BasicRAG
   - ColBERT
   - CRAG (Corrective RAG)
   - GraphRAG
   - HyDE (Hypothetical Document Embeddings)
   - HybridIFindRAG
   - NodeRAG
6. **Collect performance metrics** and results
7. **Generate comprehensive reports** comparing DBAPI performance

## Quick Start

### Using Make (Recommended)

```bash
# Run comprehensive test with default 1000 documents
make test-dbapi-comprehensive

# Quick test with 500 documents (10-15 minutes)
make test-dbapi-comprehensive-quick

# Large-scale test with 5000 documents (60-90 minutes)
make test-dbapi-comprehensive-large

# Test just DBAPI connection functionality
make test-dbapi
```

### Using the Shell Script Directly

```bash
# Default test with 1000 documents
./scripts/run_comprehensive_dbapi_test.sh

# Custom document count
./scripts/run_comprehensive_dbapi_test.sh --documents 2000

# Verbose logging
./scripts/run_comprehensive_dbapi_test.sh --verbose

# Cleanup only (remove existing containers)
./scripts/run_comprehensive_dbapi_test.sh --cleanup-only

# Show help
./scripts/run_comprehensive_dbapi_test.sh --help
```

### Using Python Directly

```bash
# Set environment variables
export TEST_DOCUMENT_COUNT=1000
export RAG_CONNECTION_TYPE=dbapi

# Run the test
python3 tests/test_comprehensive_dbapi_rag_system.py
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TEST_DOCUMENT_COUNT` | 1000 | Number of PMC documents to load |
| `IRIS_HOST` | localhost | IRIS database host |
| `IRIS_PORT` | 1972 | IRIS database port |
| `IRIS_NAMESPACE` | USER | IRIS namespace |
| `IRIS_USER` | _SYSTEM | IRIS username |
| `IRIS_PASSWORD` | SYS | IRIS password |
| `RAG_CONNECTION_TYPE` | dbapi | Connection type (set automatically) |

### Document Count Recommendations

| Document Count | Duration | Use Case |
|----------------|----------|----------|
| 500 | 10-15 min | Quick validation |
| 1000 | 15-30 min | Standard testing |
| 2000 | 30-45 min | Performance testing |
| 5000+ | 60-90 min | Large-scale validation |

## Prerequisites

### System Requirements

- **Docker** and **docker-compose** installed and running
- **Python 3.8+** with required packages
- **intersystems-irispython** package installed
- At least **4GB free disk space**
- At least **8GB RAM** (recommended for large document counts)

### Python Dependencies

```bash
# Install IRIS DBAPI driver
pip install intersystems-irispython

# Install other dependencies (if using Poetry)
poetry install

# Or using pip
pip install -r requirements.txt
```

### Docker Setup

Ensure Docker is running and you have access to the IRIS container image:

```bash
# Check Docker status
docker info

# Pull IRIS image (if needed)
docker-compose pull iris_db
```

## Test Process

### 1. Environment Setup

The test automatically:
- Stops and removes any existing IRIS containers
- Starts a fresh IRIS container using `docker-compose.yml`
- Waits for the container to become healthy
- Configures DBAPI connection parameters

### 2. Database Initialization

- Executes the complete database schema from `common/db_init_complete.sql`
- Creates all required tables for RAG techniques:
  - `RAG.SourceDocuments` - Main document storage
  - `RAG.DocumentChunks` - Document chunks
  - `RAG.Entities` - Knowledge graph entities
  - `RAG.Relationships` - Knowledge graph relationships
  - `RAG.DocumentTokenEmbeddings` - ColBERT token embeddings
  - And more...
- Creates HNSW vector indexes for optimal performance

### 3. Data Loading

- Processes PMC XML files from `data/pmc_oas_downloaded/`
- Generates embeddings using the configured embedding model
- Loads documents in batches for memory efficiency
- Shows progress and performance metrics

### 4. RAG Technique Testing

For each RAG technique:
- Initializes the pipeline with DBAPI connection
- Runs 5 standard test queries
- Measures performance metrics:
  - Initialization time
  - Query response time
  - Number of documents retrieved
  - Success rate
- Collects detailed results

### 5. Report Generation

Generates multiple output files:
- **JSON Report**: Complete test results and metrics
- **Markdown Summary**: Human-readable test summary
- **Log Files**: Detailed execution logs

## Output Files

All output files are saved in the `logs/` directory:

### JSON Report
```
logs/comprehensive_dbapi_test_report_<timestamp>.json
```

Contains:
- Test metadata (duration, configuration)
- Environment information
- Performance metrics for each technique
- Detailed query results
- Summary statistics

### Markdown Summary
```
logs/comprehensive_dbapi_test_summary_<timestamp>.md
```

Contains:
- Test overview
- Success rates by technique
- Performance comparisons
- Data loading metrics

### Log Files
```
logs/comprehensive_dbapi_test_<timestamp>.log
logs/test_run_<timestamp>.log
```

Contains:
- Detailed execution logs
- Error messages and stack traces
- Debug information (if verbose mode enabled)

## Interpreting Results

### Success Metrics

- **Overall Success Rate**: Percentage of techniques that completed successfully
- **Query Success Rate**: Percentage of queries that returned valid results
- **Average Query Time**: Mean response time across all successful queries
- **Documents Retrieved**: Average number of documents retrieved per query

### Performance Metrics

- **Initialization Time**: Time to set up each RAG technique
- **Data Loading Rate**: Documents processed per second during loading
- **Query Response Time**: Time to process individual queries
- **Memory Usage**: Peak memory consumption (if available)

### Comparison Analysis

The test compares DBAPI performance against:
- Expected baseline performance
- Performance across different RAG techniques
- Scalability with document count

## Troubleshooting

### Common Issues

#### Docker Issues
```bash
# Docker not running
sudo systemctl start docker

# Permission issues
sudo usermod -aG docker $USER
# Then log out and back in

# Port conflicts
docker-compose down -v
```

#### IRIS Connection Issues
```bash
# Check container status
docker-compose ps

# Check container logs
docker-compose logs iris_db

# Verify IRIS is responding
curl http://localhost:52773/csp/sys/UtilHome.csp
```

#### Python Dependencies
```bash
# Install IRIS DBAPI driver
pip install intersystems-irispython

# Verify installation
python -c "import iris; print('IRIS DBAPI available')"
```

#### Memory Issues
```bash
# Check available memory
free -h

# Reduce document count
./scripts/run_comprehensive_dbapi_test.sh --documents 500
```

### Debug Mode

Enable verbose logging for detailed troubleshooting:

```bash
# Using shell script
./scripts/run_comprehensive_dbapi_test.sh --verbose

# Using environment variable
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 tests/test_comprehensive_dbapi_rag_system.py
```

### Manual Cleanup

If the test fails to cleanup automatically:

```bash
# Stop containers
docker-compose down -v

# Remove containers and volumes
docker container prune -f
docker volume prune -f

# Or use the cleanup script
./scripts/run_comprehensive_dbapi_test.sh --cleanup-only
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Comprehensive DBAPI Test

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  dbapi-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install intersystems-irispython
        pip install -r requirements.txt
    
    - name: Run comprehensive DBAPI test
      run: |
        make test-dbapi-comprehensive-quick
    
    - name: Upload test reports
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: dbapi-test-reports
        path: logs/
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    stages {
        stage('Setup') {
            steps {
                sh 'pip install intersystems-irispython'
                sh 'pip install -r requirements.txt'
            }
        }
        
        stage('DBAPI Test') {
            steps {
                sh 'make test-dbapi-comprehensive'
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'logs/**/*', fingerprint: true
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: 'logs',
                reportFiles: '*.md',
                reportName: 'DBAPI Test Report'
            ])
        }
    }
}
```

## Performance Benchmarks

### Expected Performance (1000 documents)

| Technique | Init Time | Avg Query Time | Docs/Query | Success Rate |
|-----------|-----------|----------------|------------|--------------|
| BasicRAG | < 1s | < 2s | 5-10 | > 95% |
| ColBERT | < 5s | < 3s | 5-15 | > 90% |
| CRAG | < 2s | < 4s | 8-20 | > 85% |
| GraphRAG | < 10s | < 5s | 10-25 | > 80% |
| HyDE | < 2s | < 3s | 5-15 | > 90% |
| HybridIFindRAG | < 3s | < 4s | 10-30 | > 85% |
| NodeRAG | < 5s | < 4s | 8-20 | > 85% |

### Scaling Characteristics

- **Linear scaling** for data loading (documents/second remains constant)
- **Logarithmic scaling** for query performance (slight increase with more documents)
- **Memory usage** scales linearly with document count
- **Index build time** scales super-linearly for HNSW indexes

## Contributing

### Adding New RAG Techniques

To add a new RAG technique to the comprehensive test:

1. **Create the pipeline class** in `core_pipelines/`
2. **Add import and configuration** in `test_comprehensive_dbapi_rag_system.py`:

```python
techniques = [
    # ... existing techniques ...
    ("NewRAG", "core_pipelines.newrag_pipeline", "NewRAGPipeline"),
]
```

3. **Handle special initialization** if needed in `run_rag_technique()`
4. **Update documentation** and expected performance benchmarks

### Improving Test Coverage

- Add more diverse test queries
- Include edge cases and error conditions
- Test with different embedding models
- Add memory and CPU profiling
- Test concurrent access patterns

## Related Documentation

- [DBAPI Connection Guide](DBAPI_CONNECTION.md)
- [RAG Techniques Overview](RAG_TECHNIQUES.md)
- [Performance Tuning Guide](PERFORMANCE_TUNING.md)
- [Docker Setup Guide](DOCKER_SETUP.md)