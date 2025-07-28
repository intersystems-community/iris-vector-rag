# Integration Test Guide

## Overview

This guide documents the comprehensive testing framework for validating the chunking system and ColBERT token embedding auto-population fixes. The testing framework ensures that all improvements work correctly at scale with real PMC data, providing confidence in the system's reliability and performance.

## Testing Philosophy

### Real-World Validation

The testing framework follows the principle of **real-world validation**:

- ✅ **Real Data**: Tests use actual PMC documents, not synthetic data
- ✅ **Scale Testing**: Validates with 1000+ documents by default
- ✅ **End-to-End Coverage**: Tests complete pipelines from ingestion to answer generation
- ✅ **Production Scenarios**: Simulates actual usage patterns and edge cases

### Test Categories

```
┌─────────────────────────────────────────────────────────────┐
│                    Test Framework Architecture              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Unit Tests              Integration Tests    E2E Tests     │
│  ┌─────────────────┐     ┌─────────────────┐  ┌──────────┐  │
│  │ • Service logic │     │ • Component     │  │ • Full   │  │
│  │ • Utilities     │     │   integration   │  │   pipeline│  │
│  │ • Interfaces    │     │ • Database ops  │  │ • 1000+  │  │
│  │ • Mocked deps   │     │ • Real data     │  │   docs   │  │
│  └─────────────────┘     └─────────────────┘  └──────────┘  │
│           │                       │                │        │
│           v                       v                v        │
│  Fast feedback           Component validation   Production  │
│  (< 1 second)           (< 30 seconds)         (< 5 min)   │
└─────────────────────────────────────────────────────────────┘
```

## Test Structure

### Core Test Files

#### 1. ColBERT Auto-Population Tests
**File**: [`tests/test_colbert_auto_population_fix.py`](../tests/test_colbert_auto_population_fix.py)

```python
class TestColBERTAutoPopulationFix:
    """Test suite for ColBERT auto-population fixes."""
    
    def test_colbert_interface_uses_768d_embeddings(self):
        """Verify ColBERT interface uses proper 768D token embeddings."""
        
    def test_token_embedding_service_initialization(self):
        """Test TokenEmbeddingService initializes correctly."""
        
    def test_colbert_pipeline_auto_population_integration(self):
        """Test ColBERT pipeline integrates auto-population correctly."""
        
    def test_no_manual_population_required(self):
        """Test ColBERT works without manual token embedding population."""
```

#### 2. Chunking Integration Tests
**File**: [`tests/test_pipelines/test_basic.py`](../tests/test_pipelines/test_basic.py)

```python
def test_chunking_service_integration():
    """Test that BasicRAGPipeline uses DocumentChunkingService correctly."""
    
def test_text_chunking():
    """Test text chunking functionality without heavy mocks."""
    
def test_basic_pipeline_connection_uses_config_manager():
    """Test BasicRAGPipeline uses the new connection manager."""
```

#### 3. 1000-Document Validation Tests
**File**: [`tests/test_all_with_1000_docs.py`](../tests/test_all_with_1000_docs.py)

```python
def test_colbert_pipeline_with_1000_docs():
    """Test ColBERT pipeline with 1000+ real PMC documents."""
    
def test_basic_pipeline_chunking_with_1000_docs():
    """Test BasicRAG chunking integration with 1000+ documents."""
```

### Test Execution Commands

#### Quick Validation Tests

```bash
# Test ColBERT auto-population fixes
uv run pytest tests/test_colbert_auto_population_fix.py -v | tee test_output/test_colbert_auto_population.log

# Test chunking system integration
uv run pytest tests/test_pipelines/test_basic.py -v | tee test_output/test_basic_chunking.log

# Test specific chunking functionality
uv run pytest tests/test_pipelines/test_basic.py::test_text_chunking -v
```

#### Scale Testing (1000+ Documents)

```bash
# Run all tests with 1000+ documents
make test-1000

# Test specific pipeline with 1000+ documents
uv run pytest tests/test_all_with_1000_docs.py::test_colbert_pipeline -v | tee test_output/test_colbert_1000_docs.log

# Test chunking with large dataset
uv run pytest tests/test_all_with_1000_docs.py::test_basic_pipeline_chunking -v | tee test_output/test_chunking_1000_docs.log
```

#### Comprehensive Integration Tests

```bash
# Run all integration tests
uv run pytest tests/test_integration_fallback_behaviors.py -v | tee test_output/test_integration_fallback.log

# Test end-to-end IRIS RAG imports
uv run pytest tests/test_e2e_iris_rag_imports.py -v | tee test_output/test_e2e_imports.log

# Test working ColBERT implementation
uv run pytest tests/working/colbert/ -v | tee test_output/test_colbert_working.log
```

## Test Fixtures and Setup

### 1000-Document Fixture

The testing framework includes a specialized fixture for 1000+ document testing:

```python
# tests/conftest_1000docs.py
@pytest.fixture(scope="session")
def documents_1000():
    """Fixture that ensures 1000+ documents are available for testing."""
    
    # Check if we have enough documents
    connection = get_iris_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    doc_count = cursor.fetchone()[0]
    
    if doc_count < 1000:
        pytest.skip(f"Need at least 1000 documents for scale testing, found {doc_count}")
    
    return doc_count
```

### Configuration Fixtures

```python
@pytest.fixture
def config_manager():
    """Create configuration manager for testing."""
    return ConfigurationManager()

@pytest.fixture
def connection_manager():
    """Create connection manager for testing."""
    return type('ConnectionManager', (), {
        'get_connection': lambda: get_iris_connection()
    })()
```

### Mock Fixtures

```python
@pytest.fixture
def mock_iris_connector():
    """Create mock IRIS connector for unit tests."""
    mock_connector = Mock()
    mock_connection = Mock()
    mock_cursor = Mock()
    
    # Setup mock chain
    mock_connector.get_connection.return_value = mock_connection
    mock_connection.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = [0]  # No existing token embeddings
    
    return mock_connector
```

## Test Scenarios

### ColBERT Auto-Population Scenarios

#### Scenario 1: Fresh Installation
```python
def test_colbert_fresh_installation():
    """Test ColBERT pipeline on fresh installation with no token embeddings."""
    
    # Setup: Clean database with no token embeddings
    # Action: Initialize ColBERT pipeline
    # Verify: Auto-population triggers and succeeds
    # Assert: Pipeline works without manual intervention
```

#### Scenario 2: Dimension Validation
```python
def test_dimension_consistency_across_components():
    """Test that all components use consistent 768D token embeddings."""
    
    # Test ColBERT interface
    colbert_interface = get_colbert_interface_from_config(config_manager, connection_manager)
    assert colbert_interface.get_token_dimension() == 768
    
    # Test TokenEmbeddingService
    token_service = TokenEmbeddingService(config_manager, connection_manager)
    assert token_service.token_dimension == 768
    
    # Test schema manager token dimension
    schema_manager = SchemaManager(connection_manager, config_manager)
    assert schema_manager.get_colbert_token_dimension() == 768
```

#### Scenario 3: Partial Population
```python
def test_partial_token_embedding_population():
    """Test auto-population when some documents already have token embeddings."""
    
    # Setup: Database with partial token embeddings
    # Action: Load new documents
    # Verify: Only missing token embeddings are generated
    # Assert: Existing embeddings are preserved
```

### Chunking Integration Scenarios

#### Scenario 1: Strategy Switching
```python
def test_chunking_strategy_switching():
    """Test switching between different chunking strategies."""
    
    pipeline = BasicRAGPipeline(config_manager=config_manager)
    
    # Test fixed_size strategy
    pipeline.chunking_strategy = "fixed_size"
    chunks_fixed = pipeline._chunk_documents(test_documents)
    
    # Test semantic strategy
    pipeline.chunking_strategy = "semantic"
    chunks_semantic = pipeline._chunk_documents(test_documents)
    
    # Verify different chunking results
    assert len(chunks_fixed) != len(chunks_semantic)
    assert all('chunking_strategy' in chunk.metadata for chunk in chunks_fixed)
```

#### Scenario 2: Metadata Preservation
```python
def test_chunk_metadata_preservation():
    """Test that chunk metadata is properly preserved and enhanced."""
    
    original_doc = Document(
        page_content="Long document content...",
        metadata={"source": "test.pdf", "author": "Test Author"}
    )
    
    chunks = pipeline._chunk_documents([original_doc])
    
    for chunk in chunks:
        # Verify original metadata preserved
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["author"] == "Test Author"
        
        # Verify chunk-specific metadata added
        assert "chunk_index" in chunk.metadata
        assert "parent_document_id" in chunk.metadata
        assert "chunking_strategy" in chunk.metadata
```

#### Scenario 3: Large Document Processing
```python
def test_large_document_chunking_performance():
    """Test chunking performance with large documents."""
    
    # Create large document (10MB text)
    large_content = "Sample text. " * 100000
    large_doc = Document(page_content=large_content, metadata={})
    
    start_time = time.time()
    chunks = pipeline._chunk_documents([large_doc])
    processing_time = time.time() - start_time
    
    # Verify reasonable performance
    assert processing_time < 30  # Should complete within 30 seconds
    assert len(chunks) > 0
    assert all(len(chunk.page_content) <= pipeline.chunk_size for chunk in chunks)
```

### End-to-End Integration Scenarios

#### Scenario 1: Complete Pipeline Flow
```python
def test_complete_pipeline_flow_with_1000_docs(documents_1000):
    """Test complete pipeline flow with 1000+ documents."""
    
    # Initialize pipeline
    pipeline = BasicRAGPipeline(config_manager=config_manager)
    
    # Load documents (should use chunking service)
    pipeline.load_documents("data/sample_10_docs", chunk_documents=True)
    
    # Query pipeline
    result = pipeline.run("What is machine learning?")
    
    # Verify complete response
    assert "query" in result
    assert "answer" in result
    assert "retrieved_documents" in result
    assert len(result["retrieved_documents"]) > 0
```

#### Scenario 2: ColBERT vs Basic RAG Comparison
```python
def test_colbert_vs_basic_rag_comparison():
    """Compare ColBERT and Basic RAG pipelines on same dataset."""
    
    # Initialize both pipelines
    basic_pipeline = BasicRAGPipeline(config_manager=config_manager)
    colbert_pipeline = ColBERTRAGPipeline(iris_connector, config_manager)
    
    # Load same documents
    test_docs = load_test_documents()
    basic_pipeline.load_documents(documents=test_docs)
    colbert_pipeline.load_documents(documents=test_docs)
    
    # Query both pipelines
    query = "What are the benefits of machine learning?"
    basic_result = basic_pipeline.run(query)
    colbert_result = colbert_pipeline.run(query)
    
    # Verify both work and return valid results
    assert basic_result["answer"] != "No relevant documents found"
    assert colbert_result["answer"] != "No relevant documents found"
    assert len(basic_result["retrieved_documents"]) > 0
    assert len(colbert_result["retrieved_documents"]) > 0
```

## Test Result Interpretation

### Success Criteria

#### ColBERT Auto-Population Tests
- ✅ **Dimension Consistency**: All components use 768D token embeddings
- ✅ **Auto-Population**: Token embeddings generated automatically
- ✅ **Pipeline Integration**: ColBERT works without manual setup
- ✅ **Error Handling**: Graceful fallback when components fail
- ✅ **Performance**: Token generation completes within reasonable time

#### Chunking Integration Tests
- ✅ **Service Integration**: Pipelines use DocumentChunkingService
- ✅ **Strategy Flexibility**: Multiple chunking strategies work
- ✅ **Metadata Preservation**: Original and chunk metadata maintained
- ✅ **Configuration Driven**: Chunking controlled via configuration
- ✅ **Performance**: Large documents processed efficiently

#### Scale Tests (1000+ Documents)
- ✅ **Data Volume**: Tests run with 1000+ real PMC documents
- ✅ **Processing Time**: Complete pipeline execution < 5 minutes
- ✅ **Memory Usage**: No memory leaks or excessive usage
- ✅ **Result Quality**: Meaningful answers generated
- ✅ **System Stability**: No crashes or errors during processing

### Failure Analysis

#### Common Failure Patterns

1. **Dimension Mismatch Errors**
   ```
   AssertionError: Expected 768D embeddings, got 384D
   Root Cause: ColBERT interface configuration issue
   Solution: Verify schema manager configuration
   ```

2. **Auto-Population Failures**
   ```
   Error: Failed to auto-populate token embeddings
   Root Cause: Database connection or permission issues
   Solution: Check IRIS connection and table permissions
   ```

3. **Chunking Service Errors**
   ```
   AttributeError: 'BasicRAGPipeline' object has no attribute 'chunking_service'
   Root Cause: Service initialization failure
   Solution: Verify DocumentChunkingService import and initialization
   ```

4. **Scale Test Timeouts**
   ```
   TimeoutError: Test exceeded maximum execution time
   Root Cause: Inefficient processing or resource constraints
   Solution: Optimize batch sizes or increase timeout limits
   ```

### Performance Benchmarks

#### Expected Performance Metrics

```python
# ColBERT Auto-Population Performance
EXPECTED_METRICS = {
    "token_generation_rate": "100-500 tokens/second",
    "document_processing_rate": "5-20 documents/second",
    "memory_usage": "< 2GB for 1000 documents",
    "setup_time": "< 60 seconds for fresh installation"
}

# Chunking Performance
CHUNKING_METRICS = {
    "chunking_rate": "1000-5000 characters/second",
    "chunk_generation": "10-50 chunks/second",
    "metadata_overhead": "< 10% of processing time",
    "strategy_switching": "< 1 second"
}
```

## Test Environment Setup

### Prerequisites

```bash
# Ensure UV is installed and configured
uv --version

# Verify IRIS database is running
docker ps | grep iris

# Check test data availability
ls data/sample_10_docs/

# Verify test output directory exists
mkdir -p test_output
```

### Environment Variables

```bash
# Set IRIS edition for testing
export IRIS_DOCKER_IMAGE="intersystemsdc/iris-ml:latest"

# Configure test database
export IRIS_HOST="localhost"
export IRIS_PORT="1972"
export IRIS_NAMESPACE="USER"

# Set test configuration
export RAG_CONFIG_PATH="config/test_config.yaml"
```

### Database Preparation

```sql
-- Ensure test schema exists
CREATE SCHEMA IF NOT EXISTS RAG;

-- Verify required tables
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = 'RAG';

-- Check document count for scale testing
SELECT COUNT(*) FROM RAG.SourceDocuments;
```

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup IRIS Database
      run: |
        docker run -d --name iris-test \
          -p 1972:1972 -p 52773:52773 \
          intersystemsdc/iris-ml:latest
    
    - name: Install Dependencies
      run: |
        pip install uv
        uv sync
    
    - name: Run Integration Tests
      run: |
        uv run pytest tests/test_colbert_auto_population_fix.py -v
        uv run pytest tests/test_pipelines/test_basic.py -v
    
    - name: Run Scale Tests (if data available)
      run: |
        if [ -f "data/test_1000_docs.flag" ]; then
          uv run pytest tests/test_all_with_1000_docs.py -v
        fi
```

### Local CI Simulation

```bash
# Simulate CI environment locally
./scripts/run_integration_tests.sh

# Run specific test suites
./scripts/test_colbert_fixes.sh
./scripts/test_chunking_integration.sh
./scripts/test_scale_validation.sh
```

## Test Data Management

### Test Document Sets

#### Small Test Set (10 documents)
```bash
# Location: data/sample_10_docs/
# Purpose: Quick validation and unit tests
# Size: ~1MB total
# Processing time: < 30 seconds
```

#### Medium Test Set (100 documents)
```bash
# Location: data/sample_100_docs/
# Purpose: Integration testing
# Size: ~10MB total
# Processing time: < 2 minutes
```

#### Large Test Set (1000+ documents)
```bash
# Location: data/pmc_1000_docs/
# Purpose: Scale testing and performance validation
# Size: ~100MB total
# Processing time: < 5 minutes
```

### Test Data Generation

```python
# Generate test documents for specific scenarios
def generate_test_documents(count: int, content_type: str) -> List[Document]:
    """Generate test documents for testing scenarios."""
    
    if content_type == "short":
        # Generate short documents for chunking tests
        return [create_short_document(i) for i in range(count)]
    elif content_type == "long":
        # Generate long documents for performance tests
        return [create_long_document(i) for i in range(count)]
    elif content_type == "mixed":
        # Generate mixed-length documents for realistic tests
        return [create_mixed_document(i) for i in range(count)]
```

## Debugging and Troubleshooting

### Debug Mode Execution

```bash
# Run tests with debug logging
uv run pytest tests/test_colbert_auto_population_fix.py -v -s --log-cli-level=DEBUG

# Run with pdb debugging
uv run pytest tests/test_colbert_auto_population_fix.py --pdb

# Run with coverage reporting
uv run pytest tests/ --cov=iris_rag --cov-report=html
```

### Common Debug Scenarios

#### 1. Token Embedding Issues
```python
# Debug token embedding generation
import logging
logging.getLogger('iris_rag.services.token_embedding_service').setLevel(logging.DEBUG)

# Check token dimensions
service = TokenEmbeddingService(config_manager, connection_manager)
print(f"Token dimension: {service.token_dimension}")

# Verify ColBERT interface
interface = service.colbert_interface
test_embeddings = interface.encode_query("test")
print(f"Generated embeddings shape: {len(test_embeddings)} x {len(test_embeddings[0])}")
```

#### 2. Chunking Service Issues
```python
# Debug chunking service
from tools.chunking.chunking_service import DocumentChunkingService

embedding_func = lambda texts: [[0.1] * 384 for _ in texts]  # Mock embedding
service = DocumentChunkingService(embedding_func=embedding_func)

# Test chunking
chunks = service.chunk_document("test_doc", "Test content", "fixed_size")
print(f"Generated {len(chunks)} chunks")
```

#### 3. Database Connection Issues
```python
# Debug database connection
from common.iris_connection_manager import get_iris_connection

try:
    connection = get_iris_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT 1")
    print("Database connection successful")
except Exception as e:
    print(f"Database connection failed: {e}")
```

## Test Reporting

### Automated Reports

```bash
# Generate comprehensive test report
uv run pytest tests/ --html=test_output/report.html --self-contained-html

# Generate coverage report
uv run pytest tests/ --cov=iris_rag --cov-report=html --cov-report=term

# Generate performance report
uv run pytest tests/ --benchmark-only --benchmark-json=test_output/benchmark.json
```

### Report Analysis

```python
# Analyze test results
import json

def analyze_test_results(report_file: str):
    """Analyze test results and generate summary."""
    
    with open(report_file, 'r') as f:
        results = json.load(f)
    
    summary = {
        "total_tests": results["summary"]["total"],
        "passed": results["summary"]["passed"],
        "failed": results["summary"]["failed"],
        "duration": results["summary"]["duration"],
        "coverage": results.get("coverage", {}).get("totals", {}).get("percent_covered", 0)
    }
    
    return summary
```

## Related Documentation

- [Chunking System Fixes](./CHUNKING_SYSTEM_FIXES.md)
- [ColBERT Auto-Population Guide](./COLBERT_AUTO_POPULATION_GUIDE.md)
- [Troubleshooting Guide](./TROUBLESHOOTING_GUIDE.md)
- [Existing Tests Guide](./EXISTING_TESTS_GUIDE.md)

## Conclusion

The integration test framework provides comprehensive validation of the chunking system and ColBERT auto-population fixes. Key achievements include:

### Test Coverage
- ✅ **Unit Tests**: Individual component validation
- ✅ **Integration Tests**: Component interaction validation
- ✅ **Scale Tests**: 1000+ document validation
- ✅ **Performance Tests**: Timing and resource usage validation
- ✅ **Error Handling Tests**: Failure scenario validation

### Quality Assurance
- ✅ **Real Data Testing**: Uses actual PMC documents
- ✅ **Production Scenarios**: Simulates real-world usage
- ✅ **Automated Validation**: CI/CD integration
- ✅ **Comprehensive Reporting**: Detailed test results and coverage
- ✅ **Debug Support**: Tools for troubleshooting issues

The testing framework ensures that the implemented fixes work reliably at scale, providing confidence for production deployment and ongoing development.