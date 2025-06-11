# ColBERT Optimization TDD Test Specification

## Overview
This document defines the Test-Driven Development (TDD) approach for implementing the ColBERT optimization specified in [`COLBERT_OPTIMIZATION_SPECIFICATION.md`](specs/COLBERT_OPTIMIZATION_SPECIFICATION.md). Following the Red-Green-Refactor cycle, these tests will guide the implementation of the optimized [`_retrieve_documents_with_colbert`](iris_rag/pipelines/colbert.py:156) method.

## Test Structure

### Test File Organization
```
tests/
├── test_colbert_optimization.py           # Main optimization tests
├── test_colbert_optimization_performance.py  # Performance-specific tests
├── test_colbert_optimization_integration.py  # Integration tests
└── fixtures/
    ├── colbert_test_data.py               # Test data fixtures
    └── performance_fixtures.py            # Performance test fixtures
```

## Phase 1: Core Functionality Tests (Red Phase)

### Test Group 1: Vector Format Conversion

#### Test 1.1: IRIS Vector Format Conversion
```python
def test_format_as_iris_vector():
    """Test conversion of Python list to IRIS VECTOR format."""
    # Test case 1: Standard embedding
    embedding = [0.1, 0.2, 0.3, 0.4]
    expected = "0.1,0.2,0.3,0.4"
    result = format_as_iris_vector(embedding)
    assert result == expected
    
    # Test case 2: Negative values
    embedding = [-0.1, 0.2, -0.3, 0.4]
    expected = "-0.1,0.2,-0.3,0.4"
    result = format_as_iris_vector(embedding)
    assert result == expected
    
    # Test case 3: High precision values
    embedding = [0.123456789, -0.987654321]
    result = format_as_iris_vector(embedding)
    assert "0.123456789" in result
    assert "-0.987654321" in result
```

#### Test 1.2: Database Vector Parsing
```python
def test_parse_vector_from_db():
    """Test parsing of VECTOR type from database."""
    # Test case 1: Bracketed format
    vector_str = "[0.1,0.2,0.3,0.4]"
    expected = [0.1, 0.2, 0.3, 0.4]
    result = parse_vector_from_db(vector_str)
    assert result == expected
    
    # Test case 2: Comma-separated format
    vector_str = "0.1,0.2,0.3,0.4"
    expected = [0.1, 0.2, 0.3, 0.4]
    result = parse_vector_from_db(vector_str)
    assert result == expected
    
    # Test case 3: Malformed input handling
    vector_str = "invalid_format"
    result = parse_vector_from_db(vector_str)
    assert result is None or result == []
```

### Test Group 2: HNSW Query Execution

#### Test 2.1: Single Query Token HNSW Search
```python
def test_hnsw_single_token_search(iris_connector):
    """Test HNSW search for a single query token."""
    # Setup
    query_token = [0.1] * 384  # Standard ColBERT dimension
    candidate_limit = 50
    
    # Execute HNSW search
    results = execute_hnsw_token_search(
        iris_connector, 
        query_token, 
        candidate_limit
    )
    
    # Assertions
    assert len(results) <= candidate_limit
    assert all('doc_id' in result for result in results)
    assert all('token_index' in result for result in results)
    assert all('similarity_score' in result for result in results)
    assert all(isinstance(result['similarity_score'], float) for result in results)
    
    # Verify similarity scores are in descending order
    scores = [result['similarity_score'] for result in results]
    assert scores == sorted(scores, reverse=True)
```

#### Test 2.2: Multiple Query Tokens HNSW Search
```python
def test_hnsw_multiple_tokens_search(iris_connector):
    """Test HNSW search for multiple query tokens."""
    # Setup
    query_tokens = [
        [0.1] * 384,  # Token 1
        [0.2] * 384,  # Token 2
        [0.3] * 384,  # Token 3
    ]
    candidate_limit = 50
    
    # Execute HNSW searches
    all_candidates = {}
    for i, query_token in enumerate(query_tokens):
        results = execute_hnsw_token_search(
            iris_connector, 
            query_token, 
            candidate_limit
        )
        
        # Collect candidates by document
        for result in results:
            doc_id = result['doc_id']
            if doc_id not in all_candidates:
                all_candidates[doc_id] = []
            all_candidates[doc_id].append({
                'query_token_index': i,
                'doc_token_index': result['token_index'],
                'similarity_score': result['similarity_score']
            })
    
    # Assertions
    assert len(all_candidates) > 0
    assert all(isinstance(doc_id, str) for doc_id in all_candidates.keys())
    
    # Verify each document has candidate tokens
    for doc_id, candidates in all_candidates.items():
        assert len(candidates) > 0
        assert all('query_token_index' in candidate for candidate in candidates)
```

### Test Group 3: Candidate Document Management

#### Test 3.1: Document Token Aggregation
```python
def test_aggregate_candidate_documents():
    """Test aggregation of candidate tokens by document."""
    # Setup mock HNSW results
    hnsw_results = [
        {'doc_id': 'doc1', 'token_index': 0, 'embedding': [0.1, 0.2], 'similarity_score': 0.9},
        {'doc_id': 'doc1', 'token_index': 1, 'embedding': [0.2, 0.3], 'similarity_score': 0.8},
        {'doc_id': 'doc2', 'token_index': 0, 'embedding': [0.3, 0.4], 'similarity_score': 0.7},
        {'doc_id': 'doc2', 'token_index': 2, 'embedding': [0.4, 0.5], 'similarity_score': 0.6},
    ]
    
    # Execute aggregation
    candidate_docs = aggregate_candidate_documents(hnsw_results)
    
    # Assertions
    assert len(candidate_docs) == 2
    assert 'doc1' in candidate_docs
    assert 'doc2' in candidate_docs
    assert len(candidate_docs['doc1']) == 2
    assert len(candidate_docs['doc2']) == 2
    
    # Verify token data structure
    for doc_id, tokens in candidate_docs.items():
        for token in tokens:
            assert 'token_index' in token
            assert 'embedding' in token
            assert 'similarity_score' in token
```

#### Test 3.2: Document Candidate Limiting
```python
def test_limit_candidate_documents():
    """Test limiting of candidate documents for performance."""
    # Setup: Create more candidates than limit
    candidate_docs = {}
    for i in range(150):  # More than typical limit of 100
        doc_id = f"doc_{i:03d}"
        # Vary token count to test sorting
        token_count = (i % 10) + 1
        candidate_docs[doc_id] = [{'token_index': j} for j in range(token_count)]
    
    # Execute limiting
    limited_docs = limit_candidate_documents(candidate_docs, limit=100)
    
    # Assertions
    assert len(limited_docs) == 100
    
    # Verify documents with more tokens are prioritized
    token_counts = [len(tokens) for tokens in limited_docs.values()]
    assert max(token_counts) >= min(token_counts)  # Some prioritization occurred
```

## Phase 2: MaxSim Calculation Tests (Red Phase)

### Test Group 4: MaxSim Score Calculation

#### Test 4.1: Basic MaxSim Calculation
```python
def test_calculate_maxsim_basic():
    """Test basic MaxSim calculation with known values."""
    # Setup: Known embeddings for predictable results
    query_tokens = [
        [1.0, 0.0, 0.0],  # Query token 1
        [0.0, 1.0, 0.0],  # Query token 2
    ]
    
    doc_tokens = [
        [1.0, 0.0, 0.0],  # Perfect match for query token 1
        [0.0, 0.0, 1.0],  # No good match
        [0.0, 1.0, 0.0],  # Perfect match for query token 2
    ]
    
    # Execute MaxSim calculation
    maxsim_score = calculate_maxsim_score(query_tokens, doc_tokens)
    
    # Expected: 
    # Query token 1 best match: similarity = 1.0 (with doc token 1)
    # Query token 2 best match: similarity = 1.0 (with doc token 3)
    # MaxSim = (1.0 + 1.0) / 2 = 1.0
    assert abs(maxsim_score - 1.0) < 0.001
```

#### Test 4.2: MaxSim with Partial Matches
```python
def test_calculate_maxsim_partial_matches():
    """Test MaxSim calculation with partial similarity matches."""
    # Setup: Embeddings with partial similarities
    query_tokens = [
        [1.0, 0.0],  # Query token 1
        [0.0, 1.0],  # Query token 2
    ]
    
    doc_tokens = [
        [0.7, 0.7],  # Partial match for both query tokens
        [0.9, 0.1],  # Better match for query token 1
    ]
    
    # Execute MaxSim calculation
    maxsim_score = calculate_maxsim_score(query_tokens, doc_tokens)
    
    # Verify result is reasonable
    assert 0.0 < maxsim_score < 1.0
    assert isinstance(maxsim_score, float)
```

#### Test 4.3: MaxSim Edge Cases
```python
def test_calculate_maxsim_edge_cases():
    """Test MaxSim calculation edge cases."""
    # Test case 1: Empty query tokens
    maxsim_score = calculate_maxsim_score([], [[1.0, 2.0]])
    assert maxsim_score == 0.0
    
    # Test case 2: Empty document tokens
    maxsim_score = calculate_maxsim_score([[1.0, 2.0]], [])
    assert maxsim_score == 0.0
    
    # Test case 3: Both empty
    maxsim_score = calculate_maxsim_score([], [])
    assert maxsim_score == 0.0
    
    # Test case 4: Single token each
    maxsim_score = calculate_maxsim_score([[1.0, 0.0]], [[1.0, 0.0]])
    assert abs(maxsim_score - 1.0) < 0.001
```

## Phase 3: Integration Tests (Red Phase)

### Test Group 5: End-to-End Optimization

#### Test 5.1: Optimized vs Original Correctness
```python
def test_optimized_vs_original_correctness(iris_connector, colbert_pipeline):
    """Test that optimized method returns similar results to original."""
    # Setup
    test_query = "machine learning algorithms"
    query_embeddings = colbert_pipeline.colbert_query_encoder(test_query)
    top_k = 5
    
    # Execute both implementations
    original_results = colbert_pipeline._retrieve_documents_with_colbert(
        query_embeddings, top_k
    )
    optimized_results = colbert_pipeline._retrieve_documents_with_colbert_optimized(
        query_embeddings, top_k
    )
    
    # Assertions
    assert len(optimized_results) == len(original_results)
    assert len(optimized_results) <= top_k
    
    # Top result should be the same or very similar
    if len(original_results) > 0 and len(optimized_results) > 0:
        original_top = original_results[0]
        optimized_top = optimized_results[0]
        
        # Either same document or similar MaxSim score
        same_doc = original_top.id == optimized_top.id
        similar_score = abs(
            original_top.metadata['maxsim_score'] - 
            optimized_top.metadata['maxsim_score']
        ) < 0.05
        
        assert same_doc or similar_score
```

#### Test 5.2: Performance Improvement Validation
```python
def test_performance_improvement(iris_connector, colbert_pipeline):
    """Test that optimized method is significantly faster."""
    # Setup
    test_query = "cardiovascular disease treatment"
    query_embeddings = colbert_pipeline.colbert_query_encoder(test_query)
    top_k = 5
    
    # Measure original implementation
    start_time = time.time()
    original_results = colbert_pipeline._retrieve_documents_with_colbert(
        query_embeddings, top_k
    )
    original_time = time.time() - start_time
    
    # Measure optimized implementation
    start_time = time.time()
    optimized_results = colbert_pipeline._retrieve_documents_with_colbert_optimized(
        query_embeddings, top_k
    )
    optimized_time = time.time() - start_time
    
    # Assertions
    assert optimized_time < 10.0  # Target: < 10 seconds
    assert len(optimized_results) > 0  # Should return results
    
    # Performance improvement (if original completes in reasonable time)
    if original_time < 60.0:  # Only compare if original doesn't timeout
        improvement_ratio = original_time / optimized_time
        assert improvement_ratio > 2.0  # At least 2x improvement
        
        print(f"Performance improvement: {improvement_ratio:.1f}x faster")
        print(f"Original: {original_time:.2f}s, Optimized: {optimized_time:.2f}s")
```

## Phase 4: Performance and Stress Tests

### Test Group 6: Performance Validation

#### Test 6.1: Memory Usage Monitoring
```python
def test_memory_usage_optimization(iris_connector, colbert_pipeline):
    """Test that optimized method uses less memory."""
    import psutil
    import os
    
    # Setup
    process = psutil.Process(os.getpid())
    test_query = "diabetes treatment options"
    query_embeddings = colbert_pipeline.colbert_query_encoder(test_query)
    
    # Measure memory before
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Execute optimized retrieval
    results = colbert_pipeline._retrieve_documents_with_colbert_optimized(
        query_embeddings, top_k=5
    )
    
    # Measure memory after
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = memory_after - memory_before
    
    # Assertions
    assert memory_used < 100  # Should not use more than 100MB
    assert len(results) > 0   # Should still return results
    
    print(f"Memory used: {memory_used:.1f} MB")
```

#### Test 6.2: Scalability with Query Token Count
```python
def test_scalability_query_tokens(iris_connector, colbert_pipeline):
    """Test performance with varying query token counts."""
    base_embedding = [0.1] * 384
    
    # Test with different query token counts
    token_counts = [1, 5, 10, 20, 50]
    execution_times = []
    
    for token_count in token_counts:
        query_embeddings = [base_embedding] * token_count
        
        start_time = time.time()
        results = colbert_pipeline._retrieve_documents_with_colbert_optimized(
            query_embeddings, top_k=5
        )
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        
        # Should complete in reasonable time regardless of token count
        assert execution_time < 15.0
        assert len(results) <= 5
    
    # Performance should scale sub-linearly
    # (More tokens shouldn't cause proportional time increase)
    max_time = max(execution_times)
    min_time = min(execution_times)
    assert max_time / min_time < 10  # Less than 10x difference
```

### Test Group 7: Error Handling and Edge Cases

#### Test 7.1: Database Connection Failures
```python
def test_database_connection_failure(colbert_pipeline):
    """Test graceful handling of database connection failures."""
    # Setup
    query_embeddings = [[0.1] * 384]
    
    # Mock connection failure
    with patch.object(colbert_pipeline.connection_manager, 'get_connection') as mock_conn:
        mock_conn.side_effect = ConnectionError("Database unavailable")
        
        # Should raise appropriate exception
        with pytest.raises(ConnectionError):
            colbert_pipeline._retrieve_documents_with_colbert_optimized(
                query_embeddings, top_k=5
            )
```

#### Test 7.2: Malformed Vector Data Handling
```python
def test_malformed_vector_data_handling(iris_connector, colbert_pipeline):
    """Test handling of malformed vector data in database."""
    # This test would require setting up test data with malformed vectors
    # and verifying the system handles them gracefully
    
    query_embeddings = [[0.1] * 384]
    
    # Execute with potential malformed data in database
    results = colbert_pipeline._retrieve_documents_with_colbert_optimized(
        query_embeddings, top_k=5
    )
    
    # Should not crash and should return valid results
    assert isinstance(results, list)
    for result in results:
        assert hasattr(result, 'id')
        assert hasattr(result, 'page_content')
        assert 'maxsim_score' in result.metadata
```

## Test Fixtures and Utilities

### Fixture 1: Test Data Setup
```python
@pytest.fixture
def colbert_test_data(iris_connector):
    """Setup test data for ColBERT optimization tests."""
    # Create minimal test dataset
    test_docs = [
        {"doc_id": "test_doc_1", "content": "machine learning algorithms"},
        {"doc_id": "test_doc_2", "content": "deep neural networks"},
        {"doc_id": "test_doc_3", "content": "natural language processing"},
    ]
    
    # Insert test documents and generate token embeddings
    # (Implementation would depend on existing data loading utilities)
    
    yield test_docs
    
    # Cleanup test data
    cleanup_test_data(iris_connector, test_docs)
```

### Fixture 2: Performance Monitoring
```python
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests."""
    import time
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield {
        'start_time': start_time,
        'start_memory': start_memory,
        'process': process
    }
    
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Test execution time: {end_time - start_time:.2f}s")
    print(f"Memory delta: {end_memory - start_memory:.1f}MB")
```

## Test Execution Strategy

### Phase 1: Red Phase (Write Failing Tests)
1. Implement all test cases above
2. Verify they fail (since optimized methods don't exist yet)
3. Commit failing tests to establish TDD baseline

### Phase 2: Green Phase (Implement Minimal Code)
1. Implement [`format_as_iris_vector()`](specs/COLBERT_OPTIMIZATION_SPECIFICATION.md:89) to pass Test Group 1
2. Implement basic HNSW search to pass Test Group 2
3. Implement candidate aggregation to pass Test Group 3
4. Implement MaxSim calculation to pass Test Group 4
5. Integrate components to pass Test Group 5

### Phase 3: Refactor Phase (Optimize Implementation)
1. Optimize performance to pass Test Group 6
2. Add error handling to pass Test Group 7
3. Refactor for maintainability and clarity
4. Add comprehensive logging and monitoring

### Test Execution Commands
```bash
# Run all optimization tests
pytest tests/test_colbert_optimization.py -v

# Run performance tests only
pytest tests/test_colbert_optimization_performance.py -v

# Run with coverage
pytest tests/test_colbert_optimization*.py --cov=iris_rag.pipelines.colbert

# Run with performance monitoring
pytest tests/test_colbert_optimization*.py -s --tb=short
```

## Success Criteria

### Test Coverage Requirements
- **Unit Tests**: > 95% code coverage for new optimization methods
- **Integration Tests**: Complete pipeline functionality validation
- **Performance Tests**: All performance targets met
- **Edge Case Tests**: All error conditions handled gracefully

### Performance Test Targets
- **Execution Time**: All tests complete in < 10 seconds per query
- **Memory Usage**: Peak memory < 100MB during optimization tests
- **Accuracy**: MaxSim scores within 1% tolerance of original implementation
- **Reliability**: Zero test failures on edge cases

This TDD specification ensures that the ColBERT optimization is implemented with comprehensive test coverage, performance validation, and robust error handling.