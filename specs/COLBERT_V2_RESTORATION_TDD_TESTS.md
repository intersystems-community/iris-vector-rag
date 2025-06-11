# ColBERT V2 Restoration TDD Test Specification

## Overview
This document defines the Test-Driven Development (TDD) approach for restoring the optimized [`ColBERTPipelineV2`](archived_pipelines/colbert/pipeline_v2.py:12) implementation. The focus is on adapting the proven V2 hybrid approach to the current [`iris_rag`](iris_rag/pipelines/colbert.py) architecture.

## Test Strategy: Restore → Validate → Optimize

### Phase 1: Restoration Tests (Red Phase)
Verify that the V2 approach can be successfully restored in the current architecture.

### Phase 2: Performance Tests (Green Phase)  
Validate that the restored implementation achieves the expected performance improvements.

### Phase 3: Integration Tests (Refactor Phase)
Ensure seamless integration with the existing ColBERT pipeline interface.

## Test File Organization
```
tests/
├── test_colbert_v2_restoration.py         # Core V2 restoration tests
├── test_colbert_v2_performance.py         # Performance validation tests
├── test_colbert_v2_integration.py         # Integration with current pipeline
└── fixtures/
    ├── colbert_v2_test_data.py            # V2-specific test data
    └── performance_comparison_fixtures.py  # Performance comparison utilities
```

## Phase 1: V2 Restoration Tests

### Test Group 1: Document-Level Candidate Selection

#### Test 1.1: HNSW Document Retrieval
```python
def test_hnsw_document_candidate_selection(iris_connector):
    """Test that document-level HNSW search returns appropriate candidates."""
    # Setup
    query = "machine learning algorithms"
    query_embedding = get_test_embedding(query)
    top_k = 5
    candidate_multiplier = 3
    
    # Execute V2 candidate selection
    candidates = get_document_candidates_v2(
        iris_connector, 
        query_embedding, 
        top_k * candidate_multiplier
    )
    
    # Assertions
    assert len(candidates) <= top_k * candidate_multiplier
    assert len(candidates) > 0
    
    # Verify candidate structure
    for candidate in candidates:
        assert 'doc_id' in candidate
        assert 'title' in candidate
        assert 'text_content' in candidate
        assert 'initial_score' in candidate
        assert isinstance(candidate['initial_score'], float)
        assert 0.0 <= candidate['initial_score'] <= 1.0
    
    # Verify candidates are sorted by score (descending)
    scores = [c['initial_score'] for c in candidates]
    assert scores == sorted(scores, reverse=True)
```

#### Test 1.2: Candidate Quality Validation
```python
def test_candidate_quality_vs_current_implementation(iris_connector, colbert_pipeline):
    """Test that V2 candidates include the same top documents as current implementation."""
    # Setup
    query = "cardiovascular disease treatment"
    query_embeddings = colbert_pipeline.colbert_query_encoder(query)
    top_k = 5
    
    # Get results from current implementation
    current_results = colbert_pipeline._retrieve_documents_with_colbert(
        query_embeddings, top_k
    )
    current_doc_ids = {doc.id for doc in current_results}
    
    # Get candidates from V2 approach
    query_doc_embedding = np.mean(query_embeddings, axis=0)
    v2_candidates = get_document_candidates_v2(
        iris_connector, 
        query_doc_embedding, 
        top_k * 3  # Get more candidates to ensure overlap
    )
    v2_candidate_ids = {c['doc_id'] for c in v2_candidates}
    
    # Assertions
    # V2 candidates should include most/all of the current top results
    overlap = len(current_doc_ids.intersection(v2_candidate_ids))
    overlap_percentage = overlap / len(current_doc_ids) if current_doc_ids else 0
    
    assert overlap_percentage >= 0.8  # At least 80% overlap
    assert len(v2_candidates) >= len(current_results)  # V2 should get more candidates
```

### Test Group 2: Token Loading for Candidates

#### Test 2.1: Selective Token Loading
```python
def test_load_tokens_for_candidates_only(iris_connector):
    """Test that tokens are loaded only for candidate documents."""
    # Setup
    candidate_doc_ids = ["test_doc_1", "test_doc_2", "test_doc_3"]
    
    # Execute selective token loading
    doc_tokens_map = load_tokens_for_documents(iris_connector, candidate_doc_ids)
    
    # Assertions
    assert isinstance(doc_tokens_map, dict)
    
    # Should only contain tokens for requested documents
    for doc_id in doc_tokens_map.keys():
        assert doc_id in candidate_doc_ids
    
    # Verify token structure
    for doc_id, tokens in doc_tokens_map.items():
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        for token_embedding in tokens:
            assert isinstance(token_embedding, list)
            assert len(token_embedding) == 384  # ColBERT embedding dimension
            assert all(isinstance(x, float) for x in token_embedding)
```

#### Test 2.2: Token Loading Performance
```python
def test_token_loading_performance_vs_current(iris_connector):
    """Test that selective token loading is much faster than loading all tokens."""
    # Setup
    candidate_doc_ids = ["doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]  # Small set
    
    # Measure V2 selective loading
    start_time = time.time()
    v2_tokens = load_tokens_for_documents(iris_connector, candidate_doc_ids)
    v2_time = time.time() - start_time
    
    # Measure current approach (loading all tokens)
    start_time = time.time()
    all_tokens = load_all_document_tokens(iris_connector)  # Current approach
    current_time = time.time() - start_time
    
    # Assertions
    assert v2_time < current_time * 0.1  # V2 should be at least 10x faster
    assert len(v2_tokens) <= len(candidate_doc_ids)
    assert len(all_tokens) > len(v2_tokens)  # All tokens should be much more
    
    print(f"V2 selective loading: {v2_time:.3f}s")
    print(f"Current all-tokens loading: {current_time:.3f}s")
    print(f"Speed improvement: {current_time / v2_time:.1f}x")
```

### Test Group 3: V2 MaxSim Calculation

#### Test 3.1: V2 MaxSim Accuracy
```python
def test_v2_maxsim_calculation_accuracy():
    """Test that V2 MaxSim calculation matches the current implementation."""
    # Setup with known embeddings
    query_tokens = [
        [1.0, 0.0, 0.0],  # Query token 1
        [0.0, 1.0, 0.0],  # Query token 2
    ]
    
    doc_tokens = [
        [1.0, 0.0, 0.0],  # Perfect match for query token 1
        [0.0, 1.0, 0.0],  # Perfect match for query token 2
        [0.5, 0.5, 0.0],  # Partial match
    ]
    
    # Calculate using both implementations
    current_maxsim = calculate_maxsim_score_current(query_tokens, doc_tokens)
    v2_maxsim = calculate_maxsim_score_v2(query_tokens, doc_tokens)
    
    # Assertions
    assert abs(current_maxsim - v2_maxsim) < 0.001  # Should be nearly identical
    assert 0.0 <= v2_maxsim <= 2.0  # Reasonable range for normalized MaxSim
```

#### Test 3.2: V2 MaxSim Performance
```python
def test_v2_maxsim_performance():
    """Test that V2 MaxSim calculation is efficient."""
    # Setup with realistic token counts
    query_tokens = [[random.random() for _ in range(384)] for _ in range(10)]
    doc_tokens = [[random.random() for _ in range(384)] for _ in range(50)]
    
    # Measure V2 MaxSim calculation
    start_time = time.time()
    for _ in range(100):  # Multiple iterations for stable timing
        maxsim_score = calculate_maxsim_score_v2(query_tokens, doc_tokens)
    v2_time = time.time() - start_time
    
    # Assertions
    assert v2_time < 1.0  # Should complete 100 calculations in under 1 second
    assert 0.0 <= maxsim_score <= 1.0  # Reasonable score range
```

## Phase 2: Performance Validation Tests

### Test Group 4: End-to-End Performance

#### Test 4.1: V2 vs Current Performance Comparison
```python
def test_v2_vs_current_performance_improvement(iris_connector, colbert_pipeline):
    """Test that V2 implementation is significantly faster than current."""
    # Setup
    test_queries = [
        "machine learning algorithms",
        "cardiovascular disease treatment", 
        "diabetes management strategies"
    ]
    top_k = 5
    
    current_times = []
    v2_times = []
    
    for query in test_queries:
        query_embeddings = colbert_pipeline.colbert_query_encoder(query)
        
        # Measure current implementation
        start_time = time.time()
        current_results = colbert_pipeline._retrieve_documents_with_colbert(
            query_embeddings, top_k
        )
        current_time = time.time() - start_time
        current_times.append(current_time)
        
        # Measure V2 implementation
        start_time = time.time()
        v2_results = retrieve_documents_with_colbert_v2(
            iris_connector, query_embeddings, top_k
        )
        v2_time = time.time() - start_time
        v2_times.append(v2_time)
        
        # Basic result validation
        assert len(v2_results) <= top_k
        assert len(v2_results) > 0
    
    # Performance assertions
    avg_current_time = sum(current_times) / len(current_times)
    avg_v2_time = sum(v2_times) / len(v2_times)
    
    assert avg_v2_time < 5.0  # Target: < 5 seconds per query
    assert avg_v2_time < avg_current_time * 0.2  # At least 5x improvement
    
    improvement_factor = avg_current_time / avg_v2_time
    print(f"Average performance improvement: {improvement_factor:.1f}x")
    print(f"Current avg: {avg_current_time:.2f}s, V2 avg: {avg_v2_time:.2f}s")
```

#### Test 4.2: Memory Usage Validation
```python
def test_v2_memory_usage_efficiency(iris_connector):
    """Test that V2 implementation uses significantly less memory."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Measure V2 memory usage
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    query_embeddings = [[0.1] * 384 for _ in range(5)]
    v2_results = retrieve_documents_with_colbert_v2(
        iris_connector, query_embeddings, top_k=5
    )
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    v2_memory_used = memory_after - memory_before
    
    # Assertions
    assert v2_memory_used < 50  # Should use less than 50MB
    assert len(v2_results) > 0  # Should still return results
    
    print(f"V2 memory usage: {v2_memory_used:.1f} MB")
```

### Test Group 5: Accuracy Validation

#### Test 5.1: Result Quality Comparison
```python
def test_v2_result_quality_vs_current(iris_connector, colbert_pipeline):
    """Test that V2 results maintain similar quality to current implementation."""
    # Setup
    test_queries = [
        "machine learning neural networks",
        "cancer treatment immunotherapy",
        "diabetes insulin resistance"
    ]
    top_k = 5
    
    quality_metrics = []
    
    for query in test_queries:
        query_embeddings = colbert_pipeline.colbert_query_encoder(query)
        
        # Get results from both implementations
        current_results = colbert_pipeline._retrieve_documents_with_colbert(
            query_embeddings, top_k
        )
        v2_results = retrieve_documents_with_colbert_v2(
            iris_connector, query_embeddings, top_k
        )
        
        # Calculate quality metrics
        current_doc_ids = {doc.id for doc in current_results}
        v2_doc_ids = {doc.id for doc in v2_results}
        
        # Overlap in top results
        overlap = len(current_doc_ids.intersection(v2_doc_ids))
        overlap_percentage = overlap / max(len(current_doc_ids), 1)
        
        # Score correlation for overlapping documents
        overlapping_docs = current_doc_ids.intersection(v2_doc_ids)
        if overlapping_docs:
            current_scores = {doc.id: doc.metadata.get('maxsim_score', 0) 
                            for doc in current_results if doc.id in overlapping_docs}
            v2_scores = {doc.id: doc.metadata.get('maxsim_score', 0) 
                        for doc in v2_results if doc.id in overlapping_docs}
            
            score_correlation = calculate_score_correlation(current_scores, v2_scores)
        else:
            score_correlation = 0.0
        
        quality_metrics.append({
            'query': query,
            'overlap_percentage': overlap_percentage,
            'score_correlation': score_correlation
        })
    
    # Quality assertions
    avg_overlap = sum(m['overlap_percentage'] for m in quality_metrics) / len(quality_metrics)
    avg_correlation = sum(m['score_correlation'] for m in quality_metrics) / len(quality_metrics)
    
    assert avg_overlap >= 0.6  # At least 60% overlap in top results
    assert avg_correlation >= 0.7  # Strong correlation in scores
    
    print(f"Average result overlap: {avg_overlap:.1%}")
    print(f"Average score correlation: {avg_correlation:.3f}")
```

## Phase 3: Integration Tests

### Test Group 6: Pipeline Integration

#### Test 6.1: Drop-in Replacement Test
```python
def test_v2_as_drop_in_replacement(colbert_pipeline):
    """Test that V2 method can replace current method seamlessly."""
    # Setup
    query = "artificial intelligence applications"
    top_k = 3
    
    # Store original method
    original_method = colbert_pipeline._retrieve_documents_with_colbert
    
    # Replace with V2 method
    colbert_pipeline._retrieve_documents_with_colbert = lambda q_emb, k: retrieve_documents_with_colbert_v2(
        colbert_pipeline.connection_manager.get_connection(), q_emb, k
    )
    
    try:
        # Test full pipeline execution
        result = colbert_pipeline.run(query, top_k=top_k)
        
        # Assertions
        assert "query" in result
        assert "answer" in result
        assert "retrieved_documents" in result
        assert result["query"] == query
        assert len(result["retrieved_documents"]) <= top_k
        assert len(result["answer"]) > 0
        
    finally:
        # Restore original method
        colbert_pipeline._retrieve_documents_with_colbert = original_method
```

#### Test 6.2: Configuration Compatibility
```python
def test_v2_configuration_compatibility(colbert_pipeline):
    """Test that V2 implementation respects existing configuration."""
    # Setup
    query = "medical research methodology"
    
    # Test with different configurations
    configs = [
        {"top_k": 3, "candidate_multiplier": 2},
        {"top_k": 5, "candidate_multiplier": 3},
        {"top_k": 10, "candidate_multiplier": 4},
    ]
    
    for config in configs:
        # Execute with configuration
        results = retrieve_documents_with_colbert_v2_configured(
            colbert_pipeline.connection_manager.get_connection(),
            colbert_pipeline.colbert_query_encoder(query),
            **config
        )
        
        # Assertions
        assert len(results) <= config["top_k"]
        assert len(results) > 0
        
        # Verify configuration was applied
        # (This would depend on implementation details)
```

## Test Utilities and Fixtures

### Fixture 1: V2 Test Environment
```python
@pytest.fixture
def v2_test_environment(iris_connector):
    """Setup test environment for V2 restoration tests."""
    # Ensure required data exists
    cursor = iris_connector.cursor()
    
    # Verify document-level embeddings exist
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
    doc_count = cursor.fetchone()[0]
    assert doc_count > 0, "No document embeddings found for V2 testing"
    
    # Verify token embeddings exist
    cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
    token_count = cursor.fetchone()[0]
    assert token_count > 0, "No token embeddings found for V2 testing"
    
    # Verify HNSW indexes exist
    cursor.execute("""
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES 
        WHERE TABLE_NAME = 'SourceDocuments' AND INDEX_NAME LIKE '%hnsw%'
    """)
    hnsw_count = cursor.fetchone()[0]
    
    cursor.close()
    
    yield {
        "document_count": doc_count,
        "token_count": token_count,
        "hnsw_indexes": hnsw_count > 0
    }
```

### Fixture 2: Performance Comparison Utilities
```python
@pytest.fixture
def performance_comparison_utils():
    """Utilities for comparing V2 vs current performance."""
    
    def measure_execution_time(func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    
    def calculate_improvement_factor(old_time, new_time):
        return old_time / new_time if new_time > 0 else float('inf')
    
    def calculate_score_correlation(scores1, scores2):
        if not scores1 or not scores2:
            return 0.0
        
        common_keys = set(scores1.keys()).intersection(set(scores2.keys()))
        if len(common_keys) < 2:
            return 0.0
        
        values1 = [scores1[k] for k in common_keys]
        values2 = [scores2[k] for k in common_keys]
        
        return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0
    
    return {
        "measure_execution_time": measure_execution_time,
        "calculate_improvement_factor": calculate_improvement_factor,
        "calculate_score_correlation": calculate_score_correlation
    }
```

## Test Execution Strategy

### Phase 1: Restoration Validation (Red Phase)
```bash
# Run restoration tests to verify V2 approach can be implemented
pytest tests/test_colbert_v2_restoration.py -v

# Expected: All tests should fail initially (Red phase)
```

### Phase 2: Implementation (Green Phase)
```bash
# Implement V2 methods to make tests pass
# Run tests iteratively as implementation progresses
pytest tests/test_colbert_v2_restoration.py::test_hnsw_document_candidate_selection -v
pytest tests/test_colbert_v2_restoration.py::test_load_tokens_for_candidates_only -v
# ... continue with each test
```

### Phase 3: Performance Validation (Green Phase)
```bash
# Run performance tests to validate improvements
pytest tests/test_colbert_v2_performance.py -v

# Run with performance monitoring
pytest tests/test_colbert_v2_performance.py -s --tb=short
```

### Phase 4: Integration Testing (Refactor Phase)
```bash
# Run integration tests to ensure seamless replacement
pytest tests/test_colbert_v2_integration.py -v

# Run full test suite to ensure no regressions
pytest tests/test_colbert_*.py --cov=iris_rag.pipelines.colbert
```

## Success Criteria

### Restoration Success
- All V2 restoration tests pass
- Document candidate selection works correctly
- Token loading is selective and efficient
- MaxSim calculation maintains accuracy

### Performance Success
- V2 implementation is at least 5x faster than current
- Memory usage is reduced by at least 10x
- Result quality maintains 60%+ overlap with current results
- Score correlation is 0.7+ for overlapping documents

### Integration Success
- V2 method works as drop-in replacement
- All existing pipeline tests continue to pass
- Configuration compatibility is maintained
- No breaking changes to public API

This TDD specification focuses on the practical restoration of the proven V2 approach, ensuring both performance gains and quality maintenance.