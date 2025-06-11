# ColBERT Optimization Specification

## Objective
Optimize the [`_retrieve_documents_with_colbert`](iris_rag/pipelines/colbert.py:156) method to leverage IRIS's native HNSW-indexed [`VECTOR_COSINE()`](common/db_init_complete.sql:153) function for efficient token-level similarity search, reducing query processing time from ~43 seconds to < 10 seconds.

## Current State Analysis

### Performance Bottlenecks
1. **Memory Overload**: Loading 206,000+ token embeddings into Python memory
2. **String Parsing**: Manual parsing of VECTOR type data (appears as strings to client)
3. **Brute-Force MaxSim**: Python/NumPy comparison of every query token against all document tokens
4. **Sequential Processing**: Document-by-document processing with individual database queries

### Current Implementation Flow
```
Query → Load ALL token embeddings → Parse strings → Calculate MaxSim in Python → Return top-k
```

## Target State Architecture

### Optimized Implementation Flow
```
Query → For each query token: HNSW search → Aggregate candidates → Calculate MaxSim → Return top-k
```

### Key Optimizations
1. **HNSW Leveraging**: Use native [`VECTOR_COSINE()`](common/db_init_complete.sql:153) with HNSW index
2. **Memory Efficiency**: Avoid loading all embeddings into Python memory
3. **Database-Level Processing**: Push similarity calculations to IRIS database
4. **Candidate Filtering**: Process only relevant document tokens, not all 206,000+

## Detailed Pseudocode Specification

### Main Method: `_retrieve_documents_with_colbert_optimized`

```pseudocode
FUNCTION _retrieve_documents_with_colbert_optimized(query_token_embeddings, top_k):
    // Configuration parameters
    SET candidate_tokens_per_query_token = 50  // K for TOP K searches
    SET candidate_documents_limit = 100        // Maximum documents to evaluate
    
    // Step 1: Initialize data structures
    INITIALIZE connection = get_database_connection()
    INITIALIZE cursor = connection.cursor()
    INITIALIZE candidate_doc_tokens = {}  // Map: doc_id -> List[token_data]
    INITIALIZE doc_maxsim_scores = {}     // Map: doc_id -> maxsim_score
    
    TRY:
        // Step 2: For each query token, perform HNSW similarity search
        FOR each query_token_embedding IN query_token_embeddings:
            // Convert query token to IRIS VECTOR format
            SET query_vector_str = format_as_iris_vector(query_token_embedding)
            
            // Perform TOP K similarity search using HNSW index
            EXECUTE SQL:
                SELECT TOP {candidate_tokens_per_query_token}
                    doc_id,
                    token_index, 
                    token_embedding,
                    VECTOR_COSINE(token_embedding, TO_VECTOR(?)) as similarity_score
                FROM RAG.DocumentTokenEmbeddings
                ORDER BY similarity_score DESC
            WITH PARAMETERS: [query_vector_str]
            
            // Collect candidate document tokens
            FOR each result_row IN query_results:
                SET doc_id = result_row.doc_id
                SET token_data = {
                    token_index: result_row.token_index,
                    embedding: parse_vector_from_db(result_row.token_embedding),
                    similarity_score: result_row.similarity_score
                }
                
                // Add to candidate collection
                IF doc_id NOT IN candidate_doc_tokens:
                    candidate_doc_tokens[doc_id] = []
                candidate_doc_tokens[doc_id].append(token_data)
        
        // Step 3: Limit candidate documents for performance
        IF length(candidate_doc_tokens) > candidate_documents_limit:
            // Sort documents by number of candidate tokens (relevance indicator)
            SET sorted_docs = sort_by_token_count(candidate_doc_tokens)
            SET candidate_doc_tokens = take_top_n(sorted_docs, candidate_documents_limit)
        
        // Step 4: Calculate MaxSim scores for candidate documents
        FOR each doc_id IN candidate_doc_tokens:
            SET doc_token_embeddings = extract_embeddings(candidate_doc_tokens[doc_id])
            SET maxsim_score = calculate_maxsim_score(query_token_embeddings, doc_token_embeddings)
            doc_maxsim_scores[doc_id] = maxsim_score
        
        // Step 5: Sort by MaxSim score and get top-k documents
        SET sorted_doc_scores = sort_descending(doc_maxsim_scores)
        SET top_doc_ids = take_top_n(sorted_doc_scores, top_k)
        
        // Step 6: Retrieve full document content
        SET retrieved_documents = []
        FOR each (doc_id, maxsim_score) IN top_doc_ids:
            EXECUTE SQL:
                SELECT doc_id, text_content
                FROM RAG.SourceDocuments  
                WHERE doc_id = ?
            WITH PARAMETERS: [doc_id]
            
            IF document_found:
                SET document = create_document_object(doc_id, text_content, maxsim_score)
                retrieved_documents.append(document)
        
        RETURN retrieved_documents
        
    CATCH Exception as e:
        LOG error("Optimized ColBERT retrieval failed: " + e)
        THROW e
    FINALLY:
        cursor.close()
END FUNCTION
```

### Helper Functions

#### 1. Vector Format Conversion
```pseudocode
FUNCTION format_as_iris_vector(embedding_list):
    // Convert Python list to IRIS VECTOR format
    SET vector_str = join(embedding_list, ",")
    RETURN vector_str
END FUNCTION

FUNCTION parse_vector_from_db(vector_string):
    // Parse VECTOR type from database (may appear as string to client)
    IF vector_string.startswith("[") AND vector_string.endswith("]"):
        RETURN parse_bracketed_format(vector_string)
    ELSE:
        RETURN parse_comma_separated_format(vector_string)
END FUNCTION
```

#### 2. MaxSim Calculation (Optimized)
```pseudocode
FUNCTION calculate_maxsim_score(query_tokens, doc_tokens):
    // Efficient MaxSim calculation using NumPy
    IF query_tokens.empty OR doc_tokens.empty:
        RETURN 0.0
    
    // Convert to NumPy matrices
    SET query_matrix = numpy.array(query_tokens)    // Shape: (num_query_tokens, embedding_dim)
    SET doc_matrix = numpy.array(doc_tokens)        // Shape: (num_doc_tokens, embedding_dim)
    
    // Normalize embeddings for cosine similarity
    SET query_matrix = normalize_l2(query_matrix)
    SET doc_matrix = normalize_l2(doc_matrix)
    
    // Calculate similarity matrix: query_tokens × doc_tokens
    SET similarity_matrix = matrix_multiply(query_matrix, transpose(doc_matrix))
    
    // MaxSim: For each query token, find maximum similarity with any document token
    SET max_similarities = max_along_axis(similarity_matrix, axis=1)
    
    // Average the maximum similarities across all query tokens
    SET maxsim_score = mean(max_similarities)
    
    RETURN maxsim_score
END FUNCTION
```

#### 3. Candidate Document Management
```pseudocode
FUNCTION sort_by_token_count(candidate_doc_tokens):
    // Sort documents by number of candidate tokens (relevance indicator)
    SET doc_token_counts = []
    FOR each doc_id IN candidate_doc_tokens:
        SET token_count = length(candidate_doc_tokens[doc_id])
        doc_token_counts.append((doc_id, token_count))
    
    RETURN sort_descending_by_count(doc_token_counts)
END FUNCTION

FUNCTION extract_embeddings(token_data_list):
    // Extract just the embeddings from token data structures
    SET embeddings = []
    FOR each token_data IN token_data_list:
        embeddings.append(token_data.embedding)
    RETURN embeddings
END FUNCTION
```

## Configuration Parameters

### Performance Tuning Parameters
```pseudocode
CONFIGURATION:
    // HNSW search parameters
    candidate_tokens_per_query_token: 50     // Balance between recall and performance
    candidate_documents_limit: 100          // Maximum documents to evaluate with MaxSim
    
    // Database connection parameters  
    connection_timeout: 30                   // Seconds
    query_timeout: 10                        // Seconds per HNSW query
    
    // Vector processing parameters
    embedding_dimension: 384                 // ColBERT token embedding dimension
    similarity_threshold: 0.1                // Minimum similarity to consider
    
    // Fallback parameters
    enable_fallback: true                    // Fall back to current method if optimization fails
    fallback_sample_size: 1000              // Documents to sample in fallback mode
END CONFIGURATION
```

## Database Schema Requirements

### Required HNSW Index
```sql
-- Ensure HNSW index exists on token_embedding column
CREATE INDEX IF NOT EXISTS idx_hnsw_token_embedding 
ON RAG.DocumentTokenEmbeddings (token_embedding) 
AS HNSW(M=16, efConstruction=200, Distance='COSINE');
```

### Required Vector Function Support
```sql
-- Verify VECTOR_COSINE function availability
SELECT VECTOR_COSINE(token_embedding, TO_VECTOR('0.1,0.2,0.3,...')) 
FROM RAG.DocumentTokenEmbeddings 
LIMIT 1;
```

## TDD Anchors (Test Cases)

### 1. Correctness Tests

#### Test Case 1.1: Document Retrieval Accuracy
```pseudocode
TEST test_optimized_retrieval_correctness():
    // Setup
    SET test_query = "machine learning algorithms"
    SET query_embeddings = encode_query_tokens(test_query)
    
    // Execute both implementations
    SET original_results = original_retrieve_documents_with_colbert(query_embeddings, top_k=5)
    SET optimized_results = optimized_retrieve_documents_with_colbert(query_embeddings, top_k=5)
    
    // Assertions
    ASSERT length(optimized_results) == length(original_results)
    ASSERT optimized_results[0].doc_id == original_results[0].doc_id  // Top result should match
    
    // MaxSim scores should be within tolerance
    FOR i IN range(min(3, length(results))):  // Check top 3 results
        SET score_diff = abs(optimized_results[i].maxsim_score - original_results[i].maxsim_score)
        ASSERT score_diff < 0.01  // 1% tolerance for floating point differences
END TEST
```

#### Test Case 1.2: MaxSim Score Calculation Accuracy
```pseudocode
TEST test_maxsim_score_accuracy():
    // Setup with known embeddings
    SET query_tokens = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    SET doc_tokens = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]]
    
    // Calculate expected MaxSim manually
    // Query token 1 [1,0,0] best matches doc token 1 [1,0,0] = similarity 1.0
    // Query token 2 [0,1,0] best matches doc token 3 [0.5,0.5,0] = similarity ~0.707
    // Expected MaxSim = (1.0 + 0.707) / 2 = ~0.854
    
    SET calculated_score = calculate_maxsim_score(query_tokens, doc_tokens)
    ASSERT abs(calculated_score - 0.854) < 0.01
END TEST
```

### 2. Performance Tests

#### Test Case 2.1: Execution Time Improvement
```pseudocode
TEST test_performance_improvement():
    // Setup
    SET test_query = "cardiovascular disease treatment"
    SET query_embeddings = encode_query_tokens(test_query)
    
    // Measure original implementation
    SET start_time = current_time()
    SET original_results = original_retrieve_documents_with_colbert(query_embeddings, top_k=5)
    SET original_time = current_time() - start_time
    
    // Measure optimized implementation  
    SET start_time = current_time()
    SET optimized_results = optimized_retrieve_documents_with_colbert(query_embeddings, top_k=5)
    SET optimized_time = current_time() - start_time
    
    // Assertions
    ASSERT optimized_time < 10.0  // Target: < 10 seconds
    ASSERT optimized_time < original_time * 0.25  // At least 4x improvement
    
    LOG info("Performance improvement: " + (original_time / optimized_time) + "x faster")
END TEST
```

#### Test Case 2.2: Memory Usage Efficiency
```pseudocode
TEST test_memory_efficiency():
    // Setup memory monitoring
    SET memory_before = get_memory_usage()
    
    // Execute optimized retrieval
    SET query_embeddings = encode_query_tokens("test query")
    SET results = optimized_retrieve_documents_with_colbert(query_embeddings, top_k=5)
    
    SET memory_after = get_memory_usage()
    SET memory_used = memory_after - memory_before
    
    // Assertions
    ASSERT memory_used < 100_MB  // Should not load all embeddings into memory
    ASSERT length(results) > 0   // Should still return results
END TEST
```

### 3. Edge Case Tests

#### Test Case 3.1: Empty Query Handling
```pseudocode
TEST test_empty_query_handling():
    SET empty_query_embeddings = []
    SET results = optimized_retrieve_documents_with_colbert(empty_query_embeddings, top_k=5)
    
    ASSERT length(results) == 0
    // Should not throw exception
END TEST
```

#### Test Case 3.2: No Matching Tokens
```pseudocode
TEST test_no_matching_tokens():
    // Create query embeddings that won't match any document tokens
    SET outlier_embeddings = [[999.0] * 384, [-999.0] * 384]
    SET results = optimized_retrieve_documents_with_colbert(outlier_embeddings, top_k=5)
    
    // Should return some results (even with low scores) or empty list
    ASSERT length(results) >= 0
    ASSERT length(results) <= 5
END TEST
```

#### Test Case 3.3: Large Query Token Count
```pseudocode
TEST test_large_query_token_count():
    // Test with many query tokens (stress test)
    SET large_query_embeddings = generate_random_embeddings(count=50, dimension=384)
    
    SET start_time = current_time()
    SET results = optimized_retrieve_documents_with_colbert(large_query_embeddings, top_k=5)
    SET execution_time = current_time() - start_time
    
    ASSERT execution_time < 15.0  // Should still be reasonable with many tokens
    ASSERT length(results) <= 5
END TEST
```

### 4. Integration Tests

#### Test Case 4.1: End-to-End Pipeline Test
```pseudocode
TEST test_end_to_end_pipeline():
    // Test complete ColBERT pipeline with optimization
    SET pipeline = ColBERTRAGPipeline(connection_manager, config_manager)
    SET query = "What are the latest treatments for diabetes?"
    
    SET result = pipeline.run(query, top_k=3)
    
    // Assertions
    ASSERT result["query"] == query
    ASSERT result["technique"] == "ColBERT"
    ASSERT length(result["retrieved_documents"]) <= 3
    ASSERT result["execution_time"] < 10.0
    ASSERT len(result["answer"]) > 0
END TEST
```

#### Test Case 4.2: Database Connection Resilience
```pseudocode
TEST test_database_connection_resilience():
    // Test behavior with database connection issues
    SET pipeline = ColBERTRAGPipeline(connection_manager, config_manager)
    
    // Simulate connection timeout
    WITH mock_connection_timeout():
        TRY:
            SET result = pipeline.run("test query", top_k=5)
            // Should either succeed or fail gracefully
        CATCH DatabaseError:
            // Expected behavior - should not crash the application
            PASS
END TEST
```

## Implementation Strategy

### Phase 1: Core Optimization (Week 1)
1. Implement [`format_as_iris_vector()`](specs/COLBERT_OPTIMIZATION_SPECIFICATION.md:89) helper function
2. Implement HNSW-based token similarity search loop
3. Implement candidate document aggregation logic
4. Create basic performance benchmarking

### Phase 2: MaxSim Optimization (Week 1)
1. Optimize [`calculate_maxsim_score()`](specs/COLBERT_OPTIMIZATION_SPECIFICATION.md:101) with NumPy
2. Implement candidate document filtering
3. Add configuration parameter support
4. Create correctness validation tests

### Phase 3: Integration & Testing (Week 1)
1. Integrate optimized method into [`ColBERTRAGPipeline`](iris_rag/pipelines/colbert.py:21)
2. Implement comprehensive test suite
3. Add fallback mechanism for edge cases
4. Performance validation and tuning

### Phase 4: Production Readiness (Week 1)
1. Add monitoring and logging
2. Implement error handling and recovery
3. Create documentation and usage examples
4. Final performance benchmarking

## Success Criteria

### Performance Targets
- **Execution Time**: < 10 seconds per query (down from ~43 seconds)
- **Memory Usage**: < 100MB peak memory (down from loading 206,000+ embeddings)
- **Accuracy**: MaxSim scores within 1% of original implementation
- **Scalability**: Performance should scale sub-linearly with document count

### Quality Targets
- **Test Coverage**: > 95% code coverage
- **Reliability**: Zero crashes on edge cases
- **Maintainability**: Clear, documented, and modular code
- **Compatibility**: Backward compatible with existing ColBERT pipeline interface

## Risk Mitigation

### Technical Risks
1. **HNSW Index Performance**: Monitor index effectiveness and tune parameters
2. **Vector Function Compatibility**: Implement fallback for different IRIS versions
3. **Memory Leaks**: Careful cursor and connection management
4. **Accuracy Degradation**: Comprehensive validation against original implementation

### Operational Risks
1. **Database Load**: Monitor impact of multiple HNSW queries
2. **Configuration Complexity**: Provide sensible defaults and validation
3. **Deployment Issues**: Thorough testing in staging environment
4. **Rollback Plan**: Maintain original implementation as fallback option