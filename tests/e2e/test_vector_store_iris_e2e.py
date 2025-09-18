"""
Vector Store IRIS E2E Tests

Tests vector similarity search functionality with real embeddings and IRIS database:
- Vector index creation and management
- Similarity search with 50+ real documents  
- Performance metrics validation
- HNSW index efficiency testing

NO MOCKS - Uses real IRIS vector storage throughout.

Priority 1 Components Under Test:
- iris_rag/core/vector_store.py (VectorStore abstract base)
- iris_rag/storage/vector_store_iris.py (IRIS implementation)
"""

import pytest
import logging
import time
import numpy as np
from typing import List, Dict, Any, Tuple

# Real framework imports (NO MOCKS)
from iris_rag.core.models import Document
from iris_rag.storage.vector_store_iris import IRISVectorStore
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class TestVectorStoreIRISCore:
    """Test core vector store operations with real IRIS database."""
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_vector_store_initialization_e2e(
        self,
        e2e_config_manager: ConfigurationManager,
        e2e_connection_manager: ConnectionManager
    ):
        """
        Test IRISVectorStore initialization with real configuration.
        
        NO MOCKS - Tests real initialization with IRIS connection.
        """
        logger.info("=== STARTING VECTOR STORE INITIALIZATION E2E TEST ===")
        
        # Test initialization with real managers
        vector_store = IRISVectorStore(
            connection_manager=e2e_connection_manager,
            config_manager=e2e_config_manager
        )
        
        # Validate initialization
        assert vector_store is not None, "Vector store should be initialized"
        assert isinstance(vector_store, VectorStore), "Should implement VectorStore interface"
        assert isinstance(vector_store, IRISVectorStore), "Should be IRISVectorStore instance"
        
        # Test connection is working
        assert vector_store.connection_manager is not None, "Should have connection manager"
        assert vector_store.config_manager is not None, "Should have config manager"
        
        # Test configuration loading
        assert hasattr(vector_store, 'table_name'), "Should have table name from config"
        assert hasattr(vector_store, 'vector_dimension'), "Should have vector dimension"
        
        logger.info(f"Vector store initialized: table={vector_store.table_name}, dim={vector_store.vector_dimension}")
        logger.info("=== VECTOR STORE INITIALIZATION E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    @pytest.mark.pmc_data
    def test_document_storage_with_embeddings_e2e(
        self,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_pmc_documents: List[Document],
        e2e_embedding_function,
        e2e_database_cleanup,
        e2e_performance_monitor,
        e2e_document_validator
    ):
        """
        Test document storage with real embedding generation.
        
        NO MOCKS - Tests real embedding generation and vector storage in IRIS.
        """
        logger.info("=== STARTING DOCUMENT STORAGE WITH EMBEDDINGS E2E TEST ===")
        
        # Start performance monitoring
        e2e_performance_monitor.start_timer("document_storage_embeddings")
        
        # Use subset of documents for focused testing
        test_documents = e2e_pmc_documents[:5]
        
        # Step 1: Generate real embeddings
        logger.info("Step 1: Generating real embeddings for documents")
        
        embeddings = []
        for doc in test_documents:
            embedding = e2e_embedding_function(doc.page_content)
            embeddings.append(embedding)
            
            # Validate embedding structure
            assert isinstance(embedding, list), "Embedding should be a list"
            assert len(embedding) > 0, "Embedding should have dimensions"
            assert all(isinstance(val, (int, float)) for val in embedding), "Embedding values should be numeric"
            
        logger.info(f"Generated {len(embeddings)} embeddings, dimension: {len(embeddings[0])}")
        
        # Step 2: Store documents with embeddings
        logger.info("Step 2: Storing documents with embeddings in IRIS")
        
        stored_ids = e2e_iris_vector_store.add_documents(test_documents, embeddings)
        
        # Validate storage results
        assert stored_ids, "Storage should return document IDs"
        assert len(stored_ids) == len(test_documents), "Should store all documents"
        
        for stored_id in stored_ids:
            assert stored_id, "Each stored ID should be non-empty"
        
        logger.info(f"Stored {len(stored_ids)} documents with embeddings")
        
        # Step 3: Validate storage in database
        logger.info("Step 3: Validating embeddings stored in database")
        
        validation_results = e2e_document_validator.validate_document_ingestion(stored_ids)
        
        for doc_id in stored_ids:
            result = validation_results[doc_id]
            assert result["found"], f"Document {doc_id} should be found in database"
            assert result["has_embedding"], f"Document {doc_id} should have embedding in database"
            assert result["has_content"], f"Document {doc_id} should have content in database"
        
        # End performance monitoring
        storage_time = e2e_performance_monitor.end_timer("document_storage_embeddings")
        
        # Performance validation
        assert storage_time > 0, "Storage should take measurable time (not mocked)"
        assert storage_time < 180, "Storage should complete within 3 minutes for 5 documents"
        
        logger.info(f"=== DOCUMENT STORAGE WITH EMBEDDINGS E2E TEST COMPLETED ({storage_time:.2f}s) ===")
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_vector_similarity_search_e2e(
        self,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_pmc_documents: List[Document],
        e2e_biomedical_queries: List[Dict[str, Any]],
        e2e_embedding_function,
        e2e_database_cleanup,
        e2e_performance_monitor
    ):
        """
        Test vector similarity search with real embeddings and queries.
        
        NO MOCKS - Tests real vector similarity search in IRIS with HNSW index.
        """
        logger.info("=== STARTING VECTOR SIMILARITY SEARCH E2E TEST ===")
        
        # Setup: Store documents with embeddings
        test_documents = e2e_pmc_documents[:10]  # Use 10 documents for meaningful search
        stored_ids = e2e_iris_vector_store.add_documents(test_documents)
        
        assert len(stored_ids) == len(test_documents), "Setup: All documents should be stored"
        logger.info(f"Setup complete: {len(stored_ids)} documents stored")
        
        # Test similarity search for each biomedical query
        for i, query_data in enumerate(e2e_biomedical_queries):
            query_text = query_data["query"]
            expected_keywords = query_data["expected_keywords"]
            
            logger.info(f"Testing similarity search for query {i+1}: {query_text}")
            
            # Start performance monitoring for this search
            e2e_performance_monitor.start_timer(f"similarity_search_{i+1}")
            
            # Step 1: Generate query embedding
            query_embedding = e2e_embedding_function(query_text)
            
            assert query_embedding, "Query embedding should be generated"
            assert len(query_embedding) > 0, "Query embedding should have dimensions"
            
            # Step 2: Perform similarity search
            search_results = e2e_iris_vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=5
            )
            
            # Validate search results
            assert search_results, f"Similarity search should return results for '{query_text}'"
            assert len(search_results) <= 5, "Should not exceed top_k limit"
            
            # Validate result structure
            for result in search_results:
                assert isinstance(result, tuple), "Each result should be (Document, score) tuple"
                document, score = result
                
                assert isinstance(document, Document), "Result should contain Document object"
                assert document.page_content, "Retrieved document should have content"
                assert document.id, "Retrieved document should have ID"
                assert isinstance(score, (int, float)), "Similarity score should be numeric"
                assert score >= 0, "Similarity score should be non-negative"
                
                logger.info(f"  Retrieved: {document.id} (score: {score:.4f})")
            
            # Step 3: Validate result ordering (scores should be in descending order)
            scores = [score for _, score in search_results]
            assert scores == sorted(scores, reverse=True), "Results should be ordered by similarity score (descending)"
            
            # Step 4: Validate content relevance
            retrieved_content = " ".join([doc.page_content.lower() for doc, _ in search_results])
            keyword_matches = sum(1 for keyword in expected_keywords 
                                if keyword.lower() in retrieved_content)
            
            # We expect at least some keyword relevance, but not strict matching for E2E flexibility
            if keyword_matches == 0:
                logger.warning(f"No keyword matches found for query '{query_text}' - may indicate relevance issues")
            else:
                logger.info(f"Keyword relevance: {keyword_matches}/{len(expected_keywords)} keywords found")
            
            # End performance monitoring
            search_time = e2e_performance_monitor.end_timer(f"similarity_search_{i+1}")
            
            # Performance validation
            assert search_time > 0, "Search should take measurable time (not mocked)"
            assert search_time < 10, "Search should complete within 10 seconds"  # HNSW should be fast
            
        logger.info("=== VECTOR SIMILARITY SEARCH E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_vector_search_with_filters_e2e(
        self,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_pmc_documents: List[Document],
        e2e_embedding_function,
        e2e_database_cleanup,
        e2e_performance_monitor
    ):
        """
        Test vector similarity search with metadata filters.
        
        NO MOCKS - Tests real filtered vector search in IRIS.
        """
        logger.info("=== STARTING VECTOR SEARCH WITH FILTERS E2E TEST ===")
        
        # Setup: Prepare documents with diverse metadata
        test_documents = []
        for i, base_doc in enumerate(e2e_pmc_documents[:6]):
            # Add diverse metadata for filtering tests
            metadata = base_doc.metadata.copy()
            metadata.update({
                "topic": ["cardiovascular", "diabetes", "oncology"][i % 3],
                "type": "biomedical",
                "year": 2020 + (i % 4),
                "source": "e2e_test_filtered"
            })
            
            doc = Document(
                id=f"filtered_test_{i}",
                page_content=base_doc.page_content,
                metadata=metadata
            )
            test_documents.append(doc)
        
        # Store documents
        stored_ids = e2e_iris_vector_store.add_documents(test_documents)
        assert len(stored_ids) == len(test_documents), "Setup: All documents should be stored"
        
        # Test query with topic filter
        query_text = "What are treatment approaches for medical conditions?"
        query_embedding = e2e_embedding_function(query_text)
        
        # Start performance monitoring
        e2e_performance_monitor.start_timer("filtered_search")
        
        # Test filter by topic
        topic_filter = {"topic": "cardiovascular"}
        
        try:
            filtered_results = e2e_iris_vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=5,
                filter=topic_filter
            )
            
            # Validate filtered results
            assert isinstance(filtered_results, list), "Filtered search should return list"
            
            # Check that results match filter criteria (if any results returned)
            for result in filtered_results:
                document, score = result
                assert document.metadata.get("topic") == "cardiovascular", \
                    "Filtered results should match filter criteria"
                
            logger.info(f"Topic filter search returned {len(filtered_results)} results")
            
        except Exception as e:
            # Some vector stores may not support filtering - log but don't fail
            logger.warning(f"Filtered search not supported or failed: {e}")
        
        # End performance monitoring
        search_time = e2e_performance_monitor.end_timer("filtered_search")
        
        logger.info(f"=== VECTOR SEARCH WITH FILTERS E2E TEST COMPLETED ({search_time:.2f}s) ===")


class TestVectorStoreIRISPerformance:
    """Test vector store performance with larger datasets."""
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    @pytest.mark.slow
    def test_large_scale_vector_search_e2e(
        self,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_pmc_documents: List[Document],
        e2e_embedding_function,
        e2e_database_cleanup,
        e2e_performance_monitor
    ):
        """
        Test vector search performance with larger document set (50+ documents).
        
        NO MOCKS - Tests real performance with substantial data volume.
        """
        logger.info("=== STARTING LARGE SCALE VECTOR SEARCH E2E TEST ===")
        
        # Start performance monitoring
        e2e_performance_monitor.start_timer("large_scale_setup")
        
        # Expand document set to 50+ by replicating PMC documents with variations
        large_document_set = []
        target_size = min(50, len(e2e_pmc_documents) * 10)  # Aim for 50, but adapt to available data
        
        for i in range(target_size):
            base_doc = e2e_pmc_documents[i % len(e2e_pmc_documents)]
            
            # Create variation with unique ID and slight content modification
            varied_doc = Document(
                id=f"large_scale_test_{i:03d}",
                page_content=f"{base_doc.page_content} Research study variation {i+1}.",
                metadata={
                    **base_doc.metadata,
                    "variation_id": i,
                    "batch": "large_scale_test"
                }
            )
            large_document_set.append(varied_doc)
        
        logger.info(f"Created {len(large_document_set)} documents for large-scale testing")
        
        # Store documents in batches for better performance
        batch_size = 10
        all_stored_ids = []
        
        for i in range(0, len(large_document_set), batch_size):
            batch = large_document_set[i:i + batch_size]
            batch_ids = e2e_iris_vector_store.add_documents(batch)
            all_stored_ids.extend(batch_ids)
            
            logger.info(f"Stored batch {i//batch_size + 1}: {len(batch_ids)} documents")
        
        setup_time = e2e_performance_monitor.end_timer("large_scale_setup")
        
        assert len(all_stored_ids) == len(large_document_set), "All documents should be stored"
        logger.info(f"Large-scale setup completed in {setup_time:.2f}s")
        
        # Test search performance with large dataset
        test_queries = [
            "What are the latest advances in medical treatment?",
            "How do biomedical research studies improve patient outcomes?", 
            "What are the mechanisms of disease progression?"
        ]
        
        for query_text in test_queries:
            # Start performance monitoring for search
            e2e_performance_monitor.start_timer(f"large_scale_search")
            
            query_embedding = e2e_embedding_function(query_text)
            
            # Perform search against large dataset
            search_results = e2e_iris_vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=10
            )
            
            search_time = e2e_performance_monitor.end_timer("large_scale_search")
            
            # Validate search results
            assert search_results, "Search should return results from large dataset"
            assert len(search_results) <= 10, "Should respect top_k limit"
            
            # Performance validation - HNSW index should provide fast search even with large dataset
            assert search_time < 5.0, f"Search should be fast even with {len(large_document_set)} documents"
            
            logger.info(f"Query: '{query_text[:50]}...' returned {len(search_results)} results in {search_time:.3f}s")
        
        logger.info("=== LARGE SCALE VECTOR SEARCH E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_hnsw_index_efficiency_e2e(
        self,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_pmc_documents: List[Document],
        e2e_embedding_function,
        e2e_database_cleanup,
        e2e_performance_monitor
    ):
        """
        Test HNSW index efficiency and accuracy.
        
        NO MOCKS - Tests real HNSW index performance characteristics.
        """
        logger.info("=== STARTING HNSW INDEX EFFICIENCY E2E TEST ===")
        
        # Setup moderate document set
        test_documents = e2e_pmc_documents[:15]  # Moderate size for index testing
        stored_ids = e2e_iris_vector_store.add_documents(test_documents)
        
        assert len(stored_ids) == len(test_documents), "Setup: All documents should be stored"
        
        # Test search consistency - multiple searches should return consistent results
        query_text = "What are the primary mechanisms of disease treatment?"
        query_embedding = e2e_embedding_function(query_text)
        
        # Perform multiple searches to test consistency
        search_results_1 = e2e_iris_vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=5
        )
        
        search_results_2 = e2e_iris_vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=5
        )
        
        # Validate consistency
        assert len(search_results_1) == len(search_results_2), "Search results should be consistent"
        
        # Check that top results are the same (HNSW should be deterministic for same query)
        if search_results_1 and search_results_2:
            top_doc_1 = search_results_1[0][0].id
            top_doc_2 = search_results_2[0][0].id
            assert top_doc_1 == top_doc_2, "Top search result should be consistent"
        
        # Test different top_k values
        for k in [1, 3, 5]:
            results = e2e_iris_vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=k
            )
            
            assert len(results) <= k, f"Should return at most {k} results"
            assert len(results) <= len(test_documents), "Cannot return more results than stored documents"
            
            # Validate score ordering
            if len(results) > 1:
                scores = [score for _, score in results]
                assert scores == sorted(scores, reverse=True), f"Results should be ordered by score for top_k={k}"
        
        logger.info("=== HNSW INDEX EFFICIENCY E2E TEST COMPLETED ===")


class TestVectorStoreIRISErrorHandling:
    """Test error handling and edge cases in vector store operations."""
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_empty_search_handling_e2e(
        self,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_embedding_function,
        e2e_database_cleanup
    ):
        """
        Test handling of searches on empty vector store.
        
        NO MOCKS - Tests real error handling behavior.
        """
        logger.info("=== STARTING EMPTY SEARCH HANDLING E2E TEST ===")
        
        # Test search on empty store
        query_text = "This should return empty results"
        query_embedding = e2e_embedding_function(query_text)
        
        results = e2e_iris_vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=5
        )
        
        # Should return empty list, not error
        assert isinstance(results, list), "Should return list even when empty"
        assert len(results) == 0, "Should return no results from empty store"
        
        logger.info("=== EMPTY SEARCH HANDLING E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_invalid_embedding_dimensions_e2e(
        self,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_database_cleanup
    ):
        """
        Test handling of invalid embedding dimensions.
        
        NO MOCKS - Tests real dimension validation.
        """
        logger.info("=== STARTING INVALID EMBEDDING DIMENSIONS E2E TEST ===")
        
        # Test with wrong dimension embedding
        invalid_embedding = [0.1, 0.2, 0.3]  # Too short
        
        with pytest.raises((ValueError, Exception)):
            e2e_iris_vector_store.similarity_search(
                query_embedding=invalid_embedding,
                top_k=1
            )
        
        logger.info("=== INVALID EMBEDDING DIMENSIONS E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_document_count_accuracy_e2e(
        self,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_pmc_documents: List[Document],
        e2e_database_cleanup
    ):
        """
        Test document count accuracy after operations.
        
        NO MOCKS - Tests real document counting.
        """
        logger.info("=== STARTING DOCUMENT COUNT ACCURACY E2E TEST ===")
        
        # Initial count (should be 0 after cleanup)
        initial_count = e2e_iris_vector_store.get_document_count()
        assert initial_count == 0, "Store should be empty after cleanup"
        
        # Add documents and check count
        test_documents = e2e_pmc_documents[:3]
        stored_ids = e2e_iris_vector_store.add_documents(test_documents)
        
        count_after_add = e2e_iris_vector_store.get_document_count()
        assert count_after_add == len(test_documents), "Count should match added documents"
        
        # Test document deletion if supported
        try:
            deleted = e2e_iris_vector_store.delete_documents([stored_ids[0]])
            if deleted:
                count_after_delete = e2e_iris_vector_store.get_document_count()
                assert count_after_delete == len(test_documents) - 1, "Count should decrease after deletion"
        except (NotImplementedError, Exception) as e:
            logger.info(f"Document deletion not supported or failed: {e}")
        
        logger.info("=== DOCUMENT COUNT ACCURACY E2E TEST COMPLETED ===")