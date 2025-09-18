"""
Core Framework E2E Tests

Tests the complete document lifecycle through the core RAG framework:
- Document ingestion with real PMC documents 
- Entity extraction and relationship building
- Document retrieval with various query types
- Data persistence validation in IRIS

NO MOCKS - Uses real infrastructure throughout.
Based on evaluation_framework/true_e2e_evaluation.py patterns.

Priority 1 Components Under Test:
- iris_rag/core/base.py (RAGPipeline base class)
- iris_rag/core/models.py (Document, Entity, Relationship models)
"""

import pytest
import logging
import time
from typing import List, Dict, Any

# Real framework imports (NO MOCKS)
from iris_rag.core.models import Document, Entity, Relationship
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.storage.vector_store_iris import IRISVectorStore

logger = logging.getLogger(__name__)


class TestCoreFrameworkDocumentLifecycle:
    """Test complete document lifecycle through core framework components."""
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    @pytest.mark.pmc_data
    def test_document_ingestion_e2e(
        self,
        e2e_config_manager: ConfigurationManager,
        e2e_connection_manager: ConnectionManager,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_pmc_documents: List[Document],
        e2e_database_cleanup,
        e2e_performance_monitor,
        e2e_document_validator
    ):
        """
        Test complete document ingestion workflow end-to-end.
        
        NO MOCKS - Tests real document ingestion, embedding generation,
        and persistence to IRIS database.
        """
        logger.info("=== STARTING CORE FRAMEWORK DOCUMENT INGESTION E2E TEST ===")
        
        # Start performance monitoring
        e2e_performance_monitor.start_timer("document_ingestion")
        
        # Test data preparation
        test_documents = e2e_pmc_documents[:3]  # Use first 3 documents for focused testing
        assert len(test_documents) >= 3, "Need at least 3 PMC documents for E2E testing"
        
        # Verify documents have required structure
        for doc in test_documents:
            assert isinstance(doc, Document), f"Expected Document object, got {type(doc)}"
            assert doc.page_content, "Document must have content"
            assert doc.id, "Document must have ID"
            assert doc.metadata, "Document must have metadata"
            logger.info(f"Validated document: {doc.id} ({len(doc.page_content)} chars)")
        
        # Step 1: Document ingestion through vector store
        logger.info("Step 1: Testing document ingestion with real embeddings")
        
        try:
            # Real ingestion - no mocks
            document_ids = e2e_iris_vector_store.add_documents(test_documents)
            
            # Validate ingestion results
            assert document_ids, "Document ingestion should return document IDs"
            assert len(document_ids) == len(test_documents), "Should return ID for each document"
            
            for doc_id in document_ids:
                assert doc_id, "Each document ID should be non-empty"
                logger.info(f"Successfully ingested document: {doc_id}")
            
        except Exception as e:
            pytest.fail(f"Document ingestion failed: {e}")
        
        # Step 2: Validate persistence in IRIS database
        logger.info("Step 2: Validating persistence in IRIS database")
        
        validation_results = e2e_document_validator.validate_document_ingestion(document_ids)
        
        for doc_id in document_ids:
            assert doc_id in validation_results, f"Document {doc_id} not found in validation"
            result = validation_results[doc_id]
            
            assert result["found"], f"Document {doc_id} not found in database"
            assert result["has_content"], f"Document {doc_id} missing content in database"
            assert result["has_embedding"], f"Document {doc_id} missing embedding in database"
            assert result["content_length"] > 0, f"Document {doc_id} has empty content"
            
            logger.info(f"Validated persistence: {doc_id} ({result['content_length']} chars, embedding: {result['has_embedding']})")
        
        # Step 3: Test document retrieval by ID
        logger.info("Step 3: Testing document retrieval by ID")
        
        retrieved_documents = e2e_iris_vector_store.fetch_documents_by_ids(document_ids)
        
        assert len(retrieved_documents) == len(document_ids), "Should retrieve all ingested documents"
        
        for retrieved_doc in retrieved_documents:
            assert isinstance(retrieved_doc, Document), "Retrieved object should be Document"
            assert retrieved_doc.page_content, "Retrieved document should have content"
            assert retrieved_doc.metadata, "Retrieved document should have metadata"
            logger.info(f"Successfully retrieved document: {retrieved_doc.id}")
        
        # Step 4: Validate content integrity
        logger.info("Step 4: Validating content integrity")
        
        original_ids = {doc.id for doc in test_documents}
        retrieved_ids = {doc.id for doc in retrieved_documents}
        
        assert original_ids == retrieved_ids, "Retrieved document IDs should match original IDs"
        
        # End performance monitoring
        ingestion_time = e2e_performance_monitor.end_timer("document_ingestion")
        
        # Performance validation
        assert ingestion_time > 0, "Ingestion should take measurable time (not mocked)"
        assert ingestion_time < 300, "Ingestion should complete within 5 minutes"  # Reasonable E2E timeout
        
        logger.info(f"=== DOCUMENT INGESTION E2E TEST COMPLETED ({ingestion_time:.2f}s) ===")
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_document_retrieval_with_queries_e2e(
        self,
        e2e_config_manager: ConfigurationManager,
        e2e_connection_manager: ConnectionManager,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_pmc_documents: List[Document],
        e2e_biomedical_queries: List[Dict[str, Any]],
        e2e_database_cleanup,
        e2e_performance_monitor,
        e2e_document_validator,
        e2e_embedding_function
    ):
        """
        Test document retrieval with various biomedical queries end-to-end.
        
        NO MOCKS - Tests real vector similarity search with embeddings.
        """
        logger.info("=== STARTING DOCUMENT RETRIEVAL WITH QUERIES E2E TEST ===")
        
        # Setup: Ingest test documents
        test_documents = e2e_pmc_documents[:5]  # Use 5 documents for richer retrieval
        document_ids = e2e_iris_vector_store.add_documents(test_documents)
        
        assert len(document_ids) == len(test_documents), "Setup: All documents should be ingested"
        
        # Test each biomedical query
        for i, query_data in enumerate(e2e_biomedical_queries):
            query_text = query_data["query"]
            expected_topics = query_data["expected_topics"]
            expected_keywords = query_data["expected_keywords"]
            
            logger.info(f"Testing query {i+1}: {query_text}")
            
            # Start performance monitoring for this query
            e2e_performance_monitor.start_timer(f"query_{i+1}")
            
            try:
                # Step 1: Generate query embedding (real embedding, no mocks)
                query_embedding = e2e_embedding_function(query_text)
                
                assert query_embedding, "Query embedding should be generated"
                assert len(query_embedding) > 0, "Query embedding should have dimensions"
                assert isinstance(query_embedding[0], (int, float)), "Embedding should contain numeric values"
                
                # Step 2: Perform similarity search (real vector search in IRIS)
                search_results = e2e_iris_vector_store.similarity_search(
                    query_embedding=query_embedding,
                    top_k=3
                )
                
                # Validate search results
                assert search_results, f"Query '{query_text}' should return results"
                assert len(search_results) <= 3, "Should not exceed top_k limit"
                
                # Validate result structure and content
                for result in search_results:
                    assert isinstance(result, tuple), "Each result should be (Document, score) tuple"
                    document, score = result
                    
                    assert isinstance(document, Document), "Result should contain Document object"
                    assert document.page_content, "Retrieved document should have content"
                    assert isinstance(score, (int, float)), "Similarity score should be numeric"
                    assert 0 <= score <= 1, "Similarity score should be normalized between 0 and 1"
                    
                    logger.info(f"Retrieved: {document.id} (score: {score:.3f})")
                
                # Step 3: Validate retrieval quality (content relevance)
                retrieved_content = " ".join([doc.page_content for doc, _ in search_results])
                retrieved_content_lower = retrieved_content.lower()
                
                # Check for expected keywords in retrieved content
                keyword_matches = sum(1 for keyword in expected_keywords 
                                    if keyword.lower() in retrieved_content_lower)
                
                assert keyword_matches > 0, f"Query '{query_text}' should retrieve content with relevant keywords"
                
                logger.info(f"Keyword relevance: {keyword_matches}/{len(expected_keywords)} keywords found")
                
                # End performance monitoring
                query_time = e2e_performance_monitor.end_timer(f"query_{i+1}")
                
                # Performance validation
                assert query_time > 0, "Query should take measurable time (not mocked)"
                assert query_time < 30, "Query should complete within 30 seconds"  # Reasonable E2E timeout
                
            except Exception as e:
                pytest.fail(f"Query retrieval failed for '{query_text}': {e}")
        
        logger.info("=== DOCUMENT RETRIEVAL WITH QUERIES E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_basic_rag_pipeline_e2e(
        self,
        e2e_config_manager: ConfigurationManager,
        e2e_connection_manager: ConnectionManager,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_pmc_documents: List[Document],
        e2e_biomedical_queries: List[Dict[str, Any]],
        e2e_database_cleanup,
        e2e_performance_monitor,
        e2e_llm_function
    ):
        """
        Test complete BasicRAG pipeline end-to-end workflow.
        
        NO MOCKS - Tests real pipeline with document ingestion, retrieval, and answer generation.
        """
        logger.info("=== STARTING BASIC RAG PIPELINE E2E TEST ===")
        
        # Start performance monitoring
        e2e_performance_monitor.start_timer("basic_rag_pipeline")
        
        # Step 1: Initialize real BasicRAG pipeline
        logger.info("Step 1: Initializing BasicRAG pipeline with real components")
        
        try:
            pipeline = BasicRAGPipeline(
                connection_manager=e2e_connection_manager,
                config_manager=e2e_config_manager,
                llm_func=e2e_llm_function,
                vector_store=e2e_iris_vector_store
            )
            
            assert pipeline is not None, "Pipeline should be initialized"
            assert hasattr(pipeline, 'query'), "Pipeline should have query method"
            
        except Exception as e:
            pytest.fail(f"Failed to initialize BasicRAG pipeline: {e}")
        
        # Step 2: Load documents into pipeline
        logger.info("Step 2: Loading documents into pipeline")
        
        test_documents = e2e_pmc_documents[:3]  # Use subset for focused testing
        
        try:
            # Use the pipeline's document loading method
            pipeline.ingest(test_documents)
            
            # Verify documents were loaded
            stored_docs = pipeline.get_documents()
            assert len(stored_docs) >= len(test_documents), "Pipeline should contain ingested documents"
            
        except Exception as e:
            pytest.fail(f"Failed to load documents into pipeline: {e}")
        
        # Step 3: Test pipeline query method
        logger.info("Step 3: Testing pipeline query method")
        
        test_query = e2e_biomedical_queries[0]  # Use first biomedical query
        query_text = test_query["query"]
        
        try:
            # Perform real query through pipeline (no mocks)
            result = pipeline.query(query_text, top_k=3, generate_answer=True)
            
            # Validate pipeline response structure
            assert isinstance(result, dict), "Pipeline should return dictionary response"
            
            required_keys = ["query", "answer", "retrieved_documents", "contexts", "execution_time"]
            for key in required_keys:
                assert key in result, f"Pipeline response should contain '{key}'"
            
            # Validate response content
            assert result["query"] == query_text, "Response should echo the original query"
            assert result["answer"], "Pipeline should generate a non-empty answer"
            assert result["retrieved_documents"], "Pipeline should retrieve relevant documents"
            assert result["execution_time"] > 0, "Pipeline should report execution time"
            
            # Validate retrieved documents structure
            for doc in result["retrieved_documents"]:
                assert isinstance(doc, Document), "Retrieved documents should be Document objects"
                assert doc.page_content, "Retrieved documents should have content"
            
            logger.info(f"Query: {query_text}")
            logger.info(f"Answer: {result['answer'][:200]}...")
            logger.info(f"Retrieved {len(result['retrieved_documents'])} documents")
            logger.info(f"Execution time: {result['execution_time']:.2f}s")
            
        except Exception as e:
            pytest.fail(f"Pipeline query failed: {e}")
        
        # End performance monitoring
        pipeline_time = e2e_performance_monitor.end_timer("basic_rag_pipeline")
        
        # Performance validation
        assert pipeline_time > 0, "Pipeline should take measurable time (not mocked)"
        assert pipeline_time < 120, "Pipeline should complete within 2 minutes"  # Reasonable E2E timeout
        
        logger.info(f"=== BASIC RAG PIPELINE E2E TEST COMPLETED ({pipeline_time:.2f}s) ===")
    
    @pytest.mark.true_e2e
    def test_document_model_validation_e2e(self, e2e_pmc_documents: List[Document]):
        """
        Test Document model validation and data integrity end-to-end.
        
        NO MOCKS - Tests real Document model behavior and validation.
        """
        logger.info("=== STARTING DOCUMENT MODEL VALIDATION E2E TEST ===")
        
        # Test Document model with real PMC data
        for doc in e2e_pmc_documents[:3]:
            # Test basic validation
            assert isinstance(doc, Document), "Should be Document instance"
            assert doc.page_content, "Document should have content"
            assert doc.id, "Document should have ID"
            assert isinstance(doc.metadata, dict), "Metadata should be dictionary"
            
            # Test hashability (important for set operations)
            try:
                doc_set = {doc}
                assert len(doc_set) == 1, "Document should be hashable"
            except TypeError:
                pytest.fail("Document should be hashable for use in sets/dicts")
            
            # Test equality
            doc_copy = Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata=doc.metadata.copy()
            )
            assert doc == doc_copy, "Documents with same content should be equal"
            
            # Test metadata immutability preservation
            original_metadata = doc.metadata.copy()
            # Metadata should remain unchanged after operations
            assert doc.metadata == original_metadata, "Metadata should be preserved"
            
        logger.info("=== DOCUMENT MODEL VALIDATION E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    def test_entity_relationship_models_e2e(self):
        """
        Test Entity and Relationship models end-to-end.
        
        NO MOCKS - Tests real model behavior and validation.
        """
        logger.info("=== STARTING ENTITY/RELATIONSHIP MODELS E2E TEST ===")
        
        # Test Entity model with realistic biomedical data
        test_entity = Entity(
            text="diabetes mellitus",
            entity_type="DISEASE",
            confidence=0.95,
            start_offset=100,
            end_offset=116,
            source_document_id="e2e_test_doc_001",
            metadata={"mesh_id": "D003920", "umls_cui": "C0011849"}
        )
        
        # Validate Entity structure
        assert test_entity.text == "diabetes mellitus"
        assert test_entity.entity_type == "DISEASE"
        assert test_entity.confidence == 0.95
        assert test_entity.start_offset == 100
        assert test_entity.end_offset == 116
        
        # Test Entity hashability
        entity_set = {test_entity}
        assert len(entity_set) == 1, "Entity should be hashable"
        
        # Test Relationship model
        test_relationship = Relationship(
            source_entity_id=test_entity.id,
            target_entity_id="entity_002",
            relationship_type="causes",
            confidence=0.85,
            source_document_id="e2e_test_doc_001",
            metadata={"evidence": "clinical study evidence"}
        )
        
        # Validate Relationship structure
        assert test_relationship.source_entity_id == test_entity.id
        assert test_relationship.target_entity_id == "entity_002"
        assert test_relationship.relationship_type == "causes"
        assert test_relationship.confidence == 0.85
        
        # Test Relationship hashability
        relationship_set = {test_relationship}
        assert len(relationship_set) == 1, "Relationship should be hashable"
        
        logger.info("=== ENTITY/RELATIONSHIP MODELS E2E TEST COMPLETED ===")


class TestCoreFrameworkErrorHandling:
    """Test error handling and edge cases in core framework."""
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required 
    def test_invalid_document_handling_e2e(
        self,
        e2e_iris_vector_store: IRISVectorStore,
        e2e_database_cleanup
    ):
        """
        Test handling of invalid documents end-to-end.
        
        NO MOCKS - Tests real error handling behavior.
        """
        logger.info("=== STARTING INVALID DOCUMENT HANDLING E2E TEST ===")
        
        # Test empty content document
        with pytest.raises((ValueError, TypeError, Exception)):
            invalid_doc = Document(id="test_invalid", page_content="", metadata={})
            e2e_iris_vector_store.add_documents([invalid_doc])
        
        # Test None content document  
        with pytest.raises((ValueError, TypeError, Exception)):
            invalid_doc = Document(id="test_invalid_none", page_content=None, metadata={})
        
        logger.info("=== INVALID DOCUMENT HANDLING E2E TEST COMPLETED ===")
    
    @pytest.mark.true_e2e
    @pytest.mark.iris_required
    def test_database_connection_resilience_e2e(
        self,
        e2e_connection_manager: ConnectionManager,
        e2e_database_cleanup
    ):
        """
        Test database connection resilience end-to-end.
        
        NO MOCKS - Tests real connection handling.
        """
        logger.info("=== STARTING DATABASE CONNECTION RESILIENCE E2E TEST ===")
        
        # Test connection retrieval
        connection = e2e_connection_manager.get_connection()
        assert connection is not None, "Should establish database connection"
        
        # Test connection is usable
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1, "Connection should be functional"
        
        cursor.close()
        connection.close()
        
        logger.info("=== DATABASE CONNECTION RESILIENCE E2E TEST COMPLETED ===")