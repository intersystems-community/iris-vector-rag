#!/usr/bin/env python3
"""
Comprehensive 1000-Document Integration Tests for Chunking and ColBERT Token Embedding Issues

This test suite validates and fixes critical issues with:
1. Chunking system validation across all pipelines
2. ColBERT token embedding auto-population
3. Pipeline fallback behaviors with missing chunks/tokens
4. Ingestion pipeline chunk creation validation
5. ColBERT pipeline integration with 1000+ documents

Following TDD principles: write failing tests first, then implement fixes.
"""

import pytest
import logging
import time
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Import test fixtures for 1000+ document testing
from tests.conftest_1000docs import (
    enterprise_iris_connection,
    scale_test_config,
    enterprise_schema_manager,
    scale_test_documents,
    scale_test_performance_monitor,
    enterprise_test_queries
)

# Import core components
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
from iris_rag.storage.schema_manager import SchemaManager
from tools.chunking.chunking_service import DocumentChunkingService
from common.iris_connection_manager import get_iris_connection

logger = logging.getLogger(__name__)

@pytest.mark.scale_1000
class TestChunkingSystemIntegration1000Docs:
    """Test chunking system integration across all pipelines with 1000+ documents."""
    
    def test_chunking_system_integration_1000_docs(self, enterprise_iris_connection, 
                                                   scale_test_documents, 
                                                   scale_test_config,
                                                   scale_test_performance_monitor):
        """
        CRITICAL TEST: Validate chunking across all pipelines with 1000+ documents.
        
        This test should FAIL initially to expose chunking issues:
        - Test that ALL pipelines properly create and use chunks during ingestion
        - Test that chunk_documents=True actually creates chunks in database
        - Test chunk retrieval service integration across all pipelines
        """
        # Ensure we have sufficient documents for testing
        assert scale_test_documents['document_count'] >= 1000, "Insufficient documents for chunking validation"
        
        config_manager = scale_test_config['config_manager']
        connection = enterprise_iris_connection
        
        # Test chunking with BasicRAGPipeline
        start_time = time.time()
        
        # Initialize pipeline with chunking enabled
        basic_pipeline = BasicRAGPipeline(config_manager)
        
        # Create test documents for chunking
        test_docs = [
            Document(
                page_content="This is a long test document that should be chunked into smaller pieces. " * 50,
                metadata={"source": "test_chunking_1", "doc_type": "test"}
            ),
            Document(
                page_content="Another long document for chunking validation testing. " * 50,
                metadata={"source": "test_chunking_2", "doc_type": "test"}
            )
        ]
        
        # Test ingestion with chunking enabled (should create chunks)
        basic_pipeline.load_documents("", documents=test_docs, chunk_documents=True)
        
        # Verify chunks were actually created in database
        cursor = connection.cursor()
        try:
            # Check if DocumentChunks table exists and has data
            # Look for chunks from our test documents by checking chunk_text content
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE chunk_text LIKE '%test_chunking%'")
            chunk_count = cursor.fetchone()[0]
            
            # THIS SHOULD FAIL - exposing that chunks are not being created properly
            assert chunk_count > 0, "CRITICAL ISSUE: No chunks created despite chunk_documents=True"
            assert chunk_count >= len(test_docs), "CRITICAL ISSUE: Insufficient chunks created"
            
            # Verify chunk metadata is properly stored
            cursor.execute("""
                SELECT chunk_index, parent_document_id, chunk_size 
                FROM RAG.DocumentChunks 
                WHERE source LIKE 'test_chunking_%'
                ORDER BY chunk_index
            """)
            chunk_metadata = cursor.fetchall()
            
            # Validate chunk structure
            assert len(chunk_metadata) > 0, "CRITICAL ISSUE: No chunk metadata found"
            
            for chunk_meta in chunk_metadata:
                chunk_index, parent_doc_id, chunk_size = chunk_meta
                assert chunk_index is not None, "CRITICAL ISSUE: Missing chunk_index"
                assert parent_doc_id is not None, "CRITICAL ISSUE: Missing parent_document_id"
                assert chunk_size > 0, "CRITICAL ISSUE: Invalid chunk_size"
                
        finally:
            cursor.close()
        
        # Test retrieval uses chunks properly
        query_result = basic_pipeline.execute("test document chunking")
        
        # Verify retrieved documents are chunks, not full documents
        retrieved_docs = query_result.get("retrieved_documents", [])
        assert len(retrieved_docs) > 0, "CRITICAL ISSUE: No documents retrieved"
        
        # Check if retrieved documents have chunk metadata
        chunk_docs_found = False
        for doc in retrieved_docs:
            if "chunk_index" in doc.metadata:
                chunk_docs_found = True
                break
        
        # THIS SHOULD FAIL - exposing that retrieval doesn't use chunks
        assert chunk_docs_found, "CRITICAL ISSUE: Retrieved documents are not chunks"
        
        scale_test_performance_monitor['record_operation'](
            "chunking_system_validation", 
            time.time() - start_time,
            chunks_created=chunk_count,
            documents_processed=len(test_docs)
        )

    def test_ingestion_pipeline_chunk_creation_validation(self, enterprise_iris_connection,
                                                         scale_test_config,
                                                         enterprise_schema_manager):
        """
        CRITICAL TEST: Ensure ingestion pipeline creates chunks by default and errors if it fails.
        
        This test should FAIL initially to expose ingestion issues.
        """
        config_manager = scale_test_config['config_manager']
        connection = enterprise_iris_connection
        
        # Ensure DocumentChunks table exists
        enterprise_schema_manager.ensure_table_schema("DocumentChunks")
        
        # Test with chunking service directly
        chunking_service = DocumentChunkingService()
        
        test_document = Document(
            page_content="This is a comprehensive test document for chunk creation validation. " * 100,
            metadata={"source": "ingestion_test", "doc_type": "validation"}
        )
        
        # Test chunking service creates chunks
        chunks = chunking_service.chunk_document(
            doc_id="test_doc_1",
            text=test_document.page_content,
            strategy_name="fixed_size"
        )
        
        # THIS SHOULD FAIL - exposing chunking service issues
        assert len(chunks) > 1, "CRITICAL ISSUE: Chunking service not creating multiple chunks"
        
        # Verify chunk properties (chunks are dictionaries, not objects)
        for chunk in chunks:
            assert 'chunk_text' in chunk, "CRITICAL ISSUE: Chunk missing chunk_text attribute"
            assert 'chunk_metadata' in chunk, "CRITICAL ISSUE: Chunk missing chunk_metadata attribute"
            assert len(chunk['chunk_text']) > 0, "CRITICAL ISSUE: Empty chunk text"
        
        # Test ingestion pipeline integration
        pipeline = BasicRAGPipeline(config_manager)
        
        # Mock the chunking to force failure scenario
        with patch.object(pipeline, '_chunk_documents') as mock_chunk:
            mock_chunk.side_effect = Exception("Chunking failed")
            
            # This should handle chunking failure gracefully
            try:
                pipeline.load_documents("", documents=[test_document], chunk_documents=True)
                # THIS SHOULD FAIL - pipeline should error when chunking fails
                assert False, "CRITICAL ISSUE: Pipeline should fail when chunking fails"
            except Exception as e:
                # Verify proper error handling
                assert "chunking" in str(e).lower() or "chunk" in str(e).lower(), \
                    "CRITICAL ISSUE: Error message doesn't indicate chunking failure"

    def test_pipeline_fallback_behaviors_with_missing_chunks(self, enterprise_iris_connection,
                                                           scale_test_config,
                                                           scale_test_documents):
        """
        CRITICAL TEST: Test graceful degradation when chunks are missing.
        
        This test should FAIL initially to expose fallback behavior issues.
        """
        config_manager = scale_test_config['config_manager']
        connection = enterprise_iris_connection
        
        # Ensure we have documents but simulate missing chunks
        assert scale_test_documents['document_count'] >= 1000
        
        pipeline = BasicRAGPipeline(config_manager)
        
        # Clear any existing chunks to simulate missing chunk scenario
        cursor = connection.cursor()
        try:
            cursor.execute("DELETE FROM RAG.DocumentChunks WHERE source LIKE 'fallback_test_%'")
            connection.commit()
        except Exception:
            pass  # Table might not exist
        finally:
            cursor.close()
        
        # Test document without chunks
        test_doc = Document(
            page_content="Test document for fallback behavior validation.",
            metadata={"source": "fallback_test_1", "doc_type": "test"}
        )
        
        # Load document without chunking
        pipeline.load_documents("", documents=[test_doc], chunk_documents=False)
        
        # Now try to query - should fall back to full document retrieval
        result = pipeline.execute("test document fallback")
        
        # THIS SHOULD FAIL - exposing that fallback doesn't work properly
        assert "answer" in result, "CRITICAL ISSUE: Pipeline doesn't provide fallback answer"
        assert len(result.get("retrieved_documents", [])) > 0, \
            "CRITICAL ISSUE: No fallback document retrieval"
        
        # Verify fallback uses full documents when chunks unavailable
        retrieved_docs = result["retrieved_documents"]
        full_doc_found = False
        for doc in retrieved_docs:
            if "chunk_index" not in doc.metadata and doc.page_content == test_doc.page_content:
                full_doc_found = True
                break
        
        # THIS SHOULD FAIL - exposing fallback mechanism issues
        assert full_doc_found, "CRITICAL ISSUE: Fallback doesn't retrieve full documents"


@pytest.mark.scale_1000
class TestColBERTTokenEmbeddingIntegration1000Docs:
    """Test ColBERT token embedding integration with 1000+ documents."""
    
    def test_colbert_token_embedding_auto_population(self, enterprise_iris_connection,
                                                    scale_test_config,
                                                    enterprise_schema_manager,
                                                    scale_test_documents):
        """
        CRITICAL TEST: Validate automatic token embedding creation during ingestion.
        
        This test should FAIL initially to expose token embedding issues.
        """
        # Ensure we have sufficient documents
        assert scale_test_documents['document_count'] >= 1000
        
        config_manager = scale_test_config['config_manager']
        connection = enterprise_iris_connection
        
        # Ensure DocumentTokenEmbeddings table exists
        enterprise_schema_manager.ensure_table_schema("DocumentTokenEmbeddings")
        
        # Test ColBERT pipeline initialization
        try:
            colbert_pipeline = ColBERTRAGPipeline(
                iris_connector=connection,
                config_manager=config_manager
            )
        except Exception as e:
            # THIS SHOULD FAIL - exposing ColBERT initialization issues
            pytest.fail(f"CRITICAL ISSUE: ColBERT pipeline initialization failed: {e}")
        
        # Test token embedding dimension consistency
        doc_dim = enterprise_schema_manager.get_vector_dimension("SourceDocuments")
        token_dim = enterprise_schema_manager.get_colbert_token_dimension()
        
        # THIS SHOULD FAIL - exposing dimension inconsistency
        assert doc_dim == 384, f"CRITICAL ISSUE: Document embedding dimension should be 384D, got {doc_dim}D"
        assert token_dim == 768, f"CRITICAL ISSUE: Token embedding dimension should be 768D, got {token_dim}D"
        
        # Test document ingestion creates token embeddings
        test_doc = Document(
            page_content="This is a test document for ColBERT token embedding validation. " * 20,
            metadata={"source": "colbert_test_1", "doc_type": "test"}
        )
        
        # Load document and verify token embeddings are created
        colbert_pipeline.load_documents("", documents=[test_doc])
        
        # Check if token embeddings were created
        cursor = connection.cursor()
        try:
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings 
                WHERE document_id IN (
                    SELECT id FROM RAG.SourceDocuments 
                    WHERE source = 'colbert_test_1'
                )
            """)
            token_count = cursor.fetchone()[0]
            
            # THIS SHOULD FAIL - exposing that token embeddings aren't auto-created
            assert token_count > 0, "CRITICAL ISSUE: No token embeddings created during ingestion"
            
            # Verify token embedding dimensions
            cursor.execute("""
                SELECT token_embedding FROM RAG.DocumentTokenEmbeddings 
                WHERE document_id IN (
                    SELECT id FROM RAG.SourceDocuments 
                    WHERE source = 'colbert_test_1'
                )
                LIMIT 1
            """)
            token_embedding_row = cursor.fetchone()
            
            if token_embedding_row:
                # Parse the vector to check dimensions
                token_embedding = token_embedding_row[0]
                # THIS SHOULD FAIL - exposing dimension issues
                assert token_embedding is not None, "CRITICAL ISSUE: Token embedding is NULL"
                
        finally:
            cursor.close()

    def test_colbert_pipeline_1000_docs_with_tokens(self, enterprise_iris_connection,
                                                   scale_test_config,
                                                   scale_test_documents,
                                                   enterprise_test_queries,
                                                   scale_test_performance_monitor):
        """
        CRITICAL TEST: Full ColBERT workflow validation with 1000+ documents.
        
        This test should FAIL initially to expose end-to-end ColBERT issues.
        """
        # Ensure we have sufficient documents
        assert scale_test_documents['document_count'] >= 1000
        
        config_manager = scale_test_config['config_manager']
        connection = enterprise_iris_connection
        
        start_time = time.time()
        
        # Initialize ColBERT pipeline
        try:
            colbert_pipeline = ColBERTRAGPipeline(
                iris_connector=connection,
                config_manager=config_manager
            )
        except Exception as e:
            pytest.fail(f"CRITICAL ISSUE: ColBERT pipeline initialization failed: {e}")
        
        # Validate setup
        setup_valid = colbert_pipeline.validate_setup()
        # THIS SHOULD FAIL - exposing setup validation issues
        assert setup_valid, "CRITICAL ISSUE: ColBERT setup validation failed"
        
        # Test query execution with token embeddings
        test_query = enterprise_test_queries[0]['query']
        
        try:
            result = colbert_pipeline.execute(test_query, top_k=5)
        except Exception as e:
            # THIS SHOULD FAIL - exposing query execution issues
            pytest.fail(f"CRITICAL ISSUE: ColBERT query execution failed: {e}")
        
        # Validate result structure
        assert "query" in result, "CRITICAL ISSUE: Missing query in result"
        assert "answer" in result, "CRITICAL ISSUE: Missing answer in result"
        assert "retrieved_documents" in result, "CRITICAL ISSUE: Missing retrieved_documents in result"
        assert "technique" in result, "CRITICAL ISSUE: Missing technique in result"
        
        # Verify ColBERT-specific results
        assert result["technique"] == "ColBERT", "CRITICAL ISSUE: Wrong technique reported"
        assert "token_count" in result, "CRITICAL ISSUE: Missing token_count in result"
        assert result["token_count"] > 0, "CRITICAL ISSUE: No tokens used in query"
        
        # Verify retrieved documents
        retrieved_docs = result["retrieved_documents"]
        assert len(retrieved_docs) > 0, "CRITICAL ISSUE: No documents retrieved"
        assert len(retrieved_docs) <= 5, "CRITICAL ISSUE: Too many documents retrieved"
        
        # Test performance with 1000+ documents
        execution_time = result.get("execution_time", 0)
        # THIS SHOULD FAIL - exposing performance issues
        assert execution_time < 30.0, f"CRITICAL ISSUE: ColBERT query too slow: {execution_time}s"
        
        scale_test_performance_monitor['record_operation'](
            "colbert_1000_docs_query",
            execution_time,
            documents_available=scale_test_documents['document_count'],
            documents_retrieved=len(retrieved_docs),
            token_count=result["token_count"]
        )

    def test_colbert_dimension_consistency_validation(self, enterprise_iris_connection,
                                                     scale_test_config,
                                                     enterprise_schema_manager):
        """
        CRITICAL TEST: Validate ColBERT dimension consistency (768D tokens vs 384D documents).
        
        This test should FAIL initially to expose dimension inconsistency issues.
        """
        config_manager = scale_test_config['config_manager']
        
        # Get dimensions from schema manager
        doc_dimension = enterprise_schema_manager.get_vector_dimension("SourceDocuments")
        token_dimension = enterprise_schema_manager.get_colbert_token_dimension()
        
        # THIS SHOULD FAIL - exposing dimension configuration issues
        assert doc_dimension == 384, f"CRITICAL ISSUE: Document dimension should be 384D, got {doc_dimension}D"
        assert token_dimension == 768, f"CRITICAL ISSUE: Token dimension should be 768D, got {token_dimension}D"
        
        # Test ColBERT interface dimension consistency
        try:
            from iris_rag.embeddings.colbert_interface import get_colbert_interface_from_config
            colbert_interface = get_colbert_interface_from_config(config_manager, enterprise_iris_connection)
            
            # Test query encoding produces correct dimensions
            test_tokens = colbert_interface.encode_query("test query")
            
            # THIS SHOULD FAIL - exposing interface dimension issues
            assert len(test_tokens) > 0, "CRITICAL ISSUE: No tokens generated"
            assert len(test_tokens[0]) == token_dimension, \
                f"CRITICAL ISSUE: Token dimension mismatch: expected {token_dimension}D, got {len(test_tokens[0])}D"
                
        except Exception as e:
            # THIS SHOULD FAIL - exposing ColBERT interface issues
            pytest.fail(f"CRITICAL ISSUE: ColBERT interface validation failed: {e}")


@pytest.mark.scale_1000
class TestIntegratedPipelineValidation1000Docs:
    """Test integrated pipeline validation with chunking and token embeddings."""
    
    def test_end_to_end_pipeline_integration_1000_docs(self, enterprise_iris_connection,
                                                       scale_test_config,
                                                       scale_test_documents,
                                                       enterprise_test_queries,
                                                       scale_test_performance_monitor):
        """
        CRITICAL TEST: End-to-end validation of ingestion → chunking → embedding → retrieval.
        
        This test should FAIL initially to expose integration issues.
        """
        # Ensure we have sufficient documents
        assert scale_test_documents['document_count'] >= 1000
        
        config_manager = scale_test_config['config_manager']
        connection = enterprise_iris_connection
        
        start_time = time.time()
        
        # Test complete pipeline workflow
        test_docs = [
            Document(
                page_content="Comprehensive integration test document for end-to-end validation. " * 30,
                metadata={"source": "e2e_test_1", "doc_type": "integration"}
            ),
            Document(
                page_content="Second document for complete pipeline testing and validation. " * 30,
                metadata={"source": "e2e_test_2", "doc_type": "integration"}
            )
        ]
        
        # Test Basic RAG pipeline with chunking
        basic_pipeline = BasicRAGPipeline(config_manager)
        basic_pipeline.load_documents("", documents=test_docs, chunk_documents=True)
        
        # Verify chunks were created
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE source LIKE 'e2e_test_%'")
            chunk_count = cursor.fetchone()[0]
            # THIS SHOULD FAIL - exposing chunking integration issues
            assert chunk_count > 0, "CRITICAL ISSUE: No chunks created in end-to-end test"
        finally:
            cursor.close()
        
        # Test ColBERT pipeline with token embeddings
        try:
            colbert_pipeline = ColBERTRAGPipeline(
                iris_connector=connection,
                config_manager=config_manager
            )
            colbert_pipeline.load_documents("", documents=test_docs)
            
            # Verify token embeddings were created
            cursor = connection.cursor()
            try:
                cursor.execute("""
                    SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings 
                    WHERE document_id IN (
                        SELECT id FROM RAG.SourceDocuments 
                        WHERE source LIKE 'e2e_test_%'
                    )
                """)
                token_count = cursor.fetchone()[0]
                # THIS SHOULD FAIL - exposing token embedding integration issues
                assert token_count > 0, "CRITICAL ISSUE: No token embeddings created in end-to-end test"
            finally:
                cursor.close()
                
        except Exception as e:
            # THIS SHOULD FAIL - exposing ColBERT integration issues
            pytest.fail(f"CRITICAL ISSUE: ColBERT integration failed: {e}")
        
        # Test query execution across both pipelines
        test_query = "integration test document"
        
        # Basic RAG query
        basic_result = basic_pipeline.execute(test_query)
        assert "answer" in basic_result, "CRITICAL ISSUE: Basic RAG missing answer"
        assert len(basic_result["retrieved_documents"]) > 0, "CRITICAL ISSUE: Basic RAG no retrieval"
        
        # ColBERT query
        colbert_result = colbert_pipeline.execute(test_query)
        assert "answer" in colbert_result, "CRITICAL ISSUE: ColBERT missing answer"
        assert len(colbert_result["retrieved_documents"]) > 0, "CRITICAL ISSUE: ColBERT no retrieval"
        
        # Compare results
        basic_docs = basic_result["retrieved_documents"]
        colbert_docs = colbert_result["retrieved_documents"]
        
        # THIS SHOULD FAIL - exposing result quality issues
        assert len(basic_docs) > 0 and len(colbert_docs) > 0, \
            "CRITICAL ISSUE: One or both pipelines returned no results"
        
        integration_time = time.time() - start_time
        
        scale_test_performance_monitor['record_operation'](
            "end_to_end_integration",
            integration_time,
            basic_docs_retrieved=len(basic_docs),
            colbert_docs_retrieved=len(colbert_docs),
            chunks_created=chunk_count,
            tokens_created=token_count
        )


if __name__ == "__main__":
    # Run the tests to establish the red state (failing tests)
    pytest.main([__file__, "-v", "--tb=short"])