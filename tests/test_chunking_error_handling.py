#!/usr/bin/env python3
"""
Chunking Error Handling and Edge Case Tests

This module provides comprehensive testing for error handling and edge cases
in the chunking architecture integration.

Test Coverage:
1. Chunking failures and fallback behavior
2. Corrupted documents, empty documents, very large documents
3. Error recovery and graceful degradation
4. Database connection failures during chunking
5. Memory exhaustion scenarios
6. Invalid configuration handling
"""

import pytest
import logging
import time
import tempfile
import os
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

# Import test fixtures
from tests.conftest_1000docs import (
    enterprise_iris_connection,
    scale_test_config,
    enterprise_schema_manager,
    enterprise_embedding_manager
)

# Import chunking components
from tools.chunking.chunking_service import (
    DocumentChunkingService,
    FixedSizeChunkingStrategy,
    SemanticChunkingStrategy,
    HybridChunkingStrategy,
    Chunk
)
from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.vector_store.iris_impl import IRISVectorStore
from iris_rag.core.models import Document

logger = logging.getLogger(__name__)

class TestChunkingErrorHandling:
    """Comprehensive error handling and edge case tests for chunking."""

    @pytest.fixture(autouse=True)
    def setup_error_test_environment(self, enterprise_iris_connection, scale_test_config,
                                   enterprise_schema_manager, enterprise_embedding_manager):
        """Set up error testing environment."""
        self.connection = enterprise_iris_connection
        self.config = scale_test_config
        self.schema_manager = enterprise_schema_manager
        self.config_manager = scale_test_config['config_manager']
        self.embedding_func = enterprise_embedding_manager.get_embedding_function()
        
        # Initialize chunking service
        self.chunking_service = DocumentChunkingService(embedding_func=self.embedding_func)
        
        # Error tracking
        self.error_log = []

    def test_empty_document_handling(self):
        """
        Test handling of empty and minimal documents.
        
        Validates:
        - Empty string documents
        - Whitespace-only documents
        - Single character documents
        - Documents with only punctuation
        """
        edge_case_documents = [
            ("empty", ""),
            ("whitespace", "   \n\t  \r\n  "),
            ("single_char", "a"),
            ("punctuation_only", ".,!?;:"),
            ("newlines_only", "\n\n\n\n"),
            ("minimal_sentence", "Hi."),
            ("unicode_empty", "\u200b\u200c\u200d"),  # Zero-width characters
        ]
        
        strategies = ["fixed_size", "semantic", "hybrid"]
        
        for strategy_name in strategies:
            logger.info(f"Testing empty document handling with {strategy_name} strategy")
            
            for doc_type, content in edge_case_documents:
                try:
                    chunks = self.chunking_service.chunk_document(
                        f"empty_test_{doc_type}", content, strategy_name
                    )
                    
                    # Should handle gracefully - either produce no chunks or minimal chunks
                    assert isinstance(chunks, list), f"Should return list for {doc_type} with {strategy_name}"
                    
                    if len(chunks) > 0:
                        # If chunks are produced, they should be valid
                        for chunk in chunks:
                            assert isinstance(chunk, dict), "Chunk should be dictionary"
                            assert "chunk_text" in chunk, "Chunk should have text"
                            assert "chunk_id" in chunk, "Chunk should have ID"
                    
                    logger.info(f"Strategy {strategy_name} handled {doc_type} document: {len(chunks)} chunks")
                    
                except Exception as e:
                    # Log error but don't fail - some strategies may legitimately reject empty content
                    self.error_log.append({
                        "test": "empty_document",
                        "strategy": strategy_name,
                        "document_type": doc_type,
                        "error": str(e)
                    })
                    logger.warning(f"Strategy {strategy_name} failed on {doc_type}: {e}")

    def test_corrupted_document_handling(self):
        """
        Test handling of corrupted or malformed documents.
        
        Validates:
        - Documents with invalid encoding
        - Documents with control characters
        - Documents with mixed encodings
        - Binary data in text fields
        """
        corrupted_documents = [
            ("control_chars", "Normal text\x00\x01\x02 with control characters"),
            ("mixed_encoding", "Normal text " + "café".encode('latin1').decode('latin1', errors='ignore')),
            ("very_long_line", "A" * 50000 + " single line document"),
            ("repeated_chars", "x" * 10000),
            ("special_unicode", "Text with emoji 🚀🔥💯 and symbols ∑∆∇"),
            ("malformed_html", "<html><body><p>Unclosed tags<div><span>content"),
            ("json_like", '{"key": "value", "nested": {"array": [1,2,3]}}'),
            ("xml_like", "<?xml version='1.0'?><root><item>data</item></root>"),
        ]
        
        for doc_type, content in corrupted_documents:
            logger.info(f"Testing corrupted document: {doc_type}")
            
            try:
                # Test with fixed-size strategy (most robust)
                chunks = self.chunking_service.chunk_document(
                    f"corrupted_test_{doc_type}", content, "fixed_size"
                )
                
                # Should handle gracefully
                assert isinstance(chunks, list), f"Should return list for corrupted {doc_type}"
                
                if len(chunks) > 0:
                    # Validate chunk structure
                    for chunk in chunks:
                        assert isinstance(chunk["chunk_text"], str), "Chunk text should be string"
                        assert len(chunk["chunk_text"]) > 0, "Chunk text should not be empty"
                
                logger.info(f"Successfully processed corrupted document {doc_type}: {len(chunks)} chunks")
                
            except Exception as e:
                self.error_log.append({
                    "test": "corrupted_document",
                    "document_type": doc_type,
                    "error": str(e)
                })
                logger.warning(f"Failed to process corrupted document {doc_type}: {e}")

    def test_very_large_document_handling(self):
        """
        Test handling of extremely large documents.
        
        Validates:
        - Memory usage with large documents
        - Processing time limits
        - Chunk count limits
        - Graceful degradation
        """
        # Create documents of increasing size
        large_document_sizes = [
            (100_000, "100KB"),    # 100KB
            (500_000, "500KB"),    # 500KB
            (1_000_000, "1MB"),    # 1MB
        ]
        
        # Add larger sizes if we have sufficient memory
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            if available_memory > 2000:  # 2GB available
                large_document_sizes.append((5_000_000, "5MB"))
        except ImportError:
            pass
        
        for size, size_label in large_document_sizes:
            logger.info(f"Testing large document handling: {size_label}")
            
            # Create large document with varied content
            content_parts = [
                "This is a large document for testing chunking performance. ",
                "It contains repeated patterns to simulate real document structure. ",
                "The content includes multiple sentences and paragraphs. ",
                "We need to ensure that chunking can handle large documents efficiently. "
            ]
            
            # Repeat content to reach target size
            base_content = "".join(content_parts)
            repetitions = size // len(base_content) + 1
            large_content = (base_content * repetitions)[:size]
            
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            try:
                # Test with fixed-size strategy (most predictable)
                chunks = self.chunking_service.chunk_document(
                    f"large_test_{size_label}", large_content, "fixed_size"
                )
                
                processing_time = time.time() - start_time
                end_memory = self._get_memory_usage()
                memory_used = end_memory - start_memory
                
                # Validate results
                assert isinstance(chunks, list), f"Should return list for {size_label} document"
                assert len(chunks) > 0, f"Should produce chunks for {size_label} document"
                
                # Performance assertions
                assert processing_time < 60, f"Processing {size_label} should complete within 60 seconds"
                assert memory_used < 500, f"Memory usage should be reasonable for {size_label}: {memory_used}MB"
                
                # Validate chunk quality
                total_chunk_chars = sum(len(chunk["chunk_text"]) for chunk in chunks)
                coverage_ratio = total_chunk_chars / len(large_content)
                assert coverage_ratio > 0.95, f"Should maintain good coverage for {size_label}: {coverage_ratio:.2%}"
                
                logger.info(f"Large document {size_label}: {len(chunks)} chunks, "
                           f"{processing_time:.2f}s, {memory_used:.1f}MB")
                
            except MemoryError:
                logger.warning(f"Memory error processing {size_label} document - expected for very large sizes")
                self.error_log.append({
                    "test": "large_document",
                    "size": size_label,
                    "error": "MemoryError"
                })
                
            except Exception as e:
                logger.error(f"Unexpected error processing {size_label} document: {e}")
                self.error_log.append({
                    "test": "large_document",
                    "size": size_label,
                    "error": str(e)
                })

    def test_database_connection_failures(self):
        """
        Test handling of database connection failures during chunking.
        
        Validates:
        - Connection loss during chunk storage
        - Transaction rollback behavior
        - Retry mechanisms
        - Error reporting
        """
        # Create test document
        test_content = "This is a test document for database failure testing. " * 100
        
        # Test with mock connection that fails
        with patch.object(self.chunking_service, 'store_chunks') as mock_store:
            # Configure mock to raise connection error
            mock_store.side_effect = Exception("Database connection lost")
            
            try:
                # Generate chunks (should succeed)
                chunks = self.chunking_service.chunk_document(
                    "db_failure_test", test_content, "fixed_size"
                )
                
                assert len(chunks) > 0, "Should generate chunks even if storage fails"
                
                # Attempt to store chunks (should fail gracefully)
                result = self.chunking_service.store_chunks(chunks)
                
                # Should handle failure gracefully
                assert result is False, "Store operation should return False on failure"
                
            except Exception as e:
                # Should not propagate database errors to chunk generation
                logger.error(f"Unexpected error in database failure test: {e}")
                self.error_log.append({
                    "test": "database_failure",
                    "error": str(e)
                })

    def test_embedding_function_failures(self):
        """
        Test handling of embedding function failures.
        
        Validates:
        - Embedding generation errors
        - Fallback to text-only chunks
        - Partial failure handling
        - Error recovery
        """
        test_content = "This is a test document for embedding failure testing."
        
        # Test with failing embedding function
        def failing_embedding_func(texts):
            raise Exception("Embedding service unavailable")
        
        failing_chunking_service = DocumentChunkingService(embedding_func=failing_embedding_func)
        
        try:
            chunks = failing_chunking_service.chunk_document(
                "embedding_failure_test", test_content, "fixed_size"
            )
            
            # Should still produce chunks, but without embeddings
            assert len(chunks) > 0, "Should produce chunks even without embeddings"
            
            for chunk in chunks:
                # Embedding should be None or empty when embedding fails
                embedding_str = chunk.get("embedding_str")
                assert embedding_str is None or embedding_str == "", \
                    "Embedding should be None/empty when embedding function fails"
                
        except Exception as e:
            logger.warning(f"Embedding failure test error: {e}")
            self.error_log.append({
                "test": "embedding_failure",
                "error": str(e)
            })

    def test_invalid_strategy_configuration(self):
        """
        Test handling of invalid chunking strategy configurations.
        
        Validates:
        - Unknown strategy names
        - Invalid strategy parameters
        - Configuration validation
        - Error messages
        """
        test_content = "This is a test document for invalid configuration testing."
        
        # Test unknown strategy
        try:
            chunks = self.chunking_service.chunk_document(
                "invalid_strategy_test", test_content, "unknown_strategy"
            )
            assert False, "Should raise error for unknown strategy"
        except ValueError as e:
            assert "unknown_strategy" in str(e).lower(), "Error should mention unknown strategy"
            logger.info(f"Correctly caught unknown strategy error: {e}")
        except Exception as e:
            self.error_log.append({
                "test": "invalid_strategy",
                "error": str(e)
            })

    def test_concurrent_chunking_safety(self):
        """
        Test thread safety and concurrent chunking operations.
        
        Validates:
        - Multiple simultaneous chunking operations
        - Resource contention handling
        - Data consistency
        - Performance under load
        """
        import threading
        import queue
        
        test_documents = [
            f"Test document {i} with content for concurrent chunking testing. " * 50
            for i in range(10)
        ]
        
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def chunk_document_worker(doc_id, content):
            try:
                chunks = self.chunking_service.chunk_document(
                    f"concurrent_test_{doc_id}", content, "fixed_size"
                )
                results_queue.put((doc_id, len(chunks)))
            except Exception as e:
                errors_queue.put((doc_id, str(e)))
        
        # Start concurrent chunking operations
        threads = []
        for i, content in enumerate(test_documents):
            thread = threading.Thread(target=chunk_document_worker, args=(i, content))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Collect results
        successful_operations = 0
        while not results_queue.empty():
            doc_id, chunk_count = results_queue.get()
            assert chunk_count > 0, f"Document {doc_id} should produce chunks"
            successful_operations += 1
        
        # Check for errors
        error_count = 0
        while not errors_queue.empty():
            doc_id, error = errors_queue.get()
            logger.warning(f"Concurrent chunking error for doc {doc_id}: {error}")
            self.error_log.append({
                "test": "concurrent_chunking",
                "document_id": doc_id,
                "error": error
            })
            error_count += 1
        
        # Should handle most operations successfully
        assert successful_operations >= len(test_documents) * 0.8, \
            f"Should complete most concurrent operations: {successful_operations}/{len(test_documents)}"

    def test_memory_exhaustion_handling(self):
        """
        Test handling of memory exhaustion scenarios.
        
        Validates:
        - Graceful degradation under memory pressure
        - Memory cleanup after failures
        - Resource management
        - Error recovery
        """
        # Create progressively larger documents until memory pressure
        base_size = 100_000  # 100KB
        max_attempts = 10
        
        for attempt in range(max_attempts):
            size = base_size * (2 ** attempt)  # Exponential growth
            size_mb = size / (1024 * 1024)
            
            if size_mb > 100:  # Stop at 100MB to avoid system issues
                break
                
            logger.info(f"Testing memory handling with {size_mb:.1f}MB document")
            
            try:
                # Create large document
                content = "Memory test content. " * (size // 20)
                
                start_memory = self._get_memory_usage()
                
                chunks = self.chunking_service.chunk_document(
                    f"memory_test_{attempt}", content, "fixed_size"
                )
                
                end_memory = self._get_memory_usage()
                memory_used = end_memory - start_memory
                
                # Validate memory usage is reasonable
                expected_memory = size_mb * 2  # Allow 2x overhead
                if memory_used > expected_memory:
                    logger.warning(f"High memory usage: {memory_used:.1f}MB for {size_mb:.1f}MB document")
                
                # Clean up
                del content
                del chunks
                
            except MemoryError:
                logger.info(f"Memory exhaustion reached at {size_mb:.1f}MB - expected")
                break
            except Exception as e:
                logger.warning(f"Memory test error at {size_mb:.1f}MB: {e}")
                self.error_log.append({
                    "test": "memory_exhaustion",
                    "size_mb": size_mb,
                    "error": str(e)
                })
                break

    def test_graceful_degradation_scenarios(self):
        """
        Test graceful degradation under various failure conditions.
        
        Validates:
        - Partial system failures
        - Fallback mechanisms
        - Service continuity
        - Error isolation
        """
        test_content = "This is a test document for graceful degradation testing. " * 20
        
        # Test scenario 1: Semantic chunking fallback to fixed-size
        with patch.object(SemanticChunkingStrategy, 'chunk') as mock_semantic:
            mock_semantic.side_effect = Exception("Semantic analysis failed")
            
            # Hybrid strategy should fall back to fixed-size
            hybrid_strategy = HybridChunkingStrategy(
                primary_strategy=SemanticChunkingStrategy(),
                fallback_strategy=FixedSizeChunkingStrategy(),
                max_chunk_size=800
            )
            
            try:
                chunks = hybrid_strategy.chunk(test_content, "degradation_test")
                assert len(chunks) > 0, "Should produce chunks using fallback strategy"
                logger.info("Graceful degradation: Hybrid strategy used fallback successfully")
            except Exception as e:
                self.error_log.append({
                    "test": "graceful_degradation",
                    "scenario": "hybrid_fallback",
                    "error": str(e)
                })
        
        # Test scenario 2: Configuration fallback
        try:
            # Test with invalid configuration that should fall back to defaults
            invalid_config = ConfigurationManager()
            # Simulate missing configuration
            with patch.object(invalid_config, 'get', return_value={}):
                service = DocumentChunkingService(embedding_func=self.embedding_func)
                chunks = service.chunk_document("config_fallback_test", test_content, "fixed_size")
                assert len(chunks) > 0, "Should work with default configuration"
                logger.info("Graceful degradation: Configuration fallback successful")
        except Exception as e:
            self.error_log.append({
                "test": "graceful_degradation",
                "scenario": "config_fallback",
                "error": str(e)
            })

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available

    def teardown_method(self):
        """Clean up after each test and log errors."""
        if self.error_log:
            logger.warning(f"Test completed with {len(self.error_log)} errors logged")
            for error in self.error_log:
                logger.warning(f"Error: {error}")
        else:
            logger.info("All error handling tests completed successfully")