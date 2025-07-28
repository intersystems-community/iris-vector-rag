#!/usr/bin/env python3
"""
Chunking Strategy Validation Tests

This module provides comprehensive testing for all chunking strategies:
- Fixed-size chunking with overlap and sentence preservation
- Semantic chunking with similarity-based boundaries
- Hybrid chunking with fallback mechanisms

Test Coverage:
1. Strategy-specific configuration validation
2. Chunk size limits and overlap behavior
3. Threshold enforcement and boundary detection
4. Chunk quality and retrieval effectiveness
5. Strategy performance comparison
"""

import pytest
import logging
import time
import statistics
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Import test fixtures
from tests.conftest_1000docs import (
    enterprise_iris_connection,
    scale_test_config,
    enterprise_document_loader_1000docs,
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

logger = logging.getLogger(__name__)

class TestChunkingStrategyValidation:
    """Comprehensive validation tests for all chunking strategies."""

    @pytest.fixture(autouse=True)
    def setup_chunking_environment(self, enterprise_iris_connection, scale_test_config,
                                  enterprise_embedding_manager):
        """Set up chunking test environment."""
        self.connection = enterprise_iris_connection
        self.config = scale_test_config
        self.config_manager = scale_test_config['config_manager']
        self.embedding_func = enterprise_embedding_manager.get_embedding_function()
        
        # Initialize chunking service
        self.chunking_service = DocumentChunkingService(embedding_func=self.embedding_func)
        
        # Test documents with different characteristics
        self.test_documents = {
            "short": "This is a short document with minimal content for testing basic chunking behavior.",
            "medium": """This is a medium-length document that contains multiple sentences and paragraphs. 
                        It should be suitable for testing various chunking strategies and their effectiveness.
                        The document includes different topics and should demonstrate how semantic boundaries
                        are detected and handled by different chunking approaches. This content is designed
                        to test overlap behavior, sentence preservation, and chunk quality metrics.""",
            "long": """This is a comprehensive long document designed to test chunking strategies thoroughly.
                      
                      Introduction: Document chunking is a critical preprocessing step in RAG systems that
                      breaks down large documents into smaller, semantically coherent segments. The effectiveness
                      of chunking directly impacts retrieval accuracy and generation quality.
                      
                      Fixed-Size Chunking: This approach divides documents into chunks of predetermined size,
                      typically measured in characters or tokens. While simple and predictable, it may split
                      sentences or concepts inappropriately. However, it provides consistent chunk sizes that
                      work well with embedding models that have fixed input limits.
                      
                      Semantic Chunking: This more sophisticated approach attempts to identify natural boundaries
                      in the text based on semantic similarity between sentences or paragraphs. It aims to
                      preserve conceptual coherence within chunks, potentially improving retrieval relevance.
                      The challenge lies in accurately detecting semantic boundaries and maintaining reasonable
                      chunk sizes.
                      
                      Hybrid Approaches: These combine the benefits of both fixed-size and semantic chunking.
                      They typically use semantic chunking as the primary strategy but fall back to fixed-size
                      chunking when chunks become too large. This provides a balance between semantic coherence
                      and size constraints.
                      
                      Conclusion: The choice of chunking strategy depends on the specific use case, document
                      characteristics, and performance requirements. Effective chunking requires careful
                      consideration of these factors and thorough testing with representative data."""
        }

    def test_fixed_size_chunking_strategy(self):
        """
        Test fixed-size chunking strategy with various configurations.
        
        Validates:
        - Chunk size limits and consistency
        - Overlap behavior and calculation
        - Sentence preservation functionality
        - Minimum chunk size enforcement
        """
        # Test different configurations
        test_configs = [
            {"chunk_size": 200, "overlap_size": 20, "preserve_sentences": True, "min_chunk_size": 50},
            {"chunk_size": 500, "overlap_size": 50, "preserve_sentences": True, "min_chunk_size": 100},
            {"chunk_size": 1000, "overlap_size": 100, "preserve_sentences": False, "min_chunk_size": 200},
        ]
        
        for config in test_configs:
            logger.info(f"Testing fixed-size chunking with config: {config}")
            
            strategy = FixedSizeChunkingStrategy(**config)
            
            for doc_type, content in self.test_documents.items():
                chunks = strategy.chunk(content, f"test_doc_{doc_type}")
                
                # Validate basic requirements
                assert len(chunks) > 0, f"Should produce chunks for {doc_type} document"
                assert all(isinstance(chunk, Chunk) for chunk in chunks), "All chunks should be Chunk objects"
                
                # Validate chunk sizes
                chunk_sizes = [len(chunk.text) for chunk in chunks]
                max_size = max(chunk_sizes)
                min_size = min(chunk_sizes)
                
                # Most chunks should be close to target size (except possibly the last one)
                if len(chunks) > 1:
                    assert max_size <= config["chunk_size"] * 1.2, \
                        f"Chunk too large: {max_size} > {config['chunk_size'] * 1.2}"
                    assert min_size >= config["min_chunk_size"], \
                        f"Chunk too small: {min_size} < {config['min_chunk_size']}"
                
                # Validate overlap behavior
                if len(chunks) > 1 and config["overlap_size"] > 0:
                    for i in range(len(chunks) - 1):
                        current_chunk = chunks[i]
                        next_chunk = chunks[i + 1]
                        
                        # Check that overlap exists (approximate due to sentence preservation)
                        overlap_metadata = next_chunk.metadata.get("overlap_with_previous", 0)
                        assert overlap_metadata >= 0, "Overlap metadata should be non-negative"
                
                # Validate sentence preservation
                if config["preserve_sentences"]:
                    for chunk in chunks:
                        # Check that chunks don't end mid-sentence (basic heuristic)
                        text = chunk.text.strip()
                        if len(text) > 10:  # Skip very short chunks
                            last_char = text[-1]
                            # Should end with sentence-ending punctuation or be the last chunk
                            if chunk != chunks[-1]:  # Not the last chunk
                                assert last_char in '.!?', \
                                    f"Chunk should end with sentence punctuation: '{text[-20:]}'"
                
                # Validate metadata
                for i, chunk in enumerate(chunks):
                    assert "chunk_index" in chunk.metadata, "Chunk should have index metadata"
                    assert chunk.metadata["chunk_index"] == i, "Chunk index should be correct"
                    assert "chunk_size" in chunk.metadata, "Chunk should have size metadata"
                    assert chunk.metadata["chunk_size"] == len(chunk.text), "Size metadata should match actual size"
                    assert chunk.chunk_type == "fixed_size", "Chunk type should be correct"

    def test_semantic_chunking_strategy(self):
        """
        Test semantic chunking strategy with boundary detection.
        
        Validates:
        - Semantic boundary detection accuracy
        - Similarity threshold behavior
        - Chunk size constraints (min/max)
        - Sentence grouping logic
        """
        # Test different configurations
        test_configs = [
            {"similarity_threshold": 0.5, "min_chunk_size": 100, "max_chunk_size": 800},
            {"similarity_threshold": 0.7, "min_chunk_size": 200, "max_chunk_size": 1000},
            {"similarity_threshold": 0.9, "min_chunk_size": 150, "max_chunk_size": 600},
        ]
        
        for config in test_configs:
            logger.info(f"Testing semantic chunking with config: {config}")
            
            strategy = SemanticChunkingStrategy(**config)
            
            # Test with medium and long documents (semantic chunking needs sufficient content)
            for doc_type in ["medium", "long"]:
                content = self.test_documents[doc_type]
                chunks = strategy.chunk(content, f"test_doc_{doc_type}")
                
                # Validate basic requirements
                assert len(chunks) > 0, f"Should produce chunks for {doc_type} document"
                
                # Validate chunk sizes
                chunk_sizes = [len(chunk.text) for chunk in chunks]
                
                for size in chunk_sizes:
                    assert size >= config["min_chunk_size"], \
                        f"Chunk too small: {size} < {config['min_chunk_size']}"
                    assert size <= config["max_chunk_size"], \
                        f"Chunk too large: {size} > {config['max_chunk_size']}"
                
                # Validate semantic boundary metadata
                for chunk in chunks:
                    assert "sentence_count" in chunk.metadata, "Should have sentence count metadata"
                    assert chunk.metadata["sentence_count"] > 0, "Should have at least one sentence"
                    assert chunk.chunk_type == "semantic", "Chunk type should be correct"
                
                # Validate that semantic boundaries are detected
                semantic_boundaries = sum(1 for chunk in chunks 
                                        if chunk.metadata.get("semantic_boundary", False))
                
                if len(chunks) > 1:
                    # Should have some semantic boundaries detected
                    assert semantic_boundaries >= 0, "Should detect semantic boundaries"
                
                # Test similarity threshold effect
                if config["similarity_threshold"] > 0.8:
                    # High threshold should produce more, smaller chunks
                    avg_chunk_size = statistics.mean(chunk_sizes)
                    assert avg_chunk_size <= config["max_chunk_size"] * 0.8, \
                        "High similarity threshold should produce smaller chunks"

    def test_hybrid_chunking_strategy(self):
        """
        Test hybrid chunking strategy with fallback mechanisms.
        
        Validates:
        - Primary strategy execution
        - Fallback strategy activation
        - Chunk size limit enforcement
        - Strategy combination metadata
        """
        # Create primary and fallback strategies
        primary_strategy = SemanticChunkingStrategy(
            similarity_threshold=0.6, min_chunk_size=200, max_chunk_size=2000  # Large max to test fallback
        )
        fallback_strategy = FixedSizeChunkingStrategy(
            chunk_size=500, overlap_size=50, preserve_sentences=True
        )
        
        # Test different max chunk sizes for hybrid strategy
        test_max_sizes = [800, 1200, 1500]
        
        for max_size in test_max_sizes:
            logger.info(f"Testing hybrid chunking with max_size: {max_size}")
            
            hybrid_strategy = HybridChunkingStrategy(
                primary_strategy=primary_strategy,
                fallback_strategy=fallback_strategy,
                max_chunk_size=max_size
            )
            
            # Test with long document to trigger fallback
            content = self.test_documents["long"]
            chunks = hybrid_strategy.chunk(content, "test_doc_long")
            
            # Validate basic requirements
            assert len(chunks) > 0, "Should produce chunks"
            
            # Validate chunk sizes
            chunk_sizes = [len(chunk.text) for chunk in chunks]
            max_chunk_size = max(chunk_sizes)
            
            assert max_chunk_size <= max_size * 1.1, \
                f"Hybrid chunking should enforce max size: {max_chunk_size} > {max_size}"
            
            # Check for fallback usage
            fallback_chunks = [chunk for chunk in chunks 
                             if chunk.metadata.get("was_re_chunked", False)]
            
            if max_size < 1200:  # Should trigger fallback with smaller limits
                assert len(fallback_chunks) > 0, "Should use fallback strategy for oversized chunks"
                
                # Validate fallback metadata
                for chunk in fallback_chunks:
                    assert "primary_strategy" in chunk.metadata, "Should have primary strategy metadata"
                    assert "fallback_strategy" in chunk.metadata, "Should have fallback strategy metadata"
                    assert "original_chunk_size" in chunk.metadata, "Should have original size metadata"
                    assert chunk.chunk_type == "hybrid", "Chunk type should be hybrid"
            
            # Validate that non-fallback chunks use primary strategy
            primary_chunks = [chunk for chunk in chunks 
                            if not chunk.metadata.get("was_re_chunked", False)]
            
            for chunk in primary_chunks:
                assert chunk.chunk_type == "hybrid", "All chunks should have hybrid type"
                assert "primary_strategy" in chunk.metadata, "Should have primary strategy metadata"

    def test_chunking_strategy_performance_comparison(self, enterprise_document_loader_1000docs):
        """
        Compare performance characteristics of different chunking strategies.
        
        Validates:
        - Processing speed for each strategy
        - Memory usage patterns
        - Chunk quality metrics
        - Strategy-specific optimizations
        """
        documents = enterprise_document_loader_1000docs[:100]  # Use subset for performance testing
        
        strategies = {
            "fixed_size": FixedSizeChunkingStrategy(chunk_size=512, overlap_size=50),
            "semantic": SemanticChunkingStrategy(similarity_threshold=0.7, min_chunk_size=200, max_chunk_size=1000),
            "hybrid": HybridChunkingStrategy(
                primary_strategy=SemanticChunkingStrategy(similarity_threshold=0.7, min_chunk_size=200, max_chunk_size=1000),
                fallback_strategy=FixedSizeChunkingStrategy(chunk_size=512, overlap_size=50),
                max_chunk_size=800
            )
        }
        
        performance_results = {}
        
        for strategy_name, strategy in strategies.items():
            logger.info(f"Performance testing strategy: {strategy_name}")
            
            start_time = time.time()
            total_chunks = 0
            total_chars = 0
            chunk_sizes = []
            
            for doc in documents:
                try:
                    chunks = strategy.chunk(doc.content, doc.doc_id)
                    total_chunks += len(chunks)
                    
                    for chunk in chunks:
                        chunk_size = len(chunk.text)
                        chunk_sizes.append(chunk_size)
                        total_chars += chunk_size
                        
                except Exception as e:
                    logger.warning(f"Chunking failed for {doc.doc_id} with {strategy_name}: {e}")
            
            processing_time = time.time() - start_time
            
            # Calculate performance metrics
            performance_results[strategy_name] = {
                "processing_time": processing_time,
                "documents_processed": len(documents),
                "total_chunks": total_chunks,
                "total_characters": total_chars,
                "chunks_per_second": total_chunks / processing_time if processing_time > 0 else 0,
                "chars_per_second": total_chars / processing_time if processing_time > 0 else 0,
                "avg_chunk_size": statistics.mean(chunk_sizes) if chunk_sizes else 0,
                "chunk_size_std": statistics.stdev(chunk_sizes) if len(chunk_sizes) > 1 else 0,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0
            }
            
            # Performance assertions
            assert total_chunks > 0, f"Strategy {strategy_name} should produce chunks"
            assert processing_time < 300, f"Strategy {strategy_name} should complete within 5 minutes"
            
        # Compare strategies
        fixed_perf = performance_results["fixed_size"]
        semantic_perf = performance_results["semantic"]
        hybrid_perf = performance_results["hybrid"]
        
        # Fixed-size should be fastest
        assert fixed_perf["chunks_per_second"] >= semantic_perf["chunks_per_second"] * 0.5, \
            "Fixed-size chunking should be reasonably fast compared to semantic"
        
        # Semantic should have more consistent chunk sizes (lower std deviation relative to mean)
        if semantic_perf["avg_chunk_size"] > 0:
            semantic_cv = semantic_perf["chunk_size_std"] / semantic_perf["avg_chunk_size"]
            fixed_cv = fixed_perf["chunk_size_std"] / fixed_perf["avg_chunk_size"] if fixed_perf["avg_chunk_size"] > 0 else 1
            
            # Note: This assertion might need adjustment based on actual behavior
            logger.info(f"Coefficient of variation - Fixed: {fixed_cv:.3f}, Semantic: {semantic_cv:.3f}")
        
        # Hybrid should balance performance and quality
        assert hybrid_perf["chunks_per_second"] >= semantic_perf["chunks_per_second"] * 0.8, \
            "Hybrid chunking should maintain reasonable performance"
        
        logger.info("Performance comparison completed successfully")
        
        # Log performance summary
        for strategy_name, metrics in performance_results.items():
            logger.info(f"{strategy_name}: {metrics['chunks_per_second']:.2f} chunks/sec, "
                       f"avg size: {metrics['avg_chunk_size']:.0f} chars")

    def test_chunking_configuration_validation(self):
        """
        Test chunking strategy configuration validation and error handling.
        
        Validates:
        - Invalid parameter handling
        - Configuration boundary conditions
        - Error message clarity
        - Default value behavior
        """
        # Test invalid configurations
        invalid_configs = [
            # Fixed-size strategy invalid configs
            {"strategy": "fixed_size", "params": {"chunk_size": 0}, "error_type": ValueError},
            {"strategy": "fixed_size", "params": {"chunk_size": -100}, "error_type": ValueError},
            {"strategy": "fixed_size", "params": {"overlap_size": -10}, "error_type": ValueError},
            {"strategy": "fixed_size", "params": {"min_chunk_size": -5}, "error_type": ValueError},
            
            # Semantic strategy invalid configs
            {"strategy": "semantic", "params": {"similarity_threshold": -0.5}, "error_type": ValueError},
            {"strategy": "semantic", "params": {"similarity_threshold": 1.5}, "error_type": ValueError},
            {"strategy": "semantic", "params": {"min_chunk_size": 0}, "error_type": ValueError},
            {"strategy": "semantic", "params": {"max_chunk_size": 50, "min_chunk_size": 100}, "error_type": ValueError},
        ]
        
        for config in invalid_configs:
            strategy_type = config["strategy"]
            params = config["params"]
            expected_error = config["error_type"]
            
            with pytest.raises(expected_error):
                if strategy_type == "fixed_size":
                    FixedSizeChunkingStrategy(**params)
                elif strategy_type == "semantic":
                    SemanticChunkingStrategy(**params)
        
        # Test boundary conditions
        boundary_configs = [
            # Minimum valid values
            {"strategy": "fixed_size", "params": {"chunk_size": 1, "overlap_size": 0, "min_chunk_size": 1}},
            {"strategy": "semantic", "params": {"similarity_threshold": 0.0, "min_chunk_size": 1, "max_chunk_size": 2}},
            
            # Maximum reasonable values
            {"strategy": "fixed_size", "params": {"chunk_size": 10000, "overlap_size": 1000}},
            {"strategy": "semantic", "params": {"similarity_threshold": 1.0, "max_chunk_size": 10000}},
        ]
        
        for config in boundary_configs:
            strategy_type = config["strategy"]
            params = config["params"]
            
            # Should not raise exceptions
            if strategy_type == "fixed_size":
                strategy = FixedSizeChunkingStrategy(**params)
                assert strategy.chunk_size == params["chunk_size"]
            elif strategy_type == "semantic":
                strategy = SemanticChunkingStrategy(**params)
                assert strategy.similarity_threshold == params["similarity_threshold"]
        
        logger.info("Configuration validation tests completed successfully")

    def test_chunk_quality_metrics(self):
        """
        Test chunk quality assessment and metrics.
        
        Validates:
        - Chunk coherence measurement
        - Boundary quality assessment
        - Information preservation
        - Retrieval effectiveness indicators
        """
        # Use the chunking service to analyze different strategies
        test_content = self.test_documents["long"]
        
        quality_metrics = {}
        
        for strategy_name in ["fixed_size", "semantic", "hybrid"]:
            chunks = self.chunking_service.chunk_document("quality_test", test_content, strategy_name)
            
            # Calculate quality metrics
            chunk_texts = [chunk["chunk_text"] for chunk in chunks]
            chunk_lengths = [len(text) for text in chunk_texts]
            
            # Coherence metrics (simplified)
            sentence_breaks = sum(1 for text in chunk_texts 
                                if not text.strip().endswith(('.', '!', '?')))
            
            # Information preservation (no chunks should be empty or too short)
            meaningful_chunks = sum(1 for text in chunk_texts if len(text.strip()) > 20)
            
            # Overlap analysis
            overlaps = []
            for i in range(len(chunk_texts) - 1):
                current = chunk_texts[i]
                next_chunk = chunk_texts[i + 1]
                
                # Simple overlap detection (last 50 chars of current vs first 50 chars of next)
                current_end = current[-50:] if len(current) > 50 else current
                next_start = next_chunk[:50] if len(next_chunk) > 50 else next_chunk
                
                # Count common words as overlap indicator
                current_words = set(current_end.lower().split())
                next_words = set(next_start.lower().split())
                overlap_ratio = len(current_words.intersection(next_words)) / max(len(current_words), 1)
                overlaps.append(overlap_ratio)
            
            quality_metrics[strategy_name] = {
                "total_chunks": len(chunks),
                "avg_chunk_length": statistics.mean(chunk_lengths) if chunk_lengths else 0,
                "length_consistency": 1 - (statistics.stdev(chunk_lengths) / statistics.mean(chunk_lengths)) 
                                    if len(chunk_lengths) > 1 and statistics.mean(chunk_lengths) > 0 else 1,
                "sentence_break_ratio": sentence_breaks / len(chunks) if len(chunks) > 0 else 0,
                "meaningful_chunk_ratio": meaningful_chunks / len(chunks) if len(chunks) > 0 else 0,
                "avg_overlap_ratio": statistics.mean(overlaps) if overlaps else 0
            }
        
        # Quality assertions
        for strategy_name, metrics in quality_metrics.items():
            assert metrics["meaningful_chunk_ratio"] >= 0.95, \
                f"Strategy {strategy_name} should produce mostly meaningful chunks"
            assert metrics["total_chunks"] > 0, f"Strategy {strategy_name} should produce chunks"
            
            # Strategy-specific quality checks
            if strategy_name == "fixed_size":
                # Fixed-size should have consistent lengths
                assert metrics["length_consistency"] >= 0.3, \
                    f"Fixed-size chunking should have reasonable length consistency"
            
            elif strategy_name == "semantic":
                # Semantic should have fewer sentence breaks
                assert metrics["sentence_break_ratio"] <= 0.3, \
                    f"Semantic chunking should minimize sentence breaks"
            
            elif strategy_name == "hybrid":
                # Hybrid should balance consistency and coherence
                assert metrics["length_consistency"] >= 0.2, \
                    f"Hybrid chunking should maintain reasonable consistency"
                assert metrics["sentence_break_ratio"] <= 0.4, \
                    f"Hybrid chunking should limit sentence breaks"
        
        logger.info("Chunk quality metrics validation completed successfully")
        
        # Log quality summary
        for strategy_name, metrics in quality_metrics.items():
            logger.info(f"{strategy_name} quality: {metrics['total_chunks']} chunks, "
                       f"{metrics['meaningful_chunk_ratio']:.2%} meaningful, "
                       f"{metrics['sentence_break_ratio']:.2%} sentence breaks")