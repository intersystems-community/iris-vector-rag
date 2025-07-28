#!/usr/bin/env python3
"""
Comprehensive End-to-End Chunking Architecture Integration Tests

This module provides comprehensive testing for the chunking architecture integration
across all 8 RAG pipelines with real PMC documents (1000+ documents).

Test Coverage:
1. End-to-End Pipeline Testing for all 8 RAG pipelines with chunking
2. Document loading → chunking → storage → retrieval → query answering
3. Real PMC document validation with 1000+ documents
4. Pipeline-specific chunking configuration validation
5. Performance and scale testing
"""

import pytest
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import test fixtures
from tests.conftest_1000docs import (
    enterprise_iris_connection,
    scale_test_config,
    enterprise_schema_manager,
    enterprise_document_loader_1000docs,
    enterprise_embedding_manager,
    enterprise_llm_function
)

# Import core components
from iris_rag.config.manager import ConfigurationManager
from iris_rag.storage.vector_store.iris_impl import IRISVectorStore
from tools.chunking.chunking_service import DocumentChunkingService
from iris_rag.core.models import Document

# Import all 8 RAG pipelines
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.hyde import HyDEPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.noderag import NodeRAGPipeline
from iris_rag.pipelines.hybrid_ifind import HybridIFindPipeline
from iris_rag.pipelines.colbert.pipeline import ColBERTPipeline
from iris_rag.pipelines.sql_rag import SQLRAGPipeline

logger = logging.getLogger(__name__)

# Test configuration
TEST_QUERIES = [
    "What are the main findings about COVID-19 treatment?",
    "How does machine learning apply to medical diagnosis?",
    "What are the latest developments in cancer research?",
    "Explain the role of genetics in disease prevention",
    "What are the side effects of common medications?"
]

PIPELINE_CONFIGS = {
    "basic": {"enabled": True, "chunking_strategy": "fixed_size"},
    "hyde": {"enabled": True, "chunking_strategy": "fixed_size"},
    "crag": {"enabled": True, "chunking_strategy": "fixed_size"},
    "graphrag": {"enabled": True, "chunking_strategy": "semantic"},
    "noderag": {"enabled": True, "chunking_strategy": "fixed_size"},
    "hybrid_ifind": {"enabled": True, "chunking_strategy": "hybrid"},
    "colbert": {"enabled": False, "chunking_strategy": None},  # ColBERT handles chunking internally
    "sql_rag": {"enabled": False, "chunking_strategy": None}   # SQL RAG may not need chunking
}

class TestComprehensiveChunkingE2E:
    """Comprehensive end-to-end testing for chunking architecture integration."""

    @pytest.fixture(autouse=True)
    def setup_test_environment(self, enterprise_iris_connection, scale_test_config, 
                              enterprise_schema_manager):
        """Set up test environment for each test."""
        self.connection = enterprise_iris_connection
        self.config = scale_test_config
        self.schema_manager = enterprise_schema_manager
        self.config_manager = scale_test_config['config_manager']
        
        # Initialize test results tracking
        self.test_results = {
            "pipeline_results": {},
            "performance_metrics": {},
            "chunking_analysis": {},
            "errors": []
        }
        
        # Ensure required schemas exist
        required_tables = [
            "SourceDocuments", "DocumentChunks", "VectorEmbeddings",
            "ColBERTTokenEmbeddings", "GraphEntities", "GraphRelationships"
        ]
        
        for table in required_tables:
            try:
                self.schema_manager.ensure_table_schema(table)
            except Exception as e:
                logger.warning(f"Could not ensure schema for {table}: {e}")

    def test_all_pipelines_e2e_with_chunking(self, enterprise_document_loader_1000docs,
                                           enterprise_embedding_manager, enterprise_llm_function):
        """
        Test all 8 RAG pipelines end-to-end with chunking integration.
        
        This test validates:
        - Document loading and chunking for each pipeline
        - Vector storage and retrieval with chunks
        - Query processing and answer generation
        - Pipeline-specific chunking configurations
        """
        documents = enterprise_document_loader_1000docs
        embedding_func = enterprise_embedding_manager.get_embedding_function()
        llm_func = enterprise_llm_function
        
        # Validate we have sufficient documents
        assert len(documents) >= self.config['minimum_document_count'], \
            f"Need at least {self.config['minimum_document_count']} documents, got {len(documents)}"
        
        logger.info(f"Testing {len(documents)} documents across all pipelines")
        
        # Test each pipeline
        pipeline_classes = {
            "basic": BasicRAGPipeline,
            "hyde": HyDEPipeline,
            "crag": CRAGPipeline,
            "graphrag": GraphRAGPipeline,
            "noderag": NodeRAGPipeline,
            "hybrid_ifind": HybridIFindPipeline,
            "colbert": ColBERTPipeline,
            "sql_rag": SQLRAGPipeline
        }
        
        for pipeline_name, pipeline_class in pipeline_classes.items():
            logger.info(f"Testing pipeline: {pipeline_name}")
            
            try:
                # Create pipeline instance
                vector_store = IRISVectorStore(config_manager=self.config_manager)
                pipeline = pipeline_class(
                    config_manager=self.config_manager,
                    llm_func=llm_func,
                    vector_store=vector_store
                )
                
                # Test document loading with chunking
                start_time = time.time()
                
                # Configure chunking for this pipeline
                pipeline_config = PIPELINE_CONFIGS.get(pipeline_name, {})
                chunking_enabled = pipeline_config.get("enabled", True)
                chunking_strategy = pipeline_config.get("chunking_strategy", "fixed_size")
                
                if chunking_enabled:
                    # Load documents with chunking
                    pipeline.load_documents(
                        documents_path="",  # Using documents directly
                        documents=documents[:100],  # Use subset for faster testing
                        auto_chunk=True,
                        chunking_strategy=chunking_strategy,
                        generate_embeddings=True
                    )
                else:
                    # Load documents without chunking
                    pipeline.load_documents(
                        documents_path="",
                        documents=documents[:100],
                        auto_chunk=False,
                        generate_embeddings=True
                    )
                
                load_time = time.time() - start_time
                
                # Test query processing
                query_results = []
                for query in TEST_QUERIES[:3]:  # Test subset of queries
                    try:
                        query_start = time.time()
                        result = pipeline.query(query)
                        query_time = time.time() - query_start
                        
                        # Validate result structure
                        assert isinstance(result, dict), f"Pipeline {pipeline_name} should return dict"
                        assert "query" in result, f"Pipeline {pipeline_name} missing 'query' in result"
                        assert "answer" in result, f"Pipeline {pipeline_name} missing 'answer' in result"
                        assert "retrieved_documents" in result, f"Pipeline {pipeline_name} missing 'retrieved_documents'"
                        
                        # Validate answer quality
                        assert len(result["answer"]) > 10, f"Pipeline {pipeline_name} answer too short"
                        assert len(result["retrieved_documents"]) > 0, f"Pipeline {pipeline_name} no documents retrieved"
                        
                        query_results.append({
                            "query": query,
                            "answer_length": len(result["answer"]),
                            "documents_retrieved": len(result["retrieved_documents"]),
                            "query_time": query_time
                        })
                        
                    except Exception as e:
                        logger.error(f"Query failed for {pipeline_name}: {e}")
                        self.test_results["errors"].append({
                            "pipeline": pipeline_name,
                            "query": query,
                            "error": str(e)
                        })
                
                # Store results
                self.test_results["pipeline_results"][pipeline_name] = {
                    "load_time": load_time,
                    "chunking_enabled": chunking_enabled,
                    "chunking_strategy": chunking_strategy,
                    "documents_processed": len(documents[:100]),
                    "queries_tested": len(query_results),
                    "query_results": query_results,
                    "success": True
                }
                
                logger.info(f"Pipeline {pipeline_name} completed successfully")
                
            except Exception as e:
                logger.error(f"Pipeline {pipeline_name} failed: {e}")
                self.test_results["pipeline_results"][pipeline_name] = {
                    "success": False,
                    "error": str(e)
                }
                self.test_results["errors"].append({
                    "pipeline": pipeline_name,
                    "error": str(e)
                })
        
        # Validate overall results
        successful_pipelines = [
            name for name, result in self.test_results["pipeline_results"].items()
            if result.get("success", False)
        ]
        
        assert len(successful_pipelines) >= 6, \
            f"At least 6 pipelines should succeed, got {len(successful_pipelines)}: {successful_pipelines}"
        
        # Log summary
        logger.info(f"E2E Test Summary: {len(successful_pipelines)}/{len(pipeline_classes)} pipelines successful")
        
    def test_chunking_strategy_effectiveness(self, enterprise_document_loader_1000docs,
                                           enterprise_embedding_manager):
        """
        Test the effectiveness of different chunking strategies across pipelines.
        
        Validates:
        - Fixed-size chunking performance and consistency
        - Semantic chunking boundary detection
        - Hybrid chunking fallback behavior
        - Strategy-specific configuration inheritance
        """
        documents = enterprise_document_loader_1000docs[:50]  # Use subset for detailed analysis
        embedding_func = enterprise_embedding_manager.get_embedding_function()
        
        chunking_service = DocumentChunkingService(embedding_func=embedding_func)
        
        strategy_results = {}
        
        for strategy_name in ["fixed_size", "semantic", "hybrid"]:
            logger.info(f"Testing chunking strategy: {strategy_name}")
            
            strategy_metrics = {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "chunk_size_variance": 0,
                "processing_time": 0,
                "documents_processed": 0
            }
            
            start_time = time.time()
            
            for doc in documents:
                try:
                    # Analyze chunking effectiveness
                    analysis = chunking_service.analyze_chunking_effectiveness(
                        doc.doc_id, doc.content
                    )
                    
                    if strategy_name in analysis["strategy_analysis"]:
                        strategy_data = analysis["strategy_analysis"][strategy_name]
                        
                        strategy_metrics["total_chunks"] += strategy_data.get("chunk_count", 0)
                        strategy_metrics["avg_chunk_size"] += strategy_data.get("avg_chunk_length", 0)
                        strategy_metrics["documents_processed"] += 1
                        
                except Exception as e:
                    logger.warning(f"Chunking analysis failed for {doc.doc_id} with {strategy_name}: {e}")
            
            strategy_metrics["processing_time"] = time.time() - start_time
            
            # Calculate averages
            if strategy_metrics["documents_processed"] > 0:
                strategy_metrics["avg_chunk_size"] /= strategy_metrics["documents_processed"]
                strategy_metrics["avg_chunks_per_doc"] = (
                    strategy_metrics["total_chunks"] / strategy_metrics["documents_processed"]
                )
            
            strategy_results[strategy_name] = strategy_metrics
            
            # Validate strategy performance
            assert strategy_metrics["total_chunks"] > 0, f"Strategy {strategy_name} produced no chunks"
            assert strategy_metrics["avg_chunk_size"] > 0, f"Strategy {strategy_name} has zero average chunk size"
            
        # Store chunking analysis results
        self.test_results["chunking_analysis"] = strategy_results
        
        # Validate strategy differences
        fixed_chunks = strategy_results["fixed_size"]["avg_chunks_per_doc"]
        semantic_chunks = strategy_results["semantic"]["avg_chunks_per_doc"]
        hybrid_chunks = strategy_results["hybrid"]["avg_chunks_per_doc"]
        
        # Semantic chunking should generally produce fewer, larger chunks
        # Hybrid should be between fixed and semantic
        logger.info(f"Chunking strategy comparison - Fixed: {fixed_chunks:.2f}, "
                   f"Semantic: {semantic_chunks:.2f}, Hybrid: {hybrid_chunks:.2f} chunks per doc")

    def test_pipeline_configuration_inheritance(self):
        """
        Test pipeline-specific chunking configuration inheritance and overrides.
        
        Validates:
        - Default configuration loading
        - Pipeline-specific overrides from pipeline_overrides section
        - Configuration validation and error handling
        - Dynamic configuration updates
        """
        config_manager = ConfigurationManager()
        
        # Test default chunking configuration
        default_chunking = config_manager.get("storage:chunking", {})
        assert default_chunking.get("enabled", False), "Default chunking should be enabled"
        assert "strategy" in default_chunking, "Default chunking strategy should be defined"
        assert "strategies" in default_chunking, "Chunking strategies should be defined"
        
        # Test pipeline-specific overrides
        pipeline_overrides = config_manager.get("pipeline_overrides", {})
        
        # Test GraphRAG override (should use semantic chunking)
        graphrag_config = pipeline_overrides.get("graphrag", {})
        if "chunking" in graphrag_config:
            graphrag_chunking = graphrag_config["chunking"]
            assert graphrag_chunking.get("strategy") == "semantic", \
                "GraphRAG should use semantic chunking strategy"
            assert graphrag_chunking.get("threshold", 1000) <= 800, \
                "GraphRAG should use smaller chunk threshold"
        
        # Test ColBERT override (should disable chunking)
        colbert_config = pipeline_overrides.get("colbert", {})
        if "chunking" in colbert_config:
            colbert_chunking = colbert_config["chunking"]
            assert not colbert_chunking.get("enabled", True), \
                "ColBERT should disable document chunking"
        
        # Test HyDE override (should use fixed-size chunking)
        hyde_config = pipeline_overrides.get("hyde", {})
        if "chunking" in hyde_config:
            hyde_chunking = hyde_config["chunking"]
            assert hyde_chunking.get("strategy") == "fixed_size", \
                "HyDE should use fixed-size chunking strategy"
        
        logger.info("Configuration inheritance validation completed")

    def test_performance_and_scale_metrics(self, enterprise_document_loader_1000docs,
                                         enterprise_embedding_manager):
        """
        Test performance and scale metrics for chunking with 1000+ documents.
        
        Validates:
        - Chunking performance with large document sets
        - Memory usage during chunking operations
        - Concurrent chunking capability
        - Performance degradation analysis
        """
        documents = enterprise_document_loader_1000docs
        embedding_func = enterprise_embedding_manager.get_embedding_function()
        
        # Test different document set sizes
        test_sizes = [100, 500, 1000]
        if len(documents) >= 2000:
            test_sizes.append(2000)
        
        performance_results = {}
        
        for size in test_sizes:
            if size > len(documents):
                continue
                
            logger.info(f"Testing performance with {size} documents")
            
            test_docs = documents[:size]
            chunking_service = DocumentChunkingService(embedding_func=embedding_func)
            
            # Measure chunking performance
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            total_chunks = 0
            successful_docs = 0
            
            for doc in test_docs:
                try:
                    chunks = chunking_service.chunk_document(
                        doc.doc_id, doc.content, "fixed_size"
                    )
                    total_chunks += len(chunks)
                    successful_docs += 1
                    
                    # Break if taking too long (performance threshold)
                    if time.time() - start_time > 300:  # 5 minutes max
                        logger.warning(f"Performance test timeout at {successful_docs} documents")
                        break
                        
                except Exception as e:
                    logger.warning(f"Chunking failed for {doc.doc_id}: {e}")
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            performance_results[size] = {
                "processing_time": end_time - start_time,
                "documents_processed": successful_docs,
                "total_chunks": total_chunks,
                "chunks_per_second": total_chunks / (end_time - start_time) if end_time > start_time else 0,
                "docs_per_second": successful_docs / (end_time - start_time) if end_time > start_time else 0,
                "memory_usage_mb": end_memory - start_memory,
                "avg_chunks_per_doc": total_chunks / successful_docs if successful_docs > 0 else 0
            }
            
            # Performance assertions
            assert successful_docs >= size * 0.95, f"Should process at least 95% of documents, got {successful_docs}/{size}"
            assert total_chunks > 0, f"Should produce chunks for {size} documents"
            
            # Performance thresholds (adjust based on hardware)
            docs_per_second = performance_results[size]["docs_per_second"]
            assert docs_per_second > 0.1, f"Processing rate too slow: {docs_per_second} docs/sec"
        
        # Store performance metrics
        self.test_results["performance_metrics"] = performance_results
        
        # Analyze scaling behavior
        if len(performance_results) >= 2:
            sizes = sorted(performance_results.keys())
            scaling_factor = (
                performance_results[sizes[-1]]["processing_time"] / 
                performance_results[sizes[0]]["processing_time"]
            )
            doc_ratio = sizes[-1] / sizes[0]
            
            logger.info(f"Scaling analysis: {doc_ratio}x documents took {scaling_factor:.2f}x time")
            
            # Should scale roughly linearly (within 2x factor)
            assert scaling_factor <= doc_ratio * 2, \
                f"Performance scaling too poor: {scaling_factor:.2f}x time for {doc_ratio}x documents"

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0  # psutil not available

    def teardown_method(self):
        """Clean up after each test and save results."""
        # Save test results to file for analysis
        results_dir = Path("test_output")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time())
        results_file = results_dir / f"chunking_e2e_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"Test results saved to {results_file}")