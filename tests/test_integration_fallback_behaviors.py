#!/usr/bin/env python3
"""
Integration tests for fallback behaviors with real database connections.

This test suite complements the unit tests in test_fallback_behavior_validation.py
by testing fallback behaviors with actual IRIS database connections.
"""

import pytest
import logging
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
from iris_rag.config.manager import ConfigurationManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.usefixtures("initialized_database")
class TestIntegrationFallbackBehaviors:
    """
    Integration tests for fallback behaviors using real database connections.
    These tests complement the unit tests by verifying actual database behavior.
    """
    
    @pytest.fixture
    def real_config_manager(self):
        """Create a real configuration manager for integration tests."""
        return ConfigurationManager()
    
    def test_hybrid_ifind_real_database_fallback(self, real_config_manager):
        """
        Test HybridIFind with real database - should gracefully handle missing IFind indexes.
        """
        # Create pipeline with real database connection
        pipeline = HybridIFindRAGPipeline(config_manager=real_config_manager)
        
        # Query should work even if IFind indexes don't exist yet
        # (should fall back to vector search)
        result = pipeline.query("test medical query")
        
        # Should return a valid result structure
        assert isinstance(result, dict)
        assert "query" in result
        assert "retrieved_documents" in result
        assert "pipeline_type" in result
        
        # Should indicate which search methods were used
        assert "vector_results_count" in result
        assert "ifind_results_count" in result
        
        logger.info("✅ HybridIFind real database fallback test passed")
    
    def test_graphrag_real_database_entity_fallback(self, real_config_manager):
        """
        Test GraphRAG with real database - should fall back to vector search if no entities.
        """
        pipeline = GraphRAGPipeline(config_manager=real_config_manager)
        
        # Query should work even if entity extraction fails or no entities exist
        result = pipeline.query("test query about unknown entities")
        
        # Should return valid results
        assert isinstance(result, dict)
        assert "query" in result
        assert "retrieved_documents" in result
        assert "pipeline_type" in result
        
        logger.info("✅ GraphRAG real database entity fallback test passed")
    
    def test_crag_real_database_chunking_resilience(self, real_config_manager):
        """
        Test CRAG with real database - should handle document ingestion gracefully.
        """
        pipeline = CRAGPipeline(config_manager=real_config_manager)
        
        # Test with a simple document
        from iris_rag.core.models import Document
        test_doc = Document(
            page_content="This is a test document for CRAG integration testing.",
            metadata={"source": "integration_test", "type": "test"}
        )
        
        # Should handle document ingestion
        result = pipeline.ingest_documents([test_doc])
        
        # Should return status information
        assert isinstance(result, dict)
        assert "status" in result
        
        # Query should work after ingestion
        query_result = pipeline.query("test document")
        assert isinstance(query_result, dict)
        assert "retrieved_documents" in query_result
        
        logger.info("✅ CRAG real database chunking resilience test passed")
    
    def test_hyde_real_database_no_llm_fallback(self, real_config_manager):
        """
        Test HyDE with real database - should fall back to direct search without LLM.
        """
        # Create pipeline without LLM function
        pipeline = HyDERAGPipeline(
            config_manager=real_config_manager,
            llm_func=None
        )
        
        # Should fall back to direct vector search
        result = pipeline.query("test medical query")
        
        # Should return valid results
        assert isinstance(result, dict)
        assert "query" in result
        assert "retrieved_documents" in result
        assert "pipeline_type" in result
        
        # Should indicate no hypotheses were generated
        assert result.get("hypotheses") is None or len(result.get("hypotheses", [])) == 0
        
        logger.info("✅ HyDE real database no LLM fallback test passed")
    
    def test_colbert_real_database_token_fallback(self, real_config_manager):
        """
        Test ColBERT with real database - should fall back to vector search if no token embeddings.
        """
        pipeline = ColBERTRAGPipeline(config_manager=real_config_manager)
        
        # Query should work even if token embeddings table is empty
        result = pipeline.query("test query")
        
        # Should return valid results (likely from vector fallback)
        assert isinstance(result, dict)
        assert "query" in result
        assert "retrieved_documents" in result
        assert "pipeline_type" in result
        
        logger.info("✅ ColBERT real database token fallback test passed")
    
    def test_all_pipelines_database_connectivity(self, real_config_manager):
        """
        Test that all pipelines can connect to the database and handle basic queries.
        """
        pipelines = [
            ("HybridIFind", HybridIFindRAGPipeline(config_manager=real_config_manager)),
            ("GraphRAG", GraphRAGPipeline(config_manager=real_config_manager)),
            ("CRAG", CRAGPipeline(config_manager=real_config_manager)),
            ("HyDE", HyDERAGPipeline(config_manager=real_config_manager)),
            ("ColBERT", ColBERTRAGPipeline(config_manager=real_config_manager))
        ]
        
        for name, pipeline in pipelines:
            try:
                result = pipeline.query("test connectivity query")
                
                # Should return valid structure
                assert isinstance(result, dict)
                assert "query" in result
                assert "pipeline_type" in result
                
                logger.info(f"✅ {name} database connectivity test passed")
                
            except Exception as e:
                logger.error(f"❌ {name} database connectivity failed: {e}")
                # Don't fail the test immediately - collect all failures
                pytest.fail(f"{name} pipeline failed database connectivity test: {e}")
    
    def test_database_schema_resilience(self, real_config_manager):
        """
        Test that pipelines handle missing or incomplete database schemas gracefully.
        """
        # This test verifies that pipelines don't crash when expected tables might be missing
        pipeline = HybridIFindRAGPipeline(config_manager=real_config_manager)
        
        # Should handle queries even if some expected tables don't exist
        result = pipeline.query("schema resilience test")
        
        # Should return some form of result, even if empty
        assert isinstance(result, dict)
        assert "query" in result
        assert "retrieved_documents" in result
        
        logger.info("✅ Database schema resilience test passed")


if __name__ == "__main__":
    # Run integration tests with verbose output
    pytest.main([__file__, "-v", "-s"])