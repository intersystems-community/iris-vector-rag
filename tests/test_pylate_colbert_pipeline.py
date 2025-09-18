"""
Test PyLate ColBERT Pipeline Implementation

Tests the new PyLate-based ColBERT pipeline to ensure it works correctly
with consistent configuration and resolves the original memory/type issues.
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock
from iris_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline
from iris_rag.core.models import Document

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for pipeline testing."""
    connection_manager = Mock()
    config_manager = Mock()
    config_manager.get.return_value = {
        "rerank_factor": 2,
        "model_name": "lightonai/GTE-ModernColBERT-v1",
        "batch_size": 32,
        "use_native_reranking": True
    }
    
    llm_func = Mock()
    llm_func.return_value = "This is a test answer."
    
    vector_store = Mock()
    vector_store.search.return_value = [
        Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
        Document(page_content="Test document 2", metadata={"source": "test2.txt"}),
        Document(page_content="Test document 3", metadata={"source": "test3.txt"}),
        Document(page_content="Test document 4", metadata={"source": "test4.txt"}),
    ]
    
    return {
        "connection_manager": connection_manager,
        "config_manager": config_manager,
        "llm_func": llm_func,
        "vector_store": vector_store
    }


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(page_content="Document about machine learning algorithms", metadata={"source": "ml.txt"}),
        Document(page_content="Text about deep neural networks", metadata={"source": "nn.txt"}),
        Document(page_content="Information on natural language processing", metadata={"source": "nlp.txt"}),
        Document(page_content="Content about computer vision techniques", metadata={"source": "cv.txt"}),
        Document(page_content="Research on reinforcement learning", metadata={"source": "rl.txt"}),
    ]


class TestPyLateColBERTPipeline:
    """Test suite for PyLate ColBERT pipeline."""
    
    def test_initialization_with_defaults(self, mock_dependencies):
        """Test pipeline initialization with default configuration."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )
        
        assert pipeline.rerank_factor == 2
        assert pipeline.model_name == "lightonai/GTE-ModernColBERT-v1"
        assert pipeline.use_native_reranking == True
        assert pipeline.is_initialized == True  # Should initialize even in fallback mode
        
    def test_configuration_consistency_with_basic_reranking(self, mock_dependencies):
        """Test that configuration follows same patterns as BasicRAGReranking."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )
        
        # Should access config with same pattern as BasicRAGReranking
        mock_dependencies["config_manager"].get.assert_called_with(
            "pipelines:colbert_pylate", 
            mock_dependencies["config_manager"].get.return_value
        )
        
        # Should have consistent configuration parameters
        assert hasattr(pipeline, 'rerank_factor')
        assert hasattr(pipeline, 'stats')
        
    def test_load_documents(self, mock_dependencies, sample_documents):
        """Test document loading functionality."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )
        
        # Mock parent load_documents
        pipeline.vector_store = mock_dependencies["vector_store"]
        pipeline.vector_store.add_documents = Mock(return_value={"status": "success"})
        
        result = pipeline.load_documents(sample_documents)
        
        # Should store documents for PyLate reranking
        assert len(pipeline._document_store) == len(sample_documents)
        assert pipeline.stats['documents_indexed'] == len(sample_documents)
        
    def test_query_with_reranking(self, mock_dependencies, sample_documents):
        """Test query execution with PyLate reranking."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )
        
        # Load documents first
        pipeline.load_documents(sample_documents)
        
        # Mock the parent query method
        mock_parent_result = {
            "query": "test query",
            "retrieved_documents": sample_documents[:4],  # Initial candidates
            "execution_time": 0.5
        }
        
        # Use MagicMock to handle super() calls properly
        with pytest.fixture.scope('function'):
            original_super = pipeline.__class__.__bases__[0].query
            pipeline.__class__.__bases__[0].query = Mock(return_value=mock_parent_result)
            
            result = pipeline.query("What is machine learning?", top_k=2)
            
            # Restore original method
            pipeline.__class__.__bases__[0].query = original_super
        
        # Should return consistent response format
        assert "query" in result
        assert "answer" in result
        assert "retrieved_documents" in result
        assert "metadata" in result
        assert result["metadata"]["pipeline_type"] == "colbert_pylate"
        assert "reranked" in result["metadata"]
        assert "initial_candidates" in result["metadata"]
        assert "rerank_factor" in result["metadata"]
        
    def test_fallback_mode_when_pylate_unavailable(self, mock_dependencies):
        """Test that pipeline works in fallback mode when PyLate is not available."""
        # Create pipeline - should work even without PyLate
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )
        
        # Should initialize successfully (likely in fallback mode)
        assert pipeline.is_initialized == True
        assert pipeline.model is not None  # Should have mock model
        
    def test_pipeline_info_format(self, mock_dependencies):
        """Test that get_pipeline_info returns consistent format."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )
        
        info = pipeline.get_pipeline_info()
        
        # Should have consistent fields with BasicRAGReranking
        assert info["pipeline_type"] == "colbert_pylate"
        assert "rerank_factor" in info
        assert "model_name" in info
        assert "stats" in info
        assert "is_initialized" in info
        
    def test_memory_efficiency(self, mock_dependencies, sample_documents):
        """Test that pipeline doesn't exceed memory limits like the original ColBERT."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )
        
        # Should not crash with memory issues when processing documents
        try:
            pipeline.load_documents(sample_documents * 10)  # Larger document set
            # If this doesn't crash, memory management is working
            assert True
        except MemoryError:
            pytest.fail("Pipeline exceeded memory limits - memory management failed")
            
    def test_embedding_type_consistency(self, mock_dependencies):
        """Test that embeddings are returned as proper numpy arrays, not lists."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )
        
        # Mock model should return numpy arrays
        if hasattr(pipeline.model, 'encode'):
            embeddings = pipeline.model.encode(["test text"], is_query=True)
            
            # Should be numpy arrays or at least not raw lists without ndim
            for emb in embeddings:
                assert hasattr(emb, 'shape') or hasattr(emb, 'ndim'), \
                    "Embeddings should be numpy arrays, not lists"
                    
    def test_stats_tracking(self, mock_dependencies, sample_documents):
        """Test that pipeline tracks statistics correctly."""
        pipeline = PyLateColBERTPipeline(
            mock_dependencies["connection_manager"],
            mock_dependencies["config_manager"],
            llm_func=mock_dependencies["llm_func"],
            vector_store=mock_dependencies["vector_store"]
        )
        
        # Initial stats
        assert pipeline.stats['queries_processed'] == 0
        assert pipeline.stats['documents_indexed'] == 0
        
        # Load documents
        pipeline.load_documents(sample_documents)
        assert pipeline.stats['documents_indexed'] == len(sample_documents)
        
        # Note: Query stats would be tested with actual query execution


if __name__ == "__main__":
    pytest.main([__file__, "-v"])