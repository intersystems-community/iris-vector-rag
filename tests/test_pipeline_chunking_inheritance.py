"""
Test suite for the inheritance-based chunking architecture refactor.

This module tests the base RAGPipeline class chunking behavior and validates
that all pipeline implementations correctly inherit and use the base class
load_documents method with pipeline-specific configuration overrides.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from iris_rag.core.base import RAGPipeline
from iris_rag.core.models import Document
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.noderag import NodeRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline


class TestBaseRAGPipelineChunking:
    """Test the base RAGPipeline class chunking behavior."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager with pipeline overrides."""
        config_manager = Mock(spec=ConfigurationManager)
        
        # Default chunking configuration
        default_chunking = {
            "enabled": True,
            "strategy": "fixed_size",
            "chunk_size": 512,
            "overlap": 50
        }
        
        # Pipeline-specific overrides
        pipeline_overrides = {
            "basic": {"chunking": {"enabled": True, "strategy": "fixed_size"}},
            "crag": {"chunking": {"enabled": True, "strategy": "fixed_size"}},
            "graphrag": {"chunking": {"enabled": True, "strategy": "semantic"}},
            "noderag": {"chunking": {"enabled": True, "strategy": "fixed_size"}},
            "hyde": {"chunking": {"enabled": True, "strategy": "fixed_size"}},
            "hybrid_ifind": {"chunking": {"enabled": True, "strategy": "hybrid"}},
            "sqlrag": {"chunking": {"enabled": True, "strategy": "fixed_size", "size_threshold": 2000}},
            "colbert": {"chunking": {"enabled": False}}
        }
        
        def get_config_side_effect(key, default=None):
            if key == "storage:chunking":
                return default_chunking
            elif key.startswith("pipeline_overrides:"):
                # Extract pipeline name and config path
                parts = key.split(":")
                if len(parts) >= 3:
                    pipeline_name = parts[1]
                    config_path = ":".join(parts[2:])
                    if pipeline_name in pipeline_overrides:
                        pipeline_config = pipeline_overrides[pipeline_name]
                        # Navigate to the specific config path
                        current = pipeline_config
                        for part in config_path.split(":"):
                            if isinstance(current, dict) and part in current:
                                current = current[part]
                            else:
                                return default
                        return current
            return default
        
        config_manager.get_config.side_effect = get_config_side_effect
        return config_manager
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        vector_store = Mock()
        vector_store.add_documents.return_value = {
            "chunks_created": 5,
            "documents_processed": 2,
            "status": "success"
        }
        return vector_store
    
    @pytest.fixture
    def sample_documents_dir(self):
        """Create a temporary directory with sample documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample text files
            doc1_path = os.path.join(temp_dir, "doc1.txt")
            doc2_path = os.path.join(temp_dir, "doc2.md")
            
            with open(doc1_path, 'w', encoding='utf-8') as f:
                f.write("This is the content of document 1. " * 50)  # ~1500 chars
            
            with open(doc2_path, 'w', encoding='utf-8') as f:
                f.write("This is the content of document 2. " * 30)  # ~900 chars
            
            yield temp_dir
    
    def test_get_pipeline_name(self, mock_config_manager, mock_vector_store):
        """Test that _get_pipeline_name correctly extracts pipeline name from class name."""
        
        class TestRAGPipeline(RAGPipeline):
            def run(self, query: str, **kwargs) -> Dict[str, Any]:
                return {}
        
        pipeline = TestRAGPipeline(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store
        )
        
        assert pipeline._get_pipeline_name() == "test"
    
    def test_get_chunking_config(self, mock_config_manager, mock_vector_store):
        """Test that _get_chunking_config retrieves pipeline-specific configuration."""
        
        class BasicRAGTestPipeline(RAGPipeline):
            def run(self, query: str, **kwargs) -> Dict[str, Any]:
                return {}
        
        pipeline = BasicRAGTestPipeline(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store
        )
        
        chunking_config = pipeline._get_chunking_config()
        
        # Should get basic pipeline configuration
        assert chunking_config["enabled"] is True
        assert chunking_config["strategy"] == "fixed_size"
    
    def test_load_documents_from_directory(self, mock_config_manager, mock_vector_store, sample_documents_dir):
        """Test loading documents from a directory."""
        
        class TestPipeline(RAGPipeline):
            def run(self, query: str, **kwargs) -> Dict[str, Any]:
                return {}
        
        pipeline = TestPipeline(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store
        )
        
        result = pipeline.load_documents(sample_documents_dir)
        
        # Verify vector store was called with documents
        mock_vector_store.add_documents.assert_called_once()
        call_args = mock_vector_store.add_documents.call_args
        
        # Check that documents were loaded
        documents = call_args[1]["documents"]
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)
        
        # Check chunking parameters
        assert call_args[1]["auto_chunk"] is True
        assert call_args[1]["chunking_strategy"] == "fixed_size"
        
        # Check return format
        assert "documents_loaded" in result
        assert "chunks_created" in result
        assert "chunking_enabled" in result
        assert "chunking_strategy" in result
        assert result["documents_loaded"] == 2
        assert result["chunks_created"] == 5
    
    def test_load_documents_single_file(self, mock_config_manager, mock_vector_store, sample_documents_dir):
        """Test loading a single document file."""
        
        class TestPipeline(RAGPipeline):
            def run(self, query: str, **kwargs) -> Dict[str, Any]:
                return {}
        
        pipeline = TestPipeline(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store
        )
        
        # Get path to single file
        single_file = os.path.join(sample_documents_dir, "doc1.txt")
        
        result = pipeline.load_documents(single_file)
        
        # Verify vector store was called with single document
        mock_vector_store.add_documents.assert_called_once()
        call_args = mock_vector_store.add_documents.call_args
        
        documents = call_args[1]["documents"]
        assert len(documents) == 1
        assert documents[0].page_content.startswith("This is the content of document 1")
        
        assert result["documents_loaded"] == 1
    
    def test_preprocess_documents_hook(self, mock_config_manager, mock_vector_store):
        """Test that _preprocess_documents hook is called and can modify documents."""
        
        class TestPipelineWithPreprocessing(RAGPipeline):
            def run(self, query: str, **kwargs) -> Dict[str, Any]:
                return {}
            
            def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
                # Add a custom metadata field to all documents
                for doc in documents:
                    doc.metadata["preprocessed"] = True
                return documents
        
        pipeline = TestPipelineWithPreprocessing(
            config_manager=mock_config_manager,
            vector_store=mock_vector_store
        )
        
        # Create test documents
        test_docs = [
            Document(id="1", page_content="Test content 1", metadata={"source": "test1.txt"}),
            Document(id="2", page_content="Test content 2", metadata={"source": "test2.txt"})
        ]
        
        # Mock _get_documents to return our test documents
        with patch.object(pipeline, '_get_documents', return_value=test_docs):
            result = pipeline.load_documents("dummy_path")
        
        # Verify preprocessing was applied
        mock_vector_store.add_documents.assert_called_once()
        call_args = mock_vector_store.add_documents.call_args
        documents = call_args[1]["documents"]
        
        assert all(doc.metadata.get("preprocessed") is True for doc in documents)


class TestPipelineSpecificChunking:
    """Test that specific pipeline implementations use correct chunking configurations."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Create a mock configuration manager with pipeline overrides."""
        config_manager = Mock(spec=ConfigurationManager)
        
        # Pipeline-specific overrides
        pipeline_configs = {
            "pipeline_overrides:basic:chunking": {"enabled": True, "strategy": "fixed_size"},
            "pipeline_overrides:crag:chunking": {"enabled": True, "strategy": "fixed_size"},
            "pipeline_overrides:graphrag:chunking": {"enabled": True, "strategy": "semantic"},
            "pipeline_overrides:noderag:chunking": {"enabled": True, "strategy": "fixed_size"},
            "pipeline_overrides:hyde:chunking": {"enabled": True, "strategy": "fixed_size"},
            "pipeline_overrides:hybrid_ifind:chunking": {"enabled": True, "strategy": "hybrid"},
        }
        
        def get_config_side_effect(key, default=None):
            return pipeline_configs.get(key, default)
        
        config_manager.get_config.side_effect = get_config_side_effect
        return config_manager
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        vector_store = Mock()
        vector_store.add_documents.return_value = {
            "chunks_created": 3,
            "documents_processed": 1,
            "status": "success"
        }
        return vector_store
    
    @pytest.fixture
    def sample_document_file(self):
        """Create a temporary document file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write("Sample document content for testing. " * 20)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    def test_basic_rag_chunking_config(self, mock_config_manager, mock_vector_store, sample_document_file):
        """Test that BasicRAG uses correct chunking configuration."""
        with patch('iris_rag.pipelines.basic.get_iris_connection'), \
             patch('iris_rag.pipelines.basic.get_embedding_function'), \
             patch('iris_rag.pipelines.basic.get_llm_function'):
            
            pipeline = BasicRAGPipeline(
                config_manager=mock_config_manager,
                vector_store=mock_vector_store
            )
            
            result = pipeline.load_documents(sample_document_file)
            
            # Verify chunking strategy
            mock_vector_store.add_documents.assert_called_once()
            call_args = mock_vector_store.add_documents.call_args
            assert call_args[1]["chunking_strategy"] == "fixed_size"
            assert call_args[1]["auto_chunk"] is True
    
    def test_graphrag_chunking_config(self, mock_config_manager, mock_vector_store, sample_document_file):
        """Test that GraphRAG uses semantic chunking configuration."""
        with patch('iris_rag.pipelines.graphrag.get_iris_connection'), \
             patch('iris_rag.pipelines.graphrag.get_embedding_function'), \
             patch('iris_rag.pipelines.graphrag.get_llm_function'):
            
            pipeline = GraphRAGPipeline(
                config_manager=mock_config_manager,
                vector_store=mock_vector_store
            )
            
            result = pipeline.load_documents(sample_document_file)
            
            # Verify semantic chunking strategy
            mock_vector_store.add_documents.assert_called_once()
            call_args = mock_vector_store.add_documents.call_args
            assert call_args[1]["chunking_strategy"] == "semantic"
    
    def test_hybrid_ifind_preprocessing_hook(self, mock_config_manager, mock_vector_store, sample_document_file):
        """Test that HybridIFind calls _ensure_ifind_indexes via preprocessing hook."""
        with patch('iris_rag.pipelines.hybrid_ifind.get_iris_connection'), \
             patch('iris_rag.pipelines.hybrid_ifind.get_embedding_function'), \
             patch('iris_rag.pipelines.hybrid_ifind.get_llm_function'):
            
            pipeline = HybridIFindRAGPipeline(
                config_manager=mock_config_manager,
                vector_store=mock_vector_store
            )
            
            # Mock the _ensure_ifind_indexes method
            with patch.object(pipeline, '_ensure_ifind_indexes') as mock_ensure_indexes:
                result = pipeline.load_documents(sample_document_file)
                
                # Verify that IFind indexes were ensured
                mock_ensure_indexes.assert_called_once()
                
                # Verify hybrid chunking strategy
                mock_vector_store.add_documents.assert_called_once()
                call_args = mock_vector_store.add_documents.call_args
                assert call_args[1]["chunking_strategy"] == "hybrid"


class TestSpecialCaseOverrides:
    """Test that special case pipelines (SQL RAG, ColBERT) maintain their custom overrides."""
    
    def test_sql_rag_maintains_custom_override(self):
        """Test that SQL RAG still has its custom load_documents implementation."""
        from iris_rag.pipelines.sql_rag import SQLRAGPipeline
        
        # Verify that SQL RAG has its own load_documents method
        assert hasattr(SQLRAGPipeline, 'load_documents')
        
        # Check that it's not the base class method
        base_method = RAGPipeline.load_documents
        sql_method = SQLRAGPipeline.load_documents
        
        # They should be different methods (SQL RAG has custom implementation)
        assert sql_method != base_method
    
    def test_colbert_maintains_custom_override(self):
        """Test that ColBERT still has its custom load_documents implementation."""
        from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
        
        # Verify that ColBERT has its own load_documents method
        assert hasattr(ColBERTRAGPipeline, 'load_documents')
        
        # Check that it's not the base class method
        base_method = RAGPipeline.load_documents
        colbert_method = ColBERTRAGPipeline.load_documents
        
        # They should be different methods (ColBERT has custom implementation)
        assert colbert_method != base_method


class TestInheritanceBehavior:
    """Test that refactored pipelines correctly inherit from base class."""
    
    def test_refactored_pipelines_use_base_method(self):
        """Test that refactored pipelines use the base class load_documents method."""
        
        # These pipelines should now use the base class method
        refactored_pipelines = [
            CRAGPipeline,
            NodeRAGPipeline,
            HyDERAGPipeline,
        ]
        
        base_method = RAGPipeline.load_documents
        
        for pipeline_class in refactored_pipelines:
            pipeline_method = pipeline_class.load_documents
            
            # They should be the same method (inherited from base)
            assert pipeline_method == base_method, f"{pipeline_class.__name__} should inherit base load_documents"
    
    def test_hybrid_ifind_has_preprocessing_override(self):
        """Test that HybridIFind has its own _preprocess_documents method."""
        
        # HybridIFind should have its own _preprocess_documents method
        assert hasattr(HybridIFindRAGPipeline, '_preprocess_documents')
        
        base_method = RAGPipeline._preprocess_documents
        hybrid_method = HybridIFindRAGPipeline._preprocess_documents
        
        # They should be different methods (HybridIFind has custom preprocessing)
        assert hybrid_method != base_method


if __name__ == "__main__":
    pytest.main([__file__, "-v"])