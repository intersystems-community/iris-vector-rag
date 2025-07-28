"""
Test suite for pipeline chunking architecture migrations.

This module tests that all RAG pipelines properly integrate with the chunking
architecture via IRISVectorStore, following the migration strategy defined
in docs/PIPELINE_MIGRATION_STRATEGY.md.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document


class TestPipelineChunkingMigrations:
    """Test chunking integration for all RAG pipelines."""
    
    @pytest.fixture
    def config_manager(self):
        """Create a test configuration manager."""
        return ConfigurationManager()
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store for testing."""
        mock_store = Mock()
        mock_store.add_documents.return_value = {
            'documents_added': 2,
            'chunks_created': 5,
            'chunking_enabled': True,
            'chunking_strategy': 'fixed_size'
        }
        return mock_store
    
    @pytest.fixture
    def sample_documents_dir(self):
        """Create a temporary directory with sample documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample text file (should be chunked)
            large_text_path = os.path.join(temp_dir, "large_document.txt")
            with open(large_text_path, 'w') as f:
                f.write("This is a large document. " * 200)  # Large enough to trigger chunking
            
            # Create sample JSON file (should not be chunked for SQL RAG)
            json_path = os.path.join(temp_dir, "structured_data.json")
            with open(json_path, 'w') as f:
                f.write('{"key": "value", "data": "structured"}')
            
            # Create sample markdown file
            md_path = os.path.join(temp_dir, "documentation.md")
            with open(md_path, 'w') as f:
                f.write("# Documentation\n\nThis is documentation content. " * 100)
            
            yield temp_dir
    
    def test_crag_pipeline_chunking_integration(self, config_manager, mock_vector_store, sample_documents_dir):
        """Test CRAG pipeline uses chunking via IRISVectorStore."""
        from iris_rag.pipelines.crag import CRAGPipeline
        
        with patch('iris_rag.pipelines.crag.IRISVectorStore', return_value=mock_vector_store):
            pipeline = CRAGPipeline(config_manager=config_manager)
            
            # Test document loading with chunking
            result = pipeline.load_documents(sample_documents_dir)
            
            # Verify vector store was called with chunking enabled
            mock_vector_store.add_documents.assert_called()
            call_args = mock_vector_store.add_documents.call_args
            
            assert 'auto_chunk' in call_args.kwargs
            assert 'chunking_strategy' in call_args.kwargs
            
            # Verify result contains chunking information
            assert 'documents_loaded' in result
            assert 'chunks_created' in result
            assert 'chunking_enabled' in result
    
    def test_graphrag_pipeline_semantic_chunking(self, config_manager, mock_vector_store, sample_documents_dir):
        """Test GraphRAG pipeline uses semantic chunking for entity extraction."""
        from iris_rag.pipelines.graphrag import GraphRAGPipeline
        
        with patch('iris_rag.pipelines.graphrag.IRISVectorStore', return_value=mock_vector_store):
            pipeline = GraphRAGPipeline(config_manager=config_manager)
            
            # Test document loading with semantic chunking
            result = pipeline.load_documents(sample_documents_dir)
            
            # Verify semantic chunking strategy is used
            mock_vector_store.add_documents.assert_called()
            call_args = mock_vector_store.add_documents.call_args
            
            # GraphRAG should use semantic chunking by default
            assert call_args.kwargs.get('chunking_strategy') == 'semantic'
            assert call_args.kwargs.get('auto_chunk') is True
    
    def test_noderag_pipeline_chunking_integration(self, config_manager, mock_vector_store, sample_documents_dir):
        """Test NodeRAG pipeline uses fixed-size chunking."""
        from iris_rag.pipelines.noderag import NodeRAGPipeline
        
        with patch('iris_rag.pipelines.noderag.IRISVectorStore', return_value=mock_vector_store):
            pipeline = NodeRAGPipeline(config_manager=config_manager)
            
            # Test document loading
            result = pipeline.load_documents(sample_documents_dir)
            
            # Verify chunking is enabled with fixed-size strategy
            mock_vector_store.add_documents.assert_called()
            call_args = mock_vector_store.add_documents.call_args
            
            assert call_args.kwargs.get('auto_chunk') is True
            assert call_args.kwargs.get('chunking_strategy') == 'fixed_size'
    
    def test_hybrid_ifind_pipeline_chunking_integration(self, config_manager, mock_vector_store, sample_documents_dir):
        """Test HybridIFind pipeline uses hybrid chunking strategy."""
        from iris_rag.pipelines.hybrid_ifind import HybridIFindPipeline
        
        with patch('iris_rag.pipelines.hybrid_ifind.IRISVectorStore', return_value=mock_vector_store):
            pipeline = HybridIFindPipeline(config_manager=config_manager)
            
            # Test document loading
            result = pipeline.load_documents(sample_documents_dir)
            
            # Verify hybrid chunking strategy is used
            mock_vector_store.add_documents.assert_called()
            call_args = mock_vector_store.add_documents.call_args
            
            assert call_args.kwargs.get('auto_chunk') is True
            assert call_args.kwargs.get('chunking_strategy') == 'hybrid'
    
    def test_colbert_pipeline_disabled_chunking(self, config_manager, mock_vector_store, sample_documents_dir):
        """Test ColBERT pipeline disables chunking due to token-level embeddings."""
        from iris_rag.pipelines.colbert.pipeline import ColBERTPipeline
        
        with patch('iris_rag.pipelines.colbert.pipeline.IRISVectorStore', return_value=mock_vector_store):
            pipeline = ColBERTPipeline(config_manager=config_manager)
            
            # Test document loading
            result = pipeline.load_documents(sample_documents_dir)
            
            # Verify chunking is disabled for ColBERT
            mock_vector_store.add_documents.assert_called()
            call_args = mock_vector_store.add_documents.call_args
            
            assert call_args.kwargs.get('auto_chunk') is False
    
    def test_sql_rag_conditional_chunking(self, config_manager, mock_vector_store, sample_documents_dir):
        """Test SQL RAG pipeline uses conditional chunking based on document type."""
        from iris_rag.pipelines.sql_rag import SQLRAGPipeline
        
        with patch('iris_rag.pipelines.sql_rag.IRISVectorStore', return_value=mock_vector_store):
            pipeline = SQLRAGPipeline(config_manager=config_manager)
            
            # Test document loading
            result = pipeline.load_documents(sample_documents_dir)
            
            # Verify conditional chunking logic
            assert 'chunked_documents' in result
            assert 'non_chunked_documents' in result
            
            # Should have both chunked and non-chunked documents
            assert result['chunked_documents'] > 0  # Large text files should be chunked
            assert result['non_chunked_documents'] > 0  # JSON files should not be chunked
    
    def test_hyde_pipeline_chunking_integration(self, config_manager, mock_vector_store, sample_documents_dir):
        """Test HyDE RAG pipeline uses chunking via IRISVectorStore."""
        from iris_rag.pipelines.hyde import HyDERAGPipeline
        
        with patch('iris_rag.pipelines.hyde.IRISVectorStore', return_value=mock_vector_store):
            pipeline = HyDERAGPipeline(config_manager=config_manager)
            
            # Test document loading with chunking
            result = pipeline.load_documents(sample_documents_dir)
            
            # Verify vector store was called with chunking enabled
            mock_vector_store.add_documents.assert_called()
            call_args = mock_vector_store.add_documents.call_args
            
            assert 'auto_chunk' in call_args.kwargs
            assert 'chunking_strategy' in call_args.kwargs
            
            # Verify result contains chunking information
            assert 'documents_loaded' in result
            assert 'chunks_created' in result
            assert 'chunking_enabled' in result
            assert 'chunking_strategy' in result
            
            # HyDE should use fixed-size chunking by default
            assert call_args.kwargs.get('chunking_strategy') == 'fixed_size'
            assert call_args.kwargs.get('auto_chunk') is True
    
    def test_sql_rag_document_type_detection(self, config_manager, mock_vector_store):
        """Test SQL RAG pipeline correctly detects document types."""
        from iris_rag.pipelines.sql_rag import SQLRAGPipeline
        
        with patch('iris_rag.pipelines.sql_rag.IRISVectorStore', return_value=mock_vector_store):
            pipeline = SQLRAGPipeline(config_manager=config_manager)
            
            # Test document type detection
            assert pipeline._determine_document_type("test.json", '{"key": "value"}') == "json"
            assert pipeline._determine_document_type("test.csv", "col1,col2\nval1,val2") == "csv"
            assert pipeline._determine_document_type("test.sql", "SELECT * FROM table") == "sql"
            assert pipeline._determine_document_type("test.md", "# Header\nContent") == "markdown"
            assert pipeline._determine_document_type("test.txt", "Regular text content") == "text"
            
            # Test table detection
            table_content = "| col1 | col2 |\n|------|------|\n| val1 | val2 |" * 10
            assert pipeline._determine_document_type("test.txt", table_content) == "table"
    
    def test_sql_rag_chunking_decision_logic(self, config_manager, mock_vector_store):
        """Test SQL RAG pipeline chunking decision logic."""
        from iris_rag.pipelines.sql_rag import SQLRAGPipeline
        
        with patch('iris_rag.pipelines.sql_rag.IRISVectorStore', return_value=mock_vector_store):
            pipeline = SQLRAGPipeline(config_manager=config_manager)
            
            # Small documents should not be chunked
            small_content = "Short content"
            assert not pipeline._should_chunk_document(small_content, "test.txt")
            
            # Large text documents should be chunked
            large_content = "Large content. " * 200
            assert pipeline._should_chunk_document(large_content, "test.txt")
            
            # Structured files should not be chunked regardless of size
            large_json = '{"data": "' + "x" * 3000 + '"}'
            assert not pipeline._should_chunk_document(large_json, "test.json")
            
            # Table content should not be chunked
            table_content = "| col1 | col2 |\n" * 100
            assert not pipeline._should_chunk_document(table_content, "test.txt")
    
    def test_pipeline_configuration_overrides(self, config_manager):
        """Test that pipelines read their specific chunking configurations."""
        # Mock configuration to return pipeline-specific settings
        with patch.object(config_manager, 'get_config') as mock_get_config:
            mock_get_config.return_value = {
                'enabled': True,
                'strategy': 'semantic',
                'size_threshold': 1500
            }
            
            from iris_rag.pipelines.graphrag import GraphRAGPipeline
            
            with patch('iris_rag.pipelines.graphrag.IRISVectorStore'):
                pipeline = GraphRAGPipeline(config_manager=config_manager)
                
                # Create a temporary file to test with
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write("Test content")
                    temp_path = f.name
                
                try:
                    pipeline.load_documents(temp_path)
                    
                    # Verify the pipeline requested its specific configuration
                    mock_get_config.assert_called_with("pipeline_overrides:graphrag:chunking", {})
                finally:
                    os.unlink(temp_path)
    
    def test_all_pipelines_use_vector_store_interface(self, config_manager, mock_vector_store):
        """Test that all pipelines use the IRISVectorStore interface instead of direct SQL."""
        pipeline_classes = [
            ('iris_rag.pipelines.crag', 'CRAGPipeline'),
            ('iris_rag.pipelines.graphrag', 'GraphRAGPipeline'),
            ('iris_rag.pipelines.noderag', 'NodeRAGPipeline'),
            ('iris_rag.pipelines.hybrid_ifind', 'HybridIFindPipeline'),
            ('iris_rag.pipelines.colbert.pipeline', 'ColBERTPipeline'),
            ('iris_rag.pipelines.sql_rag', 'SQLRAGPipeline'),
            ('iris_rag.pipelines.hyde', 'HyDERAGPipeline'),
        ]
        
        for module_name, class_name in pipeline_classes:
            with patch(f'{module_name}.IRISVectorStore', return_value=mock_vector_store):
                module = __import__(module_name, fromlist=[class_name])
                pipeline_class = getattr(module, class_name)
                pipeline = pipeline_class(config_manager=config_manager)
                
                # Verify the pipeline has a vector_store attribute
                assert hasattr(pipeline, 'vector_store')
                assert pipeline.vector_store == mock_vector_store
    
    def test_chunking_strategy_consistency(self, config_manager):
        """Test that chunking strategies are consistently applied across pipelines."""
        expected_strategies = {
            'crag': 'fixed_size',
            'graphrag': 'semantic',
            'noderag': 'fixed_size',
            'hybrid_ifind': 'hybrid',
            'colbert': None,  # Chunking disabled
            'sql_rag': 'fixed_size',  # Default for conditional chunking
            'hyde': 'fixed_size'  # HyDE uses fixed-size chunking
        }
        
        for pipeline_name, expected_strategy in expected_strategies.items():
            config_key = f"pipeline_overrides:{pipeline_name}:chunking"
            chunking_config = config_manager.get_config(config_key, {})
            
            if expected_strategy is None:
                # ColBERT should have chunking disabled
                assert chunking_config.get('enabled', True) is False
            else:
                # Other pipelines should have the expected strategy
                actual_strategy = chunking_config.get('strategy', 'fixed_size')
                assert actual_strategy == expected_strategy, f"Pipeline {pipeline_name} should use {expected_strategy} strategy"


class TestChunkingConfigurationIntegration:
    """Test the configuration system for pipeline-specific chunking overrides."""
    
    def test_configuration_hierarchy(self):
        """Test that pipeline overrides properly override default settings."""
        config_manager = ConfigurationManager()
        
        # Test that pipeline-specific config overrides defaults
        default_chunking = config_manager.get_config("chunking", {})
        graphrag_chunking = config_manager.get_config("pipeline_overrides:graphrag:chunking", {})
        
        # GraphRAG should have semantic chunking override
        assert graphrag_chunking.get('strategy') == 'semantic'
        
        # ColBERT should have chunking disabled
        colbert_chunking = config_manager.get_config("pipeline_overrides:colbert:chunking", {})
        assert colbert_chunking.get('enabled') is False
    
    def test_missing_pipeline_config_fallback(self):
        """Test that pipelines fall back to defaults when no override is specified."""
        config_manager = ConfigurationManager()
        
        # Test with a non-existent pipeline
        nonexistent_config = config_manager.get_config("pipeline_overrides:nonexistent:chunking", {})
        assert nonexistent_config == {}
        
        # Should fall back to default chunking behavior
        default_chunking = config_manager.get_config("chunking", {})
        assert 'enabled' in default_chunking or 'strategy' in default_chunking


if __name__ == "__main__":
    pytest.main([__file__, "-v"])