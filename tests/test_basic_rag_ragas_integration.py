"""
Test Basic RAG Pipeline integration with RAGAS evaluation.

This test suite ensures that the Basic RAG pipeline works correctly
with RAGAS evaluation framework, covering all aspects of the integration.
"""

import pytest
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.core.models import Document
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager


class TestBasicRAGRagasIntegration:
    """Test suite for Basic RAG Pipeline RAGAS integration."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager."""
        return Mock(spec=ConnectionManager)
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.get.return_value = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "default_top_k": 5,
            "embedding_batch_size": 32
        }
        return mock_config
    
    @pytest.fixture
    def mock_llm_func(self):
        """Mock LLM function for answer generation."""
        def llm_func(prompt: str) -> str:
            return "Based on the provided context, diabetes is a metabolic disorder that affects blood sugar levels."
        return llm_func
    
    @pytest.fixture
    def test_documents(self):
        """Test documents for retrieval."""
        return [
            Document(
                page_content="Diabetes is a metabolic disorder characterized by high blood sugar levels.",
                metadata={"source": "medical_doc_1.txt", "doc_id": "1"}
            ),
            Document(
                page_content="Type 2 diabetes is the most common form of diabetes.",
                metadata={"source": "medical_doc_2.txt", "doc_id": "2"}
            ),
            Document(
                page_content="Insulin resistance is a key factor in type 2 diabetes development.",
                metadata={"source": "medical_doc_3.txt", "doc_id": "3"}
            )
        ]
    
    @pytest.fixture
    def basic_rag_pipeline(self, mock_connection_manager, mock_config_manager, mock_llm_func, test_documents):
        """Create a Basic RAG pipeline with mocked components."""
        with patch('iris_rag.pipelines.basic.IRISStorage') as mock_storage_class, \
             patch('iris_rag.pipelines.basic.EmbeddingManager') as mock_embedding_class:
            
            # Configure storage mock
            mock_storage = Mock()
            mock_storage_class.return_value = mock_storage
            mock_storage.initialize_schema.return_value = None
            mock_storage.vector_search.return_value = [
                (test_documents[0], 0.92),
                (test_documents[1], 0.85),
                (test_documents[2], 0.78)
            ]
            
            # Configure embedding manager mock
            mock_embedding_manager = Mock()
            mock_embedding_class.return_value = mock_embedding_manager
            mock_embedding_manager.embed_text.return_value = [0.1] * 384
            
            # Create pipeline
            pipeline = BasicRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                llm_func=mock_llm_func
            )
            
            return pipeline
    
    def test_basic_rag_retrieval_step(self, basic_rag_pipeline):
        """Test that Basic RAG can retrieve documents correctly."""
        # Test retrieval
        query = "What is diabetes?"
        retrieved_documents = basic_rag_pipeline.query(query, top_k=3)
        
        # Verify retrieval results
        assert len(retrieved_documents) == 3
        assert all(isinstance(doc, Document) for doc in retrieved_documents)
        
        # Verify document content
        first_doc = retrieved_documents[0]
        assert "diabetes" in first_doc.page_content.lower()
        assert first_doc.metadata["source"] == "medical_doc_1.txt"
    
    def test_basic_rag_full_pipeline_execution(self, basic_rag_pipeline):
        """Test full pipeline execution for RAGAS compatibility."""
        # Execute full pipeline
        query = "What is diabetes and what causes it?"
        result = basic_rag_pipeline.execute(query, top_k=3)
        
        # Verify RAGAS-compatible result format
        required_keys = ["query", "answer", "retrieved_documents"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Verify result content
        assert result["query"] == query
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0
        assert isinstance(result["retrieved_documents"], list)
        assert len(result["retrieved_documents"]) == 3
    
    def test_ragas_context_extraction_from_documents(self, basic_rag_pipeline):
        """Test context extraction from retrieved documents for RAGAS."""
        # Execute pipeline
        query = "What is diabetes?"
        result = basic_rag_pipeline.execute(query, top_k=3)
        
        # Extract contexts (as done in RAGAS evaluation)
        retrieved_documents = result["retrieved_documents"]
        contexts = []
        
        for doc in retrieved_documents:
            if isinstance(doc, Document):
                contexts.append(doc.page_content)
            elif isinstance(doc, dict) and 'page_content' in doc:
                contexts.append(doc['page_content'])
            else:
                pytest.fail(f"Unexpected document format: {type(doc)}")
        
        # Verify contexts
        assert len(contexts) == 3
        assert all(isinstance(ctx, str) for ctx in contexts)
        assert all(len(ctx.strip()) > 0 for ctx in contexts)
        
        # Verify content relevance
        assert any("diabetes" in ctx.lower() for ctx in contexts)
    
    def test_ragas_data_structure_preparation(self, basic_rag_pipeline):
        """Test preparation of data structure for RAGAS evaluation."""
        # Execute pipeline
        query = "What is diabetes?"
        result = basic_rag_pipeline.execute(query, top_k=3)
        
        # Extract data for RAGAS
        answer = result["answer"]
        retrieved_documents = result["retrieved_documents"]
        
        # Extract contexts
        contexts = [doc.page_content for doc in retrieved_documents if isinstance(doc, Document)]
        
        # Create ground truth (normally from dataset)
        ground_truth = "Diabetes is a chronic condition affecting blood sugar levels."
        
        # Prepare RAGAS data structure
        ragas_data = {
            'question': [query],
            'answer': [answer],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        }
        
        # Verify RAGAS data structure
        assert len(ragas_data['question']) == 1
        assert len(ragas_data['answer']) == 1
        assert len(ragas_data['contexts']) == 1
        assert len(ragas_data['ground_truth']) == 1
        assert len(ragas_data['contexts'][0]) > 0
        
        # Verify data types
        assert isinstance(ragas_data['question'][0], str)
        assert isinstance(ragas_data['answer'][0], str)
        assert isinstance(ragas_data['contexts'][0], list)
        assert isinstance(ragas_data['ground_truth'][0], str)
    
    @pytest.mark.parametrize("document_format", [
        "document_objects",
        "dict_with_page_content",
        "dict_with_content",
        "mixed_formats"
    ])
    def test_context_extraction_edge_cases(self, document_format):
        """Test context extraction with different document formats."""
        # Create test documents in different formats
        if document_format == "document_objects":
            documents = [
                Document(page_content="Content 1", metadata={"source": "doc1"}),
                Document(page_content="Content 2", metadata={"source": "doc2"})
            ]
        elif document_format == "dict_with_page_content":
            documents = [
                {"page_content": "Content 1", "metadata": {"source": "doc1"}},
                {"page_content": "Content 2", "metadata": {"source": "doc2"}}
            ]
        elif document_format == "dict_with_content":
            documents = [
                {"content": "Content 1", "metadata": {"source": "doc1"}},
                {"content": "Content 2", "metadata": {"source": "doc2"}}
            ]
        elif document_format == "mixed_formats":
            documents = [
                Document(page_content="Content 1", metadata={"source": "doc1"}),
                {"page_content": "Content 2", "metadata": {"source": "doc2"}},
                {"content": "Content 3", "metadata": {"source": "doc3"}}
            ]
        
        # Extract contexts using the same logic as in RAGAS evaluation
        contexts = []
        for doc in documents:
            if isinstance(doc, Document):
                if doc.page_content.strip():
                    contexts.append(doc.page_content)
            elif isinstance(doc, dict):
                if 'page_content' in doc and doc['page_content'].strip():
                    contexts.append(doc['page_content'])
                elif 'content' in doc and doc['content'].strip():
                    contexts.append(doc['content'])
        
        # Verify contexts were extracted correctly
        expected_count = len(documents) if document_format != "mixed_formats" else 3
        assert len(contexts) == expected_count
        assert all(isinstance(ctx, str) for ctx in contexts)
        assert all(len(ctx.strip()) > 0 for ctx in contexts)
    
    def test_empty_documents_handling(self):
        """Test handling of empty documents list."""
        documents = []
        
        # Extract contexts
        contexts = []
        for doc in documents:
            if isinstance(doc, Document):
                contexts.append(doc.page_content)
            elif isinstance(doc, dict) and 'page_content' in doc:
                contexts.append(doc['page_content'])
        
        # Should handle empty list gracefully
        assert len(contexts) == 0
    
    def test_documents_with_empty_content(self):
        """Test handling of documents with empty content."""
        documents = [
            Document(page_content="", metadata={"source": "doc1"}),
            Document(page_content="Valid content", metadata={"source": "doc2"}),
            Document(page_content="   ", metadata={"source": "doc3"})  # Whitespace only
        ]
        
        # Extract contexts, filtering empty content
        contexts = []
        for doc in documents:
            if isinstance(doc, Document) and doc.page_content.strip():
                contexts.append(doc.page_content)
        
        # Should only include non-empty content
        assert len(contexts) == 1
        assert contexts[0] == "Valid content"
    
    def test_pipeline_standard_return_format(self, basic_rag_pipeline):
        """Test that pipeline returns the standard format required by the project."""
        # Execute pipeline
        query = "What is diabetes?"
        result = basic_rag_pipeline.execute(query)
        
        # Verify standard return format as per project rules
        required_keys = ["query", "answer", "retrieved_documents"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"
        
        # Verify data types
        assert isinstance(result["query"], str)
        assert isinstance(result["answer"], str)
        assert isinstance(result["retrieved_documents"], list)
        
        # Verify content
        assert result["query"] == query
        assert len(result["answer"]) > 0
        assert len(result["retrieved_documents"]) > 0


class TestBasicRAGRagasCompatibility:
    """Test RAGAS compatibility specifically."""
    
    def test_ragas_imports_available(self):
        """Test that RAGAS can be imported (if installed)."""
        try:
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision
            from datasets import Dataset
            ragas_available = True
        except ImportError:
            ragas_available = False
            pytest.skip("RAGAS not installed, skipping RAGAS-specific tests")
        
        assert ragas_available
    
    @pytest.mark.skipif(
        not pytest.importorskip("ragas", reason="RAGAS not installed"),
        reason="RAGAS not available"
    )
    def test_ragas_dataset_creation(self):
        """Test creating a RAGAS dataset from pipeline results."""
        from datasets import Dataset
        
        # Mock pipeline result
        pipeline_result = {
            "query": "What is diabetes?",
            "answer": "Diabetes is a metabolic disorder.",
            "retrieved_documents": [
                Document(page_content="Diabetes info 1", metadata={}),
                Document(page_content="Diabetes info 2", metadata={})
            ]
        }
        
        # Extract data for RAGAS
        query = pipeline_result["query"]
        answer = pipeline_result["answer"]
        contexts = [doc.page_content for doc in pipeline_result["retrieved_documents"]]
        ground_truth = "Diabetes is a chronic condition."
        
        # Create RAGAS dataset
        ragas_data = {
            'question': [query],
            'answer': [answer],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        }
        
        dataset = Dataset.from_dict(ragas_data)
        
        # Verify dataset creation
        assert len(dataset) == 1
        assert 'question' in dataset.column_names
        assert 'answer' in dataset.column_names
        assert 'contexts' in dataset.column_names
        assert 'ground_truth' in dataset.column_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])