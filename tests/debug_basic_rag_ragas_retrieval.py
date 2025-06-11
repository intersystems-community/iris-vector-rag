#!/usr/bin/env python3
"""
Debug script for Basic RAG Pipeline RAGAS retrieval step.

This script focuses on debugging the specific issue with basic RAG pipeline
retrieval when used in RAGAS evaluation context.
"""

import logging
import sys
import os
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_rag_retrieval_step():
    """Test the basic RAG retrieval step in isolation."""
    logger.info("🔍 Testing Basic RAG retrieval step...")
    
    try:
        # Import the basic RAG pipeline
        from iris_rag.pipelines.basic import BasicRAGPipeline
        from iris_rag.core.models import Document
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        
        logger.info("✅ Successfully imported BasicRAGPipeline")
        
        # Create mock components
        mock_connection_manager = Mock(spec=ConnectionManager)
        mock_config_manager = Mock(spec=ConfigurationManager)
        
        # Configure mock config manager
        mock_config_manager.get.return_value = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "default_top_k": 5,
            "embedding_batch_size": 32
        }
        
        # Mock the storage and embedding components
        with patch('iris_rag.pipelines.basic.IRISStorage') as mock_storage_class, \
             patch('iris_rag.pipelines.basic.EmbeddingManager') as mock_embedding_class:
            
            # Configure storage mock
            mock_storage = Mock()
            mock_storage_class.return_value = mock_storage
            mock_storage.initialize_schema.return_value = None
            
            # Configure embedding manager mock
            mock_embedding_manager = Mock()
            mock_embedding_class.return_value = mock_embedding_manager
            
            # Create test documents for retrieval
            test_documents = [
                Document(
                    page_content="Diabetes is a metabolic disorder characterized by high blood sugar levels.",
                    metadata={"source": "medical_doc_1.txt", "doc_id": "1"}
                ),
                Document(
                    page_content="Cancer treatment involves chemotherapy, radiation, and surgery.",
                    metadata={"source": "medical_doc_2.txt", "doc_id": "2"}
                ),
                Document(
                    page_content="Heart disease is the leading cause of death worldwide.",
                    metadata={"source": "medical_doc_3.txt", "doc_id": "3"}
                )
            ]
            
            # Mock vector search to return documents with scores
            mock_storage.vector_search.return_value = [
                (test_documents[0], 0.85),
                (test_documents[1], 0.72),
                (test_documents[2], 0.68)
            ]
            
            # Mock embedding generation
            mock_embedding_manager.embed_text.return_value = [0.1] * 384  # Mock embedding vector
            
            # Create pipeline instance
            pipeline = BasicRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager
            )
            
            logger.info("✅ Successfully created BasicRAGPipeline instance")
            
            # Test the query method (retrieval step)
            test_query = "What is diabetes?"
            logger.info(f"🔍 Testing retrieval with query: '{test_query}'")
            
            retrieved_documents = pipeline.query(test_query, top_k=3)
            
            logger.info(f"📊 Retrieved {len(retrieved_documents)} documents")
            
            # Verify retrieval results
            assert len(retrieved_documents) == 3, f"Expected 3 documents, got {len(retrieved_documents)}"
            assert all(isinstance(doc, Document) for doc in retrieved_documents), "All results should be Document objects"
            
            # Log document details for debugging
            for i, doc in enumerate(retrieved_documents):
                logger.info(f"  Document {i+1}:")
                logger.info(f"    Content: {doc.page_content[:100]}...")
                logger.info(f"    Metadata: {doc.metadata}")
                logger.info(f"    ID: {doc.id}")
            
            logger.info("✅ Basic retrieval test passed")
            return True, retrieved_documents
            
    except Exception as e:
        logger.error(f"❌ Basic RAG retrieval test failed: {e}")
        logger.exception("Full traceback:")
        return False, []


def test_basic_rag_ragas_integration():
    """Test Basic RAG pipeline integration with RAGAS evaluation format."""
    logger.info("🔍 Testing Basic RAG integration with RAGAS format...")
    
    try:
        # First run the basic retrieval test
        success, retrieved_documents = test_basic_rag_retrieval_step()
        if not success:
            logger.error("❌ Basic retrieval failed, cannot test RAGAS integration")
            return False
        
        # Import RAGAS evaluation components
        from iris_rag.pipelines.basic import BasicRAGPipeline
        from iris_rag.core.models import Document
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        
        # Create mock LLM function for answer generation
        def mock_llm_func(prompt: str) -> str:
            return f"Based on the provided context, diabetes is a metabolic disorder that affects blood sugar levels."
        
        # Create mock components
        mock_connection_manager = Mock(spec=ConnectionManager)
        mock_config_manager = Mock(spec=ConfigurationManager)
        mock_config_manager.get.return_value = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "default_top_k": 5,
            "embedding_batch_size": 32
        }
        
        # Mock the storage and embedding components
        with patch('iris_rag.pipelines.basic.IRISStorage') as mock_storage_class, \
             patch('iris_rag.pipelines.basic.EmbeddingManager') as mock_embedding_class:
            
            # Configure mocks
            mock_storage = Mock()
            mock_storage_class.return_value = mock_storage
            mock_storage.initialize_schema.return_value = None
            
            mock_embedding_manager = Mock()
            mock_embedding_class.return_value = mock_embedding_manager
            
            # Mock vector search to return documents with scores
            mock_storage.vector_search.return_value = [
                (retrieved_documents[0], 0.85),
                (retrieved_documents[1], 0.72),
                (retrieved_documents[2], 0.68)
            ]
            
            # Mock embedding generation
            mock_embedding_manager.embed_text.return_value = [0.1] * 384
            
            # Create pipeline with LLM function
            pipeline = BasicRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                llm_func=mock_llm_func
            )
            
            # Test full pipeline execution (as used in RAGAS)
            test_query = "What is diabetes?"
            logger.info(f"🔍 Testing full pipeline execution with query: '{test_query}'")
            
            result = pipeline.execute(test_query, top_k=3)
            
            # Verify RAGAS-compatible result format
            required_keys = ["query", "answer", "retrieved_documents"]
            for key in required_keys:
                assert key in result, f"Missing required key: {key}"
            
            logger.info("📊 Pipeline result structure:")
            logger.info(f"  Query: {result['query']}")
            logger.info(f"  Answer: {result['answer'][:100]}...")
            logger.info(f"  Retrieved documents: {len(result['retrieved_documents'])}")
            
            # Verify retrieved documents format for RAGAS
            retrieved_docs = result['retrieved_documents']
            assert isinstance(retrieved_docs, list), "Retrieved documents should be a list"
            
            if retrieved_docs:
                # Check first document structure
                first_doc = retrieved_docs[0]
                logger.info(f"  First document type: {type(first_doc)}")
                
                if isinstance(first_doc, Document):
                    logger.info(f"  First document content: {first_doc.page_content[:100]}...")
                    logger.info(f"  First document metadata: {first_doc.metadata}")
                elif isinstance(first_doc, dict):
                    logger.info(f"  First document keys: {list(first_doc.keys())}")
                
                # Extract contexts for RAGAS (this is the critical step)
                contexts = []
                for doc in retrieved_docs:
                    if isinstance(doc, Document):
                        contexts.append(doc.page_content)
                    elif isinstance(doc, dict) and 'content' in doc:
                        contexts.append(doc['content'])
                    elif isinstance(doc, dict) and 'page_content' in doc:
                        contexts.append(doc['page_content'])
                    else:
                        logger.warning(f"⚠️ Unexpected document format: {type(doc)}")
                
                logger.info(f"📄 Extracted {len(contexts)} contexts for RAGAS")
                for i, context in enumerate(contexts[:2]):  # Show first 2
                    logger.info(f"  Context {i+1}: {context[:100]}...")
                
                # Verify we have valid contexts
                assert len(contexts) > 0, "Should have extracted at least one context"
                assert all(isinstance(ctx, str) and len(ctx) > 0 for ctx in contexts), "All contexts should be non-empty strings"
                
                logger.info("✅ RAGAS integration test passed")
                return True
            else:
                logger.warning("⚠️ No documents retrieved")
                return False
                
    except Exception as e:
        logger.error(f"❌ RAGAS integration test failed: {e}")
        logger.exception("Full traceback:")
        return False


def test_ragas_evaluation_data_preparation():
    """Test the data preparation step for RAGAS evaluation."""
    logger.info("🔍 Testing RAGAS evaluation data preparation...")
    
    try:
        # Simulate the data that would come from basic RAG pipeline
        pipeline_result = {
            "query": "What is diabetes?",
            "answer": "Diabetes is a metabolic disorder characterized by high blood sugar levels.",
            "retrieved_documents": [
                {
                    "page_content": "Diabetes is a metabolic disorder characterized by high blood sugar levels.",
                    "metadata": {"source": "medical_doc_1.txt"}
                },
                {
                    "page_content": "Type 2 diabetes is the most common form of diabetes.",
                    "metadata": {"source": "medical_doc_2.txt"}
                }
            ]
        }
        
        # Extract data for RAGAS evaluation (mimicking the evaluation framework)
        query = pipeline_result["query"]
        answer = pipeline_result["answer"]
        retrieved_documents = pipeline_result["retrieved_documents"]
        
        # Extract contexts (this is where issues often occur)
        contexts = []
        for doc in retrieved_documents:
            if isinstance(doc, dict):
                if 'page_content' in doc:
                    contexts.append(doc['page_content'])
                elif 'content' in doc:
                    contexts.append(doc['content'])
                else:
                    logger.warning(f"⚠️ Document missing content field: {list(doc.keys())}")
            else:
                logger.warning(f"⚠️ Unexpected document type: {type(doc)}")
        
        # Create ground truth (normally this would come from a dataset)
        ground_truth = "Diabetes is a chronic condition that affects how your body processes blood sugar."
        
        # Prepare RAGAS evaluation data structure
        ragas_data = {
            'question': [query],
            'answer': [answer],
            'contexts': [contexts],
            'ground_truth': [ground_truth]
        }
        
        logger.info("📊 RAGAS data preparation results:")
        logger.info(f"  Question: {ragas_data['question'][0]}")
        logger.info(f"  Answer: {ragas_data['answer'][0][:100]}...")
        logger.info(f"  Contexts: {len(ragas_data['contexts'][0])} contexts")
        logger.info(f"  Ground truth: {ragas_data['ground_truth'][0][:100]}...")
        
        # Verify data structure
        assert len(ragas_data['question']) == 1, "Should have one question"
        assert len(ragas_data['answer']) == 1, "Should have one answer"
        assert len(ragas_data['contexts']) == 1, "Should have one contexts list"
        assert len(ragas_data['ground_truth']) == 1, "Should have one ground truth"
        assert len(ragas_data['contexts'][0]) > 0, "Should have at least one context"
        
        logger.info("✅ RAGAS data preparation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ RAGAS data preparation test failed: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    """Run all debugging tests."""
    logger.info("🚀 Starting Basic RAG RAGAS retrieval debugging...")
    
    tests = [
        ("Basic RAG Retrieval", test_basic_rag_retrieval_step),
        ("RAGAS Integration", test_basic_rag_ragas_integration),
        ("RAGAS Data Preparation", test_ragas_evaluation_data_preparation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            success = test_func()
            results[test_name] = success
            if success:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DEBUGGING SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All debugging tests passed! Basic RAG RAGAS retrieval should be working.")
        return True
    else:
        logger.error("🔧 Some tests failed. Check the logs above for debugging information.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)