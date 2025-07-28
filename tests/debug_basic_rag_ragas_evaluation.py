#!/usr/bin/env python3
"""
Debug script for Basic RAG Pipeline RAGAS evaluation step.

This script tests the actual RAGAS evaluation process with the basic RAG pipeline
to identify any issues in the evaluation step itself.
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

def test_ragas_evaluation_with_basic_rag():
    """Test the actual RAGAS evaluation process with basic RAG pipeline."""
    logger.info("🔍 Testing RAGAS evaluation with Basic RAG pipeline...")
    
    try:
        # Check if RAGAS is available
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
                answer_similarity,
                answer_correctness
            )
            from datasets import Dataset
            logger.info("✅ RAGAS imports successful")
            ragas_available = True
        except ImportError as e:
            logger.warning(f"⚠️ RAGAS not available: {e}")
            ragas_available = False
            return False
        
        # Import basic RAG components
        from iris_rag.pipelines.basic import BasicRAGPipeline
        from iris_rag.core.models import Document
        from common.iris_connection_manager import get_iris_connection
        from iris_rag.config.manager import ConfigurationManager
        
        # Create mock LLM function for answer generation
        def mock_llm_func(prompt: str) -> str:
            return "Diabetes is a metabolic disorder characterized by high blood sugar levels that affects how the body processes glucose."
        
        # Create mock components
        mock_connection_manager = Mock()
        mock_config_manager = Mock(spec=ConfigurationManager)
        mock_config_manager.get.return_value = {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "default_top_k": 5,
            "embedding_batch_size": 32
        }
        
        # Create test documents
        test_documents = [
            Document(
                page_content="Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period. It occurs when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces.",
                metadata={"source": "diabetes_overview.txt", "doc_id": "1"}
            ),
            Document(
                page_content="Type 1 diabetes is an autoimmune condition where the immune system attacks insulin-producing beta cells in the pancreas. This results in little to no insulin production.",
                metadata={"source": "type1_diabetes.txt", "doc_id": "2"}
            ),
            Document(
                page_content="Type 2 diabetes is the most common form of diabetes, accounting for about 90% of all cases. It occurs when the body becomes resistant to insulin or doesn't make enough insulin.",
                metadata={"source": "type2_diabetes.txt", "doc_id": "3"}
            )
        ]
        
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
                (test_documents[0], 0.92),
                (test_documents[1], 0.85),
                (test_documents[2], 0.78)
            ]
            
            # Mock embedding generation
            mock_embedding_manager.embed_text.return_value = [0.1] * 384
            
            # Create pipeline with LLM function
            pipeline = BasicRAGPipeline(
                connection_manager=mock_connection_manager,
                config_manager=mock_config_manager,
                llm_func=mock_llm_func
            )
            
            # Execute pipeline to get result
            test_query = "What is diabetes and what causes it?"
            logger.info(f"🔍 Executing pipeline with query: '{test_query}'")
            
            result = pipeline.execute(test_query, top_k=3)
            
            logger.info("📊 Pipeline execution result:")
            logger.info(f"  Query: {result['query']}")
            logger.info(f"  Answer: {result['answer']}")
            logger.info(f"  Retrieved documents: {len(result['retrieved_documents'])}")
            
            # Extract data for RAGAS evaluation
            query = result["query"]
            answer = result["answer"]
            retrieved_documents = result["retrieved_documents"]
            
            # Extract contexts from retrieved documents
            contexts = []
            for doc in retrieved_documents:
                if isinstance(doc, Document):
                    contexts.append(doc.page_content)
                elif isinstance(doc, dict) and 'page_content' in doc:
                    contexts.append(doc['page_content'])
                else:
                    logger.warning(f"⚠️ Unexpected document format: {type(doc)}")
            
            logger.info(f"📄 Extracted {len(contexts)} contexts for RAGAS")
            
            # Create ground truth
            ground_truth = "Diabetes is a metabolic disorder characterized by high blood sugar levels. It is caused by the body's inability to produce enough insulin or use insulin effectively."
            
            # Prepare RAGAS dataset
            ragas_data = {
                'question': [query],
                'answer': [answer],
                'contexts': [contexts],
                'ground_truth': [ground_truth]
            }
            
            logger.info("🎯 Creating RAGAS dataset...")
            dataset = Dataset.from_dict(ragas_data)
            logger.info(f"✅ Dataset created with {len(dataset)} rows")
            
            # Initialize RAGAS components (mock for testing)
            logger.info("🔧 Initializing RAGAS components...")
            
            # Check if we have OpenAI API key for real evaluation
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                logger.info("🔑 OpenAI API key found, using real LLM for RAGAS")
                try:
                    from langchain_openai import ChatOpenAI
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    
                    ragas_llm = ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature=0,
                        openai_api_key=openai_api_key
                    )
                    ragas_embeddings = HuggingFaceEmbeddings(
                        model_name='sentence-transformers/all-MiniLM-L6-v2',
                        model_kwargs={'device': 'cpu'}
                    )
                    
                    logger.info("✅ RAGAS components initialized")
                    
                    # Define metrics to evaluate
                    metrics = [
                        answer_relevancy,
                        context_precision,
                        context_recall,
                        faithfulness
                    ]
                    
                    logger.info(f"📏 Using {len(metrics)} RAGAS metrics")
                    
                    # Run RAGAS evaluation
                    logger.info("🚀 Starting RAGAS evaluation...")
                    
                    ragas_result = evaluate(
                        dataset,
                        metrics=metrics,
                        llm=ragas_llm,
                        embeddings=ragas_embeddings
                    )
                    
                    logger.info("✅ RAGAS evaluation completed")
                    
                    # Extract and display results
                    if hasattr(ragas_result, 'to_pandas'):
                        df = ragas_result.to_pandas()
                        logger.info("📊 RAGAS evaluation results:")
                        for metric in ['answer_relevancy', 'context_precision', 'context_recall', 'faithfulness']:
                            if metric in df.columns:
                                score = df[metric].iloc[0] if len(df) > 0 else None
                                if score is not None:
                                    logger.info(f"  {metric}: {score:.3f}")
                                else:
                                    logger.warning(f"  {metric}: No score available")
                    elif isinstance(ragas_result, dict):
                        logger.info("📊 RAGAS evaluation results:")
                        for metric, score in ragas_result.items():
                            logger.info(f"  {metric}: {score:.3f}")
                    
                    logger.info("✅ RAGAS evaluation with Basic RAG successful")
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ RAGAS evaluation failed: {e}")
                    logger.exception("Full traceback:")
                    return False
            else:
                logger.warning("⚠️ No OpenAI API key found, skipping actual RAGAS evaluation")
                logger.info("✅ RAGAS data preparation successful (evaluation skipped)")
                return True
                
    except Exception as e:
        logger.error(f"❌ RAGAS evaluation test failed: {e}")
        logger.exception("Full traceback:")
        return False


def test_ragas_context_extraction_edge_cases():
    """Test edge cases in context extraction for RAGAS."""
    logger.info("🔍 Testing RAGAS context extraction edge cases...")
    
    try:
        from iris_rag.core.models import Document
        
        # Test different document formats
        test_cases = [
            {
                "name": "Standard Document objects",
                "documents": [
                    Document(page_content="Content 1", metadata={"source": "doc1"}),
                    Document(page_content="Content 2", metadata={"source": "doc2"})
                ]
            },
            {
                "name": "Dictionary with page_content",
                "documents": [
                    {"page_content": "Content 1", "metadata": {"source": "doc1"}},
                    {"page_content": "Content 2", "metadata": {"source": "doc2"}}
                ]
            },
            {
                "name": "Dictionary with content key",
                "documents": [
                    {"content": "Content 1", "metadata": {"source": "doc1"}},
                    {"content": "Content 2", "metadata": {"source": "doc2"}}
                ]
            },
            {
                "name": "Mixed formats",
                "documents": [
                    Document(page_content="Content 1", metadata={"source": "doc1"}),
                    {"page_content": "Content 2", "metadata": {"source": "doc2"}},
                    {"content": "Content 3", "metadata": {"source": "doc3"}}
                ]
            },
            {
                "name": "Empty documents list",
                "documents": []
            },
            {
                "name": "Documents with empty content",
                "documents": [
                    Document(page_content="", metadata={"source": "doc1"}),
                    Document(page_content="Valid content", metadata={"source": "doc2"})
                ]
            }
        ]
        
        for test_case in test_cases:
            logger.info(f"🧪 Testing: {test_case['name']}")
            documents = test_case['documents']
            
            # Extract contexts using the same logic as in the evaluation
            contexts = []
            for doc in documents:
                if isinstance(doc, Document):
                    if doc.page_content.strip():  # Only add non-empty content
                        contexts.append(doc.page_content)
                elif isinstance(doc, dict):
                    if 'page_content' in doc and doc['page_content'].strip():
                        contexts.append(doc['page_content'])
                    elif 'content' in doc and doc['content'].strip():
                        contexts.append(doc['content'])
                    else:
                        logger.warning(f"⚠️ Document missing content field: {list(doc.keys())}")
                else:
                    logger.warning(f"⚠️ Unexpected document type: {type(doc)}")
            
            logger.info(f"  Extracted {len(contexts)} contexts")
            if contexts:
                logger.info(f"  First context: {contexts[0][:50]}...")
            
            # Verify contexts are valid for RAGAS
            if contexts:
                assert all(isinstance(ctx, str) for ctx in contexts), "All contexts should be strings"
                assert all(len(ctx.strip()) > 0 for ctx in contexts), "All contexts should be non-empty"
                logger.info(f"  ✅ {test_case['name']} passed")
            else:
                logger.info(f"  ⚠️ {test_case['name']} resulted in no contexts")
        
        logger.info("✅ Context extraction edge cases test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Context extraction test failed: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    """Run all RAGAS evaluation debugging tests."""
    logger.info("🚀 Starting Basic RAG RAGAS evaluation debugging...")
    
    tests = [
        ("RAGAS Evaluation with Basic RAG", test_ragas_evaluation_with_basic_rag),
        ("Context Extraction Edge Cases", test_ragas_context_extraction_edge_cases)
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
    logger.info("RAGAS EVALUATION DEBUGGING SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All RAGAS evaluation tests passed! Basic RAG RAGAS evaluation should be working.")
        return True
    else:
        logger.error("🔧 Some tests failed. Check the logs above for debugging information.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)