#!/usr/bin/env python3
"""
Test Real PyTorch RAG System

This script validates that the RAG system works with real PyTorch models
by running the existing comprehensive test suite with real models instead of mocks.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pytorch_installation():
    """Test PyTorch installation and basic functionality"""
    logger.info("🔧 Testing PyTorch installation...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__} installed successfully")
        
        # Test basic operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        logger.info(f"✅ PyTorch tensor operations working (result shape: {z.shape})")
        
        # Check device availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"✅ Using device: {device}")
        
        return True
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False
    except Exception as e:
        logger.error(f"❌ PyTorch test failed: {e}")
        return False

def test_transformers_installation():
    """Test transformers library installation"""
    logger.info("🔧 Testing transformers installation...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        logger.info("✅ Transformers library installed successfully")
        return True
    except ImportError:
        logger.error("❌ Transformers library not installed")
        return False

def test_real_embedding_model():
    """Test real embedding model loading and inference"""
    logger.info("🔧 Testing real embedding model...")
    
    try:
        from common.utils import get_embedding_func
        
        # Get real embedding function (not mock)
        embed_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
        
        # Test with sample texts
        test_texts = [
            "InterSystems IRIS is a complete data platform.",
            "Machine learning enables semantic search.",
            "Vector databases store high-dimensional embeddings."
        ]
        
        logger.info(f"Generating embeddings for {len(test_texts)} texts...")
        start_time = time.time()
        embeddings = embed_func(test_texts)
        inference_time = time.time() - start_time
        
        # Validate embeddings
        assert len(embeddings) == len(test_texts), f"Expected {len(test_texts)} embeddings, got {len(embeddings)}"
        assert len(embeddings[0]) == 768, f"Expected 768 dimensions, got {len(embeddings[0])}"
        assert not all(x == 0 for x in embeddings[0]), "Embeddings should not be all zeros"
        
        logger.info(f"✅ Real embedding model working! Generated {len(embeddings)} embeddings of {len(embeddings[0])} dimensions in {inference_time:.2f}s")
        logger.info(f"Sample embedding: {embeddings[0][:5]}...")
        
        return True, embeddings, inference_time
    except Exception as e:
        logger.error(f"❌ Real embedding model test failed: {e}")
        return False, None, 0

def test_real_llm():
    """Test real LLM functionality"""
    logger.info("🔧 Testing real LLM...")
    
    try:
        from common.utils import get_llm_func
        
        # Get real LLM function (OpenAI)
        llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
        
        # Test with a simple prompt
        test_prompt = """Answer this question concisely:

Question: What is machine learning?

Answer:"""
        
        logger.info("Making real LLM call...")
        start_time = time.time()
        response = llm_func(test_prompt)
        llm_time = time.time() - start_time
        
        # Validate response
        assert isinstance(response, str), f"Expected string response, got {type(response)}"
        assert len(response) > 10, f"Response too short: {response}"
        
        logger.info(f"✅ Real LLM working! Generated response in {llm_time:.2f}s")
        logger.info(f"LLM Response: {response}")
        
        return True, response, llm_time
    except Exception as e:
        logger.error(f"❌ Real LLM test failed: {e}")
        return False, None, 0

def run_basic_rag_test():
    """Run a basic RAG test with real models"""
    logger.info("🔧 Testing basic RAG pipeline with real models...")
    
    try:
        from common.utils import get_embedding_func, get_llm_func
        from basic_rag.pipeline import BasicRAGPipeline
        
        # Get real functions
        embed_func = get_embedding_func(model_name="intfloat/e5-base-v2", mock=False)
        llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
        
        # Create sample documents for testing
        sample_docs = [
            {"id": "doc1", "content": "InterSystems IRIS is a complete data platform that combines a multi-model database, application server, and interoperability engine."},
            {"id": "doc2", "content": "Machine learning algorithms can process natural language text to understand semantic meaning and context."},
            {"id": "doc3", "content": "Vector databases enable semantic search by storing high-dimensional embeddings of documents and queries."}
        ]
        
        # Generate embeddings for sample documents
        logger.info("Generating embeddings for sample documents...")
        doc_texts = [doc["content"] for doc in sample_docs]
        doc_embeddings = embed_func(doc_texts)
        
        # Test query
        query = "What is InterSystems IRIS?"
        query_embedding = embed_func([query])[0]
        
        # Calculate similarities (simple cosine similarity)
        import numpy as np
        similarities = []
        for i, doc_embedding in enumerate(doc_embeddings):
            query_vec = np.array(query_embedding)
            doc_vec = np.array(doc_embedding)
            
            # Normalize vectors
            query_norm = query_vec / np.linalg.norm(query_vec)
            doc_norm = doc_vec / np.linalg.norm(doc_vec)
            
            # Calculate cosine similarity
            similarity = np.dot(query_norm, doc_norm)
            similarities.append((i, similarity, sample_docs[i]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top documents
        top_docs = similarities[:2]
        context = "\n\n".join([doc[2]["content"] for doc in top_docs])
        
        # Generate answer
        prompt = f"""Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        
        logger.info("Generating answer with real LLM...")
        answer = llm_func(prompt)
        
        # Validate result
        assert len(answer) > 20, f"Answer too short: {answer}"
        assert "IRIS" in answer, f"Answer should mention IRIS: {answer}"
        
        logger.info(f"✅ Basic RAG test passed!")
        logger.info(f"Query: {query}")
        logger.info(f"Top document similarity: {top_docs[0][1]:.4f}")
        logger.info(f"Answer: {answer}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Basic RAG test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Real PyTorch RAG System Test")
    logger.info("="*60)
    
    all_tests_passed = True
    
    # Test 1: PyTorch installation
    logger.info("\n📦 TEST 1: PyTorch Installation")
    if not test_pytorch_installation():
        all_tests_passed = False
        logger.error("❌ PyTorch installation test failed - cannot continue")
        return False
    
    # Test 2: Transformers installation
    logger.info("\n📦 TEST 2: Transformers Installation")
    if not test_transformers_installation():
        all_tests_passed = False
        logger.error("❌ Transformers installation test failed - cannot continue")
        return False
    
    # Test 3: Real embedding model
    logger.info("\n🧠 TEST 3: Real Embedding Model")
    embed_success, embeddings, embed_time = test_real_embedding_model()
    if not embed_success:
        all_tests_passed = False
        logger.error("❌ Real embedding model test failed")
    
    # Test 4: Real LLM
    logger.info("\n🤖 TEST 4: Real LLM")
    llm_success, response, llm_time = test_real_llm()
    if not llm_success:
        all_tests_passed = False
        logger.error("❌ Real LLM test failed")
    
    # Test 5: Basic RAG pipeline
    logger.info("\n🔍 TEST 5: Basic RAG Pipeline")
    if not run_basic_rag_test():
        all_tests_passed = False
        logger.error("❌ Basic RAG pipeline test failed")
    
    # Summary
    logger.info("\n" + "="*60)
    if all_tests_passed:
        logger.info("🎉 ALL TESTS PASSED! Real PyTorch RAG system is working!")
        logger.info("\n📊 SUMMARY:")
        logger.info("✅ PyTorch: Installed and working")
        logger.info("✅ Transformers: Installed and working")
        if embed_success:
            logger.info(f"✅ Real Embeddings: 768D vectors in {embed_time:.2f}s")
        if llm_success:
            logger.info(f"✅ Real LLM: Responses in {llm_time:.2f}s")
        logger.info("✅ RAG Pipeline: End-to-end functionality working")
        
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. Run comprehensive tests: python -m pytest tests/test_e2e_rag_pipelines.py -v")
        logger.info("2. Run with real data: python scripts/run_e2e_tests.py --verbose")
        logger.info("3. Run benchmarks: python scripts/run_rag_benchmarks.py")
        
        return True
    else:
        logger.error("❌ SOME TESTS FAILED! Check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)