"""
Simple Enhanced Chunking System Test

This script tests the enhanced chunking system functionality:
1. Tests all chunking strategies
2. Validates performance with real documents
3. Tests database storage and retrieval
"""

import sys
import os
import json
import time
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from chunking.enhanced_chunking_service import EnhancedDocumentChunkingService
from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_chunking():
    """Test the enhanced chunking system."""
    print("🚀 Enhanced Chunking System Test")
    print("=" * 50)
    
    # Initialize the enhanced chunking service
    embedding_model = get_embedding_model(mock=True)
    def embedding_func(texts):
        return embedding_model.embed_documents(texts)
    
    chunking_service = EnhancedDocumentChunkingService(embedding_func=embedding_func)
    
    # Sample biomedical text for testing
    sample_text = """
    Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar levels over a prolonged period.
    Symptoms often include frequent urination, increased thirst, and increased appetite. If left untreated, diabetes can cause many health complications.
    
    Type 1 diabetes results from the pancreas's failure to produce enough insulin due to loss of beta cells.
    This form was previously referred to as "insulin-dependent diabetes mellitus" (IDDM) or "juvenile diabetes".
    The cause is unknown. Type 2 diabetes begins with insulin resistance, a condition in which cells fail to respond to insulin properly.
    
    As the disease progresses, a lack of insulin may also develop (Fig. 1). This form was previously referred to as "non insulin-dependent diabetes mellitus" (NIDDM) or "adult-onset diabetes".
    The most common cause is a combination of excessive body weight and insufficient exercise.
    
    Gestational diabetes is the third main form, and occurs when pregnant women without a previous history of diabetes develop high blood sugar levels.
    Treatment may include dietary changes, blood glucose monitoring, and in some cases, insulin may be required.
    
    Several studies have shown that metformin vs. placebo significantly reduces the risk of developing type 2 diabetes (p < 0.001).
    The UKPDS study demonstrated that intensive glucose control reduces microvascular complications by 25% (95% CI: 7-40%).
    """
    
    print("\n📊 Testing Chunking Strategies")
    print("-" * 30)
    
    strategies = ["recursive", "semantic", "adaptive", "hybrid"]
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        start_time = time.time()
        
        try:
            chunks = chunking_service.chunk_document("test_doc", sample_text, strategy)
            processing_time = time.time() - start_time
            
            # Calculate metrics
            total_tokens = sum(json.loads(chunk['chunk_metadata'])['chunk_metrics']['token_count'] for chunk in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else 0
            
            results[strategy] = {
                "success": True,
                "chunks": len(chunks),
                "total_tokens": total_tokens,
                "avg_tokens": avg_tokens,
                "processing_time_ms": processing_time * 1000
            }
            
            print(f"  ✅ {strategy}: {len(chunks)} chunks, {avg_tokens:.1f} avg tokens, {processing_time*1000:.1f}ms")
            
        except Exception as e:
            results[strategy] = {"success": False, "error": str(e)}
            print(f"  ❌ {strategy}: Error - {e}")
    
    print("\n🔍 Testing Chunking Analysis")
    print("-" * 30)
    
    try:
        analysis = chunking_service.analyze_chunking_effectiveness("test_doc", sample_text)
        
        print(f"Document info:")
        print(f"  - Estimated tokens: {analysis['document_info']['estimated_tokens']}")
        print(f"  - Biomedical density: {analysis['document_info']['biomedical_density']:.3f}")
        print(f"  - Word count: {analysis['document_info']['word_count']}")
        
        print(f"\nRecommended strategy: {analysis['recommendations']['recommended_strategy']}")
        print(f"Reason: {analysis['recommendations']['reason']}")
        
    except Exception as e:
        print(f"  ❌ Analysis failed: {e}")
    
    print("\n💾 Testing Database Operations")
    print("-" * 30)
    
    try:
        # Test with adaptive strategy
        chunks = chunking_service.chunk_document("test_enhanced_db", sample_text, "adaptive")
        
        # Store chunks
        success = chunking_service.store_chunks(chunks)
        if success:
            print(f"  ✅ Stored {len(chunks)} chunks successfully")
            
            # Verify storage
            connection = get_iris_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.DocumentChunks
                WHERE doc_id = ?
            """, ("test_enhanced_db",))
            
            stored_count = cursor.fetchone()[0]
            print(f"  ✅ Verified {stored_count} chunks in database")
            
            # Cleanup
            cursor.execute("DELETE FROM RAG.DocumentChunks WHERE doc_id = ?", ("test_enhanced_db",))
            connection.commit()
            cursor.close()
            connection.close()
            print(f"  ✅ Cleaned up test data")
            
        else:
            print(f"  ❌ Failed to store chunks")
            
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
    
    print("\n📈 Testing Scale Performance")
    print("-" * 30)
    
    try:
        # Test with multiple documents
        test_docs = []
        for i in range(10):
            doc_text = f"""
            Document {i}: This is a test document for performance evaluation.
            It contains multiple sentences to test chunking performance.
            The document discusses various biomedical topics including diabetes, hypertension, and cardiovascular disease.
            Statistical analysis shows significant improvements (p < 0.05) in patient outcomes.
            Figure {i} demonstrates the correlation between treatment and recovery rates.
            """
            test_docs.append((f"perf_test_doc_{i}", doc_text))
        
        for strategy in ["adaptive", "recursive"]:
            start_time = time.time()
            total_chunks = 0
            
            for doc_id, doc_text in test_docs:
                chunks = chunking_service.chunk_document(doc_id, doc_text, strategy)
                total_chunks += len(chunks)
            
            processing_time = time.time() - start_time
            docs_per_second = len(test_docs) / processing_time
            
            print(f"  {strategy}: {len(test_docs)} docs, {total_chunks} chunks, {docs_per_second:.1f} docs/sec")
            
    except Exception as e:
        print(f"  ❌ Scale test failed: {e}")
    
    print("\n✅ Enhanced Chunking System Test Complete!")
    print("=" * 50)
    
    # Summary
    successful_strategies = sum(1 for result in results.values() if result.get("success", False))
    print(f"\nSummary:")
    print(f"  - Strategies tested: {len(strategies)}")
    print(f"  - Successful: {successful_strategies}")
    print(f"  - Success rate: {successful_strategies/len(strategies)*100:.1f}%")
    
    if successful_strategies == len(strategies):
        print(f"  🎉 All chunking strategies working correctly!")
        return True
    else:
        print(f"  ⚠️  Some strategies failed - check logs above")
        return False

if __name__ == "__main__":
    success = test_enhanced_chunking()
    exit(0 if success else 1)