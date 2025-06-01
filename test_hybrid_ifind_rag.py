"""
Test HybridIFindRAG Pipeline to ensure it's working correctly
"""

import sys
import os # Added for path manipulation
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.experimental.hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline # Updated import
from src.common.iris_connector import get_iris_connection # Updated import
from src.common.embedding_utils import get_embedding_model # Updated import
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hybrid_ifind_rag():
    """Test the HybridIFindRAG pipeline with various queries"""
    
    print("=== Testing HybridIFindRAG Pipeline ===\n")
    
    # Initialize components
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f'Based on the provided context, this is a response to: {prompt[:100]}...'
    
    # Create pipeline
    pipeline = HybridiFindRAGPipeline(
        iris_connector=iris,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test queries
    test_queries = [
        "What is diabetes and how is it treated?",
        "How do honeybees process neural signals?",
        "What are microRNAs?",
        "cancer treatment options",
        "neural plasticity mechanisms"
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}/{len(test_queries)}: {query}")
        print("-" * 60)
        
        try:
            result = pipeline.run(query)
            
            # Check result structure
            assert 'query' in result, "Missing 'query' in result"
            assert 'answer' in result, "Missing 'answer' in result"
            assert 'retrieved_documents' in result, "Missing 'retrieved_documents' in result"
            
            print(f"✅ Query executed successfully")
            print(f"   Retrieved documents: {len(result['retrieved_documents'])}")
            
            # Show retrieval methods used
            methods_used = set()
            for doc in result['retrieved_documents']:
                if 'method' in doc:
                    methods_used.add(doc['method'])
            
            print(f"   Retrieval methods used: {', '.join(methods_used)}")
            
            # Show first document
            if result['retrieved_documents']:
                doc = result['retrieved_documents'][0]
                print(f"\n   First document:")
                print(f"   - ID: {doc.get('document_id', 'N/A')}")
                print(f"   - Title: {doc.get('title', 'N/A')[:80]}...")
                print(f"   - Method: {doc.get('method', 'N/A')}")
                print(f"   - Score: {doc.get('similarity', doc.get('rank_score', 'N/A'))}")
                
                # Show content preview
                content = doc.get('content', '')
                if content:
                    print(f"   - Content preview: {content[:150]}...")
            
            # Show answer preview
            print(f"\n   Answer preview: {result['answer'][:200]}...")
            
            results.append({
                'query': query,
                'success': True,
                'doc_count': len(result['retrieved_documents']),
                'methods': list(methods_used)
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            
            results.append({
                'query': query,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"\nTotal queries: {len(test_queries)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(test_queries) - successful}")
    
    print("\nDetails:")
    for r in results:
        if r['success']:
            print(f"  ✅ {r['query'][:50]}... - {r['doc_count']} docs via {', '.join(r['methods'])}")
        else:
            print(f"  ❌ {r['query'][:50]}... - Error: {r['error']}")
    
    # Check which tables are being used
    print("\n" + "="*60)
    print("TABLE USAGE CHECK")
    print("="*60)
    
    cursor = iris.cursor()
    
    # Check SourceDocumentsIFind
    cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = 'RAG' 
        AND TABLE_NAME = 'SourceDocumentsIFind'
    """)
    has_ifind = cursor.fetchone()[0] > 0
    
    if has_ifind:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocumentsIFind")
        ifind_count = cursor.fetchone()[0]
        print(f"✅ SourceDocumentsIFind exists with {ifind_count:,} documents")
    else:
        print("❌ SourceDocumentsIFind table not found")
    
    # Check DocumentChunks
    cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
    chunk_count = cursor.fetchone()[0]
    print(f"✅ DocumentChunks has {chunk_count:,} chunks")
    
    # Check Entities_V2
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities_V2")
    entity_count = cursor.fetchone()[0]
    print(f"✅ Entities_V2 has {entity_count:,} entities")
    
    cursor.close()
    iris.close()
    
    print("\n🎉 HybridIFindRAG test complete!")
    
    return successful == len(test_queries)

if __name__ == "__main__":
    success = test_hybrid_ifind_rag()
    sys.exit(0 if success else 1)