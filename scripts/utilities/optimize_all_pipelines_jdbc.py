#!/usr/bin/env python3
"""
Optimize All Pipelines for JDBC - Ensure proper vector operations
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_jdbc_safe_chunk_retrieval():
    """Create a JDBC-safe chunk retrieval module"""
    
    content = '''"""
JDBC-Safe Chunk Retrieval Module
Handles vector operations without parameter binding issues
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from common.utils import Document

logger = logging.getLogger(__name__)

def retrieve_chunks_jdbc_safe(connection, query_embedding: List[float], 
                             top_k: int = 20, threshold: float = 0.1,
                             chunk_types: List[str] = None) -> List[Document]:
    """
    Retrieve chunks using JDBC-safe vector operations
    """
    if chunk_types is None:
        chunk_types = ['content', 'mixed']
    
    cursor = None
    chunks = []
    
    try:
        cursor = connection.cursor()
        
        # Convert embedding to string
        vector_str = ','.join(map(str, query_embedding))
        chunk_types_str = ','.join([f"'{ct}'" for ct in chunk_types])
        
        # Use direct SQL without parameter binding
        query = f"""
            SELECT TOP {top_k}
                chunk_id,
                chunk_text,
                doc_id,
                chunk_type,
                chunk_index,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) AS score
            FROM RAG.DocumentChunks
            WHERE embedding IS NOT NULL
              AND chunk_type IN ({chunk_types_str})
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) > {threshold}
            ORDER BY score DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        for chunk_id, chunk_text, doc_id, chunk_type, chunk_index, score in results:
            # Handle potential stream objects
            if hasattr(chunk_text, 'read'):
                chunk_text = chunk_text.read()
            if isinstance(chunk_text, bytes):
                chunk_text = chunk_text.decode('utf-8', errors='ignore')
            
            chunks.append(Document(
                id=f"{doc_id}_chunk_{chunk_id}",
                content=str(chunk_text),
                score=float(score) if score else 0.0,
                metadata={
                    'doc_id': doc_id,
                    'chunk_type': chunk_type,
                    'chunk_index': chunk_index
                }
            ))
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
    finally:
        if cursor:
            cursor.close()
    
    return chunks

def retrieve_documents_jdbc_safe(connection, query_embedding: List[float],
                                top_k: int = 20, threshold: float = 0.1) -> List[Document]:
    """
    Retrieve documents using JDBC-safe vector operations
    """
    cursor = None
    documents = []
    
    try:
        cursor = connection.cursor()
        
        # Convert embedding to string
        vector_str = ','.join(map(str, query_embedding))
        
        # Use direct SQL without parameter binding
        query = f"""
            SELECT TOP {top_k}
                doc_id,
                text_content,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) AS score
            FROM RAG.SourceDocuments_V2
            WHERE embedding IS NOT NULL
              AND LENGTH(embedding) > 1000
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{vector_str}')) > {threshold}
            ORDER BY score DESC
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        for doc_id, content, score in results:
            # Handle potential stream objects
            if hasattr(content, 'read'):
                content = content.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            
            documents.append(Document(
                id=doc_id,
                content=str(content),
                score=float(score) if score else 0.0
            ))
        
        logger.info(f"Retrieved {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
    finally:
        if cursor:
            cursor.close()
    
    return documents
'''
    
    with open('common/jdbc_safe_retrieval.py', 'w') as f:
        f.write(content)
    
    logger.info("✅ Created JDBC-safe retrieval module")

def update_pipeline_imports():
    """Update pipeline imports to use JDBC connections"""
    
    pipelines = [
        'basic_rag/pipeline.py',
        'hyde/pipeline.py',
        'crag/pipeline.py',
        'noderag/pipeline.py',
        'colbert/pipeline.py',
        'graphrag/pipeline.py',
        'hybrid_ifind_rag/pipeline.py'
    ]
    
    for pipeline_path in pipelines:
        if os.path.exists(pipeline_path):
            try:
                with open(pipeline_path, 'r') as f:
                    content = f.read()
                
                # Check if already using JDBC
                if 'iris_connector_jdbc' in content:
                    logger.info(f"✅ {pipeline_path} already using JDBC")
                    continue
                
                # Update import
                if 'from common.iris_connector import' in content:
                    content = content.replace(
                        'from common.iris_connector import',
                        'from common.iris_connector_jdbc import'
                    )
                    
                    with open(pipeline_path, 'w') as f:
                        f.write(content)
                    
                    logger.info(f"✅ Updated {pipeline_path} to use JDBC")
                else:
                    logger.warning(f"⚠️ {pipeline_path} doesn't have standard import")
                    
            except Exception as e:
                logger.error(f"❌ Error updating {pipeline_path}: {e}")

def create_performance_test_script():
    """Create a script to test all pipelines performance"""
    
    content = '''#!/usr/bin/env python3
"""
Test All Pipelines Performance with JDBC
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import logging
from typing import Dict, Any

# Import all pipelines
from src.deprecated.basic_rag.pipeline import BasicRAGPipeline # Updated import
from src.experimental.hyde.pipeline import HyDEPipeline # Updated import
from src.experimental.crag.pipeline import CRAGPipeline # Updated import
from src.experimental.noderag.pipeline import NodeRAGPipeline # Updated import
from src.deprecated.colbert.pipeline import OptimizedColbertRAGPipeline as ColBERTPipeline # Updated import
from src.experimental.graphrag.pipeline import GraphRAGPipeline # Updated import
from src.experimental.hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline # Updated import

from common.iris_connector_jdbc import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline(name: str, pipeline: Any, query: str) -> Dict[str, Any]:
    """Test a single pipeline"""
    logger.info(f"Testing {name}...")
    
    start_time = time.time()
    try:
        if name == "CRAG":
            # CRAG doesn't accept similarity_threshold
            result = pipeline.run(query, top_k=10)
        else:
            result = pipeline.run(query, top_k=10, similarity_threshold=0.1)
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "time": elapsed,
            "documents": len(result.get("retrieved_documents", [])),
            "answer_length": len(result.get("answer", ""))
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"{name} failed: {e}")
        return {
            "success": False,
            "time": elapsed,
            "error": str(e)
        }

def main():
    """Test all pipelines"""
    print("🚀 Testing All Pipelines with JDBC")
    print("=" * 60)
    
    # Initialize connection and functions
    conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Initialize pipelines
    pipelines = {}
    
    try:
        pipelines["BasicRAG"] = BasicRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize BasicRAG: {e}")
    
    try:
        pipelines["HyDE"] = HyDEPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize HyDE: {e}")
    
    try:
        pipelines["CRAG"] = CRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize CRAG: {e}")
    
    try:
        pipelines["NodeRAG"] = NodeRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize NodeRAG: {e}")
    
    try:
        pipelines["ColBERT"] = ColBERTPipeline(
            conn, embedding_func, embedding_func, llm_func
        )
    except Exception as e:
        logger.error(f"Failed to initialize ColBERT: {e}")
    
    try:
        pipelines["GraphRAG"] = GraphRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize GraphRAG: {e}")
    
    try:
        pipelines["HybridIFind"] = HybridiFindRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize HybridIFind: {e}")
    
    # Test query
    test_query = "What are the symptoms of diabetes?"
    
    # Test each pipeline
    results = {}
    for name, pipeline in pipelines.items():
        results[name] = test_pipeline(name, pipeline, test_query)
    
    # Print results
    print("\\n📊 Results Summary")
    print("=" * 60)
    
    for name, result in results.items():
        if result["success"]:
            print(f"✅ {name}: {result['time']:.2f}s, {result['documents']} docs")
        else:
            print(f"❌ {name}: Failed - {result.get('error', 'Unknown error')}")
    
    print("\\n✅ Testing complete!")

if __name__ == "__main__":
    main()
'''
    
    with open('scripts/test_all_pipelines_jdbc.py', 'w') as f:
        f.write(content)
    
    os.chmod('scripts/test_all_pipelines_jdbc.py', 0o755)
    logger.info("✅ Created pipeline performance test script")

def main():
    """Main optimization process"""
    print("🔧 Optimizing All Pipelines for JDBC")
    print("=" * 60)
    
    # Step 1: Create JDBC-safe retrieval module
    create_jdbc_safe_chunk_retrieval()
    
    # Step 2: Update pipeline imports
    update_pipeline_imports()
    
    # Step 3: Create performance test script
    create_performance_test_script()
    
    print("\n✅ Optimization complete!")
    print("\n📌 Next steps:")
    print("1. Run: python scripts/test_all_pipelines_jdbc.py")
    print("2. Check results and fix any remaining issues")
    print("3. Run full benchmark: python eval/enterprise_rag_benchmark_final.py")

if __name__ == "__main__":
    main()