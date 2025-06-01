"""
BasicRAG Pipeline using JDBC for improved vector support
This is a proof-of-concept for migrating from ODBC to JDBC
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')) # Adjusted for new location
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Dict, Any
import time
import logging
from jdbc_exploration.iris_jdbc_connector import get_iris_jdbc_connection

logger = logging.getLogger(__name__)

class BasicRAGPipelineJDBC:
    """Basic RAG pipeline using JDBC connection for vector operations"""
    
    def __init__(self, iris_connector=None, embedding_func=None, llm_func=None, schema="RAG"):
        """Initialize the pipeline with JDBC connection"""
        # Use provided connector or create new JDBC connection
        self.iris_connector = iris_connector or get_iris_jdbc_connection()
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        
        # Test connection
        try:
            result = self.iris_connector.execute("SELECT 1")
            logger.info("JDBC connection test successful")
        except Exception as e:
            logger.error(f"JDBC connection test failed: {e}")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using vector similarity search via JDBC"""
        
        # Generate query embedding
        if not self.embedding_func:
            raise ValueError("Embedding function not provided")
        
        query_embedding = self.embedding_func([query])[0]
        
        # Convert embedding to string format
        # Using fixed decimal format to avoid scientific notation issues
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        # SQL query with string formatting (safer for JDBC)
        sql = f"""
            SELECT TOP {top_k}
                doc_id,
                title,
                text_content,
                VECTOR_COSINE(
                    embedding,
                    TO_VECTOR('[{query_embedding_str}]', 'DOUBLE', 384)
                ) as similarity_score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY VECTOR_COSINE(
                embedding,
                TO_VECTOR('[{query_embedding_str}]', 'DOUBLE', 384)
            ) DESC
        """
        
        try:
            # Execute query using cursor directly
            cursor = self.iris_connector.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            
            # Format results
            documents = []
            for row in results:
                documents.append({
                    'doc_id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'similarity_score': float(row[3])
                })
            
            logger.info(f"Retrieved {len(documents)} documents via JDBC")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents via JDBC: {e}")
            raise
    
    def generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate answer using LLM based on retrieved documents"""
        
        if not self.llm_func:
            raise ValueError("LLM function not provided")
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents[:3]):  # Use top 3 documents
            context_parts.append(f"Document {i+1} (Score: {doc['similarity_score']:.3f}):")
            context_parts.append(f"Title: {doc['title']}")
            context_parts.append(f"Content: {doc['content'][:500]}...")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following documents, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate answer
        answer = self.llm_func(prompt)
        return answer
    
    def run(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """Run the complete RAG pipeline"""
        
        start_time = time.time()
        
        # Retrieve documents
        documents = self.retrieve_documents(query, top_k=top_k, similarity_threshold=similarity_threshold)
        retrieval_time = time.time() - start_time
        
        # Generate answer
        answer_start = time.time()
        answer = self.generate_answer(query, documents)
        generation_time = time.time() - answer_start
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": documents,
            "metadata": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "num_documents": len(documents),
                "connection_type": "JDBC"
            }
        }

def test_jdbc_pipeline():
    """Test the JDBC-based BasicRAG pipeline"""
    
    print("🔍 Testing BasicRAG Pipeline with JDBC")
    print("=" * 50)
    
    # Import required functions
    from common.utils import get_embedding_func, get_llm_func
    
    try:
        # Initialize pipeline
        pipeline = BasicRAGPipelineJDBC(
            embedding_func=get_embedding_func(),
            llm_func=get_llm_func()
        )
        
        # Test queries
        test_queries = [
            "What are the symptoms of diabetes?",
            "How is COVID-19 transmitted?",
            "What are the treatment options for hypertension?"
        ]
        
        for query in test_queries:
            print(f"\n📊 Query: {query}")
            
            result = pipeline.run(query, top_k=3)
            
            print(f"⏱️  Retrieval time: {result['metadata']['retrieval_time']:.3f}s")
            print(f"⏱️  Generation time: {result['metadata']['generation_time']:.3f}s")
            print(f"⏱️  Total time: {result['metadata']['total_time']:.3f}s")
            print(f"📄 Documents retrieved: {result['metadata']['num_documents']}")
            print(f"💬 Answer: {result['answer'][:200]}...")
            
            # Show document scores
            print("\n📊 Document Scores:")
            for doc in result['retrieved_documents']:
                print(f"   - {doc['doc_id']}: {doc['similarity_score']:.4f}")
        
        print("\n✅ JDBC Pipeline Test Successful!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def compare_odbc_vs_jdbc():
    """Compare performance between ODBC and JDBC pipelines"""
    
    print("\n🔍 Comparing ODBC vs JDBC Performance")
    print("=" * 50)
    
    from src.common.utils import get_embedding_func, get_llm_func # Updated import
    from src.deprecated.basic_rag.pipeline import BasicRAGPipeline as ODBCBasicRAGPipeline  # Updated import and aliased
    
    # Initialize both pipelines
    jdbc_pipeline = BasicRAGPipelineJDBC(
        embedding_func=get_embedding_func(),
        llm_func=get_llm_func()
    )
    
    # Note: ODBC pipeline might fail with vector operations
    try:
        from src.common.iris_connector import get_iris_connection # Updated import
        odbc_pipeline = ODBCBasicRAGPipeline( # Updated class name
            iris_connector=get_iris_connection(),
            embedding_func=get_embedding_func(),
            llm_func=get_llm_func()
        )
        odbc_available = True
    except:
        odbc_available = False
        print("⚠️  ODBC pipeline not available for comparison")
    
    # Test query
    query = "What are the symptoms of diabetes?"
    
    # Test JDBC
    print("\n📊 JDBC Pipeline:")
    jdbc_start = time.time()
    jdbc_result = jdbc_pipeline.run(query, top_k=3)
    jdbc_time = time.time() - jdbc_start
    print(f"   - Total time: {jdbc_time:.3f}s")
    print(f"   - Documents: {len(jdbc_result['retrieved_documents'])}")
    
    # Test ODBC if available
    if odbc_available:
        print("\n📊 ODBC Pipeline:")
        try:
            odbc_start = time.time()
            odbc_result = odbc_pipeline.run(query, top_k=3)
            odbc_time = time.time() - odbc_start
            print(f"   - Total time: {odbc_time:.3f}s")
            print(f"   - Documents: {len(odbc_result['retrieved_documents'])}")
            
            # Compare
            print(f"\n📊 Performance Difference:")
            print(f"   - JDBC is {odbc_time/jdbc_time:.2f}x the speed of ODBC")
            
        except Exception as e:
            print(f"   - ❌ Failed: {e}")
            print("   - This is expected due to ODBC vector function issues")
    
    print("\n✅ Comparison Complete!")

if __name__ == "__main__":
    # Test the JDBC pipeline
    test_jdbc_pipeline()
    
    # Compare with ODBC
    compare_odbc_vs_jdbc()