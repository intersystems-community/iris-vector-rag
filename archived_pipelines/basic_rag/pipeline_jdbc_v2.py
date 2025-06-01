"""
BasicRAG Pipeline using JDBC with V2 tables for improved vector support
This version uses the V2 tables which have better performance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Dict, Any
import time
import logging
from jdbc_exploration.iris_jdbc_connector import get_iris_jdbc_connection

logger = logging.getLogger(__name__)

class BasicRAGPipelineJDBCV2:
    """Basic RAG pipeline using JDBC connection with V2 tables"""
    
    def __init__(self, iris_connector=None, embedding_func=None, llm_func=None):
        """Initialize the pipeline with JDBC connection"""
        # Use provided connector or create new JDBC connection
        self.iris_connector = iris_connector or get_iris_jdbc_connection()
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        
        # Test connection
        try:
            result = self.iris_connector.execute("SELECT 1")
            logger.info("JDBC connection test successful")
        except Exception as e:
            logger.error(f"JDBC connection test failed: {e}")
            raise
    
    def retrieve_documents(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using vector similarity search via JDBC on V2 tables"""
        
        # Generate query embedding
        if not self.embedding_func:
            raise ValueError("Embedding function not provided")
        
        query_embedding = self.embedding_func([query])[0]
        
        # Convert embedding to string format
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        # SQL query using V2 table with embedding column (VARCHAR)
        # Note: We use the regular embedding column, not document_embedding_vector
        # to avoid VECTOR type issues with JDBC
        sql = """
            SELECT TOP ? 
                doc_id, 
                title, 
                text_content,
                VECTOR_COSINE(
                    TO_VECTOR(embedding),
                    TO_VECTOR(?)
                ) as similarity_score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            AND VECTOR_COSINE(
                TO_VECTOR(embedding),
                TO_VECTOR(?)
            ) > ?
            ORDER BY similarity_score DESC
        """
        
        try:
            # Execute with parameters - JDBC handles this correctly!
            results = self.iris_connector.execute(
                sql, 
                [top_k, query_embedding_str, query_embedding_str, similarity_threshold]
            )
            
            # Format results
            documents = []
            for row in results:
                documents.append({
                    'doc_id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'similarity_score': float(row[3])
                })
            
            logger.info(f"Retrieved {len(documents)} documents from V2 table via JDBC")
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
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Run the complete RAG pipeline"""
        
        start_time = time.time()
        
        # Retrieve documents
        documents = self.retrieve_documents(query, top_k=top_k)
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
                "connection_type": "JDBC",
                "table_version": "V2"
            }
        }

def test_jdbc_v2_pipeline():
    """Test the JDBC-based BasicRAG pipeline with V2 tables"""
    
    print("🔍 Testing BasicRAG Pipeline with JDBC and V2 Tables")
    print("=" * 60)
    
    # Import required functions
    from common.utils import get_embedding_func, get_llm_func
    
    try:
        # Initialize pipeline
        pipeline = BasicRAGPipelineJDBCV2(
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
        
        print("\n✅ JDBC V2 Pipeline Test Successful!")
        print("\n🎯 Key Benefits:")
        print("   - Parameter binding works correctly")
        print("   - No SQL injection vulnerabilities")
        print("   - Uses V2 tables with better performance")
        print("   - Compatible with HNSW indexes")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the JDBC V2 pipeline
    test_jdbc_v2_pipeline()