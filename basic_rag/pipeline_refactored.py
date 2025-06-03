"""
Refactored BasicRAG Pipeline using centralized connection management
"""

import logging
from typing import List, Dict, Any, Optional

from common.base_pipeline import BaseRAGPipeline
from common.utils import Document

logger = logging.getLogger(__name__)

class BasicRAGPipeline(BaseRAGPipeline):
    """Basic RAG pipeline with centralized connection management"""
    
    def retrieve_documents(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.1,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant documents using vector similarity search
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"BasicRAG: Retrieving documents for query: '{query[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        query_embedding_str = self.format_embedding_for_iris(query_embedding)
        
        # Build SQL query based on connection type
        if self.connection_manager.connection_type == "jdbc":
            # JDBC supports parameterized queries
            sql = """
                SELECT TOP ? 
                    doc_id, 
                    title, 
                    text_content,
                    VECTOR_COSINE(
                        TO_VECTOR(embedding, 'FLOAT', 384),
                        TO_VECTOR(?, 'FLOAT', 384)
                    ) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                AND VECTOR_COSINE(
                    TO_VECTOR(embedding, 'FLOAT', 384),
                    TO_VECTOR(?, 'FLOAT', 384)
                ) > ?
                ORDER BY similarity_score DESC
            """
            params = [top_k, query_embedding_str, query_embedding_str, similarity_threshold]
        else:
            # ODBC needs the query formatted differently
            sql = f"""
                SELECT TOP {top_k} 
                    doc_id, 
                    title, 
                    text_content,
                    VECTOR_COSINE(
                        TO_VECTOR(embedding, 'FLOAT', 384),
                        TO_VECTOR('{query_embedding_str}', 'FLOAT', 384)
                    ) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                AND VECTOR_COSINE(
                    TO_VECTOR(embedding, 'FLOAT', 384),
                    TO_VECTOR('{query_embedding_str}', 'FLOAT', 384)
                ) > {similarity_threshold}
                ORDER BY similarity_score DESC
            """
            params = None
        
        try:
            # Execute query using connection manager
            results = self.execute_query(sql, params)
            
            # Convert to Document objects
            documents = []
            for row in results:
                doc = Document(
                    doc_id=row[0],
                    title=row[1],
                    content=row[2],
                    score=float(row[3])
                )
                documents.append(doc)
            
            logger.info(f"BasicRAG: Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"BasicRAG: Error retrieving documents: {e}")
            raise
    
    def generate_answer(
        self, 
        query: str, 
        documents: List[Document],
        **kwargs
    ) -> str:
        """
        Generate answer using LLM based on retrieved documents
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Generated answer
        """
        if not self.llm_func:
            raise ValueError("LLM function not provided")
        
        # Use base class prompt creation
        prompt = self.create_prompt(query, documents)
        
        # Generate answer
        answer = self.llm_func(prompt)
        return answer


def test_refactored_pipeline():
    """Test the refactored BasicRAG pipeline"""
    
    print("🔍 Testing Refactored BasicRAG Pipeline")
    print("=" * 50)
    
    from common.utils import get_embedding_func, get_llm_func
    from common.connection_manager import get_connection_manager
    
    try:
        # Test with default connection (ODBC for stability)
        print("\n📊 Testing with default connection:")
        pipeline = BasicRAGPipeline(
            embedding_func=get_embedding_func(),
            llm_func=get_llm_func()
        )
        
        query = "What are the symptoms of diabetes?"
        result = pipeline.run(query, top_k=3)
        
        print(f"✅ Pipeline successful with {result['metadata']['connection_type']}!")
        print(f"   - Retrieved {result['metadata']['num_documents']} documents")
        print(f"   - Total time: {result['metadata']['total_time']:.3f}s")
        print(f"   - Answer preview: {result['answer'][:100]}...")
        
        # Show document scores
        print("\n📊 Document Scores:")
        for doc in result['retrieved_documents']:
            print(f"   - {doc.doc_id}: {doc.score:.4f}")
        
        # Try JDBC if available
        print("\n📊 Testing with JDBC connection:")
        try:
            jdbc_pipeline = BasicRAGPipeline(
                connection_manager=get_connection_manager("jdbc"),
                embedding_func=get_embedding_func(),
                llm_func=get_llm_func()
            )
            
            result = jdbc_pipeline.run(query, top_k=3)
            print(f"✅ JDBC Pipeline successful!")
            print(f"   - Retrieved {result['metadata']['num_documents']} documents")
            print(f"   - Total time: {result['metadata']['total_time']:.3f}s")
        except Exception as e:
            print(f"⚠️  JDBC Pipeline failed: {e}")
            print("   This is expected if JDBC is not properly configured")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_refactored_pipeline()