"""
BasicRAG Pipeline using IRIS Vector Functions
This version uses TO_VECTOR() and VECTOR_COSINE() for similarity search
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from typing import List, Dict, Any, Callable
from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection

logger = logging.getLogger(__name__)

class BasicRAGPipelineVector:
    def __init__(self, iris_connector,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipelineVector initialized with schema: {schema}")

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieves documents from IRIS using native vector similarity functions.
        """
        logger.debug(f"BasicRAG Vector: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        query_embedding_str = ','.join(map(str, query_embedding))
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Use IRIS vector functions with _V2 tables and VECTOR columns
            # Note: We use direct SQL construction because IRIS has issues with long parameter strings
            sql = f"""
                SELECT TOP {top_k}
                    doc_id,
                    title,
                    text_content,
                    VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE')) as similarity_score
                FROM RAG.SourceDocuments
                WHERE document_embedding_vector IS NOT NULL
                AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE')) > {similarity_threshold}
                ORDER BY similarity_score DESC
            """
            
            logger.info(f"Executing vector similarity search with IRIS functions...")
            cursor.execute(sql)
            results = cursor.fetchall()
            
            logger.info(f"Found {len(results)} documents above threshold {similarity_threshold}")
            
            for row in results:
                doc_id = row[0]
                title = row[1]
                content = row[2]
                score = row[3]
                
                doc = Document(
                    page_content=content or "",
                    doc_id=doc_id
                )
                # Store title as private attribute
                doc._title = title or ""
                doc._score = score
                retrieved_docs.append(doc)
                
                logger.debug(f"Retrieved doc {doc_id} with score {score:.4f}")
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
                
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query: str, retrieved_docs: List[Document]) -> str:
        """Generate answer using LLM with retrieved context."""
        if not retrieved_docs:
            return "No relevant documents found to answer your query."
            
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3], 1):  # Use top 3 documents
            title = getattr(doc, '_title', 'Untitled')
            score = getattr(doc, '_score', 0.0)
            context_parts.append(f"Document {i} (Score: {score:.4f}, Title: {title}):\n{doc.page_content[:500]}...")
            
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate answer
        answer = self.llm_func(prompt)
        return answer

    @timing_decorator
    def run(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """Run the complete RAG pipeline."""
        logger.info(f"BasicRAG Vector: Running pipeline for query: '{query[:50]}...'")
        
        # Retrieve relevant documents using IRIS vector functions
        retrieved_docs = self.retrieve_documents(query, top_k, similarity_threshold)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_docs)
        
        # Prepare result
        result = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": getattr(doc, '_title', 'Untitled'),
                    "score": getattr(doc, '_score', 0.0),
                    "content": doc.page_content[:200] + "..."
                }
                for doc in retrieved_docs
            ],
            "metadata": {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "num_retrieved": len(retrieved_docs),
                "method": "IRIS Vector Functions (TO_VECTOR + VECTOR_COSINE)"
            }
        }
        
        logger.info(f"BasicRAG Vector: Retrieved {len(retrieved_docs)} documents")
        return result


# Demo usage
if __name__ == "__main__":
    print("Running BasicRAGPipelineVector Demo...")
    
    # Initialize components
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create pipeline
    pipeline = BasicRAGPipelineVector(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test query
    query = "What is diabetes?"
    print(f"\nExecuting RAG pipeline for query: '{query}'")
    
    # Run pipeline
    start_time = time.time()
    result = pipeline.run(query, top_k=5, similarity_threshold=0.1)
    end_time = time.time()
    
    # Display results
    print("\n--- RAG Pipeline Result ---")
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
    for i, doc in enumerate(result['retrieved_documents'], 1):
        print(f"  Doc {i}: ID={doc['doc_id']}, Score={doc['score']:.4f}")
        print(f"         Title: {doc['title'][:50]}...")
    print(f"Method: {result['metadata']['method']}")
    print(f"Total Pipeline Latency: {(end_time - start_time) * 1000:.2f} ms")
    
    # Close connection
    iris_connector.close()
    print("\nDatabase connection closed.")
    print("\nBasicRAGPipelineVector Demo Finished.")