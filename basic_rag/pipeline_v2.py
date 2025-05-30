"""
Basic RAG Pipeline V2 - Optimized with VECTOR columns and HNSW indexes
Uses the new _V2 tables with native vector search capabilities
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable
import logging

try:
    import iris
    IRISConnection = iris.IRISConnection
except ImportError:
    IRISConnection = Any

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection

logger = logging.getLogger(__name__)

class BasicRAGPipelineV2:
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipelineV2 initialized with schema: {schema}")

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Document]:
        """
        Retrieves documents from IRIS using native VECTOR search with HNSW index.
        Much faster than manual similarity calculation.
        """
        logger.debug(f"BasicRAG V2: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Use native VECTOR search with HNSW index
            sql = f"""
                SELECT TOP {top_k} 
                    doc_id, 
                    title, 
                    text_content,
                    VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE document_embedding_vector IS NOT NULL
                AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) > {similarity_threshold}
                ORDER BY similarity_score DESC
            """
            
            cursor.execute(sql)
            results = cursor.fetchall()
            
            logger.info(f"Retrieved {len(results)} documents using HNSW index")
            
            for row in results:
                doc_id = row[0]
                title = row[1] or ""
                content = row[2] or ""
                similarity = row[3]
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    score=similarity
                )
                # Store metadata separately for later use
                doc._metadata = {
                    "title": title,
                    "similarity_score": similarity,
                    "source": "BasicRAG_V2_HNSW"
                }
                retrieved_docs.append(doc)
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer using LLM based on retrieved documents."""
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents[:3], 1):  # Use top 3 documents
            # Get metadata from _metadata attribute if it exists
            metadata = getattr(doc, '_metadata', {})
            title = metadata.get('title', 'Untitled')
            score = doc.score or 0
            content_preview = doc.content[:500] if doc.content else ""
            context_parts.append(f"Document {i} (Score: {score:.3f}, Title: {title}):\n{content_preview}...")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question.
 
Context:
{context}
 
Question: {query}
 
Please provide a comprehensive answer based on the information provided in the context. If the context doesn't contain enough information to fully answer the question, please state what information is available and what is missing.
 
Answer:"""
        
        # Generate answer
        try:
            answer = self.llm_func(prompt)
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Run the complete RAG pipeline."""
        logger.info(f"Running BasicRAG V2 pipeline for query: '{query}'")
        
        # Retrieve documents
        documents = self.retrieve_documents(query, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        # Prepare response
        result = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": getattr(doc, '_metadata', {})
                }
                for doc in documents
            ],
            "metadata": {
                "pipeline": "BasicRAG_V2",
                "top_k": top_k,
                "num_retrieved": len(documents),
                "uses_hnsw": True
            }
        }
        
        return result


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create pipeline
    pipeline = BasicRAGPipelineV2(
        iris_connector=iris_conn,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test query
    test_query = "What are the symptoms of diabetes?"
    result = pipeline.run(test_query)
    
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Retrieved {result['metadata']['num_retrieved']} documents")