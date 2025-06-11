"""
BasicRAG V2 Pipeline - Fixed Version
Uses IRIS native vector search with proper SQL formatting to avoid parameter issues
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) # Adjusted for new location

from typing import List, Dict, Any, Callable
import logging
import json

# IRISConnection type hint will be 'Any' as the actual type can vary.
# The actual connection object is passed in.
IRISConnection = Any

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection # Assuming JDBC for consistency

logger = logging.getLogger(__name__)

class BasicRAGPipelineV2Fixed:
    def __init__(self, iris_connector: Any, # Changed IRISConnection to Any
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipelineV2Fixed initialized with schema: {schema}")

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieves documents using IRIS native vector search with HNSW index.
        Fixed version that avoids parameter parsing issues.
        """
        logger.debug(f"BasicRAG V2 Fixed: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        
        # Convert to string format with fixed decimal places to avoid scientific notation
        # query_embedding_str_for_log = ','.join([f'{x:.10f}' for x in query_embedding]) # Original, for logging
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Use working VECTOR_COSINE syntax with proper parameter binding
            # Format vector as JSON array string for TO_VECTOR function
            query_vector_param_str = '[' + ','.join(map(str, query_embedding)) + ']' # For use in SQL parameter
            
            sql = f"""
                SELECT TOP {top_k}
                    doc_id,
                    title,
                    text_content,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score 
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity_score DESC
            """
            # Note: The original SQL had TO_VECTOR(embedding) which implies embedding column is string.
            # If embedding column is already VECTOR type, TO_VECTOR(embedding) is not needed.
            # For archival, keeping it as is, but this might be a point of failure/inefficiency.
            # Also, the original SQL did not filter by similarity_threshold here, it was done in Python.
            # This version will also filter in Python after fetching top_k * (some factor) if needed, or just top_k.
            # For simplicity, fetching top_k and then filtering.
            
            # Execute with parameter binding
            cursor.execute(sql, [query_vector_param_str])
            results = cursor.fetchall()
            
            logger.info(f"Retrieved {len(results)} documents using HNSW index (before threshold filter)")
            
            for row in results:
                doc_id = row[0]
                title = row[1] or ""
                content = row[2] or "" # Assuming content is string, or handle stream
                similarity = row[3]

                current_score = float(similarity) if similarity is not None else 0.0
                if current_score >= similarity_threshold:
                    # Create Document with proper fields
                    doc = Document(
                        id=str(doc_id),
                        content=f"Title: {str(title)}\n\n{str(content)}", # Ensure string
                        score=current_score
                    )
                    retrieved_docs.append(doc)
            
            # Sort again if more than top_k were fetched and then filtered
            retrieved_docs.sort(key=lambda x: x.score, reverse=True)
            retrieved_docs = retrieved_docs[:top_k]
            logger.info(f"Filtered to {len(retrieved_docs)} documents after threshold")

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}", exc_info=True)
            # Fall back to the original approach if vector search fails
            logger.info("Falling back to manual similarity calculation")
            return self._fallback_retrieve(query_text, query_embedding, top_k, similarity_threshold)
            
        finally:
            if cursor:
                cursor.close()
                
        return retrieved_docs
    
    def _fallback_retrieve(self, query_text: str, query_embedding: List[float], 
                          top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Fallback retrieval using manual similarity calculation (like original pipeline).
        """
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Get sample of documents with embeddings
            sql = f"""
                SELECT TOP 100 doc_id, title, text_content, embedding
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                AND embedding NOT LIKE '0.1,0.1,0.1%'
                ORDER BY doc_id
            """
            
            cursor.execute(sql)
            sample_docs = cursor.fetchall()
            
            logger.info(f"Processing {len(sample_docs)} sample documents for fallback")
            
            # Calculate similarities
            doc_scores = []
            
            for row in sample_docs:
                doc_id = row[0]
                title = row[1]
                content = row[2]
                embedding_str = row[3]
                
                try:
                    # Parse the stored embedding
                    if embedding_str and embedding_str.startswith('['):
                        doc_embedding = json.loads(embedding_str)
                    else:
                        doc_embedding = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
                    
                    # Ensure embeddings have the same dimension
                    if len(doc_embedding) != len(query_embedding):
                        continue
                    
                    # Calculate cosine similarity
                    dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                    query_norm = sum(a * a for a in query_embedding) ** 0.5
                    doc_norm = sum(a * a for a in doc_embedding) ** 0.5
                    
                    if query_norm > 0 and doc_norm > 0:
                        similarity = dot_product / (query_norm * doc_norm)
                    else:
                        similarity = 0.0
                    
                    if similarity >= similarity_threshold:
                        doc_scores.append((doc_id, title, content, similarity))
                        
                except Exception as e:
                    logger.debug(f"Error processing document {doc_id}: {e}")
                    continue
            
            # Sort by similarity and take top_k
            doc_scores.sort(key=lambda x: x[3], reverse=True)
            
            for doc_id, title, content, similarity in doc_scores[:top_k]:
                doc = Document(
                    id=str(doc_id),
                    content=f"Title: {str(title) or 'Untitled'}\n\n{str(content)}",
                    score=float(similarity)
                )
                retrieved_docs.append(doc)
                
        except Exception as e:
            logger.error(f"Error in fallback retrieval: {e}", exc_info=True)
            raise
            
        finally:
            if cursor:
                cursor.close()
                
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generates an answer using the LLM based on retrieved documents.
        """
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Extract title from content if present
            content_lines = doc.content.split('\n')
            title_str = 'Untitled'
            actual_content = doc.content
            if content_lines[0].startswith('Title: '):
                title_str = content_lines[0][7:]
                actual_content = '\n'.join(content_lines[2:])  # Skip title and empty line
            
            context_parts.append(f"Document {i} (Title: {title_str}, Score: {doc.score:.3f}):\n{actual_content[:1000]}...")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following documents, please answer the question.

Question: {query}

Context:
{context}

Please provide a comprehensive answer based on the information provided. If the information is not sufficient to answer the question, please state that clearly."""

        try:
            answer = self.llm_func(prompt)
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return f"I encountered an error while generating the answer: {str(e)}"

    @timing_decorator
    def run(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Runs the complete RAG pipeline.
        """
        # Retrieve relevant documents
        documents = self.retrieve_documents(query, top_k=top_k, similarity_threshold=similarity_threshold)
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        # Determine retrieval method (simplified)
        retrieval_method = "vector_search_hnsw_or_fallback"
        
        # Prepare response
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.content[:500] + "..." if doc.content and len(doc.content) > 500 else doc.content,
                    "score": doc.score
                }
                for doc in documents
            ],
            "metadata": {
                "retrieval_method": retrieval_method,
                "num_documents": len(documents)
            }
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Set to INFO for demo
    # Initialize components
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create pipeline
    pipeline = BasicRAGPipelineV2Fixed(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test query
    query = "What are the symptoms of diabetes?"
    result = pipeline.run(query, top_k=3)
    
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Retrieved {result['metadata']['num_documents']} documents")
    for doc_res in result['retrieved_documents']:
        print(f"  ID: {doc_res['id']}, Score: {doc_res['score']:.4f}")

    if iris_connector:
        iris_connector.close()