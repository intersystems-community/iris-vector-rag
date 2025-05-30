"""
BasicRAG Pipeline with TO_VECTOR workaround
Uses unquoted DOUBLE in TO_VECTOR to avoid the IRIS parser bug
"""

import logging
from typing import List, Dict, Any, Callable
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class BasicRAGPipelineVectorFix:
    """
    Basic RAG pipeline using IRIS vector search with unquoted DOUBLE workaround
    """
    
    def __init__(self, iris_connector, embedding_func: Callable, llm_func: Callable):
        self.conn = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.cursor = self.conn.cursor()
        
    def retrieve_documents_vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using IRIS vector search with unquoted DOUBLE
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_func([query])[0]
            query_embedding_str = ','.join(map(str, query_embedding))
            
            # Use TO_VECTOR without quotes around DOUBLE
            sql_query = f"""
                SELECT doc_id, title, text_content,
                       VECTOR_COSINE(
                           TO_VECTOR(embedding, DOUBLE, 384),
                           TO_VECTOR('{query_embedding_str}', DOUBLE, 384)
                       ) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity_score DESC
            """
            
            # Add TOP clause for IRIS
            sql_query = sql_query.replace("SELECT", f"SELECT TOP {top_k}")
            
            logger.info(f"Executing vector search with unquoted DOUBLE...")
            self.cursor.execute(sql_query)
            
            results = []
            for row in self.cursor.fetchall():
                results.append({
                    "doc_id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "similarity_score": float(row[3])
                })
            
            logger.info(f"Retrieved {len(results)} documents using vector search")
            return results
            
        except Exception as e:
            logger.warning(f"Vector search failed: {e}, falling back to Python similarity")
            return self.retrieve_documents_fallback(query, top_k)
    
    def retrieve_documents_fallback(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Fallback: Retrieve documents using Python-based similarity calculation
        """
        # Generate query embedding
        query_embedding = self.embedding_func([query])[0]
        
        # Fetch all documents with embeddings
        self.cursor.execute("""
            SELECT doc_id, title, text_content, embedding 
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL
        """)
        
        documents = []
        embeddings = []
        
        for row in self.cursor.fetchall():
            doc_id, title, content, embedding_str = row
            
            # Parse embedding string
            try:
                embedding = [float(x.strip()) for x in embedding_str.split(',')]
                documents.append({
                    "doc_id": doc_id,
                    "title": title,
                    "content": content
                })
                embeddings.append(embedding)
            except Exception as e:
                logger.warning(f"Failed to parse embedding for {doc_id}: {e}")
                continue
        
        if not embeddings:
            logger.warning("No valid embeddings found")
            return []
        
        # Calculate similarities
        embeddings_array = np.array(embeddings)
        query_array = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_array, embeddings_array)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            doc = documents[idx]
            doc["similarity_score"] = float(similarities[idx])
            results.append(doc)
        
        return results
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate answer using LLM based on retrieved documents
        """
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"Document {i+1} (ID: {doc['doc_id']}):")
            if doc.get('title'):
                context_parts.append(f"Title: {doc['title']}")
            context_parts.append(f"Content: {doc['content'][:2000]}...")  # Limit content length
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following documents, please answer the question.

Question: {query}

Context:
{context}

Please provide a comprehensive answer based on the information in the documents."""
        
        # Generate answer
        answer = self.llm_func(prompt)
        return answer
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Run the BasicRAG pipeline with vector search workaround
        """
        logger.info(f"Running BasicRAG with vector fix for query: {query}")
        
        # Try vector search first, fallback to Python if needed
        retrieved_docs = self.retrieve_documents_vector_search(query, top_k)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_docs)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "method": "vector_search" if "vector_search" in str(retrieved_docs) else "python_similarity"
        }