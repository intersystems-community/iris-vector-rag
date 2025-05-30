# crag/pipeline_jdbc_fixed.py
"""
JDBC-Fixed CRAG Pipeline that properly handles vector parameter binding
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable, Optional
import logging
import numpy as np

try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any

logger = logging.getLogger(__name__)

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection
from common.jdbc_stream_utils import read_iris_stream

class JDBCFixedCRAGPipeline:
    """
    JDBC-Fixed Corrective RAG (CRAG) Pipeline
    
    This implementation fixes vector parameter binding issues by:
    1. Using direct SQL without parameter binding for vector operations
    2. Properly handling JDBC stream objects
    3. Implementing fallback strategies for retrieval
    """
    
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 web_search_func: Optional[Callable[[str], List[Document]]] = None):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.web_search_func = web_search_func
        logger.info("JDBCFixedCRAGPipeline initialized")

    @timing_decorator
    def _retrieve_chunks_jdbc_safe(self, query_text: str, top_k: int = 20, 
                                   similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieve chunks using JDBC-safe vector operations
        """
        logger.info(f"CRAG: Retrieving chunks for query: '{query_text[:50]}...'")
        
        cursor = None
        retrieved_chunks = []
        
        try:
            cursor = self.iris_connector.cursor()
            
            # Generate query embedding
            query_embedding = self.embedding_func([query_text])[0]
            query_vector_str = ','.join(map(str, query_embedding))
            
            # Use direct SQL without parameter binding for vectors
            vector_query = f"""
                SELECT TOP {top_k}
                    chunk_id,
                    chunk_text,
                    doc_id,
                    chunk_type,
                    chunk_index,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{query_vector_str}')) AS score
                FROM RAG.DocumentChunks
                WHERE embedding IS NOT NULL
                  AND chunk_type IN ('content', 'mixed')
                  AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{query_vector_str}')) > {similarity_threshold}
                ORDER BY score DESC
            """
            
            cursor.execute(vector_query)
            results = cursor.fetchall()
            
            for chunk_id, chunk_text, doc_id, chunk_type, chunk_index, score in results:
                # Handle JDBC stream objects
                chunk_text_str = read_iris_stream(chunk_text)
                
                if chunk_text_str:
                    retrieved_chunks.append(Document(
                        id=f"{doc_id}_chunk_{chunk_id}",
                        content=chunk_text_str,
                        score=float(score) if score else 0.0,
                        metadata={
                            'doc_id': doc_id,
                            'chunk_type': chunk_type,
                            'chunk_index': chunk_index
                        }
                    ))
            
            logger.info(f"CRAG: Retrieved {len(retrieved_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"CRAG: Error retrieving chunks: {e}")
            # Fallback to document-level retrieval
            retrieved_chunks = self._fallback_document_retrieval(query_text, top_k)
            
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_chunks

    @timing_decorator
    def _fallback_document_retrieval(self, query_text: str, top_k: int = 20) -> List[Document]:
        """
        Fallback to document-level retrieval if chunk retrieval fails
        """
        logger.info("CRAG: Falling back to document-level retrieval")
        
        cursor = None
        retrieved_docs = []
        
        try:
            cursor = self.iris_connector.cursor()
            
            # Generate query embedding
            query_embedding = self.embedding_func([query_text])[0]
            query_vector_str = ','.join(map(str, query_embedding))
            
            # Direct SQL for document retrieval
            doc_query = f"""
                SELECT TOP {top_k}
                    doc_id,
                    text_content,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{query_vector_str}')) AS score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                  AND LENGTH(embedding) > 1000
                ORDER BY score DESC
            """
            
            cursor.execute(doc_query)
            results = cursor.fetchall()
            
            for doc_id, content, score in results:
                # Handle JDBC stream objects
                content_str = read_iris_stream(content)
                
                if content_str:
                    # Create chunks from document
                    chunk_size = 512
                    for i in range(0, len(content_str), chunk_size):
                        chunk = content_str[i:i+chunk_size]
                        retrieved_docs.append(Document(
                            id=f"{doc_id}_fallback_chunk_{i//chunk_size}",
                            content=chunk,
                            score=float(score) if score else 0.0,
                            metadata={'doc_id': doc_id, 'is_fallback': True}
                        ))
                        if len(retrieved_docs) >= top_k:
                            break
                
                if len(retrieved_docs) >= top_k:
                    break
            
            logger.info(f"CRAG: Fallback retrieved {len(retrieved_docs)} document chunks")
            
        except Exception as e:
            logger.error(f"CRAG: Error in fallback retrieval: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs[:top_k]

    @timing_decorator
    def _evaluate_retrieval(self, documents: List[Document], query: str) -> str:
        """
        Evaluate the quality of retrieved documents
        """
        if not documents:
            return "disoriented"
        
        avg_score = np.mean([doc.score for doc in documents])
        
        if avg_score > 0.7:
            return "correct"
        elif avg_score > 0.4:
            return "ambiguous"
        else:
            return "disoriented"

    @timing_decorator
    def _augment_with_web_search(self, query: str) -> List[Document]:
        """
        Augment retrieval with web search (if available)
        """
        if self.web_search_func:
            logger.info("CRAG: Augmenting with web search")
            try:
                return self.web_search_func(query)
            except Exception as e:
                logger.error(f"CRAG: Web search failed: {e}")
        return []

    @timing_decorator
    def _decompose_recompose_filter(self, documents: List[Document], query: str) -> List[Document]:
        """
        Decompose, filter, and recompose documents
        """
        if not documents:
            return []
        
        # Simple filtering based on relevance
        filtered_docs = [doc for doc in documents if doc.score > 0.3]
        
        # Sort by score
        filtered_docs.sort(key=lambda x: x.score, reverse=True)
        
        return filtered_docs

    @timing_decorator
    def retrieve_and_correct(self, query_text: str, top_k: int = 20) -> List[Document]:
        """
        Main CRAG retrieval and correction method
        """
        logger.info(f"CRAG: Starting retrieve and correct for query: '{query_text[:50]}...'")
        
        # Initial retrieval
        initial_docs = self._retrieve_chunks_jdbc_safe(query_text, top_k)
        
        # Evaluate retrieval quality
        retrieval_status = self._evaluate_retrieval(initial_docs, query_text)
        logger.info(f"CRAG: Retrieval status: {retrieval_status}")
        
        if retrieval_status == "correct":
            # Good retrieval, just filter and return
            return self._decompose_recompose_filter(initial_docs, query_text)
        
        elif retrieval_status == "ambiguous":
            # Ambiguous, try to improve with filtering
            filtered_docs = self._decompose_recompose_filter(initial_docs, query_text)
            
            # If still not enough good docs, augment
            if len(filtered_docs) < top_k // 2:
                web_docs = self._augment_with_web_search(query_text)
                filtered_docs.extend(web_docs)
            
            return filtered_docs[:top_k]
        
        else:  # disoriented
            # Poor retrieval, augment with web search
            web_docs = self._augment_with_web_search(query_text)
            
            # Combine with any decent initial docs
            combined_docs = initial_docs + web_docs
            
            # Filter and recompose
            return self._decompose_recompose_filter(combined_docs, query_text)[:top_k]

    @timing_decorator
    def generate_answer(self, query_text: str, context_docs: List[Document]) -> str:
        """
        Generate answer using LLM with corrected context
        """
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context
        context_parts = []
        for i, doc in enumerate(context_docs[:10]):  # Limit context
            context_parts.append(f"[{i+1}] {doc.content[:500]}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query_text}

Answer:"""
        
        try:
            answer = self.llm_func(prompt)
            return answer
        except Exception as e:
            logger.error(f"CRAG: Error generating answer: {e}")
            return "I encountered an error while generating the answer."

    @timing_decorator
    def run(self, query_text: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Run the complete JDBC-fixed CRAG pipeline
        """
        logger.info(f"CRAG: Running JDBC-fixed pipeline for query: '{query_text[:50]}...'")
        
        # Retrieve and correct
        corrected_docs = self.retrieve_and_correct(query_text, top_k)
        
        # Generate answer
        answer = self.generate_answer(query_text, corrected_docs)
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in corrected_docs],
            "document_count": len(corrected_docs),
            "method": "corrective_rag_jdbc"
        }

def create_jdbc_fixed_crag_pipeline(iris_connector=None, llm_func=None, web_search_func=None):
    """
    Factory function to create a JDBC-fixed CRAG pipeline
    """
    if iris_connector is None:
        iris_connector = get_iris_connection()
    
    if llm_func is None:
        llm_func = get_llm_func()
    
    embedding_func = get_embedding_func()
    
    return JDBCFixedCRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func,
        web_search_func=web_search_func
    )