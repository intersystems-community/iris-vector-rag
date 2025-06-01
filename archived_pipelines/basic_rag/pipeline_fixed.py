"""
BasicRAG Pipeline - Fixed Version
Works with the actual Document class structure
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
import json
from typing import List, Dict, Any, Callable
import numpy as np
from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection

logger = logging.getLogger(__name__)

class BasicRAGPipelineFixed:
    def __init__(self, iris_connector,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipelineFixed initialized with schema: {schema}")

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieves documents using SQL-based vector search with VECTOR_COSINE.
        """
        logger.debug(f"BasicRAG SQL: Retrieving documents for query: '{query_text[:50]}...' with top_k={top_k}, threshold={similarity_threshold}")
        
        # 1. Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        if not query_embedding or not all(isinstance(x, (float, int)) for x in query_embedding):
            logger.error(f"BasicRAG SQL: Failed to generate a valid query embedding for: '{query_text[:50]}...'")
            return []
        
        embedding_str = ','.join(map(str, query_embedding))
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # 2. Construct and execute SQL query for vector search
            # We fetch TOP top_k results ordered by similarity, then filter by threshold in Python.
            sql_query = f"""
                SELECT TOP {top_k} doc_id, title, text_content,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                  AND embedding NOT LIKE '0.1,0.1,0.1%' -- Project-specific filter for invalid embeddings
                ORDER BY similarity_score DESC
            """
            
            logger.debug(f"BasicRAG SQL: Executing SQL query. Embedding (first 50 chars): {embedding_str[:50]}")
            cursor.execute(sql_query, (embedding_str,))
            results = cursor.fetchall()
            
            logger.info(f"BasicRAG SQL: Fetched {len(results)} candidate documents from DB.")

            # 3. Process results, handle potential streams, and filter by similarity_threshold
            for row in results:
                doc_id = row[0]
                title = row[1]
                raw_text_content = row[2]
                score = row[3]

                text_content_str = ""
                if raw_text_content is not None:
                    if hasattr(raw_text_content, 'read') and callable(raw_text_content.read):
                        try:
                            data = raw_text_content.read()
                            if isinstance(data, bytes):
                                text_content_str = data.decode('utf-8', errors='replace')
                            elif isinstance(data, str):
                                text_content_str = data
                            else:
                                text_content_str = str(data)
                                logger.warning(f"BasicRAG SQL: Unexpected data type from stream read for doc_id {doc_id}: {type(data)}")
                        except Exception as e_read:
                            logger.warning(f"BasicRAG SQL: Error reading stream for doc_id {doc_id}: {e_read}")
                            text_content_str = "[Content Read Error]"
                    elif isinstance(raw_text_content, bytes):
                        text_content_str = raw_text_content.decode('utf-8', errors='replace')
                    else:
                        text_content_str = str(raw_text_content)
                
                current_score = 0.0
                if score is not None:
                    try:
                        current_score = float(score)
                    except (ValueError, TypeError):
                        logger.warning(f"BasicRAG SQL: Could not convert score '{score}' to float for doc_id {doc_id}. Using 0.0.")
                        current_score = 0.0
                
                if current_score >= similarity_threshold:
                    doc = Document(
                        id=str(doc_id),
                        content=text_content_str,
                        score=current_score
                    )
                    doc._title = str(title) if title is not None else ""
                    retrieved_docs.append(doc)
            
            logger.info(f"BasicRAG SQL: Retrieved {len(retrieved_docs)} documents after applying threshold {similarity_threshold}.")
            
        except Exception as e:
            logger.error(f"BasicRAG SQL: Error retrieving documents: {e}", exc_info=True)
            raise
        finally:
            if cursor:
                cursor.close()
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query: str, retrieved_docs: List[Document]) -> str:
        """
        Generates an answer using the LLM based on the query and retrieved documents.
        """
        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."
        
        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3], 1):  # Use top 3 documents
            title = getattr(doc, '_title', 'Untitled')
            content = doc.content[:1000]  # Limit content length
            context_parts.append(f"Document {i} (Title: {title}):\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following documents, please answer the question.

Question: {query}

Context:
{context}

Please provide a comprehensive answer based on the information in the documents. If the documents don't contain enough information to fully answer the question, please state that clearly."""

        # Generate answer
        answer = self.llm_func(prompt)
        return answer

    @timing_decorator
    def run(self, query: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Runs the complete RAG pipeline.
        """
        logger.info(f"Running BasicRAG Fixed pipeline for query: '{query}'")
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k, similarity_threshold)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_docs)
        
        # Prepare result
        result = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.content[:500],  # Truncate for response
                    "score": doc.score,
                    "title": getattr(doc, '_title', 'Untitled')
                }
                for doc in retrieved_docs
            ],
            "metadata": {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "num_retrieved": len(retrieved_docs),
                "pipeline": "BasicRAG_Fixed"
            }
        }
        
        return result


# Demo execution
if __name__ == "__main__":
    print("Running BasicRAGPipelineFixed Demo...")
    
    # Initialize components
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create pipeline
    pipeline = BasicRAGPipelineFixed(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test query
    query = "What is diabetes?"
    print(f"\nExecuting RAG pipeline for query: '{query}'")
    
    try:
        result = pipeline.run(query, top_k=5, similarity_threshold=0.1)
        
        print(f"\nAnswer: {result['answer'][:200]}...")
        print(f"\nRetrieved {len(result['retrieved_documents'])} documents")
        for i, doc in enumerate(result['retrieved_documents'], 1):
            print(f"  Doc {i}: {doc['id']} (score: {doc['score']:.4f}) - {doc['title'][:50]}...")
        print(f"\nMetadata: {result['metadata']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()