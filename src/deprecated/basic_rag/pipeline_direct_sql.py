"""
BasicRAG Pipeline - Direct SQL Version
Uses direct SQL without parameters to work with IRIS vector functions
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) # Adjusted for new location

import logging
import time
from typing import List, Dict, Any, Callable
import numpy as np
from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection

logger = logging.getLogger(__name__)

class BasicRAGPipelineDirectSQL:
    def __init__(self, iris_connector,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipelineDirectSQL initialized with schema: {schema}")

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieves documents using direct SQL with IRIS vector functions.
        """
        logger.debug(f"BasicRAG DirectSQL: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        
        # Convert embedding to string
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        query_embedding_str = ','.join(map(str, query_embedding))
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Build SQL directly without parameters
            sql = f"""
                SELECT TOP {top_k} 
                    doc_id, 
                    title, 
                    text_content,
                    VECTOR_COSINE(
                        document_embedding_vector, 
                        TO_VECTOR('{query_embedding_str}', 'FLOAT')
                    ) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE document_embedding_vector IS NOT NULL
                AND VECTOR_COSINE(
                    document_embedding_vector, 
                    TO_VECTOR('{query_embedding_str}', 'FLOAT')
                ) > {similarity_threshold}
                ORDER BY similarity_score DESC
            """
            
            logger.info(f"Executing direct SQL vector similarity search...")
            logger.debug(f"SQL length: {len(sql)} characters")
            
            cursor.execute(sql)
            
            results = cursor.fetchall()
            logger.info(f"Retrieved {len(results)} documents")
            
            for row in results:
                doc_id = row[0]
                title = row[1] or "Untitled"
                content = row[2] or ""
                score = row[3]
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    metadata={
                        "title": title,
                        "similarity_score": float(score),
                        "doc_id": doc_id
                    }
                )
                retrieved_docs.append(doc)
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            # Fall back to Python-based similarity calculation
            logger.info("Falling back to Python-based similarity calculation...")
            return self._retrieve_documents_fallback(query_text, top_k, similarity_threshold)
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs

    def _retrieve_documents_fallback(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Fallback method using Python-based similarity calculation.
        """
        logger.debug(f"BasicRAG Fallback: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Get documents with embeddings
            sql = f"""
                SELECT TOP 100 doc_id, title, text_content, embedding
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                AND embedding NOT LIKE '0.1,0.1,0.1%'
                ORDER BY doc_id
            """
            
            cursor.execute(sql)
            sample_docs = cursor.fetchall()
            
            logger.info(f"Processing {len(sample_docs)} sample documents")
            
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
                        import json
                        doc_embedding = json.loads(embedding_str)
                    else:
                        doc_embedding = [float(x.strip()) for x in embedding_str.split(',') if x.strip()]
                    
                    # Ensure embeddings have the same dimension
                    if len(doc_embedding) != len(query_embedding):
                        logger.debug(f"Dimension mismatch for doc {doc_id}: {len(doc_embedding)} vs {len(query_embedding)}")
                        continue
                    
                    # Calculate cosine similarity
                    dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
                    query_norm = sum(a * a for a in query_embedding) ** 0.5
                    doc_norm = sum(a * a for a in doc_embedding) ** 0.5
                    
                    if query_norm > 0 and doc_norm > 0:
                        similarity = dot_product / (query_norm * doc_norm)
                    else:
                        similarity = 0.0
                    
                    if similarity > similarity_threshold:
                        doc_scores.append({
                            'doc_id': doc_id,
                            'title': title or "",
                            'content': content or "",
                            'score': similarity
                        })
                        
                except Exception as e:
                    logger.debug(f"Could not process embedding for doc {doc_id}: {e}")
                    continue
            
            # Sort by score and take top_k
            doc_scores.sort(key=lambda x: x['score'], reverse=True)
            doc_scores = doc_scores[:top_k]
            
            # Convert to Document objects
            for doc_data in doc_scores:
                doc = Document(
                    id=doc_data['doc_id'],
                    content=doc_data['content'],
                    metadata={
                        "title": doc_data['title'],
                        "similarity_score": doc_data['score'],
                        "doc_id": doc_data['doc_id']
                    }
                )
                retrieved_docs.append(doc)
            
            logger.info(f"BasicRAG Fallback: Retrieved {len(retrieved_docs)} documents above threshold")
            
        except Exception as e:
            logger.error(f"Error in fallback retrieval: {e}")
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
            title = doc.metadata.get('title', 'Untitled')
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
        logger.info(f"Running BasicRAG DirectSQL pipeline for query: '{query}'")
        
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
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ],
            "metadata": {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "num_retrieved": len(retrieved_docs),
                "pipeline": "BasicRAG_DirectSQL"
            }
        }
        
        return result


# Demo execution
if __name__ == "__main__":
    print("Running BasicRAGPipelineDirectSQL Demo...")
    
    # Initialize components
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create pipeline
    pipeline = BasicRAGPipelineDirectSQL(
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
        print(f"Metadata: {result['metadata']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()