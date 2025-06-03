"""
BasicRAG Pipeline - Chunked Vector Version
Splits long embedding vectors into smaller chunks to work around IRIS parameter limitations
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

class BasicRAGPipelineChunkedVector:
    def __init__(self, iris_connector,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        self.chunk_size = 50  # Number of dimensions per chunk
        logger.info(f"BasicRAGPipelineChunkedVector initialized with schema: {schema}")

    def _split_vector_to_chunks(self, vector: List[float]) -> List[str]:
        """Split a vector into smaller chunks."""
        chunks = []
        for i in range(0, len(vector), self.chunk_size):
            chunk = vector[i:i + self.chunk_size]
            chunk_str = ','.join(map(str, chunk))
            chunks.append(chunk_str)
        return chunks

    def _create_vector_table(self, cursor, vector: List[float]):
        """Create a table with the query vector split into chunks."""
        # Drop if exists
        try:
            cursor.execute("DROP TABLE RAG.QueryVectorChunks")
        except:
            pass
        
        # Create table
        cursor.execute("""
            CREATE TABLE RAG.QueryVectorChunks (
                chunk_id INTEGER,
                chunk_data VARCHAR(2000)
            )
        """)
        
        # Insert chunks
        chunks = self._split_vector_to_chunks(vector)
        for i, chunk in enumerate(chunks):
            cursor.execute(
                "INSERT INTO RAG.QueryVectorChunks (chunk_id, chunk_data) VALUES (?, ?)",
                (i, chunk)
            )

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieves documents using a chunked vector approach.
        """
        logger.debug(f"BasicRAG ChunkedVector: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Create vector chunks table
            self._create_vector_table(cursor, query_embedding)
            
            # Build the full vector from chunks
            cursor.execute("""
                SELECT chunk_data 
                FROM RAG.QueryVectorChunks 
                ORDER BY chunk_id
            """)
            chunks = cursor.fetchall()
            
            # Reconstruct the vector string
            full_vector_str = ','.join([chunk[0] for chunk in chunks])
            
            # Now use the reconstructed vector in the query
            sql = f"""
                SELECT TOP {top_k} 
                    doc_id, 
                    title, 
                    text_content,
                    VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{full_vector_str}', 'DOUBLE')) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE document_embedding_vector IS NOT NULL
                AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{full_vector_str}', 'DOUBLE')) > {similarity_threshold}
                ORDER BY similarity_score DESC
            """
            
            logger.info(f"Executing vector similarity search with chunked approach...")
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
            
            # Clean up
            cursor.execute("DROP TABLE RAG.QueryVectorChunks")
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
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
        logger.info(f"Running BasicRAG ChunkedVector pipeline for query: '{query}'")
        
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
                "pipeline": "BasicRAG_ChunkedVector"
            }
        }
        
        return result


# Demo execution
if __name__ == "__main__":
    print("Running BasicRAGPipelineChunkedVector Demo...")
    
    # Initialize components
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create pipeline
    pipeline = BasicRAGPipelineChunkedVector(
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