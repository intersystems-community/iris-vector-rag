"""
BasicRAG Pipeline - Stored Procedure Version
Uses stored procedures to handle long embedding vectors with IRIS vector functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import time
from typing import List, Dict, Any, Callable
import numpy as np
from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection

logger = logging.getLogger(__name__)

class BasicRAGPipelineStoredProc:
    def __init__(self, iris_connector,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipelineStoredProc initialized with schema: {schema}")
        
        # Create stored procedure on initialization
        self._create_stored_procedure()
    
    def _create_stored_procedure(self):
        """Create the stored procedure for vector search if it doesn't exist."""
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Drop if exists
            try:
                cursor.execute("DROP PROCEDURE RAG.VectorSearch")
            except:
                pass  # Procedure might not exist
            
            # Create stored procedure
            create_proc = """
            CREATE PROCEDURE RAG.VectorSearch(
                IN query_vector VARCHAR(50000),
                IN top_k INTEGER,
                IN threshold DOUBLE
            )
            LANGUAGE SQL
            BEGIN
                -- Create temporary result table
                CREATE TEMPORARY TABLE TempResults AS
                SELECT 
                    doc_id,
                    title,
                    text_content,
                    VECTOR_COSINE(document_embedding_vector, TO_VECTOR(query_vector, 'DOUBLE')) as similarity_score
                FROM RAG.SourceDocuments
                WHERE document_embedding_vector IS NOT NULL
                AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR(query_vector, 'DOUBLE')) > threshold
                ORDER BY similarity_score DESC
                LIMIT top_k;
                
                -- Return results
                SELECT * FROM TempResults;
                
                -- Clean up
                DROP TABLE TempResults;
            END
            """
            
            cursor.execute(create_proc)
            logger.info("Created stored procedure RAG.VectorSearch")
            
        except Exception as e:
            logger.warning(f"Could not create stored procedure: {e}")
            # We'll fall back to direct query if procedure creation fails
        finally:
            if cursor:
                cursor.close()

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieves documents using IRIS vector functions via stored procedure.
        """
        logger.debug(f"BasicRAG StoredProc: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        query_embedding_str = ','.join(map(str, query_embedding))
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Try to use stored procedure first
            try:
                logger.info("Calling stored procedure for vector search...")
                cursor.execute(
                    "CALL RAG.VectorSearch(?, ?, ?)",
                    (query_embedding_str, top_k, similarity_threshold)
                )
                results = cursor.fetchall()
                
            except Exception as sp_error:
                logger.warning(f"Stored procedure failed: {sp_error}")
                logger.info("Falling back to direct query...")
                
                # Fallback: Use direct query with string interpolation
                sql = f"""
                    SELECT TOP {top_k} 
                        doc_id, 
                        title, 
                        text_content,
                        VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE')) as similarity_score
                    FROM {self.schema}.SourceDocuments
                    WHERE document_embedding_vector IS NOT NULL
                    AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE')) > {similarity_threshold}
                    ORDER BY similarity_score DESC
                """
                
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
        logger.info(f"Running BasicRAG StoredProc pipeline for query: '{query}'")
        
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
                "pipeline": "BasicRAG_StoredProc"
            }
        }
        
        return result


# Demo execution
if __name__ == "__main__":
    print("Running BasicRAGPipelineStoredProc Demo...")
    
    # Initialize components
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create pipeline
    pipeline = BasicRAGPipelineStoredProc(
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