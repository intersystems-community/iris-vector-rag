"""
Basic RAG Pipeline - Final Working Version
Uses direct SQL construction without parameters to avoid IRIS parsing issues
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) # Adjusted for new location

from typing import List, Dict, Any, Callable
import logging

try:
    import iris
    IRISConnection = iris.IRISConnection
except ImportError:
    IRISConnection = Any

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection
from common.jdbc_stream_utils import read_iris_stream # Added for stream handling

logger = logging.getLogger(__name__)

class BasicRAGPipeline:
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipeline initialized with schema: {schema}")

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieves documents from IRIS based on vector similarity.
        Uses the old tables without VECTOR columns to avoid all the issues.
        """
        logger.debug(f"BasicRAG: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        # Format for IRIS TO_VECTOR function
        iris_vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # SQL query using database-side vector search
            # Explicitly use TO_VECTOR on both sides with type and dimension
            # Assuming dimension is 384 based on common/utils.py DEFAULT_EMBEDDING_DIMENSION
            embedding_type = 'DOUBLE'
            embedding_dim = 384 # Ensure this matches your actual embedding dimension

            # embedding_type and embedding_dim are defined above but not used as direct SQL params here
            # TO_VECTOR will infer from the string format of iris_vector_str and the embedding column
            # Temporarily removing similarity_threshold from WHERE to see if any results are returned
            sql = f"""
                SELECT doc_id, title, text_content, similarity_score
                FROM (
                    SELECT
                        doc_id,
                        title,
                        text_content,
                        VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score,
                        ROW_NUMBER() OVER (ORDER BY VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) DESC) as rn
                    FROM {self.schema}.SourceDocuments
                    WHERE embedding IS NOT NULL
                )
                WHERE rn <= ?
                ORDER BY similarity_score DESC
            """
            
            params = (
                iris_vector_str,  # For similarity_score in SELECT
                iris_vector_str,  # For ROW_NUMBER() OVER (ORDER BY ...)
                # iris_vector_str, # No longer needed for a WHERE clause vector comparison
                # similarity_threshold, # Removed
                top_k
            )
            logger.debug(f"BasicRAG: Executing SQL (no threshold): {sql} with {len(params)} params (vector truncated): (iris_vector_str[:20] + '...', iris_vector_str[:20] + '...', top_k)")
            cursor.execute(sql, params)
            
            results = cursor.fetchall()
            logger.info(f"BasicRAG: Fetched {len(results)} candidate documents from DB.")

            for i, row in enumerate(results):
                doc_id = str(row[0])
                title = str(row[1] or "")
                raw_content_from_db = row[2] # Keep original for logging
                score = float(row[3] or 0.0)

                # Log raw data fetched
                logger.debug(f"BasicRAG: Doc {i+1}/{len(results)} - ID: {doc_id}, Title: '{title}', Score: {score:.4f}")
                logger.debug(f"BasicRAG: Doc {i+1} - Raw Content Type: {type(raw_content_from_db)}")
                
                # It's crucial to see a snippet of the raw content if it's a string, or its representation
                if isinstance(raw_content_from_db, str):
                    logger.debug(f"BasicRAG: Doc {i+1} - Raw Content Snippet: '{raw_content_from_db[:200]}'")
                elif hasattr(raw_content_from_db, 'read'): # For stream-like objects
                    # Avoid consuming the stream here if it's needed later by read_iris_stream
                    # Instead, read_iris_stream will handle it.
                    logger.debug(f"BasicRAG: Doc {i+1} - Raw Content is a stream-like object.")
                else:
                    logger.debug(f"BasicRAG: Doc {i+1} - Raw Content: {str(raw_content_from_db)[:200]}")

                content = read_iris_stream(raw_content_from_db)
                logger.debug(f"BasicRAG: Doc {i+1} - Processed Content Snippet: '{content[:200]}'")
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    score=score
                )
                doc.metadata = {"title": title} # Assign metadata separately
                retrieved_docs.append(doc)
            
            logger.info(f"BasicRAG: Retrieved {len(retrieved_docs)} documents after DB-side filtering.")
            
        except Exception as e:
            logger.error(f"BasicRAG: Error during document retrieval: {e}", exc_info=True)
            return []
        finally:
            if cursor:
                cursor.close()
                
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        """
        Generates an answer using the LLM based on the query and retrieved documents.
        """
        logger.debug(f"BasicRAG: Generating answer for query: '{query_text[:50]}...'")
        if not retrieved_docs:
            logger.warning("BasicRAG: No documents retrieved. Returning a default response.")
            return "I could not find enough information to answer your question."

        # Limit context to prevent token overflow
        context_parts = []
        total_chars = 0
        max_context_chars = 8000  # Conservative limit
        
        for doc in retrieved_docs:
            doc_content = doc.content[:2000]  # Limit each document
            if total_chars + len(doc_content) > max_context_chars:
                break
            context_parts.append(doc_content)
            total_chars += len(doc_content)
        
        context = "\n\n".join(context_parts)
        
        # Basic prompt engineering
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{context}

Question: {query_text}

Answer:"""
        
        answer = self.llm_func(prompt)
        logger.debug(f"BasicRAG: Generated answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Runs the full Basic RAG pipeline: retrieve documents and generate an answer.
        """
        logger.info(f"BasicRAG: Running pipeline for query: '{query_text[:50]}...'")
        retrieved_documents = self.retrieve_documents(query_text, top_k, similarity_threshold)
        answer = self.generate_answer(query_text, retrieved_documents)
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents],
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents),
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    print("Running BasicRAGPipeline Demo...")

    # Setup
    try:
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")  # Use stub for testing

        pipeline = BasicRAGPipeline(iris_connector=db_conn, embedding_func=embed_fn, llm_func=llm_fn)

        # Example Query
        test_query = "What is diabetes?"
        print(f"\nExecuting RAG pipeline for query: '{test_query}'")
        
        result = pipeline.run(test_query, top_k=3)

        print("\n--- RAG Pipeline Result ---")
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
        for i, doc in enumerate(result['retrieved_documents']):
            print(f"  Doc {i+1}: ID={doc['id']}, Score={doc.get('score', 0):.4f}")
            if 'metadata' in doc and 'title' in doc['metadata']:
                print(f"         Title: {doc['metadata']['title'][:60]}...")
        
        if 'latency_ms' in result:
            print(f"Total Pipeline Latency: {result['latency_ms']:.2f} ms")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'db_conn' in locals() and db_conn is not None:
            db_conn.close()
            print("\nDatabase connection closed.")

    print("\nBasicRAGPipeline Demo Finished.")