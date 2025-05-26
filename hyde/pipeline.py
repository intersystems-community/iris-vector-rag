# hyde/pipeline.py

import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable
# import sqlalchemy # No longer needed for type hinting
import logging # Ensure logging is imported
# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any # Fallback to Any if the driver isn't available during static analysis


from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
# Removed: from common.db_vector_search import search_source_documents_dynamically

logger = logging.getLogger(__name__) # Add logger
logger.setLevel(logging.DEBUG) # Ensure debug messages from this module are shown

class HyDEPipeline:
    def __init__(self, iris_connector: IRISConnection, # Updated type hint
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str]):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        print("HyDEPipeline Initialized")

    @timing_decorator
    def _generate_hypothetical_document(self, query_text: str) -> str:
        """
        Generates a hypothetical document for the given query using the LLM.
        """
        # Prompt engineering is key here. This is a basic example.
        prompt = (
            f"Write a short, concise passage that directly answers the following question. "
            f"Focus on providing a factual-sounding answer, even if you need to make up plausible details. "
            f"Do not state that you are an AI or that the answer is hypothetical.\n\n"
            f"Question: {query_text}\n\n"
            f"Passage:"
        )
        hypothetical_doc_text = self.llm_func(prompt)
        print(f"HyDE: Generated hypothetical document: '{hypothetical_doc_text[:100]}...'")
        return hypothetical_doc_text

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.75) -> List[Document]:
        """
        Generates a hypothetical document, embeds it, and retrieves similar actual documents
        using vector search against the HNSW-accelerated database.
        """
        logger.info(f"HyDE: Retrieving documents for query: '{query_text[:50]}...' with threshold {similarity_threshold}")

        # 1. Generate hypothetical document
        hypothetical_doc_text = self._generate_hypothetical_document(query_text)
        logger.info(f"HyDE: Generated hypothetical document: '{hypothetical_doc_text[:100]}...'")

        # 2. Embed the hypothetical document
        hypothetical_doc_embedding = self.embedding_func([hypothetical_doc_text])[0]
        logger.debug(f"HyDE: Embedding generated for hypothetical document.")

        # 3. Construct and execute SQL query for vector search
        documents = []
        if not hypothetical_doc_embedding or not all(isinstance(x, (int, float)) for x in hypothetical_doc_embedding):
            logger.error("HyDE: Failed to generate a valid embedding for the hypothetical document.")
            return []

        # Convert embedding to comma-separated string format for IRIS
        embedding_str = ','.join(map(str, hypothetical_doc_embedding))

        # Use RAG_HNSW schema with similarity threshold
        sql_query = f"""
            SELECT TOP {top_k} doc_id, text_content,
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
            FROM RAG_HNSW.SourceDocuments
            WHERE embedding IS NOT NULL
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
            ORDER BY similarity_score DESC
        """
        
        try:
            cursor = self.iris_connector.cursor()
            logger.debug(f"HyDE: Executing SQL query with threshold {similarity_threshold}")
            cursor.execute(sql_query, (embedding_str, embedding_str, similarity_threshold))
            results = cursor.fetchall()
            
            for row in results:
                doc_id = row[0]
                doc_content = row[1]
                score = row[2]
                documents.append(Document(id=str(doc_id), content=doc_content, score=float(score)))
            
            cursor.close()
            logger.info(f"HyDE: Retrieved {len(documents)} documents above threshold {similarity_threshold}")

        except Exception as e:
            logger.error(f"HyDE: Error during database vector search: {e}", exc_info=True)
            return []
            
        return documents

    @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        """
        Generates a final answer using the LLM based on the original query and retrieved actual documents.
        """
        print(f"HyDE: Generating final answer for query: '{query_text[:50]}...'")
        if not retrieved_docs:
            print("HyDE: No documents retrieved. Returning a default response.")
            return "I could not find enough information to answer your question."

        context = "\n\n".join([doc.content for doc in retrieved_docs])
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{context}

Question: {query_text}

Answer:"""
        
        answer = self.llm_func(prompt)
        print(f"HyDE: Generated final answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.65) -> Dict[str, Any]:
        """
        Runs the full HyDE pipeline.
        """
        print(f"HyDE: Running pipeline for query: '{query_text[:50]}...'")
        
        # Generate the hypothetical document first
        hypothetical_doc = self._generate_hypothetical_document(query_text)
        
        # Then retrieve documents
        retrieved_documents = self.retrieve_documents(query_text, top_k, similarity_threshold)
        
        # Generate the final answer
        answer = self.generate_answer(query_text, retrieved_documents)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents,
            "hypothetical_document": hypothetical_doc,
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents)
        }

if __name__ == '__main__':
    print("Running HyDEPipeline Demo...")
    from common.iris_connector import get_iris_connection # For demo

    try:
        db_conn = get_iris_connection() # Uses IRIS_CONNECTION_URL or falls back to mock
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")

        if db_conn is None:
            raise ConnectionError("Failed to get IRIS connection for HyDE demo.")

        pipeline = HyDEPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn
        )

        # Example Query
        test_query = "What are the symptoms of long COVID?"
        print(f"\nExecuting HyDE pipeline for query: '{test_query}'")
        
        result = pipeline.run(test_query, top_k=3)

        print("\n--- HyDE Pipeline Result ---")
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
        for i, doc in enumerate(result['retrieved_documents']):
            print(f"  Doc {i+1}: ID={doc.id}, Score={doc.score:.4f}, Content='{doc.content[:100]}...'")
        
        if 'latency_ms' in result: # Will be added by the run decorator
             print(f"Total Pipeline Latency (run method): {result['latency_ms']:.2f} ms")

    except ConnectionError as ce:
        print(f"Demo Setup Error: {ce}")
    except ValueError as ve:
        print(f"Demo Setup Error: {ve}")
    except ImportError as ie:
        print(f"Demo Import Error: {ie}")
    except Exception as e:
        print(f"An unexpected error occurred during HyDE demo: {e}")
    finally:
        if 'db_conn' in locals() and db_conn is not None:
            try:
                db_conn.close()
                print("Database connection closed.")
            except Exception as e_close:
                print(f"Error closing DB connection: {e_close}")
    
    print("\nHyDEPipeline Demo Finished.")
