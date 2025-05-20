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
    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Document]:
        """
        Generates a hypothetical document, embeds it, and retrieves similar actual documents.
        """
        print(f"HyDE: Retrieving documents for query: '{query_text[:50]}...'")
        hypothetical_doc_text = self._generate_hypothetical_document(query_text)
        
        if not hypothetical_doc_text:
            print("HyDE: Failed to generate hypothetical document. Returning empty list.")
            return []

        hypo_embedding = self.embedding_func([hypothetical_doc_text])[0]
        
        iris_vector_str = f"[{','.join(map(str, hypo_embedding))}]"
        current_top_k = int(top_k)

        logger.info(f"HyDE: Retrieving documents for query: '{query_text[:50]}...' based on hypothetical doc using Python-generated SQL (fully inlined).")
        
        # Construct the dynamic SQL query string in Python
        # Inline TOP K and the vector string directly into the SQL query using f-strings.
        # IRIS SQL does not support parameter placeholders (?) for TOP or TO_VECTOR arguments.
        
        sql_query = f"""
            SELECT TOP {current_top_k} doc_id, text_content,
                   VECTOR_COSINE(embedding, TO_VECTOR('{iris_vector_str}', 'DOUBLE', 768)) AS score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY score DESC
        """

        retrieved_docs: List[Document] = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Log the exact SQL being executed
            logger.debug(f"Executing SQL: {sql_query}")

            # Execute the dynamically constructed SQL query with all parameters inlined
            # No parameters are passed to cursor.execute()
            cursor.execute(sql_query)
            
            fetched_rows = cursor.fetchall()
            if fetched_rows:
                for row_tuple in fetched_rows: # row_tuple is (doc_id, text_content, score)
                    retrieved_docs.append(Document(id=str(row_tuple[0]), content=str(row_tuple[1]), score=float(row_tuple[2])))
            logger.info(f"HyDE: Retrieved {len(retrieved_docs)} documents via Python-generated SQL based on hypothetical document.")

        except Exception as e:
            logger.error(f"HyDE: Error executing Python-generated SQL query based on hypothetical document: {e}")
            # Consider pipeline's error strategy; for now, errors will propagate or lead to empty list.
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs

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
    def run(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs the full HyDE pipeline.
        """
        print(f"HyDE: Running pipeline for query: '{query_text[:50]}...'")
        # Note: _generate_hypothetical_document is called within retrieve_documents
        retrieved_documents = self.retrieve_documents(query_text, top_k)
        answer = self.generate_answer(query_text, retrieved_documents)
        
        # For output, it might be useful to also return the hypothetical document
        # For now, keeping it similar to BasicRAG's output structure.
        # hypothetical_doc = self._generate_hypothetical_document(query_text) # Could be cached if needed

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents,
            # "hypothetical_document": hypothetical_doc # Optional to include
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
