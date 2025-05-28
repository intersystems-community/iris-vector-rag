# basic_rag/pipeline.py

import os
import sys
# Add the project root directory to Python path so we can import common module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable
import logging

try:
    import iris
    IRISConnection = iris.IRISConnection
except ImportError:
    IRISConnection = Any

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func, get_iris_connector

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
        Retrieves documents from IRIS based on vector similarity using HNSW acceleration.
        Uses similarity threshold for realistic document count variation.
        """
        logger.debug(f"BasicRAG: Retrieving documents for query: '{query_text[:50]}...' with threshold {similarity_threshold}")
        query_embedding = self.embedding_func([query_text])[0]

        # Convert to comma-separated string format for IRIS
        query_embedding_str_for_sql = ','.join(map(str, query_embedding))
        
        retrieved_docs = []
        try:
            cursor = self.iris_connector.cursor()
            
            # Use RAG_HNSW schema with similarity threshold instead of fixed TOP N
            sql_query = f"""
            SELECT TOP {top_k} doc_id, text_content,
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS similarity_score
            FROM {self.schema}.SourceDocuments
            WHERE embedding IS NOT NULL
              AND LENGTH(embedding) > 1000
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
            ORDER BY similarity_score DESC
            """
            
            logger.debug(f"Executing BasicRAG SQL with threshold {similarity_threshold}")
            logger.debug(f"SQL Query: {sql_query}")
            cursor.execute(sql_query, (query_embedding_str_for_sql, query_embedding_str_for_sql, similarity_threshold))
            
            for row in cursor.fetchall():
                score = float(row[2]) if isinstance(row[2], str) else row[2]
                doc = Document(id=str(row[0]), content=row[1] if row[1] is not None else "", score=score)
                retrieved_docs.append(doc)
            
            cursor.close()
            logger.info(f"BasicRAG: Retrieved {len(retrieved_docs)} documents above threshold {similarity_threshold}")
            
        except Exception as e:
            logger.error(f"BasicRAG: Error during document retrieval: {e}", exc_info=True)
            return []
            
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

        # Limit context to prevent token overflow - truncate documents if needed
        context_parts = []
        total_chars = 0
        max_context_chars = 8000  # Conservative limit to stay under 16K tokens
        
        for doc in retrieved_docs:
            doc_content = doc.content[:2000]  # Limit each document to 2000 chars
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
        # Limit documents for answer generation to prevent context overflow
        answer_docs = retrieved_documents[:top_k] if len(retrieved_documents) > top_k else retrieved_documents
        answer = self.generate_answer(query_text, answer_docs)
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents],
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents),
            # "latency_ms" will be added by timing_decorator if applied to this 'run' method
        }

if __name__ == '__main__':
    print("Running BasicRAGPipeline Demo...")

    # Setup (requires IRIS_CONNECTION_URL and optionally OPENAI_API_KEY)
    try:
        # These will raise errors if env vars not set or libraries not installed
        db_conn = get_iris_connector() # Uses IRIS_CONNECTION_URL
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub") # Use stub LLM for local testing without OpenAI
        # To use OpenAI: llm_fn = get_llm_func(provider="openai") # Requires OPENAI_API_KEY

        pipeline = BasicRAGPipeline(iris_connector=db_conn, embedding_func=embed_fn, llm_func=llm_fn)

        # --- Pre-requisite: Ensure DB is initialized and has data ---
        # This demo assumes 'common/db_init.sql' has been run and 'SourceDocuments'
        # table exists and contains some data with embeddings.
        # For a self-contained demo, you might add a small data seeding step here.
        # Example:
        # from common.db_init import initialize_database
        # initialize_database(db_conn) # Make sure schema exists
        #
        # # Seed a sample document (if table is empty)
        # cursor = db_conn.cursor()
        # cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        # if cursor.fetchone()[0] == 0:
        #     print("Seeding a sample document for demo...")
        #     sample_doc_id = "sample_doc_1"
        #     sample_content = "InterSystems IRIS is a complete data platform. It supports SQL."
        #     sample_embedding_list = embed_fn([sample_content])[0]
        #     sample_embedding_str = str(sample_embedding_list)
        #     cursor.execute("INSERT INTO SourceDocuments (doc_id, text_content, embedding) VALUES (?, ?, TO_VECTOR(?))",
        #                    (sample_doc_id, sample_content, sample_embedding_str))
        #     # db_conn.commit() # If auto-commit is not on for the SA connection's raw DBAPI part
        # cursor.close()
        # print("Demo setup: Database checked/seeded.")
        # --- End of pre-requisite ---


        # Example Query
        test_query = "What is InterSystems IRIS?"
        print(f"\nExecuting RAG pipeline for query: '{test_query}'")
        
        result = pipeline.run(test_query, top_k=3)

        print("\n--- RAG Pipeline Result ---")
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
        for i, doc in enumerate(result['retrieved_documents']):
            print(f"  Doc {i+1}: ID={doc.id}, Score={doc.score:.4f}, Content='{doc.content[:100]}...'")
        
        if 'latency_ms' in result:
            print(f"Total Pipeline Latency: {result['latency_ms']:.2f} ms")

    except ValueError as ve:
        print(f"Setup Error: {ve}")
        print("Please ensure IRIS_CONNECTION_URL (and OPENAI_API_KEY if using OpenAI) are set.")
    except ImportError as ie:
        print(f"Import Error: {ie}")
        print("Please ensure all required libraries (sentence-transformers, langchain-openai, sqlalchemy, intersystems-irispython) are installed via Poetry.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'db_conn' in locals() and db_conn is not None:
            db_conn.close()
            print("Database connection closed.")

    print("\nBasicRAGPipeline Demo Finished.")
