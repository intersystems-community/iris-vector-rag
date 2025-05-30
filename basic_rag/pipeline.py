"""
Basic RAG Pipeline - Working Version
Simple implementation that works with IRIS SQL limitations and the Document class
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable
import logging
import json

try:
    import iris
    IRISConnection = iris.IRISConnection
except ImportError:
    IRISConnection = Any

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection

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
        Retrieves documents from IRIS based on SQL vector similarity.
        """
        logger.debug(f"BasicRAG: Retrieving documents for query: '{query_text[:50]}...' using SQL vector search.")
        
        # Generate query embedding
        query_embedding_list = self.embedding_func([query_text])[0]
        query_embedding_str = ','.join(map(str, query_embedding_list)) # For SQL query
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Check if we're using JDBC (which has issues with parameter binding for vectors)
            conn_type = type(self.iris_connector).__name__
            is_jdbc = 'JDBC' in conn_type or hasattr(self.iris_connector, '_jdbc_connection')
            
            params = []
            if is_jdbc:
                # Use direct SQL for JDBC to avoid parameter binding issues with TO_VECTOR
                # The TO_VECTOR function expects a literal string for the vector data in this case.
                sql = f"""
                    SELECT TOP {top_k}
                        doc_id,
                        title,
                        text_content,
                        VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{query_embedding_str}')) as similarity_score
                    FROM {self.schema}.SourceDocuments
                    WHERE embedding IS NOT NULL
                      AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR('{query_embedding_str}')) > {similarity_threshold}
                    ORDER BY similarity_score DESC
                """
            else:
                # Use parameter binding for ODBC
                sql = f"""
                    SELECT TOP ?
                        doc_id,
                        title,
                        text_content,
                        VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                    FROM {self.schema}.SourceDocuments
                    WHERE embedding IS NOT NULL
                      AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
                    ORDER BY similarity_score DESC
                """
                params = [top_k, query_embedding_str, query_embedding_str, similarity_threshold]

            logger.debug(f"Executing SQL vector search. JDBC: {is_jdbc}. SQL: {sql[:250]}... Params: {params if not is_jdbc else 'N/A for JDBC direct SQL'}")
            if is_jdbc:
                cursor.execute(sql)
            else:
                cursor.execute(sql, params)
            
            results = cursor.fetchall()
            logger.info(f"BasicRAG: SQL vector search retrieved {len(results)} documents.")
            
            for row in results:
                doc_id_raw = row[0]
                title_raw = row[1]
                content_raw = row[2]
                score_raw = row[3]

                # Handle potential JDBC stream objects for content
                content_str = content_raw
                if hasattr(content_raw, 'read'):
                    content_str = content_raw.read()
                    if isinstance(content_str, bytes):
                        content_str = content_str.decode('utf-8', errors='ignore')
                elif hasattr(content_raw, 'toString'): # For other JDBC types like string
                    content_str = str(content_raw)
                else:
                    content_str = str(content_raw) if content_raw else ""

                doc_id_str = str(doc_id_raw) if hasattr(doc_id_raw, 'toString') else str(doc_id_raw)
                title_str = str(title_raw) if hasattr(title_raw, 'toString') else (str(title_raw) if title_raw else "")
                
                doc = Document(
                    id=doc_id_str,
                    content=content_str,
                    score=float(score_raw) if score_raw is not None else 0.0
                )
                doc._title = title_str # Store title separately
                retrieved_docs.append(doc)
            
            logger.info(f"BasicRAG: Processed {len(retrieved_docs)} documents from SQL results.")
            
        except Exception as e:
            logger.error(f"BasicRAG: Error during SQL document retrieval: {e}", exc_info=True)
            # Optionally, could add a fallback here if desired, but instructions are to restore SQL search
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
        max_context_chars = 4000  # Conservative limit
        
        for doc in retrieved_docs:
            # Handle JDBC stream objects
            content = doc.content
            if hasattr(content, 'read'):
                # It's a stream, read it
                content = content.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')
            elif hasattr(content, 'toString'):
                content = str(content)
            
            doc_content = str(content)[:1000] if content else ""  # Limit each document
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
        
        # Convert documents to dict format, including title if available
        doc_dicts = []
        for doc in retrieved_documents:
            doc_dict = doc.to_dict()
            if hasattr(doc, '_title'):
                doc_dict['metadata'] = {'title': doc._title}
            doc_dicts.append(doc_dict)
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": doc_dicts,
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents),
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
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
