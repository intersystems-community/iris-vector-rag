"""
Basic RAG Pipeline - Native VECTOR Implementation
Uses native IRIS VECTOR columns and VECTOR_COSINE functions with HNSW indexes
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
from common.iris_connector_jdbc import get_iris_connection # Assuming JDBC for consistency if native VECTOR ops are through it

logger = logging.getLogger(__name__)

class BasicRAGPipeline: # Name conflict with pipeline_final.py, but keeping for archival accuracy
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipeline (native_vector version) initialized with schema: {schema}")

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieves documents from IRIS using native VECTOR columns and VECTOR_COSINE function.
        This leverages HNSW indexes for optimal performance.
        """
        logger.debug(f"BasicRAG (native_vector): Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        query_vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Use native VECTOR_COSINE with HNSW index for optimal performance
            sql = f"""
                SELECT TOP {top_k}
                    doc_id,
                    title,
                    text_content,
                    VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                  AND VECTOR_COSINE(embedding, TO_VECTOR(?)) > ?
                ORDER BY similarity_score DESC
            """
            
            logger.debug(f"Executing native VECTOR_COSINE query with HNSW index")
            cursor.execute(sql, [query_vector_str, query_vector_str, similarity_threshold])
            
            results = cursor.fetchall()
            logger.info(f"BasicRAG (native_vector): Native VECTOR_COSINE retrieved {len(results)} documents")
            
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
                elif hasattr(content_raw, 'toString'): # For some JDBC objects
                    content_str = str(content_raw)
                else:
                    content_str = str(content_raw) if content_raw else ""

                # Handle doc_id and title
                doc_id_str = str(doc_id_raw) if hasattr(doc_id_raw, 'toString') else str(doc_id_raw)
                title_str = str(title_raw) if hasattr(title_raw, 'toString') else (str(title_raw) if title_raw else "")
                
                doc = Document(
                    id=doc_id_str,
                    content=content_str,
                    score=float(score_raw) if score_raw is not None else 0.0
                )
                doc._title = title_str  # Store title separately
                retrieved_docs.append(doc)
            
            logger.info(f"BasicRAG (native_vector): Successfully processed {len(retrieved_docs)} documents")
            
        except Exception as e:
            logger.error(f"BasicRAG (native_vector): Error during native VECTOR document retrieval: {e}", exc_info=True)
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
        logger.debug(f"BasicRAG (native_vector): Generating answer for query: '{query_text[:50]}...'")
        if not retrieved_docs:
            logger.warning("BasicRAG (native_vector): No documents retrieved. Returning a default response.")
            return "I could not find enough information to answer your question."

        # Limit context to prevent token overflow
        context_parts = []
        total_chars = 0
        max_context_chars = 4000  # Conservative limit
        
        for doc in retrieved_docs:
            # Handle JDBC stream objects
            content = doc.content
            if hasattr(content, 'read'): # Check if it's a stream-like object
                content = content.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')
            elif hasattr(content, 'toString'): # For some JDBC objects
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
        logger.debug(f"BasicRAG (native_vector): Generated answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Runs the full Basic RAG pipeline: retrieve documents and generate an answer.
        """
        logger.info(f"BasicRAG (native_vector): Running pipeline for query: '{query_text[:50]}...'")
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
    print("Running Native VECTOR BasicRAGPipeline Demo...")

    # Setup
    try:
        db_conn = get_iris_connection() # Ensure this returns a JDBC connection if that's what this pipeline expects
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")  # Use stub for testing

        pipeline = BasicRAGPipeline(iris_connector=db_conn, embedding_func=embed_fn, llm_func=llm_fn)

        # Example Query
        test_query = "What is diabetes?"
        print(f"\nExecuting RAG pipeline for query: '{test_query}'")
        
        result = pipeline.run(test_query, top_k=5)

        print("\n--- RAG Pipeline Result ---")
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
        for i, doc in enumerate(result['retrieved_documents']):
            print(f"  Doc {i+1}: ID={doc['id']}, Score={doc.get('score', 0):.4f}")
            if 'metadata' in doc and 'title' in doc['metadata']:
                print(f"         Title: {doc['metadata']['title'][:60]}...")
        
        if 'latency_ms' in result: # This key is added by timing_decorator if used on run method
            print(f"Total Pipeline Latency: {result.get('latency_ms', 'N/A'):.2f} ms")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'db_conn' in locals() and db_conn is not None:
            db_conn.close()
            print("\nDatabase connection closed.")

    print("\nNative VECTOR BasicRAGPipeline Demo Finished.")