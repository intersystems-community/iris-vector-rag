"""
Basic RAG Pipeline - Final Working Version
Uses direct SQL construction without parameters to avoid IRIS parsing issues
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable
import logging

try:
    # Attempt to import the specific IRISConnection from the intersystems_iris package
    from intersystems_iris.dbapi import IRISConnection
except ImportError:
    # Fallback if intersystems_iris or its dbapi module isn't found
    IRISConnection = Any

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection

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
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Use the original SourceDocuments table with cosine similarity calculation
            # We'll calculate cosine similarity manually in SQL
            sql = f"""
                SELECT doc_id, title, text_content, embedding
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
            """
            
            cursor.execute(sql)
            all_docs = cursor.fetchall()
            
            # Calculate similarities in Python
            import json
            doc_scores = []
            
            for row in all_docs:
                doc_id = row[0]
                title = row[1]
                content = row[2]
                embedding_str = row[3]
                
                try:
                    # Parse the stored embedding
                    if embedding_str.startswith('['):
                        doc_embedding = json.loads(embedding_str)
                    else:
                        doc_embedding = [float(x.strip()) for x in embedding_str.split(',')]
                    
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
                            'title': title,
                            'content': content,
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
                    content=doc_data['content'] or "",
                    metadata={"title": doc_data['title'] or ""},
                    score=doc_data['score']
                )
                retrieved_docs.append(doc)
            
            logger.info(f"BasicRAG: Retrieved {len(retrieved_docs)} documents")
            
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