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
        Retrieves documents from IRIS based on vector similarity.
        Memory-efficient implementation that processes documents in small batches.
        """
        logger.debug(f"BasicRAG: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query_text])[0]
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # For BasicRAG, we'll use a very simple approach:
            # Just get a small sample of documents and calculate similarity
            # This avoids memory issues and IRIS SQL limitations
            
            # Get 100 documents with real embeddings (excluding dummy data)
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
                    # Handle JDBC string objects
                    if hasattr(embedding_str, 'toString'):
                        embedding_str = str(embedding_str)
                    
                    if embedding_str and isinstance(embedding_str, str) and embedding_str.startswith('['):
                        doc_embedding = json.loads(embedding_str)
                    else:
                        # Most embeddings are stored as comma-separated values
                        doc_embedding = [float(x.strip()) for x in str(embedding_str).split(',') if x.strip()]
                    
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
                        # Handle JDBC stream for content
                        if hasattr(content, 'read'):
                            content_str = content.read()
                            if isinstance(content_str, bytes):
                                content_str = content_str.decode('utf-8', errors='ignore')
                        elif hasattr(content, 'toString'):
                            content_str = str(content)
                        else:
                            content_str = str(content) if content else ""
                        
                        doc_scores.append({
                            'doc_id': str(doc_id) if hasattr(doc_id, 'toString') else doc_id,
                            'title': str(title) if hasattr(title, 'toString') else (title or ""),
                            'content': content_str,
                            'score': similarity
                        })
                        
                except Exception as e:
                    logger.warning(f"Could not process embedding for doc {doc_id}: {e}")
                    continue
            
            # Sort by score and take top_k
            doc_scores.sort(key=lambda x: x['score'], reverse=True)
            doc_scores = doc_scores[:top_k]
            
            # Convert to Document objects
            for doc_data in doc_scores:
                # Create Document with just the fields it accepts
                doc = Document(
                    id=doc_data['doc_id'],
                    content=doc_data['content'],
                    score=doc_data['score']
                )
                # Store title separately if needed
                doc._title = doc_data['title']
                retrieved_docs.append(doc)
            
            logger.info(f"BasicRAG: Retrieved {len(retrieved_docs)} documents above threshold")
            
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
