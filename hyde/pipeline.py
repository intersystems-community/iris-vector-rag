# hyde/pipeline.py

import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable
import logging

try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func

logger = logging.getLogger(__name__)

class HyDEPipeline:
    def __init__(self, iris_connector: IRISConnection, # Updated type hint
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"): # Added schema
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema # Store schema
        logger.info(f"HyDEPipeline initialized with schema: {schema}")

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
        logger.debug(f"HyDE: Generated hypothetical document: '{hypothetical_doc_text[:100]}...'")
        return hypothetical_doc_text

    @timing_decorator
    def retrieve_documents(self, hypothetical_doc_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Embeds the hypothetical document and retrieves similar actual documents
        using SQL-based vector search with VECTOR_COSINE.
        """
        logger.debug(f"HyDE: Retrieving documents for hypothetical_doc: '{hypothetical_doc_text[:50]}...' with top_k={top_k}, threshold={similarity_threshold}")
        
        # 1. Generate embedding for the hypothetical document
        hypothetical_doc_embedding = self.embedding_func([hypothetical_doc_text])[0]
        if not hypothetical_doc_embedding or not all(isinstance(x, (float, int)) for x in hypothetical_doc_embedding):
            logger.error(f"HyDE: Failed to generate a valid embedding for hypothetical_doc: '{hypothetical_doc_text[:50]}...'")
            return []
        
        embedding_str = ','.join(map(str, hypothetical_doc_embedding))
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # 2. Construct and execute SQL query for vector search
            # We fetch TOP top_k results ordered by similarity, then filter by threshold in Python.
            sql_query = f"""
                SELECT TOP {top_k} doc_id, title, text_content,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                  AND embedding NOT LIKE '0.1,0.1,0.1%' -- Project-specific filter for invalid embeddings
                ORDER BY similarity_score DESC
            """
            
            logger.debug(f"HyDE: Executing SQL query. Embedding (first 50 chars): {embedding_str[:50]}")
            cursor.execute(sql_query, (embedding_str,))
            results = cursor.fetchall()
            
            logger.info(f"HyDE: Fetched {len(results)} candidate documents from DB.")

            # 3. Process results, handle potential streams, and filter by similarity_threshold
            for row in results:
                doc_id = row[0]
                title = row[1] # Assuming title is now fetched as in BasicRAG
                raw_text_content = row[2]
                score = row[3]

                text_content_str = ""
                if raw_text_content is not None:
                    if hasattr(raw_text_content, 'read') and callable(raw_text_content.read):
                        try:
                            data = raw_text_content.read()
                            if isinstance(data, bytes):
                                text_content_str = data.decode('utf-8', errors='replace')
                            elif isinstance(data, str):
                                text_content_str = data
                            else:
                                text_content_str = str(data)
                                logger.warning(f"HyDE: Unexpected data type from stream read for doc_id {doc_id}: {type(data)}")
                        except Exception as e_read:
                            logger.warning(f"HyDE: Error reading stream for doc_id {doc_id}: {e_read}")
                            text_content_str = "[Content Read Error]"
                    elif isinstance(raw_text_content, bytes):
                        text_content_str = raw_text_content.decode('utf-8', errors='replace')
                    else:
                        text_content_str = str(raw_text_content)
                
                current_score = 0.0
                if score is not None:
                    try:
                        current_score = float(score)
                    except (ValueError, TypeError):
                        logger.warning(f"HyDE: Could not convert score '{score}' to float for doc_id {doc_id}. Using 0.0.")
                        current_score = 0.0
                
                if current_score >= similarity_threshold:
                    doc = Document(
                        id=str(doc_id),
                        content=text_content_str,
                        score=current_score
                    )
                    doc._title = str(title) if title is not None else "" # Store title
                    retrieved_docs.append(doc)
            
            logger.info(f"HyDE: Retrieved {len(retrieved_docs)} documents after applying threshold {similarity_threshold}.")
            
        except Exception as e:
            logger.error(f"HyDE: Error retrieving documents: {e}", exc_info=True)
            # raise # Optionally re-raise, or handle as per HyDE's original error strategy (which was to return [])
            return [] # Matching original HyDE error handling for this method
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
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Runs the full HyDE pipeline.
        """
        print(f"HyDE: Running pipeline for query: '{query_text[:50]}...'")
        
        # Generate the hypothetical document first
        hypothetical_doc_text = self._generate_hypothetical_document(query_text)
        
        # Then retrieve documents using the hypothetical document's text
        retrieved_documents = self.retrieve_documents(hypothetical_doc_text, top_k, similarity_threshold)
        
        # Generate the final answer
        answer = self.generate_answer(query_text, retrieved_documents)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [ # Match BasicRAG output structure for documents
                {
                    "id": doc.id,
                    "content": doc.content[:500],  # Truncate for response
                    "score": doc.score,
                    "title": getattr(doc, '_title', 'Untitled')
                }
                for doc in retrieved_documents
            ],
            "hypothetical_document": hypothetical_doc_text, # Keep the text
            "metadata": { # Match BasicRAG output structure for metadata
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "num_retrieved": len(retrieved_documents),
                "pipeline": "HyDE"
            }
        }

if __name__ == '__main__':
    print("Running HyDEPipeline Demo...")
    from common.iris_connector_jdbc import get_iris_connection # For demo

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
        print(f"Hypothetical Document: {result['hypothetical_document'][:200]}...") # Show hypothetical doc
        
        retrieved_docs_list = result['retrieved_documents'] # Access the list of dicts
        print(f"Retrieved Documents ({len(retrieved_docs_list)}):")
        for i, doc_dict in enumerate(retrieved_docs_list):
            print(f"  Doc {i+1}: ID={doc_dict['id']}, Score={doc_dict['score']:.4f}, Title='{doc_dict['title'][:50]}', Content='{doc_dict['content'][:50]}...'")
        
        print(f"Metadata: {result['metadata']}")
        # Latency is usually added by the timing_decorator itself if it's part of the result dict directly from run.
        # If 'run' itself is decorated, its timing would be logged, not typically returned in the dict unless explicitly added.

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
