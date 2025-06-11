# hyde/pipeline.py

import os
import sys
# Add the project root directory to Python path
# Assuming this file is in src/experimental/hyde/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Dict, Any, Callable
import logging

try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any

# Adjust imports for new structure (e.g. common/)
from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection # For demo
from common.jdbc_stream_utils import read_iris_stream # Added for stream handling

logger = logging.getLogger(__name__)

class HyDEPipeline:
    def __init__(self, iris_connector: IRISConnection, 
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"): 
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema 
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
        if hypothetical_doc_embedding is None or len(hypothetical_doc_embedding) == 0 or not all(isinstance(x, (float, int)) for x in hypothetical_doc_embedding):
            logger.error(f"HyDE: Failed to generate a valid embedding for hypothetical_doc: '{hypothetical_doc_text[:50]}...'")
            return []
        
        embedding_str = f"[{','.join(map(str, hypothetical_doc_embedding))}]" # Format for TO_VECTOR
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # 2. Construct and execute SQL query for vector search
            # Fetch more initially and filter in Python for robustness
            sql_query = f"""
                SELECT TOP {int(top_k * 2)} doc_id, title, text_content,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity_score DESC
            """
            # Assumes 'embedding' column is string-like and needs TO_VECTOR.
            
            logger.debug(f"HyDE: Executing SQL query. Embedding (first 50 chars): {embedding_str[:50]}")
            cursor.execute(sql_query, (embedding_str,))
            results = cursor.fetchall()
            
            logger.info(f"HyDE: Fetched {len(results)} candidate documents from DB.")

            # 3. Process results, handle potential streams, and filter by similarity_threshold
            for row in results:
                doc_id = row[0]
                title = row[1] 
                raw_text_content = row[2]
                score = row[3]

                # Use the robust read_iris_stream utility
                text_content_str = read_iris_stream(raw_text_content)
                if not text_content_str: # If stream was empty or unreadable
                    logger.warning(f"HyDE: Content for doc_id {doc_id} is empty or unreadable after stream processing.")
                    # Decide on a placeholder or skip if content is critical
                    # For now, let it be an empty string if read_iris_stream returns that.
                
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
                    doc._title = str(title) if title is not None else "" 
                    retrieved_docs.append(doc)
            
            # Ensure final list is sorted and respects top_k after thresholding
            retrieved_docs.sort(key=lambda d: d.score, reverse=True)
            retrieved_docs = retrieved_docs[:top_k]
            logger.info(f"HyDE: Retrieved {len(retrieved_docs)} documents after applying threshold {similarity_threshold} and top_k.")
            
        except Exception as e:
            logger.error(f"HyDE: Error retrieving documents: {e}", exc_info=True)
            return [] 
        finally:
            if cursor:
                cursor.close()
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        """
        Generates a final answer using the LLM based on the original query and retrieved actual documents.
        """
        logger.info(f"HyDE: Generating final answer for query: '{query_text[:50]}...'")
        if not retrieved_docs:
            logger.warning("HyDE: No documents retrieved. Returning a default response.")
            return "I could not find enough information to answer your question."

        context_parts = []
        for doc in retrieved_docs: # Using all retrieved_docs (already top_k)
            title = getattr(doc, '_title', 'Untitled')
            content_preview = (doc.content[:1000] + "...") if doc.content and len(doc.content) > 1000 else (doc.content or "")
            context_parts.append(f"Title: {title}\nContent: {content_preview}")
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{context}

Question: {query_text}

Answer:"""
        
        answer = self.llm_func(prompt)
        logger.info(f"HyDE: Generated final answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Runs the full HyDE pipeline.
        """
        logger.info(f"HyDE: Running pipeline for query: '{query_text[:50]}...'")
        
        hypothetical_doc_text = self._generate_hypothetical_document(query_text)
        retrieved_documents = self.retrieve_documents(hypothetical_doc_text, top_k, similarity_threshold)
        answer = self.generate_answer(query_text, retrieved_documents)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents, # Return list of Document objects
            "hypothetical_document": hypothetical_doc_text,
            "metadata": {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "num_retrieved": len(retrieved_documents),
                "pipeline": "HyDE"
            }
        }
def run_hyde_rag(query_text: str, top_k: int = 5, similarity_threshold: float = 0.0, schema: str = "RAG") -> Dict[str, Any]:
    """
    Helper function to instantiate and run the HyDEPipeline.
    """
    db_conn = None
    try:
        # Note: get_iris_connection is imported from common.iris_connector_jdbc in this file
        db_conn = get_iris_connection() 
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub") # Use stub for this helper context
        
        pipeline = HyDEPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn,
            schema=schema
        )
        return pipeline.run(query_text, top_k=top_k, similarity_threshold=similarity_threshold)
    except Exception as e:
        logger.error(f"Error in run_hyde_rag helper: {e}", exc_info=True)
        return {
            "query": query_text,
            "answer": "Error occurred in HyDE pipeline.",
            "retrieved_documents": [],
            "hypothetical_document": "",
            "error": str(e),
            "metadata": {}
        }
    finally:
        if db_conn:
            db_conn.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Running HyDEPipeline Demo...")
    
    # Adjust imports for __main__ execution
    current_dir_main = os.path.dirname(os.path.abspath(__file__))
    path_to_src_main = os.path.abspath(os.path.join(current_dir_main, '../../..'))
    if path_to_src_main not in sys.path:
        sys.path.insert(0, path_to_src_main)

    from common.iris_connector_jdbc import get_iris_connection as get_jdbc_conn_main_hyde
    from common.utils import get_embedding_func as get_embed_fn_main_hyde, get_llm_func as get_llm_fn_main_hyde

    db_conn_main_hyde = None
    try:
        db_conn_main_hyde = get_jdbc_conn_main_hyde()
        embed_fn_main_hyde = get_embed_fn_main_hyde()
        llm_fn_main_hyde = get_llm_fn_main_hyde(provider="stub")

        if db_conn_main_hyde is None:
            raise ConnectionError("Failed to get IRIS connection for HyDE demo.")

        pipeline_main_hyde = HyDEPipeline(
            iris_connector=db_conn_main_hyde,
            embedding_func=embed_fn_main_hyde,
            llm_func=llm_fn_main_hyde
        )

        test_query_main_hyde = "What are the symptoms of long COVID?"
        logger.info(f"\nExecuting HyDE pipeline for query: '{test_query_main_hyde}'")
        
        result_main_hyde = pipeline_main_hyde.run(test_query_main_hyde, top_k=3)

        print("\n--- HyDE Pipeline Result ---")
        print(f"Query: {result_main_hyde['query']}")
        print(f"Answer: {result_main_hyde['answer']}")
        print(f"Hypothetical Document: {result_main_hyde['hypothetical_document'][:200]}...")
        
        retrieved_docs_list_main_hyde = result_main_hyde['retrieved_documents']
        print(f"Retrieved Documents ({len(retrieved_docs_list_main_hyde)}):")
        for i_main, doc_dict_main in enumerate(retrieved_docs_list_main_hyde):
            print(f"  Doc {i_main+1}: ID={doc_dict_main['id']}, Score={doc_dict_main['score']:.4f}, Title='{doc_dict_main['title'][:50]}', Content='{doc_dict_main['content'][:50]}...'")
        
        print(f"Metadata: {result_main_hyde['metadata']}")

    except ConnectionError as ce_main_hyde:
        logger.error(f"HyDE Demo Setup Error: {ce_main_hyde}", exc_info=True)
    except ValueError as ve_main_hyde:
        logger.error(f"HyDE Demo Setup Error: {ve_main_hyde}", exc_info=True)
    except ImportError as ie_main_hyde:
        logger.error(f"HyDE Demo Import Error: {ie_main_hyde}", exc_info=True)
    except Exception as e_main_hyde:
        logger.error(f"An unexpected error occurred during HyDE demo: {e_main_hyde}", exc_info=True)
    finally:
        if 'db_conn_main_hyde' in locals() and db_conn_main_hyde is not None:
            try:
                db_conn_main_hyde.close()
                logger.info("Database connection closed for HyDE demo.")
            except Exception as e_close_main_hyde:
                logger.error(f"Error closing DB connection for HyDE demo: {e_close_main_hyde}", exc_info=True)
    
    logger.info("\nHyDEPipeline Demo Finished.")