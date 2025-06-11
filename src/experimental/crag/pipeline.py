# crag/pipeline.py

import os
import sys
# Add the project root directory to Python path
# Assuming this file is in src/experimental/crag/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import logging # Added
from typing import List, Dict, Any, Callable, Tuple, Literal, Optional
# import sqlalchemy # No longer needed
# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any # Fallback to Any if the driver isn't available during static analysis

# Adjust imports for new structure (e.g. common/)
from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection # For demo
from common.chunk_retrieval import ChunkRetrievalService # Added for chunk support
logger = logging.getLogger(__name__) # Added

# Define evaluation status
RetrievalStatus = Literal["confident", "ambiguous", "disoriented"]

class RetrievalEvaluator:
    """
    Evaluates the quality of retrieved documents.
    Placeholder implementation - can be replaced with LLM-based or heuristic logic.
    """
    def __init__(self, llm_func: Callable = None, embedding_func: Callable = None):
        self.llm_func = llm_func # May use an LLM for evaluation
        self.embedding_func = embedding_func # May use embeddings for relevance
        print("RetrievalEvaluator Initialized (placeholder)")

    def evaluate(self, query_text: str, documents: List[Document]) -> RetrievalStatus:
        """
        Evaluates retrieved documents and returns a status.
        """
        print(f"RetrievalEvaluator: Evaluating documents for query: '{query_text[:50]}...'")
        # Placeholder logic:
        if not documents:
            print("RetrievalEvaluator: No documents provided. Status: disoriented.")
            return "disoriented"
        
        # Simple heuristic: if average score is high, assume confident
        # This requires documents to have a score from retrieval.
        # BasicRAG/HyDE provide scores, but other methods might not.
        # Let's assume score is available for this simple mock.
        total_score = sum(doc.score for doc in documents if doc.score is not None and isinstance(doc.score, (int, float)))
        avg_score = total_score / len(documents) if documents else 0

        if avg_score > 0.8: # Arbitrary threshold
            print(f"RetrievalEvaluator: Avg score {avg_score:.2f}. Status: confident.")
            return "confident"
        elif avg_score > 0.5:
            print(f"RetrievalEvaluator: Avg score {avg_score:.2f}. Status: ambiguous.")
            return "ambiguous"
        else:
            print(f"RetrievalEvaluator: Avg score {avg_score:.2f}. Status: disoriented.")
            return "disoriented"

class CRAGPipeline:
    def __init__(self, iris_connector: IRISConnection, # Updated type hint
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 web_search_func: Callable[[str, int], List[str]] = None, # Optional web search
                 use_chunks: bool = True, # Enable chunk-based retrieval
                 chunk_types: Optional[List[str]] = None): # Chunk types to use
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.retrieval_evaluator = RetrievalEvaluator(llm_func=llm_func, embedding_func=embedding_func)
        self.web_search_func = web_search_func
        
        # Chunk support
        self.use_chunks = use_chunks
        self.chunk_types = chunk_types or ['TEXT']  # Changed from 'adaptive' to match actual data
        self.chunk_service = ChunkRetrievalService(iris_connector) if use_chunks else None
        
        logger.info(f"CRAGPipeline Initialized (use_chunks={use_chunks}, chunk_types={self.chunk_types})")

    @timing_decorator
    def _initial_retrieve(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Performs initial retrieval using embedding-based search with HNSW acceleration.
        Now supports both chunk-based and document-based retrieval.
        """
        logger.info(f"CRAG: Performing initial retrieval for query: '{query_text[:50]}...' with threshold {similarity_threshold}")

        if self.use_chunks and self.chunk_service:
            return self._retrieve_chunks(query_text, top_k, similarity_threshold)
        else:
            return self._retrieve_documents(query_text, top_k, similarity_threshold)
    
    def _retrieve_chunks(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieve relevant chunks for CRAG processing
        """
        logger.info(f"CRAG: Retrieving chunks for query: '{query_text[:50]}...'")
        
        try:
            # Check chunk availability first
            chunk_stats = self.chunk_service.check_chunk_availability()
            if not chunk_stats['available']:
                logger.warning("CRAG: No chunks available, falling back to document retrieval")
                return self._retrieve_documents(query_text, top_k, similarity_threshold)
            
            logger.info(f"CRAG: Found {chunk_stats['total_chunks']} chunks, {chunk_stats['chunks_with_embeddings']} with embeddings")
            
            # Generate query embedding
            query_embedding = self.embedding_func([query_text])[0]
            
            # Retrieve chunks using the chunk service
            retrieved_chunks = self.chunk_service.retrieve_chunks_for_query(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more chunks since they're smaller
                chunk_types=self.chunk_types,
                similarity_threshold=similarity_threshold
            )
            
            logger.info(f"CRAG: Retrieved {len(retrieved_chunks)} chunks above threshold {similarity_threshold}")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"CRAG: Error retrieving chunks: {e}", exc_info=True)
            logger.warning("CRAG: Falling back to document retrieval")
            return self._retrieve_documents(query_text, top_k, similarity_threshold)
    
    def _retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Original document-based retrieval (fallback method)
        """
        logger.info(f"CRAG: Retrieving documents for query: '{query_text[:50]}...'")

        query_embedding = self.embedding_func([query_text])[0]
        # Convert to comma-separated string format for IRIS
        # Ensure it's bracketed for TO_VECTOR
        iris_vector_str = f"[{','.join(map(str, query_embedding))}]"


        retrieved_docs: List[Document] = []
        # The SQL query was problematic. It should use TO_VECTOR(?) for the parameter.
        # Also, the WHERE clause had VECTOR_COSINE twice, which can be inefficient and error-prone with some drivers.
        # Fetching more and filtering in Python is safer.
        sql_query = f"""
            SELECT TOP {int(top_k * 2)} doc_id, text_content,
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL 
            ORDER BY score DESC
        """
        # Note: The original query had `AND LENGTH(embedding) > 1000`. This is unusual for an embedding string
        # and might be incorrect. Removing it for now, assuming `embedding` is a valid vector string.
        # Also, the original query had `VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?`
        # which is fine, but filtering in Python after fetching a slightly larger set can be more robust.
        
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            logger.debug(f"CRAG Document Retrieve with threshold {similarity_threshold}. Query vec (first 50): {iris_vector_str[:50]}")
            cursor.execute(sql_query, (iris_vector_str,)) # Pass vector string as parameter
            results = cursor.fetchall()

            filtered_results = []
            for row_data in results:
                score_val = float(row_data[2]) if row_data[2] is not None else 0.0
                if score_val >= similarity_threshold:
                    filtered_results.append(row_data)
                if len(filtered_results) >= top_k:
                    break
            
            for row_final in filtered_results:
                retrieved_docs.append(Document(id=str(row_final[0]), content=str(row_final[1] or ""), score=float(row_final[2] or 0.0)))
            
            logger.info(f"CRAG: Initial retrieval found {len(retrieved_docs)} documents after threshold {similarity_threshold}")
        except Exception as e:
            logger.error(f"CRAG: Error during initial document retrieval: {e}", exc_info=True)
        finally:
            if cursor:
                cursor.close()
        return retrieved_docs

    @timing_decorator
    def _augment_with_web_search(self, query_text: str, initial_docs: List[Document], web_top_k: int = 3) -> List[Document]:
        """
        Performs web search if web_search_func is available and augments initial documents.
        """
        logger.info(f"CRAG: Attempting web augmentation for query: '{query_text[:50]}...'")
        if not self.web_search_func:
            logger.info("CRAG: Web search function not provided. Skipping augmentation.")
            return initial_docs

        try:
            web_results_texts = self.web_search_func(query_text, num_results=web_top_k)
            logger.info(f"CRAG: Web search returned {len(web_results_texts)} results.")
            
            web_docs = [Document(id=f"web_{i}", content=text, score=0.95) for i, text in enumerate(web_results_texts)] # Assign high score
            
            # Simple augmentation: combine initial docs and web docs.
            # A real CRAG might re-rank or filter.
            combined_docs = initial_docs + web_docs
            logger.info(f"CRAG: Combined initial documents with web results. Total: {len(combined_docs)}.")
            return combined_docs

        except Exception as e:
            logger.error(f"CRAG: Error during web search augmentation: {e}", exc_info=True)
            return initial_docs # Return original docs on error

    @timing_decorator
    def _decompose_recompose_filter(self, query_text: str, documents: List[Document]) -> List[str]:
        """
        Decomposes documents, filters relevant parts, and recomposes context.
        Placeholder implementation - can be replaced with more sophisticated logic.
        """
        logger.info(f"CRAG: Decomposing, filtering, and recomposing context for query: '{query_text[:50]}...'")
        
        relevant_chunks: List[str] = []
        
        # Placeholder: Simple approach - treat each document as a chunk and filter by keyword presence
        # A real implementation would split into sentences/paragraphs and use embeddings/LLM for relevance.
        query_keywords = set(query_text.lower().split())
        for doc in documents:
            if doc.content and doc.content.strip() and doc.content != "None": # Ensure content is valid
                doc_content_lower = str(doc.content).lower()
                # Simple keyword check: if any query keyword is in the doc content
                if any(keyword in doc_content_lower for keyword in query_keywords):
                    relevant_chunks.append(str(doc.content))
            else:
                logger.warning(f"CRAG: Document {doc.id} has None or invalid content, skipping for decompose.")

        if not relevant_chunks:
            logger.info("CRAG: No relevant chunks found after filtering.")

        return relevant_chunks


    @timing_decorator
    def retrieve_and_correct(self, query_text: str, top_k: int = 5, web_top_k: int = 3, initial_threshold: float = 0.1, quality_threshold: float = 0.75) -> List[str]:
        """
        Performs initial retrieval, evaluates, potentially augments, and refines context.
        """
        logger.info(f"CRAG: Running retrieve and correct step for query: '{query_text[:50]}...'")
        
        initial_docs = self._initial_retrieve(query_text, top_k, initial_threshold)
        
        retrieval_status = self.retrieval_evaluator.evaluate(query_text, initial_docs)

        current_docs = initial_docs
        if retrieval_status == "ambiguous" or retrieval_status == "disoriented":
            logger.info(f"CRAG: Retrieval status is '{retrieval_status}'. Attempting web augmentation.")
            current_docs = self._augment_with_web_search(query_text, initial_docs, web_top_k)
        else:
            logger.info(f"CRAG: Retrieval status is '{retrieval_status}'. Skipping web augmentation.")
        
        # Always run decompose-recompose on the (potentially augmented) documents
        refined_context_list = self._decompose_recompose_filter(query_text, current_docs)
        
        logger.info(f"CRAG: Retrieve and correct step finished. Refined context has {len(refined_context_list)} chunks.")
        return refined_context_list

    @timing_decorator
    def generate_answer(self, query_text: str, refined_context_list: List[str]) -> str:
        """
        Generates a final answer using the LLM based on the refined context.
        Same as BasicRAG/HyDE, but uses a list of strings for context.
        """
        logger.info(f"CRAG: Generating final answer for query: '{query_text[:50]}...'")
        if not refined_context_list:
            logger.warning("CRAG: No refined context provided. Returning a default response.")
            return "I could not find enough information to answer your question based on the available context."

        context_str = "\n\n".join(refined_context_list)

        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{context_str}

Question: {query_text}

Answer:"""

        answer = self.llm_func(prompt)
        logger.info(f"CRAG: Generated final answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, web_top_k: int = 3, initial_threshold: float = 0.1, quality_threshold: float = 0.75) -> Dict[str, Any]:
        """
        Runs the full CRAG pipeline.
        """
        logger.info(f"CRAG: Running pipeline for query: '{query_text[:50]}...'")
        refined_context_list = self.retrieve_and_correct(query_text, top_k, web_top_k, initial_threshold, quality_threshold)
        answer = self.generate_answer(query_text, refined_context_list)

        # Convert context chunks to document format for compatibility
        retrieved_documents_for_output = []
        for i, chunk_content in enumerate(refined_context_list):
            # Ensure content is a string, provide a default if not
            content_str = str(chunk_content) if chunk_content is not None else ""
            retrieved_documents_for_output.append(
                Document(
                    id=f"crag_chunk_{i}",
                    content=content_str,
                    score=1.0  # CRAG's refined list doesn't have individual scores here
                )
            )

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents_for_output, # Now a list of Document objects
            "retrieved_context_chunks": refined_context_list, # Keep original list for inspection
            "initial_threshold": initial_threshold,
            "quality_threshold": quality_threshold,
            "document_count": len(retrieved_documents_for_output)
        }
def run_crag(query_text: str, top_k: int = 5, web_top_k: int = 3, initial_threshold: float = 0.1, quality_threshold: float = 0.75, use_chunks: bool = True) -> Dict[str, Any]:
    """
    Helper function to instantiate and run the CRAGPipeline.
    """
    db_conn = None
    try:
        # get_iris_connection is imported from common.iris_connector_jdbc in this file
        db_conn = get_iris_connection() 
        embed_fn = get_embedding_func()
        # Using stub LLM for this helper, actual LLM can be configured if CRAGPipeline is used directly
        llm_fn = get_llm_func(provider="stub") 
        
        # Mock web search for the helper, can be overridden if CRAGPipeline is used directly
        mock_web_search = lambda query, num_results: [f"Mock web result {i+1} for '{query}'" for i in range(num_results)]

        pipeline = CRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn,
            web_search_func=mock_web_search, # Provide a default web search for the helper
            use_chunks=use_chunks
        )
        return pipeline.run(
            query_text, 
            top_k=top_k, 
            web_top_k=web_top_k, 
            initial_threshold=initial_threshold, 
            quality_threshold=quality_threshold
        )
    except Exception as e:
        logger.error(f"Error in run_crag helper: {e}", exc_info=True)
        return {
            "query": query_text,
            "answer": "Error occurred in CRAG pipeline.",
            "retrieved_documents": [],
            "retrieved_context_chunks": [],
            "error": str(e)
        }
    finally:
        if db_conn:
            db_conn.close()

if __name__ == '__main__':
    # Setup basic logging for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Running CRAGPipeline Demo...")
    
    # Adjust imports for __main__ execution if common is not in PYTHONPATH
    # This is a bit redundant with the top-level sys.path modification but ensures
    # __main__ block works if common is structured as src/common
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming this file is in src/experimental/crag
    # and common is in src/common
    path_to_src = os.path.abspath(os.path.join(current_dir, '../../..')) # Go up to project root
    if path_to_src not in sys.path:
        sys.path.insert(0, path_to_src)

    from common.iris_connector_jdbc import get_iris_connection as get_jdbc_conn_main # Alias to avoid conflict
    from common.utils import get_embedding_func as get_embed_fn_main, get_llm_func as get_llm_fn_main
    # from tests.mocks.db import MockIRISConnector # For demo seeding - assuming this path is valid or not used if real DB

    db_conn_main = None
    try:
        db_conn_main = get_jdbc_conn_main()
        embed_fn_main = get_embed_fn_main()
        llm_fn_main = get_llm_fn_main(provider="stub")
        
        mock_web_search_main = lambda query, num_results: [f"Web result {i+1} for '{query}'" for i in range(num_results)]

        if db_conn_main is None:
            raise ConnectionError("Failed to get IRIS connection for CRAG demo.")

        pipeline_main = CRAGPipeline(
            iris_connector=db_conn_main,
            embedding_func=embed_fn_main,
            llm_func=llm_fn_main,
            web_search_func=mock_web_search_main
        )
        
        # Example Query 1: Should trigger confident status (assuming mock data or real data allows)
        test_query_confident_main = "What are treatments for diabetes?"
        logger.info(f"\nExecuting CRAG pipeline for query: '{test_query_confident_main}' (expecting confident)")
        result_confident_main = pipeline_main.run(test_query_confident_main, top_k=3)
        print("\n--- CRAG Pipeline Result (Confident) ---")
        print(f"Query: {result_confident_main['query']}")
        print(f"Answer: {result_confident_main['answer']}")
        print(f"Retrieved Context Chunks ({len(result_confident_main['retrieved_context_chunks'])}):")
        for i, chunk_item in enumerate(result_confident_main['retrieved_context_chunks']):
            print(f"  Chunk {i+1}: '{str(chunk_item)[:100]}...'") # Ensure chunk_item is string

        # Example Query 2: Should trigger ambiguous status (if mock scores allow or real data is sparse)
        test_query_ambiguous_main = "Tell me about a rare disease not commonly known."
        logger.info(f"\nExecuting CRAG pipeline for query: '{test_query_ambiguous_main}' (expecting ambiguous/web search)")
        result_ambiguous_main = pipeline_main.run(test_query_ambiguous_main, top_k=3)
        print("\n--- CRAG Pipeline Result (Ambiguous) ---")
        print(f"Query: {result_ambiguous_main['query']}")
        print(f"Answer: {result_ambiguous_main['answer']}")
        print(f"Retrieved Context Chunks ({len(result_ambiguous_main['retrieved_context_chunks'])}):")
        for i, chunk_item_amb in enumerate(result_ambiguous_main['retrieved_context_chunks']):
            print(f"  Chunk {i+1}: '{str(chunk_item_amb)[:100]}...'")


    except ConnectionError as ce_main:
        logger.error(f"Demo Setup Error: {ce_main}", exc_info=True)
    except ValueError as ve_main:
        logger.error(f"Demo Setup Error: {ve_main}", exc_info=True)
    except ImportError as ie_main:
        logger.error(f"Demo Import Error: {ie_main}", exc_info=True)
    except Exception as e_main:
        logger.error(f"An unexpected error occurred during CRAG demo: {e_main}", exc_info=True)
    finally:
        if 'db_conn_main' in locals() and db_conn_main is not None:
            try:
                db_conn_main.close()
                logger.info("Database connection closed for CRAG demo.")
            except Exception as e_close_main:
                logger.error(f"Error closing DB connection for CRAG demo: {e_close_main}", exc_info=True)

    logger.info("\nCRAGPipeline Demo Finished.")