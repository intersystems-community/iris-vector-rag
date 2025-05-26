# crag/pipeline.py

import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging # Added
from typing import List, Dict, Any, Callable, Tuple, Literal, Optional
# import sqlalchemy # No longer needed
# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any # Fallback to Any if the driver isn't available during static analysis

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection # For demo
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
        self.chunk_types = chunk_types or ['adaptive']
        self.chunk_service = ChunkRetrievalService(iris_connector) if use_chunks else None
        
        logger.info(f"CRAGPipeline Initialized (use_chunks={use_chunks}, chunk_types={self.chunk_types})")

    @timing_decorator
    def _initial_retrieve(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.5) -> List[Document]:
        """
        Performs initial retrieval using embedding-based search with HNSW acceleration.
        Now supports both chunk-based and document-based retrieval.
        """
        logger.info(f"CRAG: Performing initial retrieval for query: '{query_text[:50]}...' with threshold {similarity_threshold}")

        if self.use_chunks and self.chunk_service:
            return self._retrieve_chunks(query_text, top_k, similarity_threshold)
        else:
            return self._retrieve_documents(query_text, top_k, similarity_threshold)
    
    def _retrieve_chunks(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.5) -> List[Document]:
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
            logger.error(f"CRAG: Error retrieving chunks: {e}")
            logger.warning("CRAG: Falling back to document retrieval")
            return self._retrieve_documents(query_text, top_k, similarity_threshold)
    
    def _retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.5) -> List[Document]:
        """
        Original document-based retrieval (fallback method)
        """
        logger.info(f"CRAG: Retrieving documents for query: '{query_text[:50]}...'")

        query_embedding = self.embedding_func([query_text])[0]
        # Convert to comma-separated string format for IRIS
        iris_vector_str = ','.join(map(str, query_embedding))

        retrieved_docs: List[Document] = []
        sql_query = f"""
            SELECT TOP 20 doc_id, text_content,
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
              AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
            ORDER BY score DESC
        """
        
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            logger.debug(f"CRAG Document Retrieve with threshold {similarity_threshold}")
            cursor.execute(sql_query, (iris_vector_str, iris_vector_str, similarity_threshold))
            results = cursor.fetchall()

            for row in results:
                retrieved_docs.append(Document(id=row[0], content=row[1], score=row[2]))
            print(f"CRAG: Initial retrieval found {len(retrieved_docs)} documents above threshold {similarity_threshold}")
        except Exception as e:
            print(f"CRAG: Error during initial retrieval: {e}")
        finally:
            if cursor:
                cursor.close()
        return retrieved_docs

    @timing_decorator
    def _augment_with_web_search(self, query_text: str, initial_docs: List[Document], web_top_k: int = 3) -> List[Document]:
        """
        Performs web search if web_search_func is available and augments initial documents.
        """
        print(f"CRAG: Attempting web augmentation for query: '{query_text[:50]}...'")
        if not self.web_search_func:
            print("CRAG: Web search function not provided. Skipping augmentation.")
            return initial_docs

        try:
            web_results_texts = self.web_search_func(query_text, num_results=web_top_k)
            print(f"CRAG: Web search returned {len(web_results_texts)} results.")
            
            web_docs = [Document(id=f"web_{i}", content=text, score=0.95) for i, text in enumerate(web_results_texts)] # Assign high score
            
            # Simple augmentation: combine initial docs and web docs.
            # A real CRAG might re-rank or filter.
            combined_docs = initial_docs + web_docs
            print(f"CRAG: Combined initial documents with web results. Total: {len(combined_docs)}.")
            return combined_docs

        except Exception as e:
            print(f"CRAG: Error during web search augmentation: {e}")
            return initial_docs # Return original docs on error

    @timing_decorator
    def _decompose_recompose_filter(self, query_text: str, documents: List[Document]) -> List[str]:
        """
        Decomposes documents, filters relevant parts, and recomposes context.
        Placeholder implementation - can be replaced with more sophisticated logic.
        """
        print(f"CRAG: Decomposing, filtering, and recomposing context for query: '{query_text[:50]}...'")
        
        relevant_chunks: List[str] = []
        
        # Placeholder: Simple approach - treat each document as a chunk and filter by keyword presence
        # A real implementation would split into sentences/paragraphs and use embeddings/LLM for relevance.
        relevant_chunks: List[str] = []
        query_keywords = set(query_text.lower().split())
        for doc in documents:
            if doc.content: # Ensure content is not None
                doc_content_lower = str(doc.content).lower()
                # Simple keyword check: if any query keyword is in the doc content
                if any(keyword in doc_content_lower for keyword in query_keywords):
                    relevant_chunks.append(str(doc.content))
            else:
                logger.warning(f"CRAG: Document {doc.id} has None content, skipping for decompose.")

        if not relevant_chunks:
            print("CRAG: No relevant chunks found after filtering.")

        return relevant_chunks


    @timing_decorator
    def retrieve_and_correct(self, query_text: str, top_k: int = 5, web_top_k: int = 3, initial_threshold: float = 0.5, quality_threshold: float = 0.75) -> List[str]:
        """
        Performs initial retrieval, evaluates, potentially augments, and refines context.
        """
        print(f"CRAG: Running retrieve and correct step for query: '{query_text[:50]}...'")
        
        initial_docs = self._initial_retrieve(query_text, top_k, initial_threshold)
        
        retrieval_status = self.retrieval_evaluator.evaluate(query_text, initial_docs)
        # print("CRAG DEBUG: Bypassed RetrievalEvaluator.evaluate") # Removed bypass
        # retrieval_status = "confident" # Assume confident to skip web search for now # Removed bypass

        current_docs = initial_docs
        if retrieval_status == "ambiguous" or retrieval_status == "disoriented":
            print(f"CRAG: Retrieval status is '{retrieval_status}'. Attempting web augmentation.")
            current_docs = self._augment_with_web_search(query_text, initial_docs, web_top_k)
        else:
            print(f"CRAG: Retrieval status is '{retrieval_status}'. Skipping web augmentation.")
        
        # Always run decompose-recompose on the (potentially augmented) documents
        refined_context_list = self._decompose_recompose_filter(query_text, current_docs)
        
        # # For debugging, just return content from initial_docs # Removed debug block
        # refined_context_list = [str(doc.content) for doc in initial_docs if doc.content]
        # # --- End of temporary debug block ---

        print(f"CRAG: Retrieve and correct step finished. Refined context has {len(refined_context_list)} chunks.")
        return refined_context_list

    @timing_decorator
    def generate_answer(self, query_text: str, refined_context_list: List[str]) -> str:
        """
        Generates a final answer using the LLM based on the refined context.
        Same as BasicRAG/HyDE, but uses a list of strings for context.
        """
        print(f"CRAG: Generating final answer for query: '{query_text[:50]}...'")
        if not refined_context_list:
            print("CRAG: No refined context provided. Returning a default response.")
            return "I could not find enough information to answer your question based on the available context."

        context_str = "\n\n".join(refined_context_list)

        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{context_str}

Question: {query_text}

Answer:"""

        answer = self.llm_func(prompt)
        print(f"CRAG: Generated final answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, web_top_k: int = 3, initial_threshold: float = 0.5, quality_threshold: float = 0.75) -> Dict[str, Any]:
        """
        Runs the full CRAG pipeline.
        """
        print(f"CRAG: Running pipeline for query: '{query_text[:50]}...'")
        refined_context_list = self.retrieve_and_correct(query_text, top_k, web_top_k, initial_threshold, quality_threshold)
        answer = self.generate_answer(query_text, refined_context_list)

        # Convert context chunks to document format for compatibility
        retrieved_documents = []
        for i, chunk in enumerate(refined_context_list):
            retrieved_documents.append({
                "id": f"crag_chunk_{i}",
                "content": chunk,
                "score": 1.0  # CRAG doesn't provide individual scores after processing
            })

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents,
            "retrieved_context_chunks": refined_context_list,
            "initial_threshold": initial_threshold,
            "quality_threshold": quality_threshold,
            "document_count": len(retrieved_documents)
        }

if __name__ == '__main__':
    print("Running CRAGPipeline Demo...")
    from common.iris_connector import get_iris_connection # For demo
    from common.utils import get_embedding_func, get_llm_func # For demo
    from tests.mocks.db import MockIRISConnector # For demo seeding

    try:
        db_conn = get_iris_connection() # Uses IRIS_CONNECTION_URL or falls back to mock
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")
        
        # Mock web search function for demo
        mock_web_search = lambda query, num_results: [f"Web result {i+1} for '{query}'" for i in range(num_results)]

        if db_conn is None:
            raise ConnectionError("Failed to get IRIS connection for CRAG demo.")

        pipeline = CRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn,
            web_search_func=mock_web_search # Provide mock web search
        )

        # --- Pre-requisite: Ensure DB has data ---
        # This demo assumes 'common/db_init.sql' has been run and 'SourceDocuments'
        # table exists and contains some data with embeddings (as CLOB strings).
        # For a self-contained demo, you might add a small data seeding step here.
        # Example seeding (requires MockIRISConnector or real DB setup):
        if isinstance(db_conn, MockIRISConnector):
             mock_cursor = db_conn.cursor()
             # Seed sample documents with varying scores to test evaluator
             mock_cursor.stored_docs = {
                 "doc_crag_high_1": {"content": "Content highly relevant to diabetes.", "embedding": str([0.9]*384), "score": 0.95},
                 "doc_crag_high_2": {"content": "More relevant content about diabetes treatments.", "embedding": str([0.8]*384), "score": 0.90},
                 "doc_crag_med_1": {"content": "General medical information.", "embedding": str([0.5]*384), "score": 0.60},
                 "doc_crag_low_1": {"content": "Irrelevant content.", "embedding": str([0.1]*384), "score": 0.30},
             }
             # Need to mock the retrieval query result in MockIRISCursor.execute
             # to return these docs with scores for the initial retrieve step.
             # This requires coordinating the mock_iris_connector_for_crag fixture
             # with this seeding logic if using a mock.
             print("Demo setup: Mock documents seeded for CRAG.")
        else:
             print("Demo setup: Assuming real DB has data.")
        # --- End of pre-requisite ---


        # Example Query 1: Should trigger confident status
        test_query_confident = "What are treatments for diabetes?"
        print(f"\nExecuting CRAG pipeline for query: '{test_query_confident}' (expecting confident)")
        result_confident = pipeline.run(test_query_confident, top_k=3)
        print("\n--- CRAG Pipeline Result (Confident) ---")
        print(f"Query: {result_confident['query']}")
        print(f"Answer: {result_confident['answer']}")
        print(f"Retrieved Context Chunks ({len(result_confident['retrieved_context_chunks'])}):")
        for i, chunk in enumerate(result_confident['retrieved_context_chunks']):
            print(f"  Chunk {i+1}: '{chunk[:100]}...'")

        # Example Query 2: Should trigger ambiguous status (if mock scores allow)
        # This requires the mock retrieval to return docs with scores between 0.5 and 0.8
        test_query_ambiguous = "Tell me about a rare disease."
        print(f"\nExecuting CRAG pipeline for query: '{test_query_ambiguous}' (expecting ambiguous/web search)")
        result_ambiguous = pipeline.run(test_query_ambiguous, top_k=3)
        print("\n--- CRAG Pipeline Result (Ambiguous) ---")
        print(f"Query: {result_ambiguous['query']}")
        print(f"Answer: {result_ambiguous['answer']}")
        print(f"Retrieved Context Chunks ({len(result_ambiguous['retrieved_context_chunks'])}):")
        for i, chunk in enumerate(result_ambiguous['retrieved_context_chunks']):
            print(f"  Chunk {i+1}: '{chunk[:100]}...'")


    except ConnectionError as ce:
        print(f"Demo Setup Error: {ce}")
    except ValueError as ve:
        print(f"Demo Setup Error: {ve}")
    except ImportError as ie:
        print(f"Demo Import Error: {ie}")
    except Exception as e:
        print(f"An unexpected error occurred during CRAG demo: {e}")
    finally:
        if 'db_conn' in locals() and db_conn is not None:
            try:
                db_conn.close()
                print("Database connection closed.")
            except Exception as e_close:
                print(f"Error closing DB connection: {e_close}")

    print("\nCRAGPipeline Demo Finished.")
