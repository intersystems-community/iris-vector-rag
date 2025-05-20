# colbert/pipeline.py

import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable, Tuple
# import sqlalchemy # No longer needed
import numpy as np # For vector operations
import json # Import json for parsing CLOB string
import logging # Import logging

# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any # Fallback to Any if the driver isn't available during static analysis

# Configure logging
logger = logging.getLogger(__name__)

from common.utils import Document, timing_decorator, get_llm_func, get_embedding_func # get_embedding_func for query encoder mock
from common.iris_connector import get_iris_connection # For demo

class ColbertRAGPipeline:
    def __init__(self, iris_connector: IRISConnection, # Updated type hint
                 colbert_query_encoder_func: Callable[[str], List[List[float]]],
                 colbert_doc_encoder_func: Callable[[str], List[List[float]]], # Needed for offline indexing, but useful to have here
                 llm_func: Callable[[str], str]):
        self.iris_connector = iris_connector
        self.colbert_query_encoder = colbert_query_encoder_func
        self.colbert_doc_encoder = colbert_doc_encoder_func # Keep for completeness, though used in loader
        self.llm_func = llm_func
        logger.info("ColbertRAGPipeline Initialized")

    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculates cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        dot_product = np.dot(vec1_np, vec2_np)
        norm_a = np.linalg.norm(vec1_np)
        norm_b = np.linalg.norm(vec2_np)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _calculate_maxsim(self, query_embeddings: List[List[float]], doc_token_embeddings: List[List[float]]) -> float:
        """
        Calculates the MaxSim score between query token embeddings and document token embeddings.
        """
        if not query_embeddings or not doc_token_embeddings:
            return 0.0

        max_sim_scores = []
        for q_embed in query_embeddings:
            # Find max similarity between this query token and all document tokens
            max_sim = 0.0
            for d_embed in doc_token_embeddings:
                sim = self._calculate_cosine_similarity(q_embed, d_embed)
                max_sim = max(max_sim, sim)
            max_sim_scores.append(max_sim)

        # Sum the max similarities for each query token
        return sum(max_sim_scores)

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Document]:
        """
        Retrieves the top_k most relevant documents using client-side ColBERT MaxSim.
        """
        logger.info(f"ColbertRAG: Retrieving documents for query: '{query_text[:50]}...'")

        query_token_embeddings = self.colbert_query_encoder(query_text)

        if not query_token_embeddings:
            logger.warning("ColbertRAG: Failed to generate query token embeddings. Returning empty list.")
            return []

        # --- Client-side MaxSim Retrieval ---
        # This is inefficient for large datasets as it fetches all token embeddings.
        # A production implementation would use DB-side MaxSim (UDF/UDAF).

        all_doc_token_data: Dict[str, List[List[float]]] = {} # doc_id -> list of token embeddings

        # Fetch all token embeddings from IRIS
        # Assuming DocumentTokenEmbeddings table exists and embedding is CLOB string list
        sql_fetch_tokens = """
            SELECT doc_id, token_sequence_index, token_embedding
            FROM RAG.DocumentTokenEmbeddings
            ORDER BY doc_id, token_sequence_index
        """

        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            cursor.execute(sql_fetch_tokens)
            results = cursor.fetchall()

            # Group token embeddings by doc_id and parse the CLOB string back to list of floats
            for row in results:
                doc_id = row[0]
                # token_idx = row[1] # Not needed for MaxSim calculation
                token_embedding_str = row[2]

                if doc_id not in all_doc_token_data:
                    all_doc_token_data[doc_id] = []

                try:
                    # Parse the string representation of the list back to a list of floats
                    # Example string: "[0.1, 0.2, 0.3]"
                    token_embedding_list = json.loads(token_embedding_str)
                    if isinstance(token_embedding_list, list) and all(isinstance(x, (int, float)) for x in token_embedding_list):
                         all_doc_token_data[doc_id].append(token_embedding_list)
                    else:
                         logger.warning(f"ColbertRAG: Warning: Could not parse token embedding for doc {doc_id}: {token_embedding_str[:50]}...")

                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"ColbertRAG: Error parsing token embedding JSON for doc {doc_id}: {e}")
                    # Skip this embedding if parsing fails

            logger.info(f"ColbertRAG: Fetched token embeddings for {len(all_doc_token_data)} documents.")

        except Exception as e:
            logger.error(f"ColbertRAG: Error fetching token embeddings from IRIS: {e}")
            return [] # Return empty list on error
        finally:
            if cursor:
                cursor.close()

        # Calculate MaxSim score for each document
        doc_scores: List[Tuple[str, float]] = [] # List of (doc_id, score)
        for doc_id, doc_embeddings in all_doc_token_data.items():
            score = self._calculate_maxsim(query_token_embeddings, doc_embeddings)
            doc_scores.append((doc_id, score))

        # Sort documents by score in descending order
        doc_scores.sort(key=lambda item: item[1], reverse=True)

        # Select top-k document IDs and their scores
        top_k_docs_with_scores = doc_scores[:top_k]
        top_k_doc_ids = [doc_id for doc_id, score in top_k_docs_with_scores]

        # Fetch the full content for the top-k documents
        retrieved_docs: List[Document] = []
        if top_k_doc_ids:
            # Construct SQL query to get content for specific doc_ids
            # Using IN clause with placeholders
            placeholders = ', '.join(['?'] * len(top_k_doc_ids))
            sql_fetch_content = f"""
                SELECT doc_id, text_content
                FROM RAG.SourceDocuments
                WHERE doc_id IN ({placeholders})
            """
            # Need to maintain order if possible, but IN clause doesn't guarantee order.
            # Can sort in Python after fetching.

            cursor = None
            try:
                cursor = self.iris_connector.cursor()
                cursor.execute(sql_fetch_content, top_k_doc_ids)
                results = cursor.fetchall()

                # Create a dictionary for quick lookup of fetched content by doc_id
                fetched_content_map = {row[0]: row[1] for row in results}
                
                # Reconstruct retrieved_docs list using the order from top_k_docs_with_scores and fetched content
                retrieved_docs = []
                for doc_id, score in top_k_docs_with_scores:
                    if doc_id in fetched_content_map:
                        retrieved_docs.append(Document(id=doc_id, content=fetched_content_map[doc_id], score=score))
                    # else: Document not found (shouldn't happen if doc_id came from DB)

                logger.info(f"ColbertRAG: Fetched content for {len(retrieved_docs)} top-k documents.")

            except Exception as e:
                logger.error(f"ColbertRAG: Error fetching content for top-k documents: {e}")
                # retrieved_docs remains empty
            finally:
                if cursor:
                    cursor.close()

        logger.info(f"ColbertRAG: Finished retrieval. Found {len(retrieved_docs)} documents.")
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        """
        Generates a final answer using the LLM based on the original query and retrieved actual documents.
        Same as BasicRAG/HyDE.
        """
        logger.info(f"ColbertRAG: Generating final answer for query: '{query_text[:50]}...'")
        if not retrieved_docs:
            logger.warning("ColbertRAG: No documents retrieved. Returning a default response.")
            return "I could not find enough information to answer your question."

        context = "\n\n".join([doc.content for doc in retrieved_docs])

        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{context}

Question: {query_text}

Answer:"""

        answer = self.llm_func(prompt)
        print(f"ColbertRAG: Generated final answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs the full ColBERT pipeline (client-side MaxSim).
        """
        logger.info(f"ColbertRAG: Running pipeline for query: '{query_text[:50]}...'")
        retrieved_documents = self.retrieve_documents(query_text, top_k)
        answer = self.generate_answer(query_text, retrieved_documents)

        # Ensure retrieved_documents are returned in a format compatible with benchmark metrics
        # The retrieve_documents method already returns a List[Document], which has id, content, and score.
        # The benchmark runner expects a list of dicts with 'id', 'content', 'score'.
        # The Document.to_dict() method handles this conversion.
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents], # Convert Document objects to dicts
        }

if __name__ == '__main__':
    print("Running ColbertRAGPipeline Demo (Client-side MaxSim)...")
    from common.iris_connector import get_iris_connection # For demo
    from common.utils import get_embedding_func, get_llm_func # For demo
    from tests.mocks.db import MockIRISConnector # For demo seeding
    import json # Import json for parsing CLOB string
    import logging # Import logging for demo output

    # Configure logging for demo
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ColbertRAGPipeline_Demo")


    try:
        db_conn = get_iris_connection() # Uses IRIS_CONNECTION_URL or falls back to mock
        # Use the same embedding function for both query and doc encoding mock for simplicity in demo
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")

        if db_conn is None:
            raise ConnectionError("Failed to get IRIS connection for ColBERT demo.")

        # For demo, we need a mock doc encoder that returns token embeddings
        # In reality, this would be a proper ColBERT doc encoder used during loading.
        # The query encoder is used at query time.
        # Let's use the embedding_func as a stand-in for both encoders for the demo.
        # A real ColBERT encoder is different from a sentence transformer.
        
        # Mock ColBERT encoders for demo purposes
        # Returns a list of 10-dim vectors (simulating token embeddings)
        mock_colbert_encoder = lambda text: [[float(i)/10.0]*10 for i in range(min(len(text.split()), 5))] # Max 5 tokens, 10 dim
        
        pipeline = ColbertRAGPipeline(
            iris_connector=db_conn,
            colbert_query_encoder_func=mock_colbert_encoder, # Use mock for demo
            colbert_doc_encoder_func=mock_colbert_encoder, # Use mock for demo
            llm_func=llm_fn
        )

        # --- Pre-requisite: Ensure DB has data with token embeddings ---
        # This demo assumes 'common/db_init.sql' has been run and 'DocumentTokenEmbeddings'
        # table exists and contains some data with token embeddings (as CLOB strings).
        # For a self-contained demo, you might add a small data seeding step here
        # for DocumentTokenEmbeddings.
        # Example seeding (requires MockIRISConnector or real DB setup):
        if isinstance(db_conn, MockIRISConnector):
             mock_cursor = db_conn.cursor()
             mock_cursor.stored_token_embeddings = {
                 "doc_colbert_1": [{"idx": 0, "text": "token1", "embedding": str([0.1]*10), "metadata": "{}"},
                                   {"idx": 1, "text": "token2", "embedding": str([0.2]*10), "metadata": "{}"}],
                 "doc_colbert_2": [{"idx": 0, "text": "tokenA", "embedding": str([0.9]*10), "metadata": "{}"}]
             }
             mock_cursor.stored_docs = { # Also need SourceDocuments entries for content fetching
                 "doc_colbert_1": {"text_content": "Content for doc colbert 1"},
                 "doc_colbert_2": {"text_content": "Content for doc colbert 2"}
             }
             logger.info("Demo setup: Mock token embeddings seeded.")
        else:
             logger.info("Demo setup: Assuming real DB has token embeddings.")
        # --- End of pre-requisite ---


        # Example Query
        test_query = "What is the ColBERT model?"
        logger.info(f"\nExecuting ColBERT pipeline for query: '{test_query}'")

        result = pipeline.run(test_query, top_k=2)

        logger.info("\n--- ColBERT Pipeline Result ---")
        logger.info(f"Query: {result['query']}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
        for i, doc_dict in enumerate(result['retrieved_documents']): # Iterate over dicts now
            logger.info(f"  Doc {i+1}: ID={doc_dict.get('id')}, Score={doc_dict.get('score'):.4f}, Content='{doc_dict.get('content', '')[:100]}...'")

        if 'latency_ms' in result:
             logger.info(f"Total Pipeline Latency (run method): {result['latency_ms']:.2f} ms")

    except ConnectionError as ce:
        logger.error(f"Demo Setup Error: {ce}")
    except ValueError as ve:
        logger.error(f"Demo Setup Error: {ve}")
    except ImportError as ie:
        logger.error(f"Demo Import Error: {ie}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during ColBERT demo: {e}")
    finally:
        if 'db_conn' in locals() and db_conn is not None:
            try:
                db_conn.close()
                logger.info("Database connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing DB connection: {e_close}")

    logger.info("\nColbertRAGPipeline Demo Finished.")
