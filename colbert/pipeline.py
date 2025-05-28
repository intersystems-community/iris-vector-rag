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
        ColBERT's late interaction: for each query token, find max similarity with any doc token, then sum.
        """
        if not query_embeddings or not doc_token_embeddings:
            return 0.0

        max_sim_scores = []
        for q_embed in query_embeddings:
            # Find max similarity between this query token and all document tokens
            max_sim = -1.0  # Start with -1 since cosine similarity can be negative
            for d_embed in doc_token_embeddings:
                sim = self._calculate_cosine_similarity(q_embed, d_embed)
                max_sim = max(max_sim, sim)
            max_sim_scores.append(max_sim)

        # Sum the max similarities for each query token (ColBERT's late interaction)
        total_score = sum(max_sim_scores)
        
        # Normalize by query length to make scores comparable across different query lengths
        normalized_score = total_score / len(query_embeddings) if query_embeddings else 0.0
        
        return normalized_score

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Retrieves documents using ColBERT's MaxSim scoring with HNSW acceleration.
        """
        logger.info(f"ColbertRAG: Retrieving documents for query: '{query_text[:50]}...' with threshold {similarity_threshold}")
        query_token_embeddings = self.colbert_query_encoder(query_text)

        if not query_token_embeddings:
            logger.warning("ColbertRAG: Query encoder returned no embeddings.")
            return []

        candidate_docs_with_scores = []
        
        try:
            cursor = self.iris_connector.cursor()

            # Step 1: Fetch all unique doc_ids from HNSW SourceDocuments as candidates
            cursor.execute("SELECT DISTINCT doc_id FROM RAG.SourceDocuments")
            all_doc_ids = [row[0] for row in cursor.fetchall()]
            
            logger.info(f"ColbertRAG: Scoring {len(all_doc_ids)} candidate documents.")
            
            docs_with_token_embeddings = 0
            docs_with_fallback_embeddings = 0
            docs_above_threshold = 0

            for doc_id in all_doc_ids:
                # Step 2: For each doc_id, fetch its token embeddings from HNSW schema
                sql_fetch_tokens = """
                SELECT token_embedding
                FROM RAG_HNSW.DocumentTokenEmbeddings
                WHERE doc_id = ?
                ORDER BY token_sequence_index
                """
                cursor.execute(sql_fetch_tokens, (doc_id,))
                
                doc_token_embeddings_str = cursor.fetchall()
                
                # If no token embeddings, fall back to generating them on-the-fly
                if not doc_token_embeddings_str:
                    # Fallback 1: Try to get document text and generate token embeddings
                    sql_fetch_doc_text = """
                    SELECT text_content
                    FROM RAG.SourceDocuments
                    WHERE doc_id = ?
                    """
                    cursor.execute(sql_fetch_doc_text, (doc_id,))
                    doc_text_result = cursor.fetchone()
                    
                    if doc_text_result and doc_text_result[0]:
                        # Generate token embeddings on-the-fly using the ColBERT doc encoder
                        try:
                            doc_text = doc_text_result[0][:1000]  # Limit text length for performance
                            current_doc_token_embeddings = self.colbert_doc_encoder(doc_text)
                            logger.debug(f"ColbertRAG: Generated {len(current_doc_token_embeddings)} token embeddings on-the-fly for {doc_id}")
                        except Exception as e:
                            logger.warning(f"ColbertRAG: Error generating token embeddings for {doc_id}: {e}")
                            # Fallback 2: Use document-level embedding as a single "token"
                            sql_fetch_doc_embedding = """
                            SELECT embedding
                            FROM RAG.SourceDocuments
                            WHERE doc_id = ?
                            """
                            cursor.execute(sql_fetch_doc_embedding, (doc_id,))
                            doc_embedding_result = cursor.fetchone()
                            
                            if not doc_embedding_result or not doc_embedding_result[0]:
                                continue
                            
                            # Convert document embedding to token embedding format
                            try:
                                if isinstance(doc_embedding_result[0], str):
                                    doc_embedding = [float(x) for x in doc_embedding_result[0].split(',')]
                                else:
                                    doc_embedding = doc_embedding_result[0]
                                # Treat the entire document embedding as a single token
                                current_doc_token_embeddings = [doc_embedding]
                            except (ValueError, TypeError) as e:
                                logger.warning(f"ColbertRAG: Error parsing document embedding for {doc_id}: {e}")
                                continue
                    else:
                        continue
                else:
                    # Convert string embeddings to list of lists of floats
                    current_doc_token_embeddings = []
                    for token_row in doc_token_embeddings_str:
                        try:
                            # If token_embedding is already a list of floats from the driver
                            if isinstance(token_row[0], list):
                                 current_doc_token_embeddings.append(token_row[0])
                            # If it's a JSON string representation of a list
                            elif isinstance(token_row[0], str) and token_row[0].startswith('['):
                                current_doc_token_embeddings.append(json.loads(token_row[0]))
                            # If it's a comma-separated string
                            elif isinstance(token_row[0], str):
                                 current_doc_token_embeddings.append([float(x) for x in token_row[0].split(',')])
                            else:
                                logger.warning(f"ColbertRAG: Unexpected token embedding format for doc {doc_id}")
                                pass
                        except (json.JSONDecodeError, ValueError, TypeError) as e_parse:
                            logger.error(f"ColbertRAG: Error parsing token embedding for doc {doc_id}: {e_parse}")
                            continue

                if not current_doc_token_embeddings:
                    continue
                
                # Track what type of embeddings we're using
                if len(doc_token_embeddings_str) > 0:
                    docs_with_token_embeddings += 1
                else:
                    docs_with_fallback_embeddings += 1
                
                # Step 3: Calculate MaxSim score
                maxsim_score = self._calculate_maxsim(query_token_embeddings, current_doc_token_embeddings)
                
                logger.debug(f"ColbertRAG: Doc {doc_id} MaxSim score: {maxsim_score:.4f} (threshold: {similarity_threshold})")
                
                # Only include documents above threshold
                if maxsim_score > similarity_threshold:
                    docs_above_threshold += 1
                    # Fetch actual document content for context/display
                    cursor.execute("SELECT text_content FROM RAG.SourceDocuments WHERE doc_id = ?", (doc_id,))
                    content_row = cursor.fetchone()
                    doc_content = content_row[0] if content_row else "Content not found"
                    
                    candidate_docs_with_scores.append(Document(id=doc_id, content=doc_content, score=maxsim_score))

            cursor.close()
            
            # Step 4: Sort documents by MaxSim score
            candidate_docs_with_scores.sort(key=lambda doc: doc.score, reverse=True)
            retrieved_docs = candidate_docs_with_scores
            
            logger.info(f"ColbertRAG: Processed {len(all_doc_ids)} documents:")
            logger.info(f"  - {docs_with_token_embeddings} with pre-computed token embeddings")
            logger.info(f"  - {docs_with_fallback_embeddings} with on-the-fly token embeddings")
            logger.info(f"  - {docs_above_threshold} above threshold {similarity_threshold}")
            logger.info(f"ColbertRAG: Found {len(retrieved_docs)} documents above threshold {similarity_threshold}")

        except Exception as e:
            logger.error(f"ColbertRAG: Error during document retrieval: {e}", exc_info=True)
            return []

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
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Runs the full ColBERT pipeline (client-side MaxSim).
        """
        logger.info(f"ColbertRAG: Running pipeline for query: '{query_text[:50]}...'")
        retrieved_documents = self.retrieve_documents(query_text, top_k, similarity_threshold)
        answer = self.generate_answer(query_text, retrieved_documents)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents],
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents)
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
