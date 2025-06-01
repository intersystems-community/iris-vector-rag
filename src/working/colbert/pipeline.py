# colbert/pipeline_fixed.py - Optimized ColBERT Implementation

import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))) # Adjusted path for new location

from typing import List, Dict, Any, Callable, Tuple, Optional
import numpy as np
import json
import logging
import hashlib

# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any

# Configure logging
logger = logging.getLogger(__name__)

from common.utils import Document, timing_decorator, get_llm_func, get_embedding_func, get_config_value
from common.iris_connector_jdbc import get_iris_connection

class ColbertRAGPipeline: # Reverted name for consistency
    def __init__(self, iris_connector: IRISConnection,
                 colbert_query_encoder_func: Callable[[str], List[List[float]]],
                 llm_func: Callable[[str], str],
                 embedding_func: Optional[Callable[[str], List[float]]] = None, # For Stage 1
                 colbert_doc_encoder_func: Optional[Callable[[str], List[List[float]]]] = None # Keep for potential future use or specific ColBERT doc encoding
                 ):
        self.iris_connector = iris_connector
        self.colbert_query_encoder = colbert_query_encoder_func
        self.llm_func = llm_func
        
        if embedding_func:
            self.embedding_func = embedding_func
        else:
            logger.info("ColBERT: embedding_func not provided, using default from common.utils.") # Adjusted log prefix
            self.embedding_func = get_embedding_func() # Default from common.utils

        # self.colbert_doc_encoder is not strictly needed if we use pre-computed token embeddings
        # and the provided embedding_func for Stage 1.
        # Keeping it if it might be used for on-the-fly document tokenization for ColBERT in some scenarios.
        self.colbert_doc_encoder = colbert_doc_encoder_func
        
        logger.info("ColbertRAGPipeline Initialized") # Adjusted log prefix

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

        # Vectorized _calculate_maxsim
        if not query_embeddings or not doc_token_embeddings: # Should have been caught by the caller, but good check
            logger.warning("OptimizedColBERT: _calculate_maxsim called with empty query or document embeddings.")
            return 0.0

        try:
            # Ensure all doc token embeddings are valid lists/arrays of numbers
            valid_doc_tokens = [d_emb for d_emb in doc_token_embeddings if isinstance(d_emb, (list, np.ndarray)) and len(d_emb) > 0]
            if not valid_doc_tokens:
                logger.warning("OptimizedColBERT: No valid document token embeddings after filtering in _calculate_maxsim.")
                return 0.0
            
            doc_matrix = np.array(valid_doc_tokens, dtype=np.float32)
            if doc_matrix.ndim != 2:
                logger.error(f"OptimizedColBERT: doc_token_embeddings could not be formed into a 2D matrix. Shape: {doc_matrix.shape}")
                return 0.0 # Cannot proceed if doc matrix is not 2D
        except ValueError as e:
            logger.error(f"OptimizedColBERT: Error creating NumPy array from doc_token_embeddings: {e}. Check for consistent embedding dimensions.")
            return 0.0 # Cannot proceed

        doc_norms = np.linalg.norm(doc_matrix, axis=1)
        max_sim_scores = []

        for q_embed_list in query_embeddings:
            try:
                q_vec = np.array(q_embed_list, dtype=np.float32)
                if q_vec.ndim != 1:
                    logger.warning(f"OptimizedColBERT: A query embedding is not 1D. Shape: {q_vec.shape}. Using -1.0 for this token.")
                    max_sim_scores.append(-1.0)
                    continue
                if q_vec.shape[0] != doc_matrix.shape[1]:
                    logger.warning(f"OptimizedColBERT: Query embedding dim {q_vec.shape[0]} != doc embedding dim {doc_matrix.shape[1]}. Using -1.0 for this token.")
                    max_sim_scores.append(-1.0)
                    continue
            except ValueError as e:
                logger.warning(f"OptimizedColBERT: Error creating NumPy array from a query_embedding: {e}. Using -1.0 for this token.")
                max_sim_scores.append(-1.0)
                continue

            q_norm = np.linalg.norm(q_vec)

            if q_norm < 1e-9:  # Effectively zero norm for query token
                max_sim_scores.append(0.0) # Max similarity is 0 if query token is zero vector
                continue

            # Vectorized dot products: D (m, n) @ q (n,) -> (m,)
            dot_prods = np.dot(doc_matrix, q_vec)
            
            denominators = q_norm * doc_norms # doc_norms is (m,)
            similarities = np.zeros_like(denominators, dtype=np.float32)
            
            # Avoid division by zero for document tokens with zero norm
            valid_indices = denominators > 1e-9
            
            if np.any(valid_indices): # Check if there's at least one valid denominator
                similarities[valid_indices] = dot_prods[valid_indices] / denominators[valid_indices]
                max_sim_scores.append(np.max(similarities)) # Max over all doc tokens for this query token
            else: # All denominators were zero or close to zero
                max_sim_scores.append(0.0) # Or -1.0 if preferred, but 0.0 if no valid similarity

        # Sum the max similarities for each query token (ColBERT's late interaction)
        total_score = sum(max_sim_scores)
        
        # Normalize by query length to make scores comparable across different query lengths
        normalized_score = total_score / len(query_embeddings) if query_embeddings else 0.0
        
        return normalized_score

    def _limit_content_size(self, documents: List[Document], max_total_chars: int = 50000) -> List[Document]:
        """
        Limits the total content size to prevent LLM context overflow.
        Prioritizes higher-scoring documents and truncates content as needed.
        """
        if not documents:
            return documents
            
        # Sort by score (highest first)
        sorted_docs = sorted(documents, key=lambda d: d.score, reverse=True)
        
        limited_docs = []
        total_chars = 0
        
        for doc in sorted_docs:
            content = doc.content or ""
            
            # Calculate remaining space
            remaining_space = max_total_chars - total_chars
            
            if remaining_space <= 0:
                break
                
            # Truncate content if necessary
            if len(content) > remaining_space:
                content = content[:remaining_space] + "..."
                
            limited_doc = Document(
                id=doc.id,
                content=content,
                score=doc.score
            )
            limited_docs.append(limited_doc)
            total_chars += len(content)
            
        logger.info(f"ColBERT: Limited content from {len(documents)} docs to {len(limited_docs)} docs, {total_chars} chars")
        return limited_docs

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Document]:
        """
        Optimized document retrieval using ColBERT's MaxSim scoring.
        Uses efficient batching and proper top_k limiting.
        """
        logger.info(f"OptimizedColBERT: Retrieving top {top_k} documents for query: '{query_text[:50]}...'")
        
        colbert_candidate_pool_size = get_config_value("colbert.candidate_pool_size", 100)
        logger.info(f"OptimizedColBERT: Using candidate pool size: {colbert_candidate_pool_size}")
        # Generate query token embeddings
        query_token_embeddings = self.colbert_query_encoder(query_text)
        
        # Handle the case where query_token_embeddings might be a tuple (tokens, embeddings)
        if isinstance(query_token_embeddings, tuple):
            tokens, embeddings = query_token_embeddings
            if len(tokens) == 0:
                logger.warning("OptimizedColBERT: Query encoder returned no embeddings.")
                return []
        else:
            logger.warning("OptimizedColBERT: Query encoder returned unexpected format.")
            return []
# Stage 1: Retrieve candidate documents using standard vector search
        logger.info("OptimizedColBERT: Stage 1 - Retrieving candidate documents.")
        if not hasattr(self, 'embedding_func') or self.embedding_func is None:
            logger.error("OptimizedColBERT: embedding_func not available for Stage 1 retrieval.")
            return [] # Or handle error appropriately

        # self.embedding_func expects a list of texts and returns a list of embeddings.
        # For a single query, pass it as a list and take the first result.
        query_doc_embeddings_list = self.embedding_func([query_text])
        if not query_doc_embeddings_list or not query_doc_embeddings_list[0] or not isinstance(query_doc_embeddings_list[0], list):
            logger.error("OptimizedColBERT: Failed to generate valid query document embedding for Stage 1.")
            return []
        query_doc_embedding = query_doc_embeddings_list[0] # This should be List[float]

        if not query_doc_embedding: # Redundant check, but safe
            logger.error("OptimizedColBERT: Query document embedding is empty after extraction for Stage 1.")
            return []

        # Format numbers to ensure they are treated as doubles by IRIS TO_VECTOR
        query_embedding_str = "[" + ",".join([f"{x:.8f}" for x in query_doc_embedding]) + "]"
        
        candidate_doc_ids = []
        try:
            cursor = self.iris_connector.cursor()
            # Note: Ensure RAG.SourceDocuments.embedding has an HNSW index for performance
            sql_candidate_search = f"""
                SELECT TOP {colbert_candidate_pool_size} doc_id
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) DESC
            """
            logger.debug(f"OptimizedColBERT: Stage 1 SQL: {sql_candidate_search} with query_embedding_str (first 50): {query_embedding_str[:50]}...")
            cursor.execute(sql_candidate_search, (query_embedding_str,))
            candidate_rows = cursor.fetchall()
            candidate_doc_ids = [row[0] for row in candidate_rows]
            cursor.close()
            logger.info(f"OptimizedColBERT: Stage 1 retrieved {len(candidate_doc_ids)} candidate document IDs.")
            if not candidate_doc_ids:
                logger.warning("OptimizedColBERT: Stage 1 did not retrieve any candidate documents.")
                return []
        except Exception as e_stage1:
            logger.error(f"OptimizedColBERT: Error during Stage 1 candidate retrieval: {e_stage1}", exc_info=True)
            return []

        candidate_docs_with_scores = []
        
        try:
            cursor = self.iris_connector.cursor()

            # Optimized approach: Get a reasonable sample of documents to score
            # Instead of processing ALL documents, we'll use a more efficient strategy
            
            # Step 2a: Get text_content for the candidate_doc_ids that have token embeddings
            if not candidate_doc_ids: # Should have been caught earlier, but as a safeguard
                logger.warning("OptimizedColBERT: No candidate_doc_ids provided to Stage 2a.")
                return []

            placeholders = ','.join(['?' for _ in candidate_doc_ids])
            sql_get_docs_with_tokens = f"""
            SELECT DISTINCT sd.doc_id, sd.text_content
            FROM RAG.SourceDocuments sd
            INNER JOIN RAG.DocumentTokenEmbeddings dte ON sd.doc_id = dte.doc_id
            WHERE sd.doc_id IN ({placeholders})
            """
            logger.debug(f"OptimizedColBERT: Stage 2a SQL: {sql_get_docs_with_tokens} with {len(candidate_doc_ids)} candidates.")
            cursor.execute(sql_get_docs_with_tokens, candidate_doc_ids)
            docs_data = cursor.fetchall()
            
            logger.info(f"OptimizedColBERT: Found {len(docs_data)} documents with token embeddings")

            if not docs_data:
                logger.warning("OptimizedColBERT: No documents with token embeddings found in the database.")
                return []

            doc_ids = []
            doc_contents = {}
            for doc_row in docs_data:
                doc_id_raw = doc_row[0]
                doc_id = doc_id_raw.lower() if isinstance(doc_id_raw, str) else doc_id_raw
                raw_text_content = doc_row[1]
                doc_ids.append(doc_id)
                
                text_content_str = ""
                if hasattr(raw_text_content, 'read'):
                    try:
                        byte_list = []
                        while True:
                            byte_val = raw_text_content.read()
                            if byte_val == -1: break
                            byte_list.append(byte_val)
                        if byte_list:
                            text_content_str = bytes(byte_list).decode('utf-8', errors='replace')
                    except Exception as e_read:
                        logger.warning(f"OptimizedColBERT: Could not read content stream for doc_id {doc_id} during doc_contents population: {e_read}")
                        text_content_str = "" # Default to empty string if read fails
                elif isinstance(raw_text_content, str):
                    text_content_str = raw_text_content
                elif isinstance(raw_text_content, bytes):
                    try:
                        text_content_str = raw_text_content.decode('utf-8', errors='replace')
                    except Exception as e_decode:
                        logger.warning(f"OptimizedColBERT: Could not decode bytes content for doc_id {doc_id} during doc_contents population: {e_decode}")
                        text_content_str = ""
                elif raw_text_content is None:
                    text_content_str = ""
                
                doc_contents[doc_id] = text_content_str

            # Step 2b: Fetch all token embeddings for these candidate doc_ids in a single query
            # This significantly reduces database round trips compared to fetching for all docs.
            # Using a parameterized query for IN clause to prevent SQL injection and handle large lists.
            placeholders = ','.join(['?' for _ in doc_ids])
            sql_fetch_all_tokens = f"""
            SELECT doc_id, token_index, token_embedding
            FROM RAG.DocumentTokenEmbeddings
            WHERE doc_id IN ({placeholders})
            ORDER BY doc_id, token_index
            """
            
            # Ensure doc_ids is a tuple for the IN clause, some drivers are picky
            params_for_in_clause = tuple(doc_ids)
            cursor.execute(sql_fetch_all_tokens, params_for_in_clause)
            all_token_rows = cursor.fetchall()
            logger.info(f"OptimizedColBERT: Second query fetched {len(all_token_rows)} token embedding rows for {len(doc_ids)} doc_ids.")

            # Group token embeddings by doc_id
            grouped_doc_token_embeddings = {}
            processed_tokens_for_debug = 0
            first_doc_id_for_debug = None
            # Attempt to import java.util for type checking if jpype is in use
            try:
                import java.util # type: ignore
            except ImportError:
                java = None # type: ignore

            for token_row in all_token_rows:
                _doc_id_raw, token_idx, token_embedding_data = token_row
                doc_id_from_token_row = _doc_id_raw.lower() if isinstance(_doc_id_raw, str) else _doc_id_raw
                
                if first_doc_id_for_debug is None:
                    first_doc_id_for_debug = doc_id_from_token_row
                
                if doc_id_from_token_row == first_doc_id_for_debug and processed_tokens_for_debug < 5:
                    logger.debug(f"OptimizedColBERT DEBUG TOKEN: doc_id='{doc_id_from_token_row}', token_idx={token_idx}, type(token_embedding_data)={type(token_embedding_data)}, value (first 100 chars)='{str(token_embedding_data)[:100]}'")
                    processed_tokens_for_debug += 1

                if doc_id_from_token_row not in grouped_doc_token_embeddings:
                    grouped_doc_token_embeddings[doc_id_from_token_row] = []
                
                token_embedding = None
                try:
                    if isinstance(token_embedding_data, str):
                        if token_embedding_data.startswith('[') and token_embedding_data.endswith(']'):
                            token_embedding = json.loads(token_embedding_data)
                        else:
                            token_embedding = [float(x.strip()) for x in token_embedding_data.split(',')]
                    elif isinstance(token_embedding_data, (list, np.ndarray)):
                        token_embedding = list(token_embedding_data)
                    elif java and isinstance(token_embedding_data, java.util.List):
                         token_embedding = [float(x) for x in token_embedding_data]
                    elif hasattr(token_embedding_data, '_values') or (java and hasattr(token_embedding_data, 'getClass') and "array" in str(token_embedding_data.getClass()).lower()):
                        try:
                            if hasattr(token_embedding_data, 'tolist'):
                                token_embedding = token_embedding_data.tolist()
                            else:
                                logger.debug(f"OptimizedColBERT: Attempting string conversion for Java object type {type(token_embedding_data)} for doc {doc_id_from_token_row}, index {token_idx}.")
                                token_embedding_str_conv = str(token_embedding_data)
                                if token_embedding_str_conv.startswith('[') and token_embedding_str_conv.endswith(']'):
                                     token_embedding = json.loads(token_embedding_str_conv)
                                else:
                                     token_embedding = [float(x.strip()) for x in token_embedding_str_conv.split(',')]
                        except Exception as e_java_conv:
                            logger.error(f"OptimizedColBERT: FAILED to convert/parse Java-like object for token embedding for doc {doc_id_from_token_row}, index {token_idx}. Type: {type(token_embedding_data)}. Error: {e_java_conv}")
                            continue
                    else:
                        logger.warning(f"OptimizedColBERT: Unexpected type for token embedding: {type(token_embedding_data)} for doc {doc_id_from_token_row}, index {token_idx}. Value (first 100): {str(token_embedding_data)[:100]}")
                        continue
                    
                    if token_embedding:
                        grouped_doc_token_embeddings[doc_id_from_token_row].append(token_embedding)

                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(f"OptimIZEDColBERT: FAILED to parse token embedding for doc {doc_id_from_token_row}, index {token_idx}. Type: {type(token_embedding_data)}, Value: '{str(token_embedding_data)[:200]}'. Error: {e}")
                    continue
            
            logger.info(f"OptimizedColBERT: Grouped token embeddings populated. Number of doc_ids with grouped embeddings: {len(grouped_doc_token_embeddings)}. Keys (sample): {list(grouped_doc_token_embeddings.keys())[:3]}")
            # for d_id_key, embeds_list in grouped_doc_token_embeddings.items(): # Can be too verbose
            #    logger.debug(f"OptimizedColBERT: Grouped embeds for '{d_id_key}': {len(embeds_list)} embeddings.")

            processed_count = 0
            above_threshold_count = 0

            # Debugging: Compare doc_ids from both sources
            if doc_contents and grouped_doc_token_embeddings:
                dc_keys_sample = list(doc_contents.keys())[:5]
                gte_keys_sample = list(grouped_doc_token_embeddings.keys())[:5]
                logger.debug(f"OptimizedColBERT DEBUG: Sample doc_ids from doc_contents: {dc_keys_sample}")
                logger.debug(f"OptimizedColBERT DEBUG: Sample doc_ids from grouped_doc_token_embeddings: {gte_keys_sample}")
                
                # Check if the first few keys match
                for i in range(min(5, len(dc_keys_sample))):
                    dc_key = dc_keys_sample[i]
                    is_in_gte = dc_key in grouped_doc_token_embeddings
                    logger.debug(f"OptimizedColBERT DEBUG: doc_contents key '{dc_key}' (type: {type(dc_key)}) in grouped_doc_token_embeddings: {is_in_gte}")
                    if not is_in_gte and isinstance(dc_key, str):
                        # Try checking with stripped versions if string
                        stripped_dc_key = dc_key.strip()
                        is_in_gte_stripped = stripped_dc_key in grouped_doc_token_embeddings
                        if is_in_gte_stripped:
                             logger.warning(f"OptimizedColBERT DEBUG: Key '{dc_key}' NOT found, but STRIPPED key '{stripped_dc_key}' IS found in grouped_doc_token_embeddings!")
                        else:
                             # Check if any key in grouped_doc_token_embeddings is very similar
                             for gte_key in gte_keys_sample: # Check against sample for brevity
                                 if isinstance(gte_key, str) and gte_key.strip() == stripped_dc_key:
                                     logger.warning(f"OptimizedColBERT DEBUG: doc_contents key '{dc_key}' / '{stripped_dc_key}' matches STRIPPED gte_key '{gte_key}'!")
                                     break


            logger.info(f"OptimizedColBERT: Starting document scoring. Number of docs in doc_contents: {len(doc_contents)}. Doc IDs from doc_contents (sample): {list(doc_contents.keys())[:3]}")
            if not query_token_embeddings: # query_token_embeddings is actually (tokens, embeddings)
                logger.error("OptimizedColBERT: Query encoder returned empty tokens or embeddings before starting scoring loop. Cannot calculate MaxSim.")
                return []
            
            actual_query_embeddings = query_token_embeddings[1] # The second element is the list of embedding vectors
            if not actual_query_embeddings:
                logger.error("OptimizedColBERT: Actual query embeddings list is empty before starting scoring loop. Cannot calculate MaxSim.")
                return []


            # Step 3: Iterate through documents and calculate MaxSim scores
            for doc_id_iterate, doc_content_iterate in doc_contents.items():
                logger.debug(f"OptimizedColBERT: Processing doc_id_iterate '{doc_id_iterate}' (type: {type(doc_id_iterate)}) from doc_contents.")
                current_doc_token_embeddings = grouped_doc_token_embeddings.get(doc_id_iterate, [])
                logger.debug(f"OptimizedColBERT: For '{doc_id_iterate}', found {len(current_doc_token_embeddings)} token embeddings in grouped map.")
                
                if not current_doc_token_embeddings:
                    logger.warning(f"OptimizedColBERT: No token embeddings list found or list is empty for doc_id: '{doc_id_iterate}'. Skipping.")
                    # processed_count += 1 # This was incrementing even if skipped, leading to incorrect "Processed X docs" if all were skipped here.
                                         # Let's only increment processed_count if we actually attempt a MaxSim.
                    continue
                
                # Calculate MaxSim score
                # query_token_embeddings is a tuple (tokens, embeddings). _calculate_maxsim expects list of embeddings.
                maxsim_score = self._calculate_maxsim(actual_query_embeddings, current_doc_token_embeddings)
                logger.info(f"OptimizedColBERT: Doc ID '{doc_id_iterate}', Calculated MaxSim score: {maxsim_score:.4f}")
                
                processed_count += 1 # Increment only if MaxSim was calculated

                # Only include documents above threshold
                if maxsim_score > similarity_threshold:
                    limited_content = (doc_content_iterate or "")[:5000]  # Limit per document
                    candidate_docs_with_scores.append(
                        Document(id=doc_id_iterate, content=limited_content, score=maxsim_score)
                    )
                    above_threshold_count += 1
                
                processed_count += 1
                
                # Early termination if we have enough high-scoring candidates
                if len(candidate_docs_with_scores) >= top_k * 3:
                    break

            cursor.close()
            
            # Sort by score and limit to top_k
            candidate_docs_with_scores.sort(key=lambda doc: doc.score, reverse=True)
            retrieved_docs = candidate_docs_with_scores[:top_k]
            
            # Apply content size limiting
            retrieved_docs = self._limit_content_size(retrieved_docs, max_total_chars=30000)
            
            logger.info(f"OptimizedColBERT: Processed {processed_count} documents, found {above_threshold_count} above threshold, returning {len(retrieved_docs)} final documents.")
            
        except Exception as e:
            logger.error(f"OptimizedColBERT: Error during document retrieval: {e}", exc_info=True)
            return []

        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        """
        Generates a final answer using the LLM based on the original query and retrieved documents.
        """
        logger.info(f"OptimizedColBERT: Generating answer for query: '{query_text[:50]}...'")
        
        if not retrieved_docs:
            logger.warning("OptimizedColBERT: No documents retrieved. Returning default response.")
            return "I could not find enough information to answer your question."

        # Create context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs):
            context_parts.append(f"Document {i+1} (Score: {doc.score:.3f}):\n{doc.content}")
        
        context = "\n\n".join(context_parts)
        
        # Ensure context doesn't exceed reasonable limits
        if len(context) > 25000:
            context = context[:25000] + "\n\n[Content truncated for length...]"

        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.
 
Context:
{context}

Question: {query_text}
Answer:"""
        
        try:
            answer = self.llm_func(prompt)
        except Exception as e:
            logger.error(f"OptimizedColBERT: Error during LLM call: {e}", exc_info=True)
            answer = "There was an error generating the answer."
            
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """Runs the full RAG pipeline: retrieve documents and generate an answer."""
        logger.info(f"OptimizedColBERT: Running full pipeline for query: '{query_text[:50]}...'")
        retrieved_docs = self.retrieve_documents(query_text, top_k, similarity_threshold)
        answer = self.generate_answer(query_text, retrieved_docs)
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_docs
        }

# Example Usage (for testing or direct execution)
if __name__ == "__main__":
    # Basic logging setup for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    # --- Configuration for ColBERT Semantic Encoder ---
    # This function will be passed to the pipeline to encode queries into token embeddings
    # It needs to match how document tokens were encoded (e.g., using fjmgAI/reason-colBERT-150M-GTE-ModernColBERT)
    
    # This is a placeholder for the actual ColBERT query encoder function.
    # In a real setup, this would load the ColBERT model and tokenizer.
    # For the E2E test, this is provided by a fixture in conftest.py
    
    # IMPORTANT: This semantic encoder needs to be compatible with the one used for document tokenization
    # For this example, we'll simulate it. In a real scenario, you'd use the actual ColBERT model.
    
    # This function needs to return a tuple: (List[str], List[List[float]])
    # where List[str] are the tokens and List[List[float]] are their embeddings.
    def create_colbert_semantic_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", embedding_func_override: Callable = None):
        """
        Creates a ColBERT-style query encoder.
        In a real scenario, this would load a ColBERT model.
        Here, we use a standard sentence transformer and simulate token-level output for demo.
        The actual E2E test uses a fixture that loads the correct ColBERT model.
        """
        from transformers import AutoTokenizer, AutoModel # type: ignore
        
        # Use the actual ColBERT model specified in config for consistency
        # This should match the model used for document token embedding generation
        colbert_model_name = get_config_value("colbert.document_encoder_model", "fjmgAI/reason-colBERT-150M-GTE-ModernColBERT")
        logger.info(f"ColBERT Demo: Loading query encoder model: {colbert_model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(colbert_model_name)
        model = AutoModel.from_pretrained(colbert_model_name)
        model.eval() # Set to evaluation mode

        def encoder(text: str) -> Tuple[List[str], List[List[float]]]:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            # For ColBERT, we typically use the last hidden state for token embeddings
            token_embeddings_tensor = outputs.last_hidden_state.squeeze(0) # Remove batch dim
            
            # Get tokens (strings)
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0).tolist())
            
            # Detach tensor and convert to list of lists
            token_embeddings_list = token_embeddings_tensor.detach().cpu().numpy().tolist()
            
            if len(tokens) != len(token_embeddings_list):
                 logger.warning(f"ColBERT Demo Encoder: Mismatch between token count ({len(tokens)}) and embedding count ({len(token_embeddings_list)}). This can happen with special tokens.")
                 # Attempt to align if possible, e.g. by slicing based on attention mask, or just log and proceed
                 # For simplicity here, we'll proceed, but in production, this needs careful handling.

            return tokens, token_embeddings_list
        return encoder

    try:
        # Get IRIS connection
        iris_conn = get_iris_connection()
        logger.info("Successfully connected to IRIS for ColBERT demo.")

        # Get LLM function (using a simple echo for demo)
        # llm = get_llm_func() # This would use OpenAI by default
        def echo_llm(prompt: str) -> str:
            # Truncate prompt for display if too long
            displayed_prompt = prompt[:300] + "..." if len(prompt) > 300 else prompt
            logger.info(f"EchoLLM received prompt (first 300 chars): {displayed_prompt}")
            return f"LLM Echo: Based on the context, the answer to your query is derived from the provided documents."
        
        llm = echo_llm

        # Create the ColBERT query encoder function
        # This uses the actual ColBERT model for query encoding
        colbert_query_encoder = create_colbert_semantic_encoder()
        
        # Instantiate the pipeline
        # The `embedding_func` for Stage 1 will use the default from common.utils (e.g., all-MiniLM-L6-v2)
        colbert_pipeline = ColbertRAGPipeline(
            iris_connector=iris_conn,
            colbert_query_encoder_func=colbert_query_encoder,
            llm_func=llm
        )

        # Example query
        test_query = "What are the effects of metformin on type 2 diabetes?"
        logger.info(f"Running ColBERT pipeline with query: '{test_query}'")
        
        # Run the pipeline
        results = colbert_pipeline.run(test_query, top_k=3)

        print("\nOptimized ColBERT Pipeline Demo Results:")
        print(f"Query: {results['query']}")
        print(f"Answer: {results['answer']}")
        print("Retrieved Documents:")
        if results['retrieved_documents']:
            for i, doc in enumerate(results['retrieved_documents']):
                print(f"  Doc {i+1}: ID={doc.id}, Score={doc.score:.4f}")
                # print(f"    Content: {doc.content[:200]}...") # Content can be long
        else:
            print("  No documents retrieved.")

    except Exception as e:
        logger.error(f"Error during ColBERT demo: {e}", exc_info=True)
    finally:
        if 'iris_conn' in locals() and iris_conn:
            iris_conn.close()
            logger.info("IRIS connection closed for ColBERT demo.")