# colbert/pipeline_optimized.py

import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable, Tuple
import numpy as np
import json
import logging
from collections import defaultdict

# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any

# Configure logging
logger = logging.getLogger(__name__)

from common.utils import Document, timing_decorator, get_llm_func, get_embedding_func
from common.iris_connector import get_iris_connection

class OptimizedColbertRAGPipeline:
    """
    Optimized ColBERT RAG Pipeline that uses HNSW indexing on token embeddings
    for dramatically improved performance.
    """
    
    def __init__(self, iris_connector: IRISConnection,
                 colbert_query_encoder_func: Callable[[str], List[List[float]]],
                 colbert_doc_encoder_func: Callable[[str], List[List[float]]],
                 llm_func: Callable[[str], str]):
        self.iris_connector = iris_connector
        self.colbert_query_encoder = colbert_query_encoder_func
        self.colbert_doc_encoder = colbert_doc_encoder_func
        self.llm_func = llm_func
        logger.info("OptimizedColbertRAGPipeline Initialized")

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

    def _calculate_maxsim_optimized(self, query_embeddings: List[List[float]], 
                                   doc_token_embeddings: List[List[float]]) -> float:
        """
        Optimized MaxSim calculation using vectorized operations.
        """
        if not query_embeddings or not doc_token_embeddings:
            return 0.0

        # Convert to numpy arrays for vectorized operations
        query_matrix = np.array(query_embeddings)  # Shape: (num_query_tokens, embedding_dim)
        doc_matrix = np.array(doc_token_embeddings)  # Shape: (num_doc_tokens, embedding_dim)
        
        # Normalize vectors
        query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        doc_norms = np.linalg.norm(doc_matrix, axis=1, keepdims=True)
        
        # Avoid division by zero
        query_norms[query_norms == 0] = 1.0
        doc_norms[doc_norms == 0] = 1.0
        
        query_normalized = query_matrix / query_norms
        doc_normalized = doc_matrix / doc_norms
        
        # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
        similarity_matrix = np.dot(query_normalized, doc_normalized.T)
        
        # For each query token, find max similarity with any doc token
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Sum and normalize by query length
        total_score = np.sum(max_similarities)
        normalized_score = total_score / len(query_embeddings) if query_embeddings else 0.0
        
        return float(normalized_score)

    @timing_decorator
    def retrieve_documents_with_hnsw(self, query_text: str, top_k: int = 5, 
                                   similarity_threshold: float = 0.6,
                                   hnsw_candidates: int = 100) -> List[Document]:
        """
        Retrieves documents using HNSW-accelerated ColBERT MaxSim scoring.
        
        Strategy:
        1. Use HNSW to find candidate token embeddings for each query token
        2. Group candidates by document
        3. Compute MaxSim scores efficiently using vectorized operations
        4. Return top-k documents above threshold
        """
        logger.info(f"OptimizedColbertRAG: Retrieving documents for query: '{query_text[:50]}...'")
        query_token_embeddings = self.colbert_query_encoder(query_text)

        if not query_token_embeddings:
            logger.warning("OptimizedColbertRAG: Query encoder returned no embeddings.")
            return []

        try:
            cursor = self.iris_connector.cursor()
            
            # Step 1: For each query token, find candidate document tokens using HNSW
            doc_token_candidates = defaultdict(list)  # doc_id -> list of token embeddings
            
            logger.info(f"OptimizedColbertRAG: Processing {len(query_token_embeddings)} query tokens")
            
            for i, query_token_embedding in enumerate(query_token_embeddings):
                # Convert query token embedding to string format for IRIS
                query_vector_str = ','.join(map(str, query_token_embedding))
                
                # Use HNSW to find similar token embeddings
                hnsw_query = f"""
                SELECT TOP {hnsw_candidates}
                    doc_id,
                    token_embedding,
                    VECTOR_COSINE(TO_VECTOR(token_embedding), TO_VECTOR(?)) as similarity
                FROM RAG_HNSW.DocumentTokenEmbeddings
                WHERE token_embedding IS NOT NULL
                ORDER BY VECTOR_COSINE(TO_VECTOR(token_embedding), TO_VECTOR(?)) DESC
                """
                
                cursor.execute(hnsw_query, (query_vector_str, query_vector_str))
                candidates = cursor.fetchall()
                
                logger.debug(f"OptimizedColbertRAG: Query token {i} found {len(candidates)} HNSW candidates")
                
                # Group candidates by document
                for doc_id, token_embedding_str, similarity in candidates:
                    try:
                        # Parse token embedding
                        if isinstance(token_embedding_str, str):
                            if token_embedding_str.startswith('['):
                                token_embedding = json.loads(token_embedding_str)
                            else:
                                token_embedding = [float(x) for x in token_embedding_str.split(',')]
                        else:
                            token_embedding = token_embedding_str
                        
                        doc_token_candidates[doc_id].append(token_embedding)
                        
                    except (json.JSONDecodeError, ValueError, TypeError) as e:
                        logger.warning(f"OptimizedColbertRAG: Error parsing token embedding for {doc_id}: {e}")
                        continue
            
            # Step 2: Compute MaxSim scores for each document
            logger.info(f"OptimizedColbertRAG: Computing MaxSim for {len(doc_token_candidates)} candidate documents")
            
            candidate_docs_with_scores = []
            docs_above_threshold = 0
            
            for doc_id, doc_token_embeddings in doc_token_candidates.items():
                # Remove duplicates while preserving order
                unique_embeddings = []
                seen = set()
                for emb in doc_token_embeddings:
                    emb_tuple = tuple(emb)
                    if emb_tuple not in seen:
                        unique_embeddings.append(emb)
                        seen.add(emb_tuple)
                
                # Compute MaxSim score
                maxsim_score = self._calculate_maxsim_optimized(query_token_embeddings, unique_embeddings)
                
                logger.debug(f"OptimizedColbertRAG: Doc {doc_id} MaxSim score: {maxsim_score:.4f}")
                
                # Only include documents above threshold
                if maxsim_score > similarity_threshold:
                    docs_above_threshold += 1
                    
                    # Fetch document content
                    cursor.execute("SELECT text_content FROM RAG_HNSW.SourceDocuments WHERE doc_id = ?", (doc_id,))
                    content_row = cursor.fetchone()
                    doc_content = content_row[0] if content_row else "Content not found"
                    
                    candidate_docs_with_scores.append(Document(id=doc_id, content=doc_content, score=maxsim_score))
            
            # Step 3: Sort and return top-k
            candidate_docs_with_scores.sort(key=lambda doc: doc.score, reverse=True)
            retrieved_docs = candidate_docs_with_scores[:top_k]
            
            cursor.close()
            
            logger.info(f"OptimizedColbertRAG: Found {docs_above_threshold} documents above threshold {similarity_threshold}")
            logger.info(f"OptimizedColbertRAG: Returning top {len(retrieved_docs)} documents")
            
            return retrieved_docs

        except Exception as e:
            logger.error(f"OptimizedColbertRAG: Error during document retrieval: {e}", exc_info=True)
            return []

    @timing_decorator
    def retrieve_documents_fallback(self, query_text: str, top_k: int = 5, 
                                  similarity_threshold: float = 0.6) -> List[Document]:
        """
        Fallback method that uses batch processing instead of individual queries.
        Used when HNSW indexing is not available.
        """
        logger.info(f"OptimizedColbertRAG: Using fallback batch processing for query: '{query_text[:50]}...'")
        query_token_embeddings = self.colbert_query_encoder(query_text)

        if not query_token_embeddings:
            logger.warning("OptimizedColbertRAG: Query encoder returned no embeddings.")
            return []

        try:
            cursor = self.iris_connector.cursor()

            # Step 1: Batch fetch all token embeddings
            logger.info("OptimizedColbertRAG: Batch fetching all token embeddings")
            
            batch_query = """
            SELECT doc_id, token_embedding
            FROM RAG_HNSW.DocumentTokenEmbeddings
            WHERE token_embedding IS NOT NULL
            ORDER BY doc_id, token_sequence_index
            """
            
            cursor.execute(batch_query)
            all_token_data = cursor.fetchall()
            
            # Step 2: Group by document
            doc_token_embeddings = defaultdict(list)
            
            for doc_id, token_embedding_str in all_token_data:
                try:
                    if isinstance(token_embedding_str, str):
                        if token_embedding_str.startswith('['):
                            token_embedding = json.loads(token_embedding_str)
                        else:
                            token_embedding = [float(x) for x in token_embedding_str.split(',')]
                    else:
                        token_embedding = token_embedding_str
                    
                    doc_token_embeddings[doc_id].append(token_embedding)
                    
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(f"OptimizedColbertRAG: Error parsing token embedding for {doc_id}: {e}")
                    continue
            
            # Step 3: Compute MaxSim scores
            logger.info(f"OptimizedColbertRAG: Computing MaxSim for {len(doc_token_embeddings)} documents")
            
            candidate_docs_with_scores = []
            docs_above_threshold = 0
            
            for doc_id, doc_tokens in doc_token_embeddings.items():
                maxsim_score = self._calculate_maxsim_optimized(query_token_embeddings, doc_tokens)
                
                if maxsim_score > similarity_threshold:
                    docs_above_threshold += 1
                    
                    # Fetch document content
                    cursor.execute("SELECT text_content FROM RAG_HNSW.SourceDocuments WHERE doc_id = ?", (doc_id,))
                    content_row = cursor.fetchone()
                    doc_content = content_row[0] if content_row else "Content not found"
                    
                    candidate_docs_with_scores.append(Document(id=doc_id, content=doc_content, score=maxsim_score))
            
            # Step 4: Sort and return top-k
            candidate_docs_with_scores.sort(key=lambda doc: doc.score, reverse=True)
            retrieved_docs = candidate_docs_with_scores[:top_k]
            
            cursor.close()
            
            logger.info(f"OptimizedColbertRAG: Found {docs_above_threshold} documents above threshold")
            logger.info(f"OptimizedColbertRAG: Returning top {len(retrieved_docs)} documents")
            
            return retrieved_docs

        except Exception as e:
            logger.error(f"OptimizedColbertRAG: Error during fallback retrieval: {e}", exc_info=True)
            return []

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, 
                          similarity_threshold: float = 0.6) -> List[Document]:
        """
        Main retrieval method that tries HNSW first, then falls back to batch processing.
        """
        try:
            # Try HNSW-accelerated retrieval first
            return self.retrieve_documents_with_hnsw(query_text, top_k, similarity_threshold)
        except Exception as e:
            logger.warning(f"OptimizedColbertRAG: HNSW retrieval failed: {e}")
            logger.info("OptimizedColbertRAG: Falling back to batch processing")
            return self.retrieve_documents_fallback(query_text, top_k, similarity_threshold)

    @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        """
        Generates a final answer using the LLM based on the original query and retrieved documents.
        """
        logger.info(f"OptimizedColbertRAG: Generating final answer for query: '{query_text[:50]}...'")
        if not retrieved_docs:
            logger.warning("OptimizedColbertRAG: No documents retrieved. Returning a default response.")
            return "I could not find enough information to answer your question."

        context = "\n\n".join([doc.content for doc in retrieved_docs])

        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.
If the context does not contain the answer, state that you cannot answer based on the provided information.

Context:
{context}

Question: {query_text}

Answer:"""

        answer = self.llm_func(prompt)
        logger.info(f"OptimizedColbertRAG: Generated final answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Runs the full optimized ColBERT pipeline.
        """
        logger.info(f"OptimizedColbertRAG: Running pipeline for query: '{query_text[:50]}...'")
        retrieved_documents = self.retrieve_documents(query_text, top_k, similarity_threshold)
        answer = self.generate_answer(query_text, retrieved_documents)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents],
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents)
        }

    @timing_decorator
    def query(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Alias for run() method to maintain compatibility with other pipeline interfaces.
        """
        return self.run(query_text, top_k, similarity_threshold)


def check_hnsw_token_index_exists(iris_connector: IRISConnection) -> bool:
    """
    Check if HNSW index exists on DocumentTokenEmbeddings.
    """
    try:
        cursor = iris_connector.cursor()
        
        # Check if the index exists
        cursor.execute("""
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.INDEXES 
        WHERE TABLE_SCHEMA = 'RAG_HNSW' 
        AND TABLE_NAME = 'DocumentTokenEmbeddings'
        AND INDEX_TYPE LIKE '%HNSW%'
        """)
        
        result = cursor.fetchone()
        cursor.close()
        
        return result[0] > 0 if result else False
        
    except Exception as e:
        logger.warning(f"Could not check HNSW index status: {e}")
        return False


def create_hnsw_token_index(iris_connector: IRISConnection) -> bool:
    """
    Create HNSW index on DocumentTokenEmbeddings if it doesn't exist.
    """
    try:
        cursor = iris_connector.cursor()
        
        # Create HNSW index on token embeddings
        create_index_sql = """
        CREATE INDEX idx_hnsw_token_embeddings
        ON RAG_HNSW.DocumentTokenEmbeddings (token_embedding)
        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
        """
        
        cursor.execute(create_index_sql)
        cursor.close()
        
        logger.info("Successfully created HNSW index on DocumentTokenEmbeddings")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create HNSW index on token embeddings: {e}")
        return False


if __name__ == '__main__':
    print("Running OptimizedColbertRAGPipeline Demo...")
    
    # Configure logging for demo
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("OptimizedColbertRAGPipeline_Demo")

    try:
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")

        if db_conn is None:
            raise ConnectionError("Failed to get IRIS connection for optimized ColBERT demo.")

        # Mock ColBERT encoders for demo
        mock_colbert_encoder = lambda text: [[float(i)/10.0]*128 for i in range(min(len(text.split()), 5))]
        
        # Check and create HNSW index if needed
        if not check_hnsw_token_index_exists(db_conn):
            logger.info("HNSW index on token embeddings not found. Attempting to create...")
            create_hnsw_token_index(db_conn)
        else:
            logger.info("HNSW index on token embeddings already exists")
        
        pipeline = OptimizedColbertRAGPipeline(
            iris_connector=db_conn,
            colbert_query_encoder_func=mock_colbert_encoder,
            colbert_doc_encoder_func=mock_colbert_encoder,
            llm_func=llm_fn
        )

        # Example Query
        test_query = "What is the ColBERT model?"
        logger.info(f"\nExecuting optimized ColBERT pipeline for query: '{test_query}'")

        result = pipeline.run(test_query, top_k=5)

        logger.info("\n--- Optimized ColBERT Pipeline Result ---")
        logger.info(f"Query: {result['query']}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
        for i, doc_dict in enumerate(result['retrieved_documents']):
            logger.info(f"  Doc {i+1}: ID={doc_dict.get('id')}, Score={doc_dict.get('score'):.4f}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during optimized ColBERT demo: {e}")
    finally:
        if 'db_conn' in locals() and db_conn is not None:
            try:
                db_conn.close()
                logger.info("Database connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing DB connection: {e_close}")

    logger.info("\nOptimizedColbertRAGPipeline Demo Finished.")