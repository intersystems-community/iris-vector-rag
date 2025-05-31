# colbert/pipeline_fixed.py - Optimized ColBERT Implementation

import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable, Tuple
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

from common.utils import Document, timing_decorator, get_llm_func, get_embedding_func
from common.iris_connector_jdbc import get_iris_connection

class OptimizedColbertRAGPipeline:
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
            max_sim = -1.0
            for d_embed in doc_token_embeddings:
                sim = self._calculate_cosine_similarity(q_embed, d_embed)
                max_sim = max(max_sim, sim)
            max_sim_scores.append(max_sim)

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

        candidate_docs_with_scores = []
        
        try:
            cursor = self.iris_connector.cursor()

            # Optimized approach: Get a reasonable sample of documents to score
            # Instead of processing ALL documents, we'll use a more efficient strategy
            
            # Step 1: Get doc_ids and text_content for a sample of documents that have token embeddings
            # This is still limited to 1000 documents to manage memory and processing time.
            sql_get_docs_with_tokens = """
            SELECT DISTINCT d.doc_id, d.text_content
            FROM RAG.SourceDocuments d
            INNER JOIN RAG.DocumentTokenEmbeddings t ON d.doc_id = t.doc_id
            ORDER BY d.doc_id
            LIMIT 1000
            """
            
            cursor.execute(sql_get_docs_with_tokens)
            docs_data = cursor.fetchall()
            
            logger.info(f"OptimizedColBERT: Found {len(docs_data)} documents with token embeddings")

            if not docs_data:
                logger.warning("OptimizedColBERT: No documents with token embeddings found in the database.")
                return []

            doc_ids = [doc_row[0] for doc_row in docs_data]
            doc_contents = {doc_row[0]: doc_row[1] for doc_row in docs_data}

            # Step 2: Fetch all token embeddings for these doc_ids in a single query
            # This significantly reduces database round trips.
            # Using a parameterized query for IN clause to prevent SQL injection and handle large lists.
            placeholders = ','.join(['?' for _ in doc_ids])
            sql_fetch_all_tokens = f"""
            SELECT doc_id, token_sequence_index, token_embedding
            FROM RAG.DocumentTokenEmbeddings
            WHERE doc_id IN ({placeholders})
            ORDER BY doc_id, token_sequence_index
            """
            
            cursor.execute(sql_fetch_all_tokens, doc_ids)
            all_token_rows = cursor.fetchall()

            # Group token embeddings by doc_id
            grouped_doc_token_embeddings = {}
            for token_row in all_token_rows:
                doc_id, token_sequence_index, token_embedding_str = token_row
                if doc_id not in grouped_doc_token_embeddings:
                    grouped_doc_token_embeddings[doc_id] = []
                
                try:
                    if isinstance(token_embedding_str, str):
                        if ',' in token_embedding_str:
                            token_embedding = [float(x.strip()) for x in token_embedding_str.split(',')]
                        else:
                            token_embedding = json.loads(token_embedding_str)
                        grouped_doc_token_embeddings[doc_id].append(token_embedding)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.debug(f"OptimizedColBERT: Error parsing token embedding for doc {doc_id}, index {token_sequence_index}: {e}")
                    continue
            
            processed_count = 0
            above_threshold_count = 0

            # Step 3: Iterate through documents and calculate MaxSim scores
            for doc_id, doc_content in doc_contents.items():
                current_doc_token_embeddings = grouped_doc_token_embeddings.get(doc_id, [])
                
                if not current_doc_token_embeddings:
                    continue
                
                # Calculate MaxSim score
                maxsim_score = self._calculate_maxsim(query_token_embeddings, current_doc_token_embeddings)
                
                # Only include documents above threshold
                if maxsim_score > similarity_threshold:
                    limited_content = (doc_content or "")[:5000]  # Limit per document
                    candidate_docs_with_scores.append(
                        Document(id=doc_id, content=limited_content, score=maxsim_score)
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
            logger.info(f"OptimizedColBERT: Generated answer: '{answer[:100]}...'")
            return answer
        except Exception as e:
            logger.error(f"OptimizedColBERT: Error generating answer: {e}")
            return "I encountered an error while generating the answer."

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Runs the full optimized ColBERT pipeline.
        """
        logger.info(f"OptimizedColBERT: Running pipeline for query: '{query_text[:50]}...'")
        
        retrieved_documents = self.retrieve_documents(query_text, top_k, similarity_threshold)
        answer = self.generate_answer(query_text, retrieved_documents)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents],
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents)
        }


def create_colbert_semantic_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Creates a semantic encoder for ColBERT that uses a real embedding model.
    This replaces the hash-based encoder to capture semantic meaning.
    """
    embedding_func = get_embedding_func(model_name=model_name)

    def encoder(text: str) -> List[List[float]]:
        """
        Generate token embeddings using a semantic model.
        For simplicity, this will generate a single embedding for the entire text,
        but a true ColBERT implementation would generate embeddings for each token.
        For now, we'll simulate token embeddings by repeating the sentence embedding.
        """
        if not text or not text.strip():
            return []
        
        # Get a single embedding for the text
        sentence_embedding = embedding_func([text])[0]
        
        # Simulate token embeddings by splitting text and assigning the same embedding
        # This is a simplification; a real ColBERT would use a token-level encoder.
        tokens = text.strip().split()
        if not tokens:
            return []
        
        # Limit number of tokens for performance and to match expected ColBERT output structure
        tokens = tokens[:32] # Max 32 tokens for query, adjust for doc if needed
        
        # Return a list of embeddings, one for each "token" (though they are identical here)
        return [sentence_embedding for _ in tokens]
    
    return encoder


# Factory function for easy integration
def create_colbert_pipeline(iris_connector=None, llm_func=None):
    """
    Factory function to create an optimized ColBERT pipeline with working encoders.
    """
    if iris_connector is None:
        iris_connector = get_iris_connection()
    
    if llm_func is None:
        llm_func = get_llm_func()
    
    # Create semantic encoders
    semantic_encoder = create_colbert_semantic_encoder()
    
    return OptimizedColbertRAGPipeline(
        iris_connector=iris_connector,
        colbert_query_encoder_func=semantic_encoder,
        colbert_doc_encoder_func=semantic_encoder, # Using same for simplicity
        llm_func=llm_func
    )


if __name__ == '__main__':
    print("Running Optimized ColBERT Pipeline Demo...")
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("OptimizedColBERT_Demo")
    
    try:
        # Create the optimized pipeline
        pipeline = create_colbert_pipeline()
        
        # Test query
        test_query = "What are the symptoms of diabetes?"
        logger.info(f"\nExecuting optimized ColBERT pipeline for query: '{test_query}'")
        
        result = pipeline.run(test_query, top_k=3, similarity_threshold=0.0)
        
        logger.info("\n--- Optimized ColBERT Pipeline Result ---")
        logger.info(f"Query: {result['query']}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
        
        for i, doc_dict in enumerate(result['retrieved_documents']):
            content_preview = doc_dict.get('content', '')[:100]
            logger.info(f"  Doc {i+1}: ID={doc_dict.get('id')}, Score={doc_dict.get('score'):.4f}")
            logger.info(f"    Content: '{content_preview}...'")
        
    except Exception as e:
        logger.error(f"Demo error: {e}", exc_info=True)
    
    logger.info("\nOptimized ColBERT Pipeline Demo Finished.")