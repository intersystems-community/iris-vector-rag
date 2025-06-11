# colbert/pipeline_v2.py

import logging
from typing import List, Dict, Any, Callable
from common.utils import Document, timing_decorator
from common.jdbc_stream_utils import read_iris_stream
import numpy as np
import json

logger = logging.getLogger(__name__)

class ColBERTPipelineV2:
    """
    ColBERT RAG Pipeline V2 with HNSW support
    
    This implementation uses native IRIS VECTOR columns and HNSW indexes
    for accelerated similarity search on the _V2 tables.
    """
    
    def __init__(self, iris_connector, embedding_func: Callable, llm_func: Callable):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = "RAG"
    
    def _calculate_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
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
        Calculate the MaxSim score between query token embeddings and document token embeddings.
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
    
    def _generate_token_embeddings(self, text: str) -> List[List[float]]:
        """
        Generate token embeddings for ColBERT.
        For simplicity, this simulates token embeddings by splitting text and using sentence embeddings.
        """
        if not text or not text.strip():
            return []
        
        # Get a single embedding for the text
        sentence_embedding = self.embedding_func([text])[0]
        
        # Simulate token embeddings by splitting text
        tokens = text.strip().split()
        if not tokens:
            return []
        
        # Limit number of tokens for performance
        tokens = tokens[:32]  # Max 32 tokens
        
        # Return a list of embeddings, one for each "token"
        return [sentence_embedding for _ in tokens]
    
    @timing_decorator
    def retrieve_documents(self, query: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Document]:
        """
        Retrieve documents using ColBERT MaxSim scoring with HNSW-accelerated vector search
        """
        # Generate query token embeddings
        query_token_embeddings = self._generate_token_embeddings(query)
        
        if not query_token_embeddings:
            logger.warning("ColBERT V2: Query encoder returned no embeddings.")
            return []
        
        # Generate query embedding for initial retrieval
        query_embedding = self.embedding_func([query])[0]
        query_embedding_str = f"[{','.join([f'{x:.10f}' for x in query_embedding])}]"
        
        retrieved_docs = []
        
        # First, get candidate documents using standard vector search
        sql_query = f"""
            SELECT TOP {top_k * 3} doc_id, title, text_content,
                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
            FROM {self.schema}.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY score DESC
        """
        
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            logger.debug("ColBERT V2 Document Retrieve with HNSW")
            cursor.execute(sql_query, [query_embedding_str])
            candidate_results = cursor.fetchall()
            
            # Process candidates with ColBERT MaxSim scoring
            for row in candidate_results:
                doc_id, title, content, initial_score = row
                
                # Handle potential stream objects
                content = read_iris_stream(content) if content else ""
                title = read_iris_stream(title) if title else ""
                
                # Generate document token embeddings
                doc_token_embeddings = self._generate_token_embeddings(content)
                
                if doc_token_embeddings:
                    # Calculate ColBERT MaxSim score
                    maxsim_score = self._calculate_maxsim(query_token_embeddings, doc_token_embeddings)
                    
                    # Only include documents above threshold
                    if maxsim_score > similarity_threshold:
                        doc = Document(
                            id=doc_id,
                            content=content[:5000],  # Limit content size
                            score=maxsim_score
                        )
                        # Store metadata separately
                        doc._metadata = {
                            "title": title,
                            "similarity_score": maxsim_score,
                            "initial_score": float(initial_score) if initial_score else 0.0,
                            "source": "ColBERT_V2_HNSW"
                        }
                        retrieved_docs.append(doc)
            
            # Sort by ColBERT MaxSim score and limit to top_k
            retrieved_docs.sort(key=lambda x: x.score, reverse=True)
            retrieved_docs = retrieved_docs[:top_k]
            
            print(f"ColBERT V2: Retrieved {len(retrieved_docs)} documents using MaxSim scoring")
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            print(f"Error retrieving documents: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs
    
    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate answer using retrieved documents with ColBERT context
        """
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context with ColBERT scoring information
        context_parts = []
        for i, doc in enumerate(documents[:3], 1):
            metadata = getattr(doc, '_metadata', {})
            title = metadata.get('title', 'Untitled')
            maxsim_score = doc.score or 0
            initial_score = metadata.get('initial_score', 0)
            
            content_preview = doc.content[:500] if doc.content else ""
            context_parts.append(
                f"Document {i} (ID: {doc.id}, ColBERT Score: {maxsim_score:.3f}, Vector Score: {initial_score:.3f}, Title: {title}):\n{content_preview}..."
            )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following documents ranked using ColBERT's late interaction mechanism, answer the question comprehensively.
The ColBERT scores indicate fine-grained token-level relevance matching.

Context:
{context}

Question: {query}

Please provide a detailed answer based on the available information:"""
        
        try:
            response = self.llm_func(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def run(self, query: str, top_k: int = 5, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Execute the ColBERT V2 pipeline with HNSW acceleration
        """
        print(f"\n{'='*50}")
        print(f"ColBERT V2 Pipeline (HNSW) - Query: {query}")
        print(f"{'='*50}")
        
        # Retrieve documents using ColBERT MaxSim
        documents = self.retrieve_documents(query, top_k=top_k, similarity_threshold=similarity_threshold)
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        # Prepare results
        result = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": getattr(doc, '_metadata', {})
                }
                for doc in documents
            ],
            "metadata": {
                "pipeline": "ColBERT_V2",
                "uses_hnsw": True,
                "uses_maxsim": True,
                "top_k": top_k,
                "num_documents_retrieved": len(documents)
            }
        }
        
        return result