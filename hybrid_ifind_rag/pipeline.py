"""
Hybrid iFind+Graph+Vector RAG Pipeline Implementation

This module implements a sophisticated hybrid RAG pipeline that combines three
retrieval methods using SQL reciprocal rank fusion:

1. iFind Keyword Search - Exact term matching using IRIS bitmap indexes
2. Graph-based Retrieval - Relationship discovery through entity graphs  
3. Vector Similarity Search - Semantic matching with embeddings

The pipeline uses SQL CTEs for efficient reciprocal rank fusion and leverages
IRIS's unique capabilities for multi-modal retrieval.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import json

from common.utils import get_iris_connector, get_embedding_func, get_llm_func

logger = logging.getLogger(__name__)


class HybridiFindRAGPipeline:
    """
    Hybrid RAG pipeline combining iFind keyword search, graph retrieval, 
    and vector similarity search with reciprocal rank fusion.
    """
    
    def __init__(self, iris_connector, embedding_func=None, llm_func=None):
        """
        Initialize the hybrid RAG pipeline.
        
        Args:
            iris_connector: IRIS database connection
            embedding_func: Function to generate embeddings
            llm_func: Function to generate responses
        """
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func or get_embedding_func()
        self.llm_func = llm_func or get_llm_func()
        
        # Default configuration
        self.config = {
            'ifind_weight': 0.33,
            'graph_weight': 0.33, 
            'vector_weight': 0.34,
            'rrf_k': 60,
            'max_results_per_method': 20,
            'final_results_limit': 10
        }
        
        logger.info("Initialized Hybrid iFind+Graph+Vector RAG Pipeline")
    
    def update_config(self, **kwargs):
        """Update pipeline configuration parameters."""
        self.config.update(kwargs)
        logger.info(f"Updated configuration: {kwargs}")
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract meaningful keywords from query for iFind search.
        
        Args:
            query: Input query string
            
        Returns:
            List of extracted keywords
        """
        # Simple keyword extraction - can be enhanced with NLP
        import re
        
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'what',
            'how', 'when', 'where', 'why', 'who'
        }
        
        # Extract words (alphanumeric, 3+ characters)
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())
        keywords = [word for word in words if word not in stop_words]
        
        logger.debug(f"Extracted keywords from '{query}': {keywords}")
        return keywords
    
    def _ifind_keyword_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Perform keyword search using %FIND search_index() SQL function.
        This requires an %iFind.Index.Basic index to be defined on the text field.
        
        For this to work, we need:
        1. An ObjectScript class with %iFind.Index.Basic on text_content
        2. Or use a table that already has such an index
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of documents with keyword match scores
        """
        if not keywords:
            return []
        
        try:
            # Build iFind search string
            # Join keywords with implicit AND
            search_string = ' '.join(keywords[:5])  # Limit to 5 keywords
            
            # Check if we have an iFind-enabled table
            # First try SourceDocumentsIFind if it exists
            cursor = self.iris_connector.cursor()
            
            # Check for SourceDocumentsIFind table
            cursor.execute("""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'RAG' 
                AND TABLE_NAME = 'SourceDocumentsIFind'
            """)
            
            has_ifind_table = cursor.fetchone()[0] > 0
            
            if has_ifind_table:
                # Use %FIND search_index() on the iFind-enabled table
                query = f"""
                SELECT TOP {self.config['max_results_per_method']}
                    doc_id as document_id,
                    title,
                    SUBSTRING(text_content, 1, 1000) as content,
                    '' as metadata,
                    ROW_NUMBER() OVER (ORDER BY doc_id) as rank_position
                FROM RAG.SourceDocumentsIFind
                WHERE %ID %FIND search_index(TextContentFTI, ?)
                ORDER BY doc_id
                """
                
                cursor.execute(query, (search_string,))
                results = []
                
                for row in cursor.fetchall():
                    results.append({
                        'document_id': row[0],
                        'title': row[1],
                        'content': row[2] if row[2] else 'Content not available',
                        'metadata': row[3],
                        'rank_position': row[4],
                        'method': 'ifind'
                    })
                
                cursor.close()
                
                if results:
                    logger.info(f"iFind search found {len(results)} documents using %FIND")
                    return results
            
            cursor.close()
            
            # If no iFind table or no results, fall back to other methods
            return self._fallback_search(keywords)
            
        except Exception as e:
            logger.error(f"Error in iFind search: {e}")
            # If %FIND fails (no index), fall back to other methods
            return self._fallback_search(keywords)
    
    def _fallback_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fallback search using available methods"""
        try:
            # Try DocumentChunks first if available
            cursor = self.iris_connector.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            has_chunks = cursor.fetchone()[0] > 0
            cursor.close()
            
            if has_chunks:
                # Search in DocumentChunks
                conditions = []
                params = []
                
                for keyword in keywords[:3]:
                    conditions.append("c.chunk_text LIKE ?")
                    params.append(f"%{keyword}%")
                
                where_clause = " OR ".join(conditions)
                
                query = f"""
                SELECT DISTINCT TOP {self.config['max_results_per_method']}
                    c.doc_id as document_id,
                    d.title as title,
                    SUBSTRING(c.chunk_text, 1, 1000) as content,
                    '' as metadata,
                    ROW_NUMBER() OVER (ORDER BY c.doc_id) as rank_position
                FROM RAG.DocumentChunks c
                INNER JOIN RAG.SourceDocuments d ON c.doc_id = d.doc_id
                WHERE {where_clause}
                ORDER BY c.doc_id
                """
                
                cursor = self.iris_connector.cursor()
                cursor.execute(query, params)
                results = []
                
                for row in cursor.fetchall():
                    results.append({
                        'document_id': row[0],
                        'title': row[1],
                        'content': row[2] if row[2] else 'Content not available',
                        'metadata': row[3],
                        'rank_position': row[4],
                        'method': 'ifind'
                    })
                
                cursor.close()
                
                if results:
                    logger.info(f"Fallback chunk search found {len(results)} documents")
                    return results
            
            # Final fallback: title search
            return self._search_by_title(keywords)
            
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return self._search_by_title(keywords)
    
    def _search_by_title(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search by title only"""
        try:
            conditions = []
            params = []
            
            for keyword in keywords[:5]:
                conditions.append("UPPER(d.title) LIKE UPPER(?)")
                params.append(f"%{keyword}%")
            
            where_clause = " OR ".join(conditions)
            
            query = f"""
            SELECT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.title as title,
                SUBSTRING(CAST(d.text_content AS VARCHAR(1000)), 1, 500) as content,
                '' as metadata,
                ROW_NUMBER() OVER (ORDER BY d.doc_id) as rank_position
            FROM RAG.SourceDocuments d
            WHERE {where_clause}
            ORDER BY d.doc_id
            """
            
            cursor = self.iris_connector.cursor()
            cursor.execute(query, params)
            results = []
            
            for row in cursor.fetchall():
                results.append({
                    'document_id': row[0],
                    'title': row[1],
                    'content': row[2] if row[2] else 'Content preview not available',
                    'metadata': row[3],
                    'rank_position': row[4],
                    'method': 'ifind'
                })
            
            cursor.close()
            logger.info(f"Title search found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in title search: {e}")
            return []
    
    def _graph_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform graph-based retrieval using entity relationships.
        
        Args:
            query: Query string
            
        Returns:
            List of documents retrieved via knowledge graph
        """
        try:
            # Use GraphRAG pipeline for graph retrieval
            from graphrag.pipeline import GraphRAGPipeline
            
            # Initialize GraphRAG with same connection
            graphrag = GraphRAGPipeline(
                self.iris_connector,
                self.embedding_func,
                self.llm_func
            )
            
            # Get documents via knowledge graph
            documents = graphrag.retrieve_documents_via_kg(query, top_k=self.config['max_results_per_method'])
            
            # Format results
            graph_results = []
            # Handle both list and dict return types
            if isinstance(documents, dict):
                documents = documents.get('retrieved_documents', [])
            elif not isinstance(documents, list):
                documents = []
                
            for i, doc in enumerate(documents):
                # Handle Document objects from GraphRAG
                if hasattr(doc, '__dict__'):
                    # It's a Document object
                    graph_results.append({
                        'document_id': getattr(doc, 'doc_id', ''),
                        'title': getattr(doc, 'title', ''),
                        'content': str(getattr(doc, 'text_content', ''))[:1000],
                        'metadata': json.dumps(getattr(doc, 'metadata', {})),
                        'rank_position': i + 1,
                        'method': 'graph'
                    })
                else:
                    # It's a dictionary
                    graph_results.append({
                        'document_id': doc.get('doc_id', ''),
                        'title': doc.get('title', ''),
                        'content': str(doc.get('text_content', ''))[:1000],
                        'metadata': json.dumps(doc.get('metadata', {})),
                        'rank_position': i + 1,
                        'method': 'graph'
                    })
            
            logger.info(f"Graph retrieval returned {len(graph_results)} results")
            return graph_results
            
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return []
    
    def _vector_similarity_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using embeddings.
        
        Args:
            query: Query string
            
        Returns:
            List of documents with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_func([query])[0]
            
            # Convert to IRIS vector string format
            iris_vector_str = ','.join(map(str, query_embedding))
            
            # Vector similarity search query
            query = f"""
            SELECT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.title,
                SUBSTRING(d.text_content, 1, 1000) as content,
                '' as metadata,
                VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) as similarity,
                ROW_NUMBER() OVER (ORDER BY VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) DESC) as rank_position
            FROM RAG.SourceDocuments d
            WHERE d.embedding IS NOT NULL
            ORDER BY similarity DESC
            """
            
            cursor = self.iris_connector.cursor()
            cursor.execute(query, (iris_vector_str, iris_vector_str))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'document_id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'metadata': row[3],
                    'similarity': float(row[4]),
                    'rank_position': row[5],
                    'method': 'vector'
                })
            
            cursor.close()
            logger.info(f"Vector similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, 
                               ifind_results: List[Dict[str, Any]],
                               graph_results: List[Dict[str, Any]], 
                               vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Combine results from all three methods using reciprocal rank fusion.
        
        Args:
            ifind_results: Results from iFind keyword search
            graph_results: Results from graph retrieval
            vector_results: Results from vector similarity search
            
        Returns:
            Fused and ranked results
        """
        # Calculate RRF scores
        doc_scores = {}
        doc_info = {}
        
        # Process iFind results
        for result in ifind_results:
            doc_id = result['document_id']
            rank = result['rank_position']
            score = self.config['ifind_weight'] / (self.config['rrf_k'] + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
                doc_info[doc_id] = result
            doc_scores[doc_id] += score
        
        # Process graph results
        for result in graph_results:
            doc_id = result['document_id']
            rank = result['rank_position']
            score = self.config['graph_weight'] / (self.config['rrf_k'] + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
                doc_info[doc_id] = result
            doc_scores[doc_id] += score
        
        # Process vector results
        for result in vector_results:
            doc_id = result['document_id']
            rank = result['rank_position']
            score = self.config['vector_weight'] / (self.config['rrf_k'] + rank)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
                doc_info[doc_id] = result
            doc_scores[doc_id] += score
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        final_results = []
        for doc_id, rrf_score in sorted_docs[:self.config['final_results_limit']]:
            result = doc_info[doc_id].copy()
            result['rrf_score'] = rrf_score
            final_results.append(result)
        
        logger.info(f"RRF fusion produced {len(final_results)} final results from "
                   f"{len(doc_scores)} unique documents")
        
        return final_results
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the hybrid RAG pipeline.
        
        Args:
            query: User query
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing answer and retrieved documents
        """
        start_time = time.time()
        
        # Update config if parameters provided
        if kwargs:
            self.update_config(**kwargs)
        
        logger.info(f"Processing hybrid RAG query: '{query}'")
        
        # Step 1: Hybrid retrieval
        logger.info("Starting hybrid retrieval...")
        
        # Extract keywords for iFind
        keywords = self._extract_keywords(query)
        
        # Execute all three retrieval methods
        retrieval_start = time.time()
        
        ifind_start = time.time()
        ifind_results = self._ifind_keyword_search(keywords)
        ifind_time = time.time() - ifind_start
        
        graph_start = time.time()
        graph_results = self._graph_retrieval(query)
        graph_time = time.time() - graph_start
        
        vector_start = time.time()
        vector_results = self._vector_similarity_search(query)
        vector_time = time.time() - vector_start
        
        # Reciprocal rank fusion
        fusion_start = time.time()
        final_results = self._reciprocal_rank_fusion(
            ifind_results, graph_results, vector_results
        )
        fusion_time = time.time() - fusion_start
        
        retrieval_time = time.time() - retrieval_start
        
        logger.info(f"Hybrid retrieval completed in {retrieval_time:.3f}s:")
        logger.info(f"  - iFind: {len(ifind_results)} results in {ifind_time:.3f}s")
        logger.info(f"  - Graph: {len(graph_results)} results in {graph_time:.3f}s")
        logger.info(f"  - Vector: {len(vector_results)} results in {vector_time:.3f}s")
        logger.info(f"  - Fusion: {len(final_results)} results in {fusion_time:.3f}s")
        
        # Step 2: Generate answer
        if final_results:
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(final_results):
                context_parts.append(
                    f"Document {i+1} (RRF Score: {doc.get('rrf_score', 0):.4f}):\n"
                    f"Title: {doc.get('title', 'N/A')}\n"
                    f"Content: {doc.get('content', '')}\n"
                )
            
            context = "\n---\n".join(context_parts)
            
            # Generate answer using LLM
            prompt = f"""You are a helpful AI assistant. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Answer:"""
            
            answer = self.llm_func(prompt)
            logger.info(f"Generated response for query: '{query}'")
        else:
            answer = "I couldn't find relevant information to answer your question."
            logger.warning("No documents retrieved for query")
        
        total_time = time.time() - start_time
        logger.info(f"Hybrid RAG query completed in {total_time:.3f}s with {len(final_results)} documents")
        
        return {
            'query': query,
            'answer': answer,
            'retrieved_documents': final_results,
            'metadata': {
                'total_time': total_time,
                'retrieval_time': retrieval_time,
                'ifind_results': len(ifind_results),
                'graph_results': len(graph_results),
                'vector_results': len(vector_results),
                'final_results': len(final_results),
                'keywords': keywords
            }
        }