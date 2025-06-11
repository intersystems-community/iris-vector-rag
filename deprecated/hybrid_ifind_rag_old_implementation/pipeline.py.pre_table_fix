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
        Perform iFind keyword search using IRIS bitmap indexes.
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of documents with keyword match scores
        """
        if not keywords:
            return []
        
        try:
            # For now, implement basic keyword search using LIKE
            # TODO: Replace with actual iFind implementation using %FIND predicate
            keyword_conditions = []
            params = []
            
            for i, keyword in enumerate(keywords[:5]):  # Limit to 5 keywords
                # Remove UPPER function - not supported on stream fields in IRIS
                keyword_conditions.append(f"d.text_content LIKE ?")
                params.append(f"%{keyword}%")
            
            where_clause = " OR ".join(keyword_conditions)
            
            query = f"""
            SELECT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.doc_id as title,
                d.text_content as content,
                '' as metadata,
                ROW_NUMBER() OVER (ORDER BY d.doc_id) as rank_position
            FROM RAG.SourceDocuments_V2 d
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
                    'content': row[2],
                    'metadata': row[3],
                    'rank_position': row[4],
                    'method': 'ifind'
                })
            
            cursor.close()
            logger.info(f"iFind keyword search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in iFind keyword search: {e}")
            return []
    
    def _graph_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform graph-based retrieval using the fixed GraphRAG components.
        
        Args:
            query: Input query string
            
        Returns:
            List of documents from graph traversal
        """
        try:
            # Use the fixed GraphRAG pipeline for proper knowledge graph traversal
            from graphrag.pipeline import FixedGraphRAGPipeline
            
            # Create GraphRAG pipeline instance
            graph_pipeline = FixedGraphRAGPipeline(
                iris_connector=self.iris_connector,
                embedding_func=self.embedding_func,
                llm_func=self.llm_func
            )
            
            # Get documents via knowledge graph traversal
            graph_docs = graph_pipeline.retrieve_documents_via_kg(
                query,
                top_k=self.config['max_results_per_method']
            )
            
            results = []
            for i, doc in enumerate(graph_docs):
                results.append({
                    'document_id': doc.id,
                    'title': doc.id,
                    'content': doc.content,
                    'metadata': '',
                    'relationship_strength': doc.score,
                    'rank_position': i + 1,
                    'method': 'graph'
                })
            
            logger.info(f"Graph retrieval returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Graph retrieval failed: {e}")
            return []
            return results
            
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}")
            return []
    
    def _vector_similarity_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using embeddings.
        
        Args:
            query: Input query string
            
        Returns:
            List of documents with similarity scores
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_func([query])[0]
            
            # Convert embedding to string format for SQL
            embedding_str = ','.join(map(str, query_embedding))
            
            # Use similarity threshold filtering like BasicRAG for better performance
            similarity_threshold = 0.1
            
            # Check if we're using JDBC (which has issues with parameter binding for vectors)
            conn_type = type(self.iris_connector).__name__
            is_jdbc = 'JDBC' in conn_type or hasattr(self.iris_connector, '_jdbc_connection')
            
            if is_jdbc:
                # Use direct SQL for JDBC to avoid parameter binding issues
                query_sql = f"""
                SELECT TOP {self.config['max_results_per_method']}
                    d.doc_id as document_id,
                    d.doc_id as title,
                    d.text_content as content,
                    '' as metadata,
                    VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR('{embedding_str}')) as similarity_score,
                    ROW_NUMBER() OVER (ORDER BY VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR('{embedding_str}')) DESC) as rank_position
                FROM RAG.SourceDocuments_V2 d
                WHERE d.embedding IS NOT NULL
                  AND LENGTH(d.embedding) > 1000
                  AND VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR('{embedding_str}')) > {similarity_threshold}
                ORDER BY similarity_score DESC
                """
                
                cursor = self.iris_connector.cursor()
                cursor.execute(query_sql)  # No parameters for JDBC
            else:
                # Use parameter binding for ODBC
                query_sql = f"""
                SELECT TOP {self.config['max_results_per_method']}
                    d.doc_id as document_id,
                    d.doc_id as title,
                    d.text_content as content,
                    '' as metadata,
                    VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) as similarity_score,
                    ROW_NUMBER() OVER (ORDER BY VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) DESC) as rank_position
                FROM RAG.SourceDocuments_V2 d
                WHERE d.embedding IS NOT NULL
                  AND LENGTH(d.embedding) > 1000
                  AND VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) > ?
                ORDER BY similarity_score DESC
                """
                
                cursor = self.iris_connector.cursor()
                cursor.execute(query_sql, [embedding_str, embedding_str, embedding_str, similarity_threshold])
            
            results = []
            
            for row in cursor.fetchall():
                # Handle potential stream objects (for JDBC)
                content = row[2]
                if hasattr(content, 'read'):
                    content = content.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')
                
                results.append({
                    'document_id': row[0],
                    'title': row[1],
                    'content': str(content),
                    'metadata': row[3],
                    'similarity_score': float(row[4]) if row[4] else 0.0,
                    'rank_position': row[5],
                    'method': 'vector'
                })
            
            cursor.close()
            logger.info(f"Vector similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, ifind_results: List[Dict], 
                               graph_results: List[Dict], 
                               vector_results: List[Dict]) -> List[Dict[str, Any]]:
        """
        Combine results using reciprocal rank fusion.
        
        Args:
            ifind_results: Results from iFind keyword search
            graph_results: Results from graph retrieval
            vector_results: Results from vector similarity search
            
        Returns:
            Fused and ranked results
        """
        try:
            # Create document lookup maps
            ifind_map = {r['document_id']: r for r in ifind_results}
            graph_map = {r['document_id']: r for r in graph_results}
            vector_map = {r['document_id']: r for r in vector_results}
            
            # Get all unique document IDs
            all_doc_ids = set()
            all_doc_ids.update(ifind_map.keys())
            all_doc_ids.update(graph_map.keys())
            all_doc_ids.update(vector_map.keys())
            
            fused_results = []
            
            for doc_id in all_doc_ids:
                # Get document info (prefer vector, then graph, then ifind)
                doc_info = vector_map.get(doc_id) or graph_map.get(doc_id) or ifind_map.get(doc_id)
                
                # Calculate RRF score
                rrf_score = 0.0
                methods_used = []
                
                # iFind contribution
                if doc_id in ifind_map:
                    rank = ifind_map[doc_id]['rank_position']
                    rrf_score += self.config['ifind_weight'] / (self.config['rrf_k'] + rank)
                    methods_used.append('ifind')
                
                # Graph contribution  
                if doc_id in graph_map:
                    rank = graph_map[doc_id]['rank_position']
                    rrf_score += self.config['graph_weight'] / (self.config['rrf_k'] + rank)
                    methods_used.append('graph')
                
                # Vector contribution
                if doc_id in vector_map:
                    rank = vector_map[doc_id]['rank_position']
                    rrf_score += self.config['vector_weight'] / (self.config['rrf_k'] + rank)
                    methods_used.append('vector')
                
                fused_results.append({
                    'document_id': doc_id,
                    'title': doc_info['title'],
                    'content': doc_info['content'],
                    'metadata': doc_info['metadata'],
                    'rrf_score': rrf_score,
                    'methods_used': methods_used,
                    'method_count': len(methods_used),
                    'ifind_rank': ifind_map.get(doc_id, {}).get('rank_position'),
                    'graph_rank': graph_map.get(doc_id, {}).get('rank_position'),
                    'vector_rank': vector_map.get(doc_id, {}).get('rank_position'),
                    'similarity_score': vector_map.get(doc_id, {}).get('similarity_score'),
                    'relationship_strength': graph_map.get(doc_id, {}).get('relationship_strength')
                })
            
            # Sort by RRF score (descending) and method count (descending)
            fused_results.sort(key=lambda x: (x['rrf_score'], x['method_count']), reverse=True)
            
            # Limit results
            final_results = fused_results[:self.config['final_results_limit']]
            
            logger.info(f"RRF fusion produced {len(final_results)} final results from {len(all_doc_ids)} unique documents")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in reciprocal rank fusion: {e}")
            return []
    
    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search approach.
        
        Args:
            query: Input query string
            
        Returns:
            List of retrieved documents with fusion scores
        """
        start_time = time.time()
        
        try:
            # Extract keywords for iFind search
            keywords = self._extract_keywords(query)
            
            # Perform parallel retrieval (in sequence for now)
            logger.info("Starting hybrid retrieval...")
            
            # 1. iFind keyword search
            ifind_start = time.time()
            ifind_results = self._ifind_keyword_search(keywords)
            ifind_time = time.time() - ifind_start
            
            # 2. Graph-based retrieval
            graph_start = time.time()
            graph_results = self._graph_retrieval(query)
            graph_time = time.time() - graph_start
            
            # 3. Vector similarity search
            vector_start = time.time()
            vector_results = self._vector_similarity_search(query)
            vector_time = time.time() - vector_start
            
            # 4. Reciprocal rank fusion
            fusion_start = time.time()
            fused_results = self._reciprocal_rank_fusion(ifind_results, graph_results, vector_results)
            fusion_time = time.time() - fusion_start
            
            total_time = time.time() - start_time
            
            # Log performance metrics
            logger.info(f"Hybrid retrieval completed in {total_time:.3f}s:")
            logger.info(f"  - iFind: {len(ifind_results)} results in {ifind_time:.3f}s")
            logger.info(f"  - Graph: {len(graph_results)} results in {graph_time:.3f}s") 
            logger.info(f"  - Vector: {len(vector_results)} results in {vector_time:.3f}s")
            logger.info(f"  - Fusion: {len(fused_results)} results in {fusion_time:.3f}s")
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in hybrid document retrieval: {e}")
            return []
    
    def generate_response(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Generate response using retrieved documents.
        
        Args:
            query: Original query
            retrieved_docs: Documents from hybrid retrieval
            
        Returns:
            Generated response string
        """
        try:
            if not retrieved_docs:
                return "I couldn't find any relevant documents to answer your question."
            
            # Prepare context from retrieved documents
            context_parts = []
            for i, doc in enumerate(retrieved_docs[:5], 1):  # Use top 5 documents
                methods = ", ".join(doc['methods_used'])
                context_parts.append(
                    f"Document {i} (via {methods}, RRF score: {doc['rrf_score']:.4f}):\n"
                    f"Title: {doc['title']}\n"
                    f"Content: {doc['content'][:1000]}..."  # Limit content length
                )
            
            context = "\n\n".join(context_parts)
            
            # Create prompt for LLM
            prompt = f"""Based on the following documents retrieved through hybrid search (keyword, graph, and vector methods), please answer the question.

Question: {query}

Retrieved Documents:
{context}

Please provide a comprehensive answer based on the information in these documents. If the documents don't contain enough information to fully answer the question, please indicate what information is missing."""

            # Generate response
            response = self.llm_func(prompt)
            
            logger.info(f"Generated response for query: '{query[:50]}...'")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Execute complete hybrid RAG pipeline.
        
        Args:
            query_text: Input query string
            
        Returns:
            Dictionary containing query results and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing hybrid RAG query: '{query_text}'")
            
            # Retrieve documents using hybrid approach
            retrieved_docs = self.retrieve_documents(query_text)
            
            # Generate response
            response = self.generate_response(query_text, retrieved_docs)
            
            total_time = time.time() - start_time
            
            # Prepare result
            result = {
                "query": query_text,
                "answer": response,
                "retrieved_documents": retrieved_docs,
                "metadata": {
                    "total_time": total_time,
                    "num_documents": len(retrieved_docs),
                    "retrieval_methods": {
                        "ifind_weight": self.config['ifind_weight'],
                        "graph_weight": self.config['graph_weight'],
                        "vector_weight": self.config['vector_weight']
                    },
                    "fusion_config": {
                        "rrf_k": self.config['rrf_k'],
                        "max_results_per_method": self.config['max_results_per_method']
                    }
                }
            }
            
            logger.info(f"Hybrid RAG query completed in {total_time:.3f}s with {len(retrieved_docs)} documents")
            return result
            
        except Exception as e:
            logger.error(f"Error in hybrid RAG query: {e}")
            return {
                "query": query_text,
                "answer": f"Error processing query: {str(e)}",
                "retrieved_documents": [],
                "metadata": {
                    "error": str(e),
                    "total_time": time.time() - start_time
                }
            }
    
    def run(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """
        Run method for compatibility with enterprise validation framework.
        
        Args:
            query_text: Input query string
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Dictionary containing query results and metadata
        """
        return self.query(query_text)


def create_hybrid_ifind_rag_pipeline(iris_connector,
                                    embedding_func=None,
                                    llm_func=None) -> HybridiFindRAGPipeline:
    """
    Factory function to create a Hybrid iFind RAG pipeline.
    
    Args:
        iris_connector: IRIS database connection
        embedding_func: Optional embedding function
        llm_func: Optional LLM function
        
    Returns:
        Configured HybridiFindRAGPipeline instance
    """
    return HybridiFindRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )