from typing import Callable, Optional, Any
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
import os # Added for sys.path
import sys # Added for sys.path

# Add the project root directory to Python path
# Assuming this file is in src/experimental/hybrid_ifind_rag/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Adjust imports for new structure (e.g. src/common/, src/experimental/graphrag)
from common.utils import get_embedding_func, get_llm_func # get_iris_connector removed as it's passed in
from src.experimental.graphrag.pipeline import GraphRAGPipeline # Assuming GraphRAG is moved here

logger = logging.getLogger(__name__)


class HybridiFindRAGPipeline:
    """
    Hybrid RAG pipeline combining iFind keyword search, graph retrieval, 
    and vector similarity search with reciprocal rank fusion.
    """
    
    def __init__(self, iris_connector: Any, embedding_func: Optional[Callable]=None, llm_func: Optional[Callable]=None): # Added type hint for iris_connector
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
            'rrf_k': 60, # Constant for RRF calculation
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
        """
        import re
        
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'what',
            'how', 'when', 'where', 'why', 'who'
        }
        
        words = re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower())
        keywords = [word for word in words if word not in stop_words]
        
        logger.debug(f"Extracted keywords from '{query}': {keywords}")
        return keywords
    
    def _ifind_keyword_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        if not keywords:
            return []
        
        cursor = None
        try:
            search_string = ' '.join(keywords[:5])
            cursor = self.iris_connector.cursor()
            
            # Check for SourceDocumentsIFind table (assuming it's iFind enabled)
            # This is a simplification; a robust check would inspect catalog or specific iFind metadata
            # For now, we assume if it exists, it's the one to use.
            try:
                cursor.execute("SELECT TOP 1 doc_id FROM RAG.SourceDocumentsIFind")
                has_ifind_table = True
            except Exception:
                has_ifind_table = False

            if has_ifind_table:
                # Use %FIND search_index() on the iFind-enabled table
                # This SQL is specific to how iFind is set up.
                # The original query used %ID %FIND which might be for a specific class projection.
                # A more general approach for a SQL table with an iFind index on a column:
                # WHERE CONTAINS(TextContentFTI, ?)
                # For now, sticking to the provided structure, assuming TextContentFTI is the indexed virtual field.
                query_sql = f"""
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
                # Note: The original query had no explicit ORDER BY for ROW_NUMBER, which can lead to non-deterministic ranks.
                # Added ORDER BY doc_id for ROW_NUMBER consistency.
                
                cursor.execute(query_sql, (search_string,))
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'document_id': str(row[0]),
                        'title': str(row[1] or ""),
                        'content': str(row[2] or 'Content not available'),
                        'metadata': str(row[3] or ""),
                        'rank_position': int(row[4]),
                        'method': 'ifind'
                    })
                
                if results:
                    logger.info(f"iFind search found {len(results)} documents using %FIND on SourceDocumentsIFind")
                    return results
            
            # Fallback if SourceDocumentsIFind is not available or yields no results
            logger.info("iFind table RAG.SourceDocumentsIFind not used or yielded no results, trying fallback search.")
            return self._fallback_search(keywords)
            
        except Exception as e:
            logger.error(f"Error in iFind search: {e}", exc_info=True)
            return self._fallback_search(keywords) # Fallback on any error
        finally:
            if cursor:
                cursor.close()
    
    def _fallback_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fallback search using LIKE on DocumentChunks or SourceDocuments title."""
        # Try DocumentChunks first
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            # Check if DocumentChunks table exists and has data
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
                has_chunks = cursor.fetchone()[0] > 0
            except Exception:
                has_chunks = False

            if has_chunks:
                conditions = []
                params = []
                for keyword in keywords[:3]: # Limit keywords for LIKE performance
                    conditions.append("LOWER(c.chunk_text) LIKE LOWER(?)")
                    params.append(f"%{keyword}%")
                
                if not conditions: return []
                where_clause = " OR ".join(conditions)
                
                query_sql = f"""
                SELECT DISTINCT TOP {self.config['max_results_per_method']}
                    c.doc_id as document_id,
                    d.title as title,
                    SUBSTRING(c.chunk_text, 1, 1000) as content,
                    '' as metadata,
                    ROW_NUMBER() OVER (ORDER BY c.doc_id, c.chunk_id) as rank_position
                FROM RAG.DocumentChunks c
                INNER JOIN RAG.SourceDocuments d ON c.doc_id = d.doc_id
                WHERE {where_clause}
                ORDER BY c.doc_id, c.chunk_id 
                """
                cursor.execute(query_sql, params)
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'document_id': str(row[0]),
                        'title': str(row[1] or ""),
                        'content': str(row[2] or 'Content not available'),
                        'metadata': str(row[3] or ""),
                        'rank_position': int(row[4]),
                        'method': 'ifind_fallback_chunk'
                    })
                if results:
                    logger.info(f"Fallback chunk search found {len(results)} documents")
                    return results
            
            # Final fallback: title search on SourceDocuments
            return self._search_by_title(keywords)
            
        except Exception as e:
            logger.error(f"Error in fallback search: {e}", exc_info=True)
            return self._search_by_title(keywords) # Try title search even if chunk search fails
        finally:
            if cursor:
                cursor.close()
    
    def _search_by_title(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search by title only in RAG.SourceDocuments."""
        if not keywords: return []
        cursor = None
        try:
            conditions = []
            params = []
            for keyword in keywords[:5]:
                conditions.append("LOWER(d.title) LIKE LOWER(?)")
                params.append(f"%{keyword}%")

            if not conditions: return []
            where_clause = " OR ".join(conditions)
            
            query_sql = f"""
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
            cursor.execute(query_sql, params)
            results = []
            for row in cursor.fetchall():
                results.append({
                    'document_id': str(row[0]),
                    'title': str(row[1] or ""),
                    'content': str(row[2] or 'Content preview not available'),
                    'metadata': str(row[3] or ""),
                    'rank_position': int(row[4]),
                    'method': 'ifind_fallback_title'
                })
            logger.info(f"Title search found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in title search: {e}", exc_info=True)
            return []
        finally:
            if cursor:
                cursor.close()
    
    def _graph_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """Performs graph-based retrieval using the GraphRAGPipeline."""
        try:
            graphrag_pipeline = GraphRAGPipeline(
                self.iris_connector,
                self.embedding_func,
                self.llm_func # Pass LLM for potential internal use by GraphRAG, though not typical for its retrieval
            )
            
            # GraphRAG's retrieve_documents_via_kg returns Tuple[List[Document], str]
            documents_objects, method_str = graphrag_pipeline.retrieve_documents_via_kg(query, top_k=self.config['max_results_per_method'])
            
            graph_results = []
            for i, doc_obj in enumerate(documents_objects):
                graph_results.append({
                    'document_id': str(doc_obj.id),
                    'title': str(doc_obj.metadata.get('title', '') if doc_obj.metadata else ''),
                    'content': str(doc_obj.content or "")[:1000], # Truncate for consistency
                    'metadata': json.dumps(doc_obj.metadata or {}), # Ensure metadata is serializable
                    'rank_position': i + 1, # Rank based on order from GraphRAG
                    'method': 'graph'
                })
            
            logger.info(f"Graph retrieval returned {len(graph_results)} results using method: {method_str}")
            return graph_results
            
        except Exception as e:
            logger.error(f"Error in graph retrieval: {e}", exc_info=True)
            return []
    
    def _vector_similarity_search(self, query: str) -> List[Dict[str, Any]]:
        """Performs vector similarity search."""
        cursor = None
        try:
            query_embedding = self.embedding_func([query])[0]
            iris_vector_str = f"[{','.join(map(str, query_embedding))}]" # Format for TO_VECTOR
            
            sql_query = f"""
            SELECT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.title,
                SUBSTRING(CAST(d.text_content AS VARCHAR(1000)), 1, 1000) as content, 
                '' as metadata,
                VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) as similarity,
                ROW_NUMBER() OVER (ORDER BY VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) DESC) as rank_position
            FROM RAG.SourceDocuments d
            WHERE d.embedding IS NOT NULL
            ORDER BY similarity DESC
            """
            # Assumes d.embedding is string-like and needs TO_VECTOR
            
            cursor = self.iris_connector.cursor()
            cursor.execute(sql_query, (iris_vector_str, iris_vector_str)) # Pass vector string twice
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'document_id': str(row[0]),
                    'title': str(row[1] or ""),
                    'content': str(row[2] or ""),
                    'metadata': str(row[3] or ""),
                    'similarity': float(row[4]) if row[4] is not None else 0.0,
                    'rank_position': int(row[5]),
                    'method': 'vector'
                })
            
            logger.info(f"Vector similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}", exc_info=True)
            return []
        finally:
            if cursor:
                cursor.close()
    
    def _reciprocal_rank_fusion(self, 
                               ifind_results: List[Dict[str, Any]],
                               graph_results: List[Dict[str, Any]], 
                               vector_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        doc_scores: Dict[str, float] = {}
        doc_info: Dict[str, Dict[str, Any]] = {}
        
        all_results_lists = [
            (ifind_results, self.config['ifind_weight']),
            (graph_results, self.config['graph_weight']),
            (vector_results, self.config['vector_weight'])
        ]

        for results_list, weight in all_results_lists:
            for result_item in results_list:
                doc_id = str(result_item['document_id']) # Ensure consistent ID type
                rank = int(result_item['rank_position'])
                score_contribution = weight / (self.config['rrf_k'] + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    # Store the first encountered version of doc info, prefer vector/graph if available
                    if result_item['method'] != 'ifind_fallback_title' or doc_id not in doc_info :
                         doc_info[doc_id] = result_item
                doc_scores[doc_id] += score_contribution
        
        sorted_docs_by_rrf = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_fused_results = []
        for doc_id_sorted, rrf_score_val in sorted_docs_by_rrf[:self.config['final_results_limit']]:
            fused_result = doc_info[doc_id_sorted].copy() # Start with the stored info
            fused_result['rrf_score'] = rrf_score_val
            # Ensure all essential fields are present
            fused_result['document_id'] = str(fused_result.get('document_id', doc_id_sorted))
            fused_result['title'] = str(fused_result.get('title', ''))
            fused_result['content'] = str(fused_result.get('content', ''))[:1000] # Truncate content
            fused_result['metadata'] = str(fused_result.get('metadata', ''))
            final_fused_results.append(fused_result)
        
        logger.info(f"RRF fusion produced {len(final_fused_results)} final results from "
                   f"{len(doc_scores)} unique documents")
        
        return final_fused_results
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        if kwargs:
            self.update_config(**kwargs)
        
        logger.info(f"Processing hybrid RAG query: '{query}'")
        
        keywords = self._extract_keywords(query)
        
        retrieval_start_time = time.time()
        ifind_results = self._ifind_keyword_search(keywords)
        graph_results = self._graph_retrieval(query)
        vector_results = self._vector_similarity_search(query)
        retrieval_duration = time.time() - retrieval_start_time
        
        fusion_start_time = time.time()
        fused_documents = self._reciprocal_rank_fusion(ifind_results, graph_results, vector_results)
        fusion_duration = time.time() - fusion_start_time
        
        # Prepare context for LLM
        context_for_llm = "\n\n".join([
            f"Source: {doc.get('method', 'Unknown')}, Title: {doc.get('title', 'N/A')}, Score: {doc.get('rrf_score', 0.0):.4f}\nContent: {doc.get('content', '')}" 
            for doc in fused_documents
        ])
        
        generation_start_time = time.time()
        prompt_for_llm = f"""Based on the following retrieved documents, answer the question.
Context:
{context_for_llm}

Question: {query}
Answer:"""
        
        answer = self.llm_func(prompt_for_llm)
        generation_duration = time.time() - generation_start_time
        
        total_duration = time.time() - start_time
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": fused_documents, # These are the fused and ranked documents
            "timings": {
                "total_duration_ms": total_duration * 1000,
                "retrieval_duration_ms": retrieval_duration * 1000,
                "fusion_duration_ms": fusion_duration * 1000,
                "generation_duration_ms": generation_duration * 1000,
            },
            "metadata": {
                "pipeline": "HybridiFindRAG",
                "num_ifind_results": len(ifind_results),
                "num_graph_results": len(graph_results),
                "num_vector_results": len(vector_results),
                "num_fused_results": len(fused_documents)
            }
        }
def create_hybrid_ifind_rag_pipeline(iris_connector: Any, 
                                     embedding_func: Optional[Callable] = None, 
                                     llm_func: Optional[Callable] = None) -> HybridiFindRAGPipeline:
    """
    Factory function to create and return an instance of HybridiFindRAGPipeline.
    """
    logger.info("Creating HybridiFindRAGPipeline instance via factory function.")
    return HybridiFindRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
def run_hybrid_ifind_rag(query: str, **kwargs) -> Dict[str, Any]:
    """
    Helper function to instantiate and run the HybridiFindRAGPipeline using its factory.
    """
    db_conn = None
    try:
        # Assuming get_iris_connection is available from common.utils or common.iris_connector_jdbc
        # The main __init__ of HybridiFindRAGPipeline uses get_embedding_func and get_llm_func by default
        # This helper will rely on those defaults if not overridden via kwargs to create_hybrid_ifind_rag_pipeline
        
        # Need to import get_iris_connection if not already available in this scope
        # For simplicity, assuming it's accessible or the test provides it via kwargs if necessary.
        # If direct instantiation is preferred:
        from common.iris_connector_jdbc import get_iris_connection as get_jdbc_conn
        from common.utils import get_embedding_func, get_llm_func

        db_conn = get_jdbc_conn()
        embedding_func = get_embedding_func()
        llm_func = get_llm_func(provider="stub")

        pipeline = create_hybrid_ifind_rag_pipeline(
            iris_connector=db_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        return pipeline.run(query, **kwargs)
    except Exception as e:
        logger.error(f"Error in run_hybrid_ifind_rag helper: {e}", exc_info=True)
        return {
            "query": query,
            "answer": "Error occurred in HybridiFindRAG pipeline.",
            "retrieved_documents": [],
            "error": str(e)
        }
    finally:
        if db_conn:
            db_conn.close()

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Adjust imports for __main__ execution
    current_dir_main_hybrid = os.path.dirname(os.path.abspath(__file__))
    path_to_src_main_hybrid = os.path.abspath(os.path.join(current_dir_main_hybrid, '../../..'))
    if path_to_src_main_hybrid not in sys.path:
         sys.path.insert(0, path_to_src_main_hybrid)

    # These imports are already correct as they use src.common
    from common.iris_connector_jdbc import get_iris_connection as get_jdbc_conn_main_hybrid
    from common.utils import get_embedding_func as get_embed_fn_main_hybrid, get_llm_func as get_llm_fn_main_hybrid

    db_conn_main_hybrid = None
    try:
         logger.info("Attempting to run HybridiFindRAGPipeline example...")
         db_conn_main_hybrid = get_jdbc_conn_main_hybrid()
         if db_conn_main_hybrid is None:
             raise ConnectionError("Failed to get IRIS connection for HybridiFindRAG demo.")
 
         pipeline = HybridiFindRAGPipeline(
             iris_connector=db_conn_main_hybrid,
             embedding_func=get_embed_fn_main_hybrid(),
             llm_func=get_llm_fn_main_hybrid(provider="stub")
         )
         
         test_query = "What are the latest treatments for Alzheimer's disease involving graph-based drug discovery?"
         logger.info(f"Running HybridiFindRAG pipeline with test query: '{test_query}'")
         
         result = pipeline.run(test_query)
         
         print("\n--- HybridiFindRAG Pipeline Result ---")
         print(f"Query: {result['query']}")
         print(f"Answer: {result['answer']}")
         print(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
         for i, doc_res in enumerate(result['retrieved_documents']):
             print(f"  Doc {i+1}: ID={doc_res.get('document_id', 'N/A')}, RRF Score={doc_res.get('rrf_score', 0.0):.4f}, Method={doc_res.get('method', 'N/A')}")
             print(f"     Title: {doc_res.get('title', 'N/A')[:60]}...")
         print(f"Timings (ms): {result['timings']}")
         print(f"Metadata: {result['metadata']}")
 
    except Exception as e_main_hybrid:
        logger.error(f"Error during HybridiFindRAG demo: {e_main_hybrid}", exc_info=True)
    finally:
         if db_conn_main_hybrid:
             db_conn_main_hybrid.close()
             logger.info("IRIS connection closed for HybridiFindRAG demo.")