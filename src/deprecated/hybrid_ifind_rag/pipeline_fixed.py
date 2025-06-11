"""
Hybrid iFind+Graph+Vector RAG Pipeline Implementation (Fixed/Alternative Version)

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
import os # Added
import sys # Added

# Add the project root directory to Python path
# Assuming this file is in src/deprecated/hybrid_ifind_rag/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.utils import get_embedding_func, get_llm_func # get_iris_connector removed
# Assuming GraphRAGPipeline will be in src/experimental/graphrag/
from src.experimental.graphrag.pipeline import GraphRAGPipeline


logger = logging.getLogger(__name__)


class HybridiFindRAGPipeline: # Name conflict with the one moved to experimental
    """
    Hybrid RAG pipeline combining iFind keyword search, graph retrieval, 
    and vector similarity search with reciprocal rank fusion. (Fixed/Alternative Version)
    """
    
    def __init__(self, iris_connector: Any, embedding_func: Optional[Callable]=None, llm_func: Optional[Callable]=None): # Type hint for iris_connector
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
        
        logger.info("Initialized Hybrid iFind+Graph+Vector RAG Pipeline (Fixed/Alternative Version)")
    
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
        """
        Perform keyword search using SUBSTRING on stream fields and title search.
        Since IRIS doesn't support LIKE on STREAM fields, we use a combination of:
        1. Title search (VARCHAR field)
        2. SUBSTRING search on first 5000 chars of text_content
        """
        if not keywords:
            return []
        
        cursor = None # Initialize cursor
        try:
            conditions = []
            params = []
            
            for keyword in keywords[:5]: 
                conditions.append("UPPER(d.title) LIKE UPPER(?)")
                params.append(f"%{keyword}%")
                conditions.append("POSITION(UPPER(?), UPPER(SUBSTRING(CAST(d.text_content AS VARCHAR(5000)), 1, 5000))) > 0") # Ensure text_content is cast
                params.append(keyword)
            
            if not conditions: return []
            where_clause = " OR ".join(conditions)
            
            query_sql = f"""
            SELECT DISTINCT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.title as title,
                SUBSTRING(CAST(d.text_content AS VARCHAR(1000)), 1, 1000) as content, -- Cast for SUBSTRING
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
                    'method': 'ifind'
                })
            
            logger.info(f"iFind keyword search (fixed version) found {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword search (fixed version): {e}", exc_info=True)
            return self._title_only_search(keywords) # Fallback to title-only
        finally:
            if cursor:
                cursor.close()
    
    def _title_only_search(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fallback to title-only search"""
        if not keywords:
            return []
        cursor = None # Initialize
        try:
            keyword_conditions = []
            params = []
            
            for keyword in keywords[:5]:
                keyword_conditions.append("UPPER(d.title) LIKE UPPER(?)")
                params.append(f"%{keyword}%")
            
            if not keyword_conditions: return []
            where_clause = " OR ".join(keyword_conditions)
            
            query_sql = f"""
            SELECT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.title as title,
                SUBSTRING(CAST(d.text_content AS VARCHAR(500)), 1, 500) as content, -- Cast for SUBSTRING
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
                    'method': 'ifind_title_only' # Clarify method
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in title search (fixed version): {e}", exc_info=True)
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
                self.llm_func
            )
            
            documents_objects, method_str = graphrag_pipeline.retrieve_documents_via_kg(query, top_k=self.config['max_results_per_method'])
            
            graph_results = []
            for i, doc_obj in enumerate(documents_objects):
                graph_results.append({
                    'document_id': str(doc_obj.id),
                    'title': str(doc_obj.metadata.get('title', '') if doc_obj.metadata else ''),
                    'content': str(doc_obj.content or "")[:1000],
                    'metadata': json.dumps(doc_obj.metadata or {}),
                    'rank_position': i + 1,
                    'method': 'graph'
                })
            
            logger.info(f"Graph retrieval (fixed version) returned {len(graph_results)} results using method: {method_str}")
            return graph_results
            
        except Exception as e:
            logger.error(f"Error in graph retrieval (fixed version): {e}", exc_info=True)
            return []
    
    def _vector_similarity_search(self, query: str) -> List[Dict[str, Any]]:
        """Performs vector similarity search."""
        cursor = None # Initialize
        try:
            query_embedding = self.embedding_func([query])[0]
            iris_vector_str = f"[{','.join(map(str, query_embedding))}]"
            
            sql_query = f"""
            SELECT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.title,
                SUBSTRING(CAST(d.text_content AS VARCHAR(1000)), 1, 1000) as content, -- Cast for SUBSTRING
                '' as metadata,
                VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) as similarity,
                ROW_NUMBER() OVER (ORDER BY VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) DESC) as rank_position
            FROM RAG.SourceDocuments d
            WHERE d.embedding IS NOT NULL
            ORDER BY similarity DESC
            """
            
            cursor = self.iris_connector.cursor()
            cursor.execute(sql_query, (iris_vector_str, iris_vector_str))
            
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
            
            logger.info(f"Vector similarity search (fixed version) returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in vector similarity search (fixed version): {e}", exc_info=True)
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
                doc_id = str(result_item['document_id'])
                rank = int(result_item['rank_position'])
                score_contribution = weight / (self.config['rrf_k'] + rank)
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                    doc_info[doc_id] = result_item
                doc_scores[doc_id] += score_contribution
        
        sorted_docs_by_rrf = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        
        final_fused_results = []
        for doc_id_sorted, rrf_score_val in sorted_docs_by_rrf[:self.config['final_results_limit']]:
            fused_result = doc_info[doc_id_sorted].copy()
            fused_result['rrf_score'] = rrf_score_val
            fused_result['document_id'] = str(fused_result.get('document_id', doc_id_sorted))
            fused_result['title'] = str(fused_result.get('title', ''))
            fused_result['content'] = str(fused_result.get('content', ''))[:1000]
            fused_result['metadata'] = str(fused_result.get('metadata', ''))
            final_fused_results.append(fused_result)
        
        logger.info(f"RRF fusion (fixed version) produced {len(final_fused_results)} final results")
        return final_fused_results
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        if kwargs:
            self.update_config(**kwargs)
        
        logger.info(f"Processing hybrid RAG query (fixed version): '{query}'")
        
        keywords = self._extract_keywords(query)
        
        retrieval_start_time = time.time()
        ifind_results = self._ifind_keyword_search(keywords)
        graph_results = self._graph_retrieval(query)
        vector_results = self._vector_similarity_search(query)
        retrieval_duration = time.time() - retrieval_start_time
        
        fusion_start_time = time.time()
        fused_documents = self._reciprocal_rank_fusion(ifind_results, graph_results, vector_results)
        fusion_duration = time.time() - fusion_start_time
        
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
            "retrieved_documents": fused_documents,
            "timings": {
                "total_duration_ms": total_duration * 1000,
                "retrieval_duration_ms": retrieval_duration * 1000,
                "fusion_duration_ms": fusion_duration * 1000,
                "generation_duration_ms": generation_duration * 1000,
            },
            "metadata": {
                "pipeline": "HybridiFindRAG_Fixed",
                "num_ifind_results": len(ifind_results),
                "num_graph_results": len(graph_results),
                "num_vector_results": len(vector_results),
                "num_fused_results": len(fused_documents)
            }
        }

# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Adjust imports for __main__ execution
    current_dir_main_hybrid_fixed = os.path.dirname(os.path.abspath(__file__))
    path_to_src_main_hybrid_fixed = os.path.abspath(os.path.join(current_dir_main_hybrid_fixed, '../../..'))
    if path_to_src_main_hybrid_fixed not in sys.path:
         sys.path.insert(0, path_to_src_main_hybrid_fixed)

    from common.iris_connector import get_iris_connection as get_conn_main_hybrid_fixed # Use default connector
    from common.utils import get_embedding_func as get_embed_fn_main_hybrid_fixed, get_llm_func as get_llm_fn_main_hybrid_fixed

    db_conn_main_hybrid_fixed = None
    try:
        logger.info("Attempting to run HybridiFindRAGPipeline (Fixed Version) example...")
        db_conn_main_hybrid_fixed = get_conn_main_hybrid_fixed()
        if db_conn_main_hybrid_fixed is None:
            raise ConnectionError("Failed to get IRIS connection for HybridiFindRAG (Fixed Version) demo.")

        pipeline = HybridiFindRAGPipeline( # This will use the class defined in this file
            iris_connector=db_conn_main_hybrid_fixed,
            embedding_func=get_embed_fn_main_hybrid_fixed(),
            llm_func=get_llm_fn_main_hybrid_fixed(provider="stub")
        )
        
        test_query = "Latest advancements in AI for medical diagnosis"
        logger.info(f"Running HybridiFindRAG pipeline (Fixed Version) with test query: '{test_query}'")
        
        result = pipeline.run(test_query)
        
        print("\n--- HybridiFindRAG Pipeline (Fixed Version) Result ---")
        print(f"Query: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Retrieved Documents ({len(result['retrieved_documents'])}):")
        for i, doc_res in enumerate(result['retrieved_documents']):
            print(f"  Doc {i+1}: ID={doc_res.get('document_id', 'N/A')}, RRF Score={doc_res.get('rrf_score', 0.0):.4f}, Method={doc_res.get('method', 'N/A')}")
            print(f"     Title: {doc_res.get('title', 'N/A')[:60]}...")
        print(f"Timings (ms): {result['timings']}")
        print(f"Metadata: {result['metadata']}")

    except Exception as e_main_hybrid_fixed:
        logger.error(f"Error during HybridiFindRAG (Fixed Version) demo: {e_main_hybrid_fixed}", exc_info=True)
    finally:
        if db_conn_main_hybrid_fixed:
            db_conn_main_hybrid_fixed.close()
            logger.info("IRIS connection closed for HybridiFindRAG (Fixed Version) demo.")