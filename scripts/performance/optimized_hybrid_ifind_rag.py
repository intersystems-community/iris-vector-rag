#!/usr/bin/env python3
"""
URGENT: Optimized HybridiFindRAG with Parallelization and Caching
Implements concurrent execution and caching for dramatic performance improvements.
"""

import sys
import time
import threading
import concurrent.futures
from typing import Dict, List, Any, Optional
import hashlib
import json
from functools import lru_cache

sys.path.insert(0, '.')

from hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

class OptimizedHybridiFindRAGPipeline(HybridiFindRAGPipeline):
    """
    Optimized HybridiFindRAG with:
    1. Query Parallelization: Concurrent execution of retrieval methods
    2. Embedding Caching: Cache embeddings to avoid recomputation
    3. Result Caching: Cache query results for repeated queries
    4. Optimized Vector Search: Enhanced similarity filtering
    """
    
    def __init__(self, iris_connector, embedding_func=None, llm_func=None):
        super().__init__(iris_connector, embedding_func, llm_func)
        
        # Caching setup
        self.embedding_cache = {}
        self.result_cache = {}
        self.cache_max_size = 1000
        
        # Performance tracking
        self.performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'parallel_executions': 0,
            'total_queries': 0
        }
        
        print("🚀 Initialized OptimizedHybridiFindRAG with parallelization and caching")
    
    def _get_query_hash(self, query: str, top_k: int) -> str:
        """Generate hash for query caching"""
        query_data = f"{query}_{top_k}_{self.config['max_results_per_method']}"
        return hashlib.md5(query_data.encode()).hexdigest()
    
    @lru_cache(maxsize=500)
    def _cached_embedding(self, query: str) -> List[float]:
        """Cache embeddings to avoid recomputation"""
        if query in self.embedding_cache:
            return self.embedding_cache[query]
        
        embedding = self.embedding_func([query])[0]
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        # Cache management
        if len(self.embedding_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        
        self.embedding_cache[query] = embedding
        return embedding
    
    def _parallel_vector_similarity_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Optimized vector similarity search with enhanced filtering
        """
        try:
            # Use cached embedding
            query_embedding = self._cached_embedding(query)
            embedding_str = ','.join(map(str, query_embedding))
            
            # Enhanced similarity threshold for better performance
            similarity_threshold = 0.15  # Slightly higher for better filtering
            
            query_sql = f"""
            SELECT TOP {self.config['max_results_per_method']}
                d.doc_id as document_id,
                d.doc_id as title,
                d.text_content as content,
                '' as metadata,
                VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) as similarity_score
            FROM RAG.SourceDocuments d
            WHERE d.embedding IS NOT NULL
              AND LENGTH(d.embedding) > 1000
              AND VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?)) > ?
            ORDER BY similarity_score DESC
            """
            
            cursor = self.iris_connector.cursor()
            cursor.execute(query_sql, [embedding_str, embedding_str, similarity_threshold])
            results = []
            
            for i, row in enumerate(cursor.fetchall(), 1):
                results.append({
                    'document_id': row[0],
                    'title': row[1],
                    'content': row[2],
                    'metadata': row[3],
                    'similarity_score': float(row[4]) if row[4] else 0.0,
                    'rank_position': i,
                    'method': 'vector'
                })
            
            cursor.close()
            return results
            
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return []
    
    def _parallel_retrieval(self, query: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute all retrieval methods in parallel for maximum performance
        """
        results = {
            'ifind': [],
            'graph': [],
            'vector': []
        }
        
        def ifind_task():
            try:
                keywords = self._extract_keywords(query)
                return self._ifind_keyword_search(keywords)
            except Exception as e:
                print(f"❌ iFind error: {e}")
                return []
        
        def graph_task():
            try:
                return self._graph_retrieval(query)
            except Exception as e:
                print(f"❌ Graph error: {e}")
                return []
        
        def vector_task():
            try:
                return self._parallel_vector_similarity_search(query)
            except Exception as e:
                print(f"❌ Vector error: {e}")
                return []
        
        # Execute all retrieval methods concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_ifind = executor.submit(ifind_task)
            future_graph = executor.submit(graph_task)
            future_vector = executor.submit(vector_task)
            
            # Collect results
            results['ifind'] = future_ifind.result()
            results['graph'] = future_graph.result()
            results['vector'] = future_vector.result()
        
        self.performance_stats['parallel_executions'] += 1
        return results
    
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Optimized run method with caching and parallelization
        """
        start_time = time.time()
        self.performance_stats['total_queries'] += 1
        
        # Check cache first
        query_hash = self._get_query_hash(query_text, top_k)
        if query_hash in self.result_cache:
            self.performance_stats['cache_hits'] += 1
            cached_result = self.result_cache[query_hash].copy()
            cached_result['execution_time'] = time.time() - start_time
            cached_result['cache_hit'] = True
            print(f"🎯 Cache hit! Returning cached result in {cached_result['execution_time']:.3f}s")
            return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        # Parallel retrieval
        print("🚀 Executing parallel retrieval...")
        retrieval_start = time.time()
        parallel_results = self._parallel_retrieval(query_text)
        retrieval_time = time.time() - retrieval_start
        
        # Fusion
        fusion_start = time.time()
        fused_results = self._reciprocal_rank_fusion(
            parallel_results['ifind'],
            parallel_results['graph'],
            parallel_results['vector']
        )
        fusion_time = time.time() - fusion_start
        
        # Select top results
        final_results = fused_results[:top_k]
        
        # Generate answer
        llm_start = time.time()
        answer = self.generate_response(query_text, final_results)
        llm_time = time.time() - llm_start
        
        total_time = time.time() - start_time
        
        # Prepare result
        result = {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": final_results,
            "method": "optimized_hybrid_ifind_rag",
            "execution_time": total_time,
            "performance_breakdown": {
                "retrieval_time": retrieval_time,
                "fusion_time": fusion_time,
                "llm_time": llm_time
            },
            "retrieval_stats": {
                "ifind_results": len(parallel_results['ifind']),
                "graph_results": len(parallel_results['graph']),
                "vector_results": len(parallel_results['vector']),
                "fused_results": len(fused_results),
                "final_results": len(final_results)
            },
            "cache_hit": False
        }
        
        # Cache result
        if len(self.result_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.result_cache))
            del self.result_cache[oldest_key]
        
        self.result_cache[query_hash] = result.copy()
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        total_queries = self.performance_stats['total_queries']
        cache_hit_rate = (self.performance_stats['cache_hits'] / total_queries * 100) if total_queries > 0 else 0
        
        return {
            **self.performance_stats,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'embedding_cache_size': len(self.embedding_cache),
            'result_cache_size': len(self.result_cache)
        }

def test_optimized_pipeline():
    """Test the optimized pipeline"""
    print("🧪 Testing Optimized HybridiFindRAG Pipeline...")
    
    # Initialize
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    pipeline = OptimizedHybridiFindRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test queries
    test_queries = [
        'What are the symptoms of diabetes?',
        'How is diabetes treated?',
        'What causes diabetes?',
        'What are the symptoms of diabetes?',  # Repeat for cache test
    ]
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        
        start_time = time.time()
        result = pipeline.run(query, top_k=3)
        end_time = time.time()
        
        print(f"📊 Execution time: {result['execution_time']:.2f}s")
        print(f"📊 Cache hit: {result.get('cache_hit', False)}")
        print(f"📊 Retrieved documents: {len(result['retrieved_documents'])}")
        
        results.append(result)
    
    # Performance summary
    print(f"\n📈 PERFORMANCE SUMMARY:")
    stats = pipeline.get_performance_stats()
    for key, value in stats.items():
        print(f"📊 {key}: {value}")
    
    # Calculate average performance
    non_cached_times = [r['execution_time'] for r in results if not r.get('cache_hit', False)]
    cached_times = [r['execution_time'] for r in results if r.get('cache_hit', False)]
    
    if non_cached_times:
        avg_non_cached = sum(non_cached_times) / len(non_cached_times)
        print(f"📊 Average non-cached time: {avg_non_cached:.2f}s")
    
    if cached_times:
        avg_cached = sum(cached_times) / len(cached_times)
        print(f"📊 Average cached time: {avg_cached:.3f}s")
        print(f"📊 Cache speedup: {avg_non_cached/avg_cached:.1f}x faster")

if __name__ == "__main__":
    test_optimized_pipeline()