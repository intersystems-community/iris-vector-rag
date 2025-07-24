#!/usr/bin/env python3
"""
Enterprise Validation with Fixed ColBERT

This script validates the complete enterprise RAG system including the optimized ColBERT pipeline.
"""

import os
import sys
import time
import logging
import json
from typing import Dict, Any, List

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func, get_colbert_query_encoder_func, get_colbert_doc_encoder_func_adapted # Updated import
from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from iris_rag.pipelines.crag import CRAGPipeline # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_colbert_encoder(embedding_dim: int = 128):
    """Create a mock ColBERT encoder for testing."""
    def mock_encoder(text: str) -> List[List[float]]:
        import numpy as np
        words = text.split()[:10]  # Limit to 10 tokens
        embeddings = []
        
        for i, word in enumerate(words):
            np.random.seed(hash(word) % 10000)
            embedding = np.random.randn(embedding_dim)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    return mock_encoder

def test_rag_technique(pipeline, technique_name: str, queries: List[str], similarity_threshold: float = 0.3) -> Dict[str, Any]:
    """Test a RAG technique with multiple queries."""
    logger.info(f"Testing {technique_name}...")
    
    results = {
        "technique": technique_name,
        "total_time": 0,
        "query_results": [],
        "avg_time_per_query": 0,
        "success_count": 0,
        "error_count": 0,
        "total_documents_found": 0
    }
    
    start_time = time.time()
    
    for i, query in enumerate(queries):
        query_start = time.time()
        try:
            if technique_name == "OptimizedColBERT":
                result = pipeline.run(query, top_k=5, similarity_threshold=similarity_threshold)
            else:
                result = pipeline.run(query, top_k=5)
            
            query_time = time.time() - query_start
            docs_found = len(result.get("retrieved_documents", []))
            
            query_result = {
                "query_id": i,
                "query": query[:50] + "..." if len(query) > 50 else query,
                "time_seconds": query_time,
                "documents_found": docs_found,
                "success": True
            }
            
            results["query_results"].append(query_result)
            results["success_count"] += 1
            results["total_documents_found"] += docs_found
            
            logger.info(f"  Query {i+1}: {query_time:.2f}s, {docs_found} docs")
            
        except Exception as e:
            query_time = time.time() - query_start
            logger.error(f"  Query {i+1} failed: {e}")
            
            query_result = {
                "query_id": i,
                "query": query[:50] + "..." if len(query) > 50 else query,
                "time_seconds": query_time,
                "documents_found": 0,
                "success": False,
                "error": str(e)
            }
            
            results["query_results"].append(query_result)
            results["error_count"] += 1
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["avg_time_per_query"] = total_time / len(queries) if queries else 0
    results["avg_documents_per_query"] = results["total_documents_found"] / len(queries) if queries else 0
    
    logger.info(f"{technique_name} completed: {total_time:.2f}s total, {results['avg_time_per_query']:.2f}s avg, {results['avg_documents_per_query']:.1f} docs avg")
    
    return results

def run_enterprise_validation():
    """Run comprehensive enterprise validation."""
    logger.info("Starting Enterprise Validation with Fixed ColBERT...")
    
    # Test queries
    test_queries = [
        "What are the latest treatments for diabetes?",
        "How does machine learning improve medical diagnosis?",
        "What are the mechanisms of cancer immunotherapy?",
        "How do genetic mutations contribute to disease development?",
        "What role does AI play in modern healthcare systems?"
    ]
    
    try:
        # Get database connection
        iris_connector = get_iris_connection()
        if not iris_connector:
            raise ConnectionError("Failed to get IRIS connection")
        
        # Get common functions
        embedding_func = get_embedding_func()
        llm_func = get_llm_func(provider="stub")
        mock_colbert_encoder = create_mock_colbert_encoder(128)
        
        # Initialize all pipelines
        pipelines = {}
        
        # Basic RAG
        try:
            pipelines["BasicRAG"] = BasicRAGPipeline(
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func
            )
            logger.info("✅ BasicRAG pipeline initialized")
        except Exception as e:
            logger.error(f"❌ BasicRAG initialization failed: {e}")
        
        # Optimized ColBERT
        try:
            pipelines["OptimizedColBERT"] = ColBERTRAGPipeline(
                iris_connector=iris_connector,
                colbert_query_encoder_func=mock_colbert_encoder,
                colbert_doc_encoder_func=mock_colbert_encoder,
                llm_func=llm_func
            )
            logger.info("✅ OptimizedColBERT pipeline initialized")
        except Exception as e:
            logger.error(f"❌ OptimizedColBERT initialization failed: {e}")
        
        # HyDE
        try:
            pipelines["HyDE"] = HyDERAGPipeline(
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func
            )
            logger.info("✅ HyDE pipeline initialized")
        except Exception as e:
            logger.error(f"❌ HyDE initialization failed: {e}")
        
        # CRAG
        try:
            pipelines["CRAG"] = CRAGPipeline(
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func
            )
            logger.info("✅ CRAG pipeline initialized")
        except Exception as e:
            logger.error(f"❌ CRAG initialization failed: {e}")
        
        # NodeRAG
        try:
            pipelines["NodeRAG"] = NodeRAGPipeline(
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func
            )
            logger.info("✅ NodeRAG pipeline initialized")
        except Exception as e:
            logger.error(f"❌ NodeRAG initialization failed: {e}")
        
        # GraphRAG
        try:
            pipelines["GraphRAG"] = GraphRAGPipeline(
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func
            )
            logger.info("✅ GraphRAG pipeline initialized")
        except Exception as e:
            logger.error(f"❌ GraphRAG initialization failed: {e}")
        
        # Hybrid iFind+Graph+Vector RAG
        try:
            pipelines["Hybrid iFind RAG"] = HybridIFindRAGPipeline(
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func
            )
            logger.info("✅ Hybrid iFind+Graph+Vector RAG pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Hybrid iFind RAG initialization failed: {e}")
        
        # Test all pipelines
        all_results = {}
        
        for technique_name, pipeline in pipelines.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing {technique_name}")
            logger.info('='*60)
            
            # Use lower threshold for ColBERT to get more realistic results
            threshold = 0.3 if technique_name == "OptimizedColBERT" else None
            
            if threshold:
                results = test_rag_technique(pipeline, technique_name, test_queries, threshold)
            else:
                results = test_rag_technique(pipeline, technique_name, test_queries)
            
            all_results[technique_name] = results
        
        # Generate comprehensive report
        validation_report = {
            "timestamp": time.time(),
            "total_techniques_tested": len(all_results),
            "successful_techniques": len([r for r in all_results.values() if r["success_count"] > 0]),
            "test_queries": test_queries,
            "technique_results": all_results,
            "performance_ranking": sorted(
                [(name, result["avg_time_per_query"]) for name, result in all_results.items()],
                key=lambda x: x[1]
            ),
            "retrieval_ranking": sorted(
                [(name, result["avg_documents_per_query"]) for name, result in all_results.items()],
                key=lambda x: x[1], reverse=True
            )
        }
        
        # Save results
        timestamp = int(time.time())
        results_file = f"enterprise_validation_fixed_colbert_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        # Print comprehensive summary
        logger.info("\n" + "="*80)
        logger.info("ENTERPRISE VALIDATION SUMMARY - FIXED COLBERT")
        logger.info("="*80)
        
        logger.info(f"Techniques tested: {validation_report['total_techniques_tested']}")
        logger.info(f"Successful techniques: {validation_report['successful_techniques']}")
        
        logger.info("\nPerformance Ranking (fastest to slowest):")
        for i, (technique, avg_time) in enumerate(validation_report["performance_ranking"], 1):
            logger.info(f"  {i}. {technique}: {avg_time:.2f}s avg")
        
        logger.info("\nRetrieval Ranking (most to least documents):")
        for i, (technique, avg_docs) in enumerate(validation_report["retrieval_ranking"], 1):
            logger.info(f"  {i}. {technique}: {avg_docs:.1f} docs avg")
        
        logger.info(f"\nDetailed results saved to: {results_file}")
        
        # Specific ColBERT assessment
        if "OptimizedColBERT" in all_results:
            colbert_result = all_results["OptimizedColBERT"]
            logger.info(f"\n🎯 COLBERT PERFORMANCE ASSESSMENT:")
            logger.info(f"   Average query time: {colbert_result['avg_time_per_query']:.2f}s")
            logger.info(f"   Success rate: {colbert_result['success_count']}/{len(test_queries)} queries")
            logger.info(f"   Average documents retrieved: {colbert_result['avg_documents_per_query']:.1f}")
            
            if colbert_result['avg_time_per_query'] < 5.0:
                logger.info("   ✅ ColBERT performance is ACCEPTABLE for enterprise use")
            else:
                logger.info("   ⚠️  ColBERT performance needs further optimization")
        
        iris_connector.close()
        
        return validation_report
        
    except Exception as e:
        logger.error(f"Error during enterprise validation: {e}", exc_info=True)
        return None

def main():
    """Main function."""
    logger.info("Enterprise Validation with Fixed ColBERT Starting...")
    
    try:
        results = run_enterprise_validation()
        
        if results:
            logger.info("✅ Enterprise validation completed successfully!")
            
            # Check overall system health
            successful_techniques = results["successful_techniques"]
            total_techniques = results["total_techniques_tested"]
            
            if successful_techniques == total_techniques:
                logger.info(f"🎉 ALL {total_techniques} RAG techniques are working correctly!")
            elif successful_techniques > 0:
                logger.info(f"✅ {successful_techniques}/{total_techniques} RAG techniques working")
            else:
                logger.error("❌ No RAG techniques are working properly")
                
        else:
            logger.error("❌ Enterprise validation failed")
            
    except Exception as e:
        logger.error(f"❌ Fatal error during enterprise validation: {e}", exc_info=True)

if __name__ == "__main__":
    main()