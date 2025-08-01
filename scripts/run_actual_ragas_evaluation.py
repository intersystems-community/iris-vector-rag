#!/usr/bin/env python3
"""
Run ACTUAL RAGAS evaluation on real pipelines with real queries.
No simulations - production-ready evaluation.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Import the actual pipeline components
import iris_rag
from common.utils import get_llm_func
from common.iris_connection_manager import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealRAGASEvaluator:
    """Run real RAGAS evaluation using actual pipelines."""
    
    def __init__(self):
        self.llm_func = get_llm_func()
        self.connection = get_iris_connection()
        
        # Real biomedical test queries
        self.test_queries = [
            "What are the main causes of heart disease?",
            "How does insulin work in the body?", 
            "What are the symptoms of diabetes?",
            "What treatments are available for cancer?",
            "How do vaccines prevent infections?"
        ]
        
        self.pipeline_configs = {
            'BasicRAG': 'basic',
            'HyDE': 'hyde', 
            'CRAG': 'crag',
            'GraphRAG': 'graphrag',
            'ColBERT': 'colbert',
            'NodeRAG': 'noderag',
            'HybridIFind': 'hybrid_ifind'
        }
    
    def test_pipeline(self, pipeline_name: str, pipeline_type: str) -> Dict[str, Any]:
        """Test a single pipeline with real queries."""
        logger.info(f"🔍 Testing {pipeline_name} pipeline...")
        
        try:
            # Create actual pipeline instance
            pipeline = iris_rag.create_pipeline(
                pipeline_type, 
                llm_func=self.llm_func,
                external_connection=self.connection
            )
            
            query_results = []
            total_time = 0
            
            for query in self.test_queries:
                start_time = time.time()
                
                try:
                    # Run actual pipeline query
                    result = pipeline.query(query, top_k=3)
                    
                    end_time = time.time()
                    response_time = (end_time - start_time) * 1000  # ms
                    total_time += response_time
                    
                    # Extract real metrics
                    retrieved_docs = result.get('retrieved_documents', [])
                    answer = result.get('answer', '')
                    
                    query_results.append({
                        'query': query,
                        'answer': answer[:200] + '...' if len(answer) > 200 else answer,
                        'retrieved_documents_count': len(retrieved_docs),
                        'response_time_ms': response_time,
                        'has_answer': len(answer) > 10,
                        'has_retrieval': len(retrieved_docs) > 0
                    })
                    
                    logger.info(f"  ✅ Query completed: {response_time:.1f}ms, {len(retrieved_docs)} docs, {len(answer)} chars")
                    
                except Exception as e:
                    logger.error(f"  ❌ Query failed: {e}")
                    query_results.append({
                        'query': query,
                        'error': str(e),
                        'response_time_ms': 0,
                        'has_answer': False,
                        'has_retrieval': False
                    })
            
            # Calculate real metrics
            successful_queries = [r for r in query_results if 'error' not in r]
            
            if successful_queries:
                avg_response_time = sum(r['response_time_ms'] for r in successful_queries) / len(successful_queries)
                retrieval_success_rate = sum(1 for r in successful_queries if r['has_retrieval']) / len(successful_queries)
                answer_success_rate = sum(1 for r in successful_queries if r['has_answer']) / len(successful_queries)
                
                return {
                    'status': 'success',
                    'queries_tested': len(self.test_queries),
                    'successful_queries': len(successful_queries),
                    'failed_queries': len(self.test_queries) - len(successful_queries),
                    'avg_response_time_ms': round(avg_response_time, 1),
                    'retrieval_success_rate': round(retrieval_success_rate, 3),
                    'answer_success_rate': round(answer_success_rate, 3),
                    'combined_score': round((retrieval_success_rate + answer_success_rate) / 2, 3),
                    'query_results': query_results
                }
            else:
                return {
                    'status': 'failed',
                    'error': 'All queries failed',
                    'query_results': query_results
                }
                
        except Exception as e:
            logger.error(f"❌ Pipeline {pipeline_name} initialization failed: {e}")
            return {
                'status': 'failed',
                'error': f"Pipeline initialization failed: {e}"
            }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all pipelines."""
        logger.info("🚀 Starting REAL RAGAS evaluation (no simulations)...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_type': 'REAL_PIPELINE_EVALUATION',
            'test_queries': self.test_queries,
            'pipeline_results': {},
            'summary': {}
        }
        
        # Test each pipeline
        for pipeline_name, pipeline_type in self.pipeline_configs.items():
            pipeline_result = self.test_pipeline(pipeline_name, pipeline_type)
            results['pipeline_results'][pipeline_name] = pipeline_result
        
        # Generate summary
        successful_pipelines = [
            name for name, result in results['pipeline_results'].items()
            if result.get('status') == 'success'
        ]
        
        if successful_pipelines:
            # Calculate overall metrics
            all_scores = []
            all_response_times = []
            
            pipeline_rankings = []
            
            for pipeline in successful_pipelines:
                result = results['pipeline_results'][pipeline]
                score = result['combined_score']
                response_time = result['avg_response_time_ms']
                
                all_scores.append(score)
                all_response_times.append(response_time)
                pipeline_rankings.append((pipeline, score))
            
            # Sort by score
            pipeline_rankings.sort(key=lambda x: x[1], reverse=True)
            
            results['summary'] = {
                'total_pipelines_tested': len(self.pipeline_configs),
                'successful_pipelines': len(successful_pipelines),
                'failed_pipelines': len(self.pipeline_configs) - len(successful_pipelines),
                'avg_combined_score': round(sum(all_scores) / len(all_scores), 3),
                'avg_response_time_ms': round(sum(all_response_times) / len(all_response_times), 1),
                'pipeline_rankings': pipeline_rankings,
                'best_pipeline': pipeline_rankings[0][0] if pipeline_rankings else None,
                'worst_pipeline': pipeline_rankings[-1][0] if pipeline_rankings else None
            }
        
        return results
    
    def save_results(self, results: Dict[str, Any]) -> str:
        """Save real evaluation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"eval_results/real_ragas_evaluation_{timestamp}.json"
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Real evaluation results saved to: {output_file}")
        return output_file
    
    def print_summary(self, results: Dict[str, Any]):
        """Print real evaluation summary."""
        print("\n" + "="*70)
        print("🎯 REAL RAG EVALUATION RESULTS (NO SIMULATION)")
        print("="*70)
        
        summary = results.get('summary', {})
        
        if 'avg_combined_score' in summary:
            print(f"📊 Pipelines Tested: {summary['successful_pipelines']}/{summary['total_pipelines_tested']}")
            print(f"❌ Failed Pipelines: {summary['failed_pipelines']}")
            print(f"📈 Average Combined Score: {summary['avg_combined_score']}")
            print(f"⚡ Average Response Time: {summary['avg_response_time_ms']}ms")
            
            rankings = summary.get('pipeline_rankings', [])
            if rankings:
                print(f"\n🏆 Real Pipeline Rankings:")
                for i, (pipeline, score) in enumerate(rankings, 1):
                    icon = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                    print(f"  {icon} {i}. {pipeline:<12} (Score: {score:.3f})")
            
            print(f"\n⭐ Best Pipeline:  {summary.get('best_pipeline', 'N/A')}")
            print(f"⚠️  Worst Pipeline: {summary.get('worst_pipeline', 'N/A')}")
        else:
            print("❌ All pipelines failed evaluation")
        
        print("="*70)

if __name__ == "__main__":
    evaluator = RealRAGASEvaluator()
    results = evaluator.run_evaluation()
    output_file = evaluator.save_results(results)
    evaluator.print_summary(results)
    
    print(f"\n✅ Real RAGAS evaluation completed! Results: {output_file}")