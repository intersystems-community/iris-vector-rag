#!/usr/bin/env python3
"""
Simple Pipeline Evaluation Script
Focuses on performance metrics and basic quality assessment without RAGAS dependencies
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline modules
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.colbert import ColBERTRAGPipeline
from iris_rag.pipelines.noderag import NodeRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePipelineEvaluator:
    """Simple pipeline evaluator focused on performance and basic quality metrics"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"simple_evaluation_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Test queries for evaluation
        self.test_queries = [
            "What are the effects of metformin on type 2 diabetes?",
            "How does SGLT2 inhibition affect kidney function?", 
            "What is the mechanism of action of GLP-1 receptor agonists?",
            "What are the cardiovascular benefits of SGLT2 inhibitors?",
            "How do statins prevent cardiovascular disease?",
            "What are the mechanisms of antibiotic resistance?",
            "How do biologics treat rheumatoid arthritis?",
            "What is the mechanism of action of levodopa in Parkinson's disease?",
            "How do glucocorticoids suppress inflammation?",
            "What is the role of ACE inhibitors in heart failure?"
        ]
        
        # Initialize pipelines
        self.pipelines = self._initialize_pipelines()
        
    def _initialize_pipelines(self) -> Dict[str, Any]:
        """Initialize all RAG pipelines"""
        pipelines = {}
        
        # Import connection and config managers
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        
        # Initialize managers
        connection_manager = ConnectionManager()
        config_manager = ConfigurationManager()
        
        try:
            pipelines['basic'] = BasicRAGPipeline(connection_manager, config_manager)
            logger.info("✅ Basic RAG pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Basic RAG: {e}")
            
        try:
            pipelines['hyde'] = HyDERAGPipeline(connection_manager, config_manager)
            logger.info("✅ HyDE pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HyDE: {e}")
            
        try:
            pipelines['crag'] = CRAGPipeline(connection_manager, config_manager)
            logger.info("✅ CRAG pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize CRAG: {e}")
            
        try:
            pipelines['colbert'] = ColBERTRAGPipeline(connection_manager, config_manager)
            logger.info("✅ ColBERT pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ColBERT: {e}")
            
        try:
            pipelines['noderag'] = NodeRAGPipeline(connection_manager, config_manager)
            logger.info("✅ NodeRAG pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize NodeRAG: {e}")
            
        try:
            pipelines['graphrag'] = GraphRAGPipeline(connection_manager, config_manager)
            logger.info("✅ GraphRAG pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize GraphRAG: {e}")
            
        try:
            pipelines['hybrid_ifind'] = HybridIFindRAGPipeline(connection_manager, config_manager)
            logger.info("✅ Hybrid IFind pipeline initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Hybrid IFind: {e}")
            
        return pipelines
    
    def _execute_pipeline(self, pipeline_name: str, pipeline: Any, query: str) -> Dict[str, Any]:
        """Execute a single pipeline query"""
        try:
            start_time = time.time()
            
            # Try different method names
            if hasattr(pipeline, 'run'):
                result = pipeline.run(query)
            elif hasattr(pipeline, 'query'):
                result = pipeline.query(query)
            elif hasattr(pipeline, 'execute'):
                result = pipeline.execute(query)
            else:
                # Try calling the pipeline directly
                result = pipeline(query)
            
            execution_time = time.time() - start_time
            
            # Standardize result format
            if isinstance(result, dict):
                answer = result.get('answer', str(result))
                contexts = result.get('retrieved_documents', result.get('contexts', []))
                docs_count = len(contexts) if contexts else 0
            else:
                answer = str(result)
                contexts = []
                docs_count = 0
            
            # Basic quality metrics
            answer_length = len(answer) if answer else 0
            has_answer = bool(answer and answer.strip() and not answer.startswith('Error'))
            
            return {
                'answer': answer,
                'contexts': contexts,
                'execution_time': execution_time,
                'success': True,
                'error': None,
                'docs_retrieved': docs_count,
                'answer_length': answer_length,
                'has_valid_answer': has_answer
            }
            
        except Exception as e:
            logger.error(f"❌ Error executing {pipeline_name}: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'contexts': [],
                'execution_time': 0,
                'success': False,
                'error': str(e),
                'docs_retrieved': 0,
                'answer_length': 0,
                'has_valid_answer': False
            }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete pipeline evaluation"""
        logger.info("🚀 Starting Simple Pipeline Evaluation")
        start_time = time.time()
        
        # Execute all pipelines
        pipeline_results = {}
        performance_metrics = {}
        
        for pipeline_name, pipeline in self.pipelines.items():
            logger.info(f"🔄 Evaluating {pipeline_name} pipeline...")
            
            results = []
            total_time = 0
            total_docs = 0
            total_answer_length = 0
            valid_answers = 0
            
            for i, query in enumerate(self.test_queries):
                logger.info(f"  Query {i+1}/{len(self.test_queries)}: {query[:50]}...")
                result = self._execute_pipeline(pipeline_name, pipeline, query)
                results.append(result)
                
                if result['success']:
                    total_time += result['execution_time']
                    total_docs += result['docs_retrieved']
                    total_answer_length += result['answer_length']
                    if result['has_valid_answer']:
                        valid_answers += 1
            
            pipeline_results[pipeline_name] = results
            
            # Calculate performance metrics
            successful_results = [r for r in results if r['success']]
            performance_metrics[pipeline_name] = {
                'total_queries': len(results),
                'successful_queries': len(successful_results),
                'success_rate': len(successful_results) / len(results) if results else 0,
                'avg_execution_time': total_time / len(successful_results) if successful_results else 0,
                'total_execution_time': total_time,
                'avg_docs_retrieved': total_docs / len(successful_results) if successful_results else 0,
                'avg_answer_length': total_answer_length / len(successful_results) if successful_results else 0,
                'valid_answer_rate': valid_answers / len(results) if results else 0
            }
            
            logger.info(f"✅ {pipeline_name}: {len(successful_results)}/{len(results)} successful, "
                       f"avg time: {performance_metrics[pipeline_name]['avg_execution_time']:.2f}s")
        
        # Compile final results
        final_results = {
            'timestamp': self.timestamp,
            'evaluation_duration': time.time() - start_time,
            'performance_metrics': performance_metrics,
            'pipeline_results': pipeline_results,
            'test_queries': self.test_queries
        }
        
        # Save results
        results_file = os.path.join(self.results_dir, f'simple_evaluation_results_{self.timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(final_results)
        
        logger.info(f"✅ Simple pipeline evaluation completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"📁 Results saved to: {self.results_dir}")
        
        return final_results
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate a summary report"""
        report_file = os.path.join(self.results_dir, f'summary_report_{self.timestamp}.md')
        
        with open(report_file, 'w') as f:
            f.write(f"# Simple Pipeline Evaluation Report\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Performance Summary\n\n")
            f.write(f"| Pipeline | Success Rate | Avg Time (s) | Total Time (s) | Avg Docs | Avg Answer Length | Valid Answer Rate |\n")
            f.write(f"|----------|--------------|--------------|----------------|----------|-------------------|-------------------|\n")
            
            for pipeline_name, metrics in results['performance_metrics'].items():
                f.write(f"| {pipeline_name} | {metrics['success_rate']:.1%} | "
                       f"{metrics['avg_execution_time']:.2f} | {metrics['total_execution_time']:.2f} | "
                       f"{metrics['avg_docs_retrieved']:.1f} | {metrics['avg_answer_length']:.0f} | "
                       f"{metrics['valid_answer_rate']:.1%} |\n")
            
            f.write(f"\n## Evaluation Details\n\n")
            f.write(f"- **Total Duration:** {results['evaluation_duration']:.2f} seconds\n")
            f.write(f"- **Test Queries:** {len(results['test_queries'])}\n")
            f.write(f"- **Pipelines Evaluated:** {len(results['performance_metrics'])}\n")
            f.write(f"- **Timestamp:** {results['timestamp']}\n")
            
            # ColBERT specific analysis
            if 'colbert' in results['performance_metrics']:
                colbert_metrics = results['performance_metrics']['colbert']
                f.write(f"\n## ColBERT Pipeline Analysis\n\n")
                f.write(f"- **Success Rate:** {colbert_metrics['success_rate']:.1%}\n")
                f.write(f"- **Average Query Time:** {colbert_metrics['avg_execution_time']:.2f} seconds\n")
                f.write(f"- **Total Execution Time:** {colbert_metrics['total_execution_time']:.2f} seconds\n")
                f.write(f"- **Average Documents Retrieved:** {colbert_metrics['avg_docs_retrieved']:.1f}\n")
                f.write(f"- **Valid Answer Rate:** {colbert_metrics['valid_answer_rate']:.1%}\n")
                
                # Compare with fastest pipeline
                fastest_pipeline = min(results['performance_metrics'].items(),
                                     key=lambda x: x[1]['avg_execution_time'])
                if fastest_pipeline[1]['avg_execution_time'] > 0:
                    speed_ratio = colbert_metrics['avg_execution_time'] / fastest_pipeline[1]['avg_execution_time']
                    f.write(f"- **Speed Comparison:** {speed_ratio:.1f}x slower than {fastest_pipeline[0]}\n")
                else:
                    f.write(f"- **Speed Comparison:** Cannot compare - fastest pipeline ({fastest_pipeline[0]}) has 0 execution time\n")

def main():
    """Main execution function"""
    evaluator = SimplePipelineEvaluator()
    results = evaluator.run_evaluation()
    
    print("\n" + "="*80)
    print("🎉 SIMPLE PIPELINE EVALUATION COMPLETED!")
    print("="*80)
    
    print(f"📊 Performance Summary:")
    for pipeline_name, metrics in results['performance_metrics'].items():
        print(f"  {pipeline_name:15} | Success: {metrics['success_rate']:6.1%} | "
              f"Time: {metrics['avg_execution_time']:6.2f}s | "
              f"Valid Answers: {metrics['valid_answer_rate']:6.1%}")
    
    # Highlight ColBERT performance
    if 'colbert' in results['performance_metrics']:
        colbert_metrics = results['performance_metrics']['colbert']
        print(f"\n🎯 ColBERT Pipeline Highlights:")
        print(f"  • Average Query Time: {colbert_metrics['avg_execution_time']:.2f} seconds")
        print(f"  • Success Rate: {colbert_metrics['success_rate']:.1%}")
        print(f"  • Valid Answer Rate: {colbert_metrics['valid_answer_rate']:.1%}")
        print(f"  • Documents Retrieved: {colbert_metrics['avg_docs_retrieved']:.1f} per query")
    
    print(f"\n📁 Results saved to: {evaluator.results_dir}")
    print("="*80)

if __name__ == "__main__":
    main()