#!/usr/bin/env python3
"""
Focused RAGAS Evaluation Script
Addresses the LangchainIRISCacheWrapper issues and calculates proper RAGAS metrics
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import RAGAS components
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision, 
    context_recall,
    faithfulness,
    answer_similarity,
    answer_correctness
)

# Import datasets
from datasets import Dataset

# Import LangChain components without caching
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

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

class FocusedRAGASEvaluator:
    """Focused RAGAS evaluator that avoids caching issues"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"focused_ragas_results_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize LLM and embeddings WITHOUT caching
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=1000
        )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        
        # Test queries for evaluation
        self.test_queries = [
            "What are the effects of metformin on type 2 diabetes?",
            "How does SGLT2 inhibition affect kidney function?", 
            "What is the mechanism of action of GLP-1 receptor agonists?",
            "What are the cardiovascular benefits of SGLT2 inhibitors?",
            "How do statins prevent cardiovascular disease?"
        ]
        
        # Ground truth answers
        self.ground_truths = [
            "Metformin helps treat type 2 diabetes by reducing glucose production in the liver and increasing insulin sensitivity in peripheral tissues.",
            "SGLT2 inhibitors protect kidney function by reducing hyperfiltration, decreasing albuminuria, and providing nephroprotection through mechanisms independent of glycemic control.",
            "GLP-1 receptor agonists work by stimulating insulin secretion, suppressing glucagon secretion, slowing gastric emptying, and promoting satiety, ultimately improving glycemic control and often leading to weight loss.",
            "SGLT2 inhibitors provide cardiovascular benefits by reducing heart failure hospitalizations, cardiovascular death, and major adverse cardiovascular events through mechanisms including improved cardiac metabolism and reduced preload.",
            "Statins prevent cardiovascular disease by inhibiting HMG-CoA reductase, reducing cholesterol synthesis, lowering LDL cholesterol levels, and providing pleiotropic anti-inflammatory effects."
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
            
            if hasattr(pipeline, 'query'):
                result = pipeline.query(query)
            elif hasattr(pipeline, 'run'):
                result = pipeline.query(query)
            else:
                # Try calling the pipeline directly
                result = pipeline(query)
            
            execution_time = time.time() - start_time
            
            # Standardize result format - prioritize retrieved_documents over contexts
            if isinstance(result, dict):
                answer = result.get('answer', str(result))
                
                # PRIORITY 1: Extract contexts from retrieved_documents (reliable source)
                retrieved_documents = result.get('retrieved_documents', [])
                context_strings = []
                
                if retrieved_documents:
                    for doc in retrieved_documents:
                        if hasattr(doc, 'content'):
                            # Document object with content attribute
                            if doc.content and doc.content.strip():
                                context_strings.append(str(doc.content))
                        elif hasattr(doc, 'page_content'):
                            # Document object with page_content attribute
                            if doc.page_content and doc.page_content.strip():
                                context_strings.append(str(doc.page_content))
                        elif isinstance(doc, dict):
                            # Dictionary format document
                            content_val = doc.get('content', doc.get('text', doc.get('page_content', '')))
                            if content_val and str(content_val).strip():
                                context_strings.append(str(content_val))
                        elif isinstance(doc, str):
                            # String content directly
                            if doc.strip():
                                context_strings.append(doc)
                
                # FALLBACK: Use contexts field only if retrieved_documents didn't provide content
                if not context_strings:
                    contexts_field = result.get('contexts', [])
                    for ctx in contexts_field:
                        if isinstance(ctx, str) and ctx.strip():
                            context_strings.append(ctx)
                        elif hasattr(ctx, 'content') and ctx.content and ctx.content.strip():
                            context_strings.append(str(ctx.content))
                        elif hasattr(ctx, 'page_content') and ctx.page_content and ctx.page_content.strip():
                            context_strings.append(str(ctx.page_content))
                
                contexts = context_strings
            else:
                answer = str(result)
                contexts = []
            
            return {
                'answer': answer,
                'contexts': contexts,
                'execution_time': execution_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"❌ Error executing {pipeline_name}: {e}")
            return {
                'answer': f"Error: {str(e)}",
                'contexts': [],
                'execution_time': 0,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_ragas_metrics(self, pipeline_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Calculate RAGAS metrics for all pipelines"""
        ragas_results = {}
        
        for pipeline_name, results in pipeline_results.items():
            logger.info(f"📊 Calculating RAGAS metrics for {pipeline_name}")
            
            try:
                # Prepare data for RAGAS
                questions = []
                answers = []
                contexts = []
                ground_truths = []
                
                for i, result in enumerate(results):
                    if result['success']:
                        questions.append(self.test_queries[i])
                        answers.append(result['answer'])
                        contexts.append(result['contexts'])
                        ground_truths.append(self.ground_truths[i])
                
                if not questions:
                    logger.warning(f"⚠️ No successful results for {pipeline_name}")
                    continue
                
                # Create dataset
                dataset = Dataset.from_dict({
                    'question': questions,
                    'answer': answers,
                    'contexts': contexts,
                    'ground_truth': ground_truths
                })
                
                # Define metrics
                metrics = [
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_similarity,
                    answer_correctness
                ]
                
                # Run evaluation
                logger.info(f"🔄 Running RAGAS evaluation for {pipeline_name}...")
                evaluation_result = evaluate(
                    dataset=dataset,
                    metrics=metrics,
                    llm=self.llm,
                    embeddings=self.embeddings
                )
                
                # Extract scores
                ragas_results[pipeline_name] = {
                    'answer_relevancy': evaluation_result['answer_relevancy'],
                    'context_precision': evaluation_result['context_precision'],
                    'context_recall': evaluation_result['context_recall'],
                    'faithfulness': evaluation_result['faithfulness'],
                    'answer_similarity': evaluation_result['answer_similarity'],
                    'answer_correctness': evaluation_result['answer_correctness'],
                    'avg_score': sum([
                        evaluation_result['answer_relevancy'],
                        evaluation_result['context_precision'],
                        evaluation_result['context_recall'],
                        evaluation_result['faithfulness'],
                        evaluation_result['answer_similarity'],
                        evaluation_result['answer_correctness']
                    ]) / 6
                }
                
                logger.info(f"✅ RAGAS metrics calculated for {pipeline_name}")
                
            except Exception as e:
                logger.error(f"❌ Error calculating RAGAS metrics for {pipeline_name}: {e}")
                ragas_results[pipeline_name] = {
                    'error': str(e),
                    'answer_relevancy': None,
                    'context_precision': None,
                    'context_recall': None,
                    'faithfulness': None,
                    'answer_similarity': None,
                    'answer_correctness': None,
                    'avg_score': None
                }
        
        return ragas_results
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the complete focused RAGAS evaluation"""
        logger.info("🚀 Starting Focused RAGAS Evaluation")
        start_time = time.time()
        
        # Execute all pipelines
        pipeline_results = {}
        performance_metrics = {}
        
        for pipeline_name, pipeline in self.pipelines.items():
            logger.info(f"🔄 Evaluating {pipeline_name} pipeline...")
            
            results = []
            total_time = 0
            
            for i, query in enumerate(self.test_queries):
                logger.info(f"  Query {i+1}/{len(self.test_queries)}: {query[:50]}...")
                result = self._execute_pipeline(pipeline_name, pipeline, query)
                results.append(result)
                total_time += result['execution_time']
            
            pipeline_results[pipeline_name] = results
            
            # Calculate performance metrics
            successful_results = [r for r in results if r['success']]
            performance_metrics[pipeline_name] = {
                'total_queries': len(results),
                'successful_queries': len(successful_results),
                'success_rate': len(successful_results) / len(results) if results else 0,
                'avg_execution_time': total_time / len(results) if results else 0,
                'total_execution_time': total_time
            }
            
            logger.info(f"✅ {pipeline_name}: {len(successful_results)}/{len(results)} successful")
        
        # Calculate RAGAS metrics
        logger.info("📊 Calculating RAGAS metrics...")
        ragas_results = self._calculate_ragas_metrics(pipeline_results)
        
        # Compile final results
        final_results = {
            'timestamp': self.timestamp,
            'evaluation_duration': time.time() - start_time,
            'performance_metrics': performance_metrics,
            'ragas_metrics': ragas_results,
            'pipeline_results': pipeline_results
        }
        
        # Save results
        results_file = os.path.join(self.results_dir, f'focused_ragas_results_{self.timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        # Generate summary report
        self._generate_summary_report(final_results)
        
        logger.info(f"✅ Focused RAGAS evaluation completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"📁 Results saved to: {self.results_dir}")
        
        return final_results
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate a summary report"""
        report_file = os.path.join(self.results_dir, f'summary_report_{self.timestamp}.md')
        
        with open(report_file, 'w') as f:
            f.write(f"# Focused RAGAS Evaluation Report\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Performance Summary\n\n")
            f.write(f"| Pipeline | Success Rate | Avg Time (s) | Total Time (s) |\n")
            f.write(f"|----------|--------------|--------------|----------------|\n")
            
            for pipeline_name, metrics in results['performance_metrics'].items():
                f.write(f"| {pipeline_name} | {metrics['success_rate']:.1%} | "
                       f"{metrics['avg_execution_time']:.2f} | {metrics['total_execution_time']:.2f} |\n")
            
            f.write(f"\n## RAGAS Quality Metrics\n\n")
            f.write(f"| Pipeline | Avg Score | Answer Relevancy | Context Precision | Context Recall | Faithfulness | Answer Similarity | Answer Correctness |\n")
            f.write(f"|----------|-----------|------------------|-------------------|----------------|--------------|-------------------|--------------------|\n")
            
            for pipeline_name, metrics in results['ragas_metrics'].items():
                if 'error' not in metrics:
                    f.write(f"| {pipeline_name} | {metrics['avg_score']:.3f} | "
                           f"{metrics['answer_relevancy']:.3f} | {metrics['context_precision']:.3f} | "
                           f"{metrics['context_recall']:.3f} | {metrics['faithfulness']:.3f} | "
                           f"{metrics['answer_similarity']:.3f} | {metrics['answer_correctness']:.3f} |\n")
                else:
                    f.write(f"| {pipeline_name} | ERROR | - | - | - | - | - | - |\n")
            
            f.write(f"\n## Evaluation Details\n\n")
            f.write(f"- **Total Duration:** {results['evaluation_duration']:.2f} seconds\n")
            f.write(f"- **Test Queries:** {len(self.test_queries)}\n")
            f.write(f"- **Pipelines Evaluated:** {len(results['performance_metrics'])}\n")
            f.write(f"- **Timestamp:** {results['timestamp']}\n")

def main():
    """Main execution function"""
    evaluator = FocusedRAGASEvaluator()
    results = evaluator.run_evaluation()
    
    print("\n" + "="*80)
    print("🎉 FOCUSED RAGAS EVALUATION COMPLETED!")
    print("="*80)
    
    print(f"📊 Performance Summary:")
    for pipeline_name, metrics in results['performance_metrics'].items():
        print(f"  {pipeline_name:15} | Success: {metrics['success_rate']:6.1%} | Time: {metrics['avg_execution_time']:6.2f}s")
    
    print(f"\n📈 RAGAS Quality Metrics:")
    for pipeline_name, metrics in results['ragas_metrics'].items():
        if 'error' not in metrics:
            print(f"  {pipeline_name:15} | Avg Score: {metrics['avg_score']:.3f}")
        else:
            print(f"  {pipeline_name:15} | ERROR: {metrics['error']}")
    
    print(f"\n📁 Results saved to: {evaluator.results_dir}")
    print("="*80)

if __name__ == "__main__":
    main()