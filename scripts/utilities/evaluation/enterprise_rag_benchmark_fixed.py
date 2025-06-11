#!/usr/bin/env python3
"""
Enterprise RAG Benchmark - Fixed Version for All Pipelines
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import traceback

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming eval is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Core imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# RAG imports - using existing working pipelines
from src.deprecated.basic_rag.pipeline import BasicRAGPipeline # Updated import
from src.experimental.hyde.pipeline import HyDEPipeline # Updated import
from src.experimental.crag.pipeline import CRAGPipeline # Updated import
from src.deprecated.colbert.pipeline import OptimizedColbertRAGPipeline # Updated import
from src.experimental.noderag.pipeline import NodeRAGPipeline # Updated import
from src.experimental.graphrag.pipeline import GraphRAGPipeline # Updated import
from src.experimental.hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline # Updated import

# Common utilities
from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnterpriseRAGBenchmarkFixed:
    """Fixed benchmark that works with all existing pipelines"""
    
    def __init__(self):
        load_dotenv()
        
        self.connection = get_iris_connection()
        self.embedding_func = get_embedding_func()
        
        # Try to use real LLM, fallback to stub
        try:
            if os.getenv("OPENAI_API_KEY"):
                self.llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
                self.real_llm = True
                logger.info("✅ Using OpenAI GPT-3.5-turbo for evaluation")
            else:
                self.llm_func = get_llm_func(provider="stub")
                self.real_llm = False
                logger.warning("⚠️ Using stub LLM (set OPENAI_API_KEY for real evaluation)")
        except Exception as e:
            self.llm_func = get_llm_func(provider="stub")
            self.real_llm = False
            logger.warning(f"⚠️ LLM setup failed, using stub: {e}")
        
        # Initialize pipelines without schema parameter
        self.pipelines = self._initialize_pipelines()
        
        # Test queries for evaluation
        self.test_queries = [
            "What are the main treatments for diabetes?",
            "How does cancer affect the immune system?",
            "What are the side effects of chemotherapy?",
            "How do vaccines work in the human body?",
            "What causes heart disease?"
        ]
        
    def _initialize_pipelines(self) -> Dict[str, Any]:
        """Initialize all RAG pipelines"""
        pipelines = {}
        
        try:
            pipelines['BasicRAG'] = BasicRAGPipeline(
                self.connection, self.embedding_func, self.llm_func, schema="RAG"
            )
            logger.info("✅ BasicRAG initialized")
        except Exception as e:
            logger.error(f"❌ BasicRAG failed: {e}")
        
        try:
            pipelines['HyDE'] = HyDEPipeline(
                self.connection, self.embedding_func, self.llm_func
            )
            logger.info("✅ HyDE initialized")
        except Exception as e:
            logger.error(f"❌ HyDE failed: {e}")
        
        try:
            pipelines['CRAG'] = CRAGPipeline(
                self.connection, self.embedding_func, self.llm_func
            )
            logger.info("✅ CRAG initialized")
        except Exception as e:
            logger.error(f"❌ CRAG failed: {e}")
        
        try:
            # OptimizedColBERT needs specific ColBERT encoders
            pipelines['OptimizedColBERT'] = OptimizedColbertRAGPipeline(
                self.connection,
                self.embedding_func,  # colbert_query_encoder_func
                self.embedding_func,  # colbert_doc_encoder_func
                self.llm_func
            )
            logger.info("✅ OptimizedColBERT initialized")
        except Exception as e:
            logger.error(f"❌ OptimizedColBERT failed: {e}")
        
        try:
            pipelines['NodeRAG'] = NodeRAGPipeline(
                self.connection, self.embedding_func, self.llm_func
            )
            logger.info("✅ NodeRAG initialized")
        except Exception as e:
            logger.error(f"❌ NodeRAG failed: {e}")
        
        try:
            pipelines['GraphRAG'] = GraphRAGPipeline(
                self.connection, self.embedding_func, self.llm_func
            )
            logger.info("✅ GraphRAG initialized")
        except Exception as e:
            logger.error(f"❌ GraphRAG failed: {e}")
        
        try:
            pipelines['HybridiFindRAG'] = HybridiFindRAGPipeline(
                self.connection, self.embedding_func, self.llm_func
            )
            logger.info("✅ HybridiFindRAG initialized")
        except Exception as e:
            logger.error(f"❌ HybridiFindRAG failed: {e}")
        
        logger.info(f"🚀 Initialized {len(pipelines)} RAG pipelines")
        return pipelines
    
    def run_single_query(self, pipeline_name: str, query: str) -> Dict[str, Any]:
        """Run a single query and collect metrics"""
        pipeline = self.pipelines[pipeline_name]
        
        start_time = time.time()
        try:
            result = pipeline.run(query, top_k=10, similarity_threshold=0.1)
            response_time = time.time() - start_time
            
            # Extract metrics
            documents = result.get('retrieved_documents', [])
            answer = result.get('answer', '')
            
            # Calculate similarity scores
            similarity_scores = []
            for doc in documents:
                if isinstance(doc, dict) and 'score' in doc:
                    similarity_scores.append(doc['score'])
                elif hasattr(doc, 'score'):
                    similarity_scores.append(doc.score)
            
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            
            return {
                'success': True,
                'response_time': response_time,
                'documents_retrieved': len(documents),
                'avg_similarity_score': avg_similarity,
                'answer_length': len(answer),
                'answer': answer,
                'documents': documents,
                'query': query
            }
            
        except Exception as e:
            logger.error(f"❌ {pipeline_name} failed for query '{query[:50]}...': {e}")
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'documents_retrieved': 0,
                'avg_similarity_score': 0.0,
                'answer_length': 0,
                'answer': '',
                'documents': [],
                'query': query,
                'error': str(e)
            }
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark across all techniques"""
        logger.info("🚀 Starting comprehensive RAG benchmark...")
        
        benchmark_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for pipeline_name in self.pipelines.keys():
            logger.info(f"📊 Benchmarking {pipeline_name}...")
            
            pipeline_results = []
            total_time = 0
            successful_queries = 0
            
            for i, query in enumerate(self.test_queries):
                logger.info(f"  Query {i+1}/{len(self.test_queries)}: {query[:50]}...")
                
                result = self.run_single_query(pipeline_name, query)
                pipeline_results.append(result)
                
                if result['success']:
                    successful_queries += 1
                    total_time += result['response_time']
                
                time.sleep(1)  # Brief pause between queries
            
            # Calculate aggregate metrics
            successful_results = [r for r in pipeline_results if r['success']]
            
            if successful_results:
                avg_response_time = np.mean([r['response_time'] for r in successful_results])
                avg_documents = np.mean([r['documents_retrieved'] for r in successful_results])
                avg_similarity = np.mean([r['avg_similarity_score'] for r in successful_results])
                avg_answer_length = np.mean([r['answer_length'] for r in successful_results])
                
                benchmark_results[pipeline_name] = {
                    'success_rate': successful_queries / len(self.test_queries),
                    'avg_response_time': avg_response_time,
                    'avg_documents_retrieved': avg_documents,
                    'avg_similarity_score': avg_similarity,
                    'avg_answer_length': avg_answer_length,
                    'individual_results': pipeline_results
                }
                
                logger.info(f"✅ {pipeline_name}: {successful_queries}/{len(self.test_queries)} successful")
            else:
                logger.error(f"❌ {pipeline_name}: No successful queries")
                benchmark_results[pipeline_name] = {
                    'success_rate': 0,
                    'avg_response_time': 0,
                    'avg_documents_retrieved': 0,
                    'avg_similarity_score': 0,
                    'avg_answer_length': 0,
                    'individual_results': pipeline_results
                }
        
        # Save results
        results_file = f"benchmark_results_fixed_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {results_file}")
        
        return benchmark_results
    
    def create_visualizations(self, results: Dict[str, Any]) -> None:
        """Create comprehensive visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for visualization
        techniques = list(results.keys())
        
        # Extract metrics
        response_times = [results[t]['avg_response_time'] for t in techniques]
        documents_retrieved = [results[t]['avg_documents_retrieved'] for t in techniques]
        similarity_scores = [results[t]['avg_similarity_score'] for t in techniques]
        success_rates = [results[t]['success_rate'] for t in techniques]
        answer_lengths = [results[t]['avg_answer_length'] for t in techniques]
        
        # Create performance comparison charts
        self._create_performance_charts(techniques, response_times, documents_retrieved, 
                                      similarity_scores, success_rates, timestamp)
        
        # Create spider chart with performance metrics
        self._create_performance_spider_chart(techniques, response_times, documents_retrieved,
                                            similarity_scores, success_rates, timestamp)
        
        logger.info(f"📊 Visualizations created with timestamp: {timestamp}")
    
    def _create_performance_charts(self, techniques: List[str], response_times: List[float], 
                                 documents_retrieved: List[float], similarity_scores: List[float], 
                                 success_rates: List[float], timestamp: str):
        """Create performance comparison charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Response Time
        bars1 = ax1.bar(techniques, response_times, color='skyblue', alpha=0.7)
        ax1.set_title('Average Response Time (seconds)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Seconds')
        ax1.tick_params(axis='x', rotation=45)
        for bar, time in zip(bars1, response_times):
            if time > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{time:.1f}s', ha='center', va='bottom')
        
        # Documents Retrieved
        bars2 = ax2.bar(techniques, documents_retrieved, color='lightgreen', alpha=0.7)
        ax2.set_title('Average Documents Retrieved', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Documents')
        ax2.tick_params(axis='x', rotation=45)
        for bar, docs in zip(bars2, documents_retrieved):
            if docs > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        f'{docs:.1f}', ha='center', va='bottom')
        
        # Similarity Scores
        bars3 = ax3.bar(techniques, similarity_scores, color='orange', alpha=0.7)
        ax3.set_title('Average Similarity Score', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Similarity Score')
        ax3.tick_params(axis='x', rotation=45)
        for bar, score in zip(bars3, similarity_scores):
            if score > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{score:.3f}', ha='center', va='bottom')
        
        # Success Rate
        bars4 = ax4.bar(techniques, [sr * 100 for sr in success_rates], color='lightcoral', alpha=0.7)
        ax4.set_title('Success Rate (%)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Success Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 105)
        for bar, rate in zip(bars4, success_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate*100:.0f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"rag_performance_comparison_fixed_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Performance charts saved: rag_performance_comparison_fixed_{timestamp}.png")
    
    def _create_performance_spider_chart(self, techniques: List[str], response_times: List[float],
                                       documents_retrieved: List[float], similarity_scores: List[float],
                                       success_rates: List[float], timestamp: str):
        """Create spider chart with normalized performance metrics"""
        fig = go.Figure()
        
        # Normalize metrics (0-1 scale, higher is better)
        max_response_time = max(response_times) if max(response_times) > 0 else 1
        max_documents = max(documents_retrieved) if max(documents_retrieved) > 0 else 1
        max_similarity = max(similarity_scores) if max(similarity_scores) > 0 else 1
        
        metrics = ['Speed', 'Document Retrieval', 'Similarity Quality', 'Success Rate']
        
        for i, technique in enumerate(techniques):
            if success_rates[i] > 0:  # Only plot techniques that worked
                # Normalize values (invert response time so lower is better)
                speed_score = 1 - (response_times[i] / max_response_time) if max_response_time > 0 else 0
                doc_score = documents_retrieved[i] / max_documents if max_documents > 0 else 0
                sim_score = similarity_scores[i] / max_similarity if max_similarity > 0 else 0
                success_score = success_rates[i]
                
                values = [speed_score, doc_score, sim_score, success_score]
                values.append(values[0])  # Close the polygon
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=technique,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="RAG Techniques Performance Comparison",
            font=dict(size=14)
        )
        
        fig.write_html(f"rag_spider_chart_fixed_{timestamp}.html")
        try:
            fig.write_image(f"rag_spider_chart_fixed_{timestamp}.png", width=800, height=600)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")
        
        logger.info(f"✅ Spider chart saved: rag_spider_chart_fixed_{timestamp}.html")

def main():
    """Main function to run the enterprise benchmark"""
    print("🚀 Enterprise RAG Benchmark - Fixed Version")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = EnterpriseRAGBenchmarkFixed()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Create visualizations
    benchmark.create_visualizations(results)
    
    # Print summary
    print("\n📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    for technique, metrics in results.items():
        print(f"\n🔹 {technique}:")
        print(f"   Success Rate: {metrics['success_rate']*100:.1f}%")
        print(f"   Avg Response Time: {metrics['avg_response_time']:.2f}s")
        print(f"   Avg Documents: {metrics['avg_documents_retrieved']:.1f}")
        print(f"   Avg Similarity: {metrics['avg_similarity_score']:.3f}")
        print(f"   Avg Answer Length: {metrics['avg_answer_length']:.0f} chars")
    
    print(f"\n🎉 Benchmark completed! Check the generated visualization files.")

if __name__ == "__main__":
    main()