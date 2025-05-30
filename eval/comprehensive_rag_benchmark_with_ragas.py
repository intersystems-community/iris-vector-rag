#!/usr/bin/env python3
"""
Comprehensive RAG Benchmark with RAGAS Evaluation
Tests all 7 RAG techniques with realistic queries and quality metrics
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import traceback
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false" # Suppress parallelism warning

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️ RAGAS not installed. Install with: pip install ragas datasets")

# RAG imports - using JDBC-compatible pipelines
from basic_rag.pipeline_jdbc import BasicRAGPipelineJDBC as BasicRAGPipeline
from hyde.pipeline import HyDEPipeline
from crag.pipeline_jdbc_fixed import JDBCFixedCRAGPipeline as CRAGPipeline
from colbert.pipeline import OptimizedColbertRAGPipeline as ColBERTPipeline
from noderag.pipeline import NodeRAGPipeline
from graphrag.pipeline_jdbc_fixed import JDBCFixedGraphRAGPipeline as GraphRAGPipeline
from hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline as HybridIFindRAGPipeline

# Common utilities
from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func, get_llm_func, DEFAULT_EMBEDDING_MODEL
from dotenv import load_dotenv

# Langchain for RAGAS LLM/Embeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv() # Ensure .env is loaded at the very beginning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveRAGBenchmark:
    """Comprehensive benchmark with RAGAS evaluation"""
    
    def __init__(self):
        load_dotenv()
        
        self.connection = get_iris_connection()
        self.embedding_func = get_embedding_func()
        
        # Try to use real LLM for better evaluation
        try:
            if os.getenv("OPENAI_API_KEY"):
                self.llm_func = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
                self.embedding_func_ragas = HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
                self.real_llm = True
                logger.info("✅ Using OpenAI GPT-3.5-turbo for evaluation and HuggingFaceEmbeddings for RAGAS")
            else:
                self.llm_func = get_llm_func(provider="stub")
                self.embedding_func_ragas = None # RAGAS won't use this if real_llm is False
                self.real_llm = False
                logger.warning("⚠️ Using stub LLM (set OPENAI_API_KEY for real evaluation)")
        except Exception as e:
            self.llm_func = get_llm_func(provider="stub")
            self.embedding_func_ragas = None
            self.real_llm = False
            logger.warning(f"⚠️ LLM setup failed, using stub: {e}")
        
        # Initialize pipelines
        self.pipelines = self._initialize_pipelines()
        
        # Realistic test queries based on database content
        self.test_queries = [
            {
                "query": "What is the role of olfactory perception in honeybee behavior?",
                "ground_truth": "Olfactory perception plays a crucial role in honeybee behavior, enabling them to identify flowers, communicate through pheromones, and navigate their environment.",
                "keywords": ["olfactory", "honeybee", "perception", "behavior"]
            },
            {
                "query": "How do honeybees process neural signals related to smell?",
                "ground_truth": "Honeybees process olfactory neural signals through their antennal lobes and mushroom bodies, which integrate sensory information for behavioral responses.",
                "keywords": ["honeybee", "neural", "olfactory", "smell", "signal"]
            },
            {
                "query": "What are the similarities between honeybee and human olfactory systems?",
                "ground_truth": "Both honeybee and human olfactory systems use similar molecular mechanisms for odor detection and neural processing, despite structural differences.",
                "keywords": ["honeybee", "human", "olfactory", "similarity", "system"]
            },
            {
                "query": "How do microRNAs regulate gene expression?",
                "ground_truth": "MicroRNAs regulate gene expression by binding to complementary sequences on target mRNAs, leading to translational repression or mRNA degradation.",
                "keywords": ["microRNA", "gene", "regulation", "expression", "mRNA"]
            },
            {
                "query": "What is the relationship between microRNAs and disease?",
                "ground_truth": "MicroRNAs are involved in various diseases including cancer, cardiovascular disease, and neurological disorders through dysregulation of gene expression.",
                "keywords": ["microRNA", "disease", "cancer", "regulation"]
            },
            {
                "query": "How do sensory neurons transmit information?",
                "ground_truth": "Sensory neurons transmit information through electrical signals called action potentials, which travel along axons to relay sensory input to the central nervous system.",
                "keywords": ["sensory", "neuron", "transmit", "signal", "action potential"]
            },
            {
                "query": "What are the mechanisms of neural plasticity?",
                "ground_truth": "Neural plasticity involves synaptic changes, neurogenesis, and structural modifications that allow the nervous system to adapt to experience and injury.",
                "keywords": ["neural", "plasticity", "synapse", "adaptation", "neurogenesis"]
            },
            {
                "query": "How do biological systems process sensory information?",
                "ground_truth": "Biological systems process sensory information through specialized receptors, neural pathways, and brain regions that integrate and interpret sensory inputs.",
                "keywords": ["biological", "sensory", "process", "receptor", "neural"]
            },
            {
                "query": "What are the latest findings in neuroscience research?",
                "ground_truth": "Recent neuroscience research has revealed new insights into brain connectivity, neural coding, and the molecular basis of neurological disorders.",
                "keywords": ["neuroscience", "research", "brain", "neural", "findings"]
            },
            {
                "query": "How do insects use chemical signals for communication?",
                "ground_truth": "Insects use chemical signals called pheromones for various forms of communication including mating, alarm signaling, and trail marking.",
                "keywords": ["insect", "chemical", "signal", "pheromone", "communication"]
            }
        ]
        
    def _initialize_pipelines(self) -> Dict[str, Any]:
        """Initialize all RAG pipelines with correct parameters"""
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
            pipelines['ColBERT'] = ColBERTPipeline(
                iris_connector=self.connection,
                colbert_query_encoder_func=self.embedding_func,
                colbert_doc_encoder_func=self.embedding_func,
                llm_func=self.llm_func
            )
            logger.info("✅ ColBERT initialized")
        except Exception as e:
            logger.error(f"❌ ColBERT failed: {e}")
        
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
            pipelines['HybridIFindRAG'] = HybridIFindRAGPipeline(
                self.connection, self.embedding_func, self.llm_func
            )
            logger.info("✅ HybridIFindRAG initialized")
        except Exception as e:
            logger.error(f"❌ HybridIFindRAG failed: {e}")
        
        logger.info(f"🚀 Initialized {len(pipelines)} RAG pipelines")
        return pipelines
    
    def run_single_query(self, pipeline_name: str, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single query and collect metrics"""
        pipeline = self.pipelines[pipeline_name]
        query = query_data["query"]
        
        start_time = time.time()
        try:
            # Use different parameters based on pipeline
            if pipeline_name == 'CRAG':
                result = pipeline.run(query, top_k=10)
            else:
                result = pipeline.run(query, top_k=10, similarity_threshold=0.1)
            
            response_time = time.time() - start_time
            
            # Extract metrics
            documents = result.get('retrieved_documents', [])
            answer = result.get('answer', '')
            
            # Extract context texts for RAGAS
            contexts = []
            for doc in documents:
                if isinstance(doc, dict):
                    text = doc.get('text', '') or doc.get('content', '') or doc.get('chunk_text', '')
                elif hasattr(doc, 'text'):
                    text = doc.text
                elif hasattr(doc, 'content'):
                    text = doc.content
                else:
                    text = str(doc)
                if text:
                    contexts.append(text)
            
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
                'contexts': contexts,
                'query': query,
                'ground_truth': query_data.get('ground_truth', ''),
                'keywords': query_data.get('keywords', [])
            }
            
        except Exception as e:
            logger.error(f"❌ {pipeline_name} failed for query '{query[:50]}...': {e}")
            traceback.print_exc()
            return {
                'success': False,
                'response_time': time.time() - start_time,
                'documents_retrieved': 0,
                'avg_similarity_score': 0.0,
                'answer_length': 0,
                'answer': '',
                'contexts': [],
                'query': query,
                'ground_truth': query_data.get('ground_truth', ''),
                'keywords': query_data.get('keywords', []),
                'error': str(e)
            }
    
    def evaluate_with_ragas(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Evaluate results using RAGAS metrics"""
        if not RAGAS_AVAILABLE:
            logger.warning("⚠️ RAGAS not available, skipping quality evaluation")
            return None
        
        if not self.real_llm:
            logger.warning("⚠️ RAGAS evaluation requires real LLM, skipping")
            return None
        
        # Filter successful results with answers
        valid_results = [r for r in results if r['success'] and r['answer'] and r['contexts']]
        
        if not valid_results:
            logger.warning("⚠️ No valid results for RAGAS evaluation")
            return None
        
        try:
            # Prepare data for RAGAS
            data = {
                'question': [r['query'] for r in valid_results],
                'answer': [r['answer'] for r in valid_results],
                'contexts': [r['contexts'] for r in valid_results],
                'ground_truth': [r['ground_truth'] for r in valid_results]
            }
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics based on available data
            metrics = [answer_relevancy, faithfulness]
            if all(r['ground_truth'] for r in valid_results):
                metrics.extend([answer_similarity, answer_correctness])
            if all(r['contexts'] for r in valid_results):
                metrics.extend([context_precision])
            
            # Run RAGAS evaluation
            logger.info("🔍 Running RAGAS evaluation...")
            ragas_results = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm_func,
                embeddings=self.embedding_func_ragas
            )
            
            return ragas_results
            
        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {e}")
            traceback.print_exc()
            return None
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark with RAGAS evaluation"""
        logger.info("🚀 Starting comprehensive RAG benchmark with RAGAS...")
        
        benchmark_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for pipeline_name in self.pipelines.keys():
            logger.info(f"\n📊 Benchmarking {pipeline_name}...")
            
            pipeline_results = []
            total_time = 0
            successful_queries = 0
            
            for i, query_data in enumerate(self.test_queries):
                logger.info(f"  Query {i+1}/{len(self.test_queries)}: {query_data['query'][:50]}...")
                
                result = self.run_single_query(pipeline_name, query_data)
                pipeline_results.append(result)
                
                if result['success']:
                    successful_queries += 1
                    total_time += result['response_time']
                
                time.sleep(0.5)  # Brief pause between queries
            
            # Calculate aggregate metrics
            successful_results = [r for r in pipeline_results if r['success']]
            
            if successful_results:
                # Performance metrics
                avg_response_time = np.mean([r['response_time'] for r in successful_results])
                avg_documents = np.mean([r['documents_retrieved'] for r in successful_results])
                avg_similarity = np.mean([r['avg_similarity_score'] for r in successful_results])
                avg_answer_length = np.mean([r['answer_length'] for r in successful_results])
                
                # RAGAS evaluation
                ragas_scores = self.evaluate_with_ragas(successful_results)
                
                benchmark_results[pipeline_name] = {
                    'success_rate': successful_queries / len(self.test_queries),
                    'avg_response_time': avg_response_time,
                    'avg_documents_retrieved': avg_documents,
                    'avg_similarity_score': avg_similarity,
                    'avg_answer_length': avg_answer_length,
                    'ragas_scores': ragas_scores,
                    'individual_results': pipeline_results
                }
                
                logger.info(f"✅ {pipeline_name}: {successful_queries}/{len(self.test_queries)} successful")
                if ragas_scores:
                    logger.info(f"   RAGAS Scores: {ragas_scores}")
            else:
                logger.error(f"❌ {pipeline_name}: No successful queries")
                benchmark_results[pipeline_name] = {
                    'success_rate': 0,
                    'avg_response_time': 0,
                    'avg_documents_retrieved': 0,
                    'avg_similarity_score': 0,
                    'avg_answer_length': 0,
                    'ragas_scores': None,
                    'individual_results': pipeline_results
                }
        
        # Save results
        results_file = f"comprehensive_benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Convert RAGAS results to serializable format
            serializable_results = {}
            for technique, data in benchmark_results.items():
                serializable_data = data.copy()
                if data['ragas_scores'] is not None:
                    serializable_data['ragas_scores'] = {
                        k: float(v) for k, v in data['ragas_scores'].items()
                    }
                serializable_results[technique] = serializable_data
            
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {results_file}")
        
        return benchmark_results
    
    def create_comprehensive_visualizations(self, results: Dict[str, Any]) -> None:
        """Create comprehensive visualizations including RAGAS scores"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data
        techniques = list(results.keys())
        
        # Performance metrics
        response_times = [results[t]['avg_response_time'] for t in techniques]
        documents_retrieved = [results[t]['avg_documents_retrieved'] for t in techniques]
        similarity_scores = [results[t]['avg_similarity_score'] for t in techniques]
        success_rates = [results[t]['success_rate'] for t in techniques]
        
        # Create performance comparison
        self._create_performance_comparison(techniques, response_times, documents_retrieved,
                                          similarity_scores, success_rates, timestamp)
        
        # Create RAGAS comparison if available
        if any(results[t]['ragas_scores'] for t in techniques):
            self._create_ragas_comparison(results, timestamp)
        
        # Create comprehensive spider chart
        self._create_comprehensive_spider_chart(results, timestamp)
        
        logger.info(f"📊 Visualizations created with timestamp: {timestamp}")
    
    def _create_performance_comparison(self, techniques: List[str], response_times: List[float],
                                     documents_retrieved: List[float], similarity_scores: List[float],
                                     success_rates: List[float], timestamp: str):
        """Create performance comparison charts"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Response Time
        bars1 = ax1.bar(techniques, response_times, color='skyblue', alpha=0.8)
        ax1.set_title('Average Response Time', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Seconds', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        for bar, time in zip(bars1, response_times):
            if time > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f'{time:.2f}s', ha='center', va='bottom', fontsize=10)
        
        # Documents Retrieved
        bars2 = ax2.bar(techniques, documents_retrieved, color='lightgreen', alpha=0.8)
        ax2.set_title('Average Documents Retrieved', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Number of Documents', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        for bar, docs in zip(bars2, documents_retrieved):
            if docs > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{docs:.1f}', ha='center', va='bottom', fontsize=10)
        
        # Similarity Scores
        bars3 = ax3.bar(techniques, similarity_scores, color='orange', alpha=0.8)
        ax3.set_title('Average Similarity Score', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Similarity Score', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        for bar, score in zip(bars3, similarity_scores):
            if score > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Success Rate
        bars4 = ax4.bar(techniques, [sr * 100 for sr in success_rates], color='lightcoral', alpha=0.8)
        ax4.set_title('Success Rate', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Success Rate (%)', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 105)
        for bar, rate in zip(bars4, success_rates):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate*100:.0f}%', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('RAG Techniques Performance Comparison', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"rag_performance_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Performance comparison saved: rag_performance_comparison_{timestamp}.png")
    
    def _create_ragas_comparison(self, results: Dict[str, Any], timestamp: str):
        """Create RAGAS scores comparison"""
        # Collect RAGAS metrics
        techniques_with_ragas = []
        ragas_metrics = {}
        
        for technique, data in results.items():
            if data['ragas_scores']:
                techniques_with_ragas.append(technique)
                for metric, score in data['ragas_scores'].items():
                    if metric not in ragas_metrics:
                        ragas_metrics[metric] = []
                    ragas_metrics[metric].append(score)
        
        if not techniques_with_ragas:
            return
        
        # Create RAGAS comparison chart
        n_metrics = len(ragas_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(techniques_with_ragas)))
        
        for i, (metric, scores) in enumerate(ragas_metrics.items()):
            ax = axes[i]
            bars = ax.bar(techniques_with_ragas, scores, color=colors, alpha=0.8)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('RAGAS Quality Metrics Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"ragas_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ RAGAS comparison saved: ragas_comparison_{timestamp}.png")
    
    def _create_comprehensive_spider_chart(self, results: Dict[str, Any], timestamp: str):
        """Create comprehensive spider chart with all metrics"""
        fig = go.Figure()
        
        # Prepare metrics
        metrics = ['Speed', 'Doc Retrieval', 'Similarity', 'Success Rate']
        
        # Add RAGAS metrics if available
        ragas_metric_names = set()
        for data in results.values():
            if data['ragas_scores']:
                ragas_metric_names.update(data['ragas_scores'].keys())
        
        for ragas_metric in sorted(ragas_metric_names):
            metrics.append(ragas_metric.replace('_', ' ').title())
        
        # Normalize values
        max_response_time = max([r['avg_response_time'] for r in results.values() if r['avg_response_time'] > 0], default=1)
        max_documents = max([r['avg_documents_retrieved'] for r in results.values() if r['avg_documents_retrieved'] > 0], default=1)
        max_similarity = max([r['avg_similarity_score'] for r in results.values() if r['avg_similarity_score'] > 0], default=1)
        
        for technique, data in results.items():
            if data['success_rate'] > 0:
                # Performance metrics (normalized)
                speed_score = 1 - (data['avg_response_time'] / max_response_time) if max_response_time > 0 else 0
                doc_score = data['avg_documents_retrieved'] / max_documents if max_documents > 0 else 0
                sim_score = data['avg_similarity_score'] / max_similarity if max_similarity > 0 else 0
                success_score = data['success_rate']
                
                values = [speed_score, doc_score, sim_score, success_score]
                
                # Add RAGAS scores
                for ragas_metric in sorted(ragas_metric_names):
                    if data['ragas_scores'] and ragas_metric in data['ragas_scores']:
                        values.append(data['ragas_scores'][ragas_metric])
                    else:
                        values.append(0)
                
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
            title="Comprehensive RAG Techniques Comparison",
            font=dict(size=14),
            height=800,
            width=1000
        )
        
        fig.write_html(f"comprehensive_spider_chart_{timestamp}.html")
        try:
            fig.write_image(f"comprehensive_spider_chart_{timestamp}.png", width=1000, height=800)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")
        
        logger.info(f"✅ Comprehensive spider chart saved: comprehensive_spider_chart_{timestamp}.html")
    
    def generate_report(self, results: Dict[str, Any], timestamp: str) -> None:
        """Generate comprehensive markdown report"""
        report = f"""# Comprehensive RAG Benchmark Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents a comprehensive evaluation of 7 RAG (Retrieval-Augmented Generation) techniques with both performance metrics and quality evaluation using RAGAS.

## Techniques Evaluated

1. **BasicRAG**: Standard vector similarity search
2. **HyDE**: Hypothetical Document Embeddings
3. **CRAG**: Corrective RAG with relevance assessment
4. **ColBERT**: Late interaction neural ranking
5. **NodeRAG**: Node-based retrieval
6. **GraphRAG**: Knowledge graph enhanced retrieval
7. **HybridIFindRAG**: Hybrid approach with multiple strategies

## Performance Results

| Technique | Success Rate | Avg Response Time | Avg Documents | Avg Similarity |
|-----------|-------------|-------------------|---------------|----------------|
"""
        
        for technique, data in results.items():
            report += f"| {technique} | {data['success_rate']*100:.1f}% | {data['avg_response_time']:.3f}s | {data['avg_documents_retrieved']:.1f} | {data['avg_similarity_score']:.3f} |\n"
        
        # Add RAGAS results if available
        if any(data['ragas_scores'] for data in results.values()):
            report += "\n## RAGAS Quality Evaluation\n\n"
            report += "| Technique |"
            
            # Get all RAGAS metrics
            all_metrics = set()
            for data in results.values():
                if data['ragas_scores']:
                    all_metrics.update(data['ragas_scores'].keys())
            
            for metric in sorted(all_metrics):
                report += f" {metric.replace('_', ' ').title()} |"
            report += "\n|" + "-|" * (len(all_metrics) + 1) + "\n"
            
            for technique, data in results.items():
                if data['ragas_scores']:
                    report += f"| {technique} |"
                    for metric in sorted(all_metrics):
                        score = data['ragas_scores'].get(metric, 0)
                        report += f" {score:.3f} |"
                    report += "\n"
        
        report += f"""
## Key Findings

1. **Best Overall Performance**: {max(results.items(), key=lambda x: x[1]['success_rate'])[0]}
2. **Fastest Response Time**: {min((k, v) for k, v in results.items() if v['avg_response_time'] > 0)[0]}
3. **Most Documents Retrieved**: {max(results.items(), key=lambda x: x[1]['avg_documents_retrieved'])[0]}
"""
        
        if any(data['ragas_scores'] for data in results.values()):
            # Find best for each RAGAS metric
            for metric in sorted(all_metrics):
                best_technique = max(
                    ((k, v) for k, v in results.items() if v['ragas_scores'] and metric in v['ragas_scores']),
                    key=lambda x: x[1]['ragas_scores'][metric],
                    default=(None, None)
                )
                if best_technique[0]:
                    report += f"4. **Best {metric.replace('_', ' ').title()}**: {best_technique[0]}\n"
        
        report += "\n## Conclusion\n\n"
        report += "This comprehensive benchmark demonstrates the strengths and weaknesses of each RAG technique "
        report += "across both performance metrics and quality evaluation. The results can guide the selection "
        report += "of appropriate techniques based on specific requirements for speed, accuracy, and quality.\n"
        
        # Save report
        report_file = f"comprehensive_benchmark_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to {report_file}")


def main():
    """Main function to run comprehensive benchmark"""
    print("🚀 Comprehensive RAG Benchmark with RAGAS Evaluation")
    print("=" * 60)
    print("📌 Testing all 7 RAG techniques")
    print("📌 Using realistic queries based on database content")
    print("📌 Including RAGAS quality metrics (if available)")
    print("=" * 60)
    
    # Initialize benchmark
    benchmark = ComprehensiveRAGBenchmark()
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    # Create visualizations
    benchmark.create_comprehensive_visualizations(results)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark.generate_report(results, timestamp)
    
    # Print summary
    print("\n📊 COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 60)
    
    total_techniques = len(results)
    successful_techniques = sum(1 for m in results.values() if m['success_rate'] > 0)
    
    print(f"\n✅ Techniques Working: {successful_techniques}/{total_techniques}")
    
    for technique, metrics in results.items():
        status = "✅" if metrics['success_rate'] > 0 else "❌"
        print(f"\n{status} {technique}:")
        print(f"   Success Rate: {metrics['success_rate']*100:.1f}%")
        if metrics['success_rate'] > 0:
            print(f"   Avg Response Time: {metrics['avg_response_time']:.2f}s")
            print(f"   Avg Documents: {metrics['avg_documents_retrieved']:.1f}")
            print(f"   Avg Similarity: {metrics['avg_similarity_score']:.3f}")
            print(f"   Avg Answer Length: {metrics['avg_answer_length']:.0f} chars")
            if metrics['ragas_scores']:
                print("   RAGAS Scores:")
                for metric, score in metrics['ragas_scores'].items():
                    print(f"     - {metric}: {score:.3f}")
    
    print(f"\n🎉 Comprehensive benchmark completed!")
    print(f"📊 Check the generated visualization files and report")


if __name__ == "__main__":
    main()