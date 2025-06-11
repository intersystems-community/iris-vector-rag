#!/usr/bin/env python3
"""
Enhanced Benchmark Runner for Production Scale RAG Testing

This script provides comprehensive benchmarking capabilities for production-scale RAG systems,
including performance monitoring, memory usage tracking, and detailed metrics collection.

Usage:
    python scripts/enhanced_benchmark_runner.py --techniques basic_rag,graphrag --queries 50
    python scripts/enhanced_benchmark_runner.py --full-benchmark --output-dir results/
"""

import os
import sys
import logging
import time
import json
import argparse
import psutil
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func
from eval.metrics import calculate_retrieval_metrics, calculate_answer_quality_metrics
from basic_rag.pipeline import BasicRAGPipeline
from graphrag.pipeline import GraphRAGPipeline
from hyde.pipeline import HyDEPipeline
from crag.pipeline import CRAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    technique: str
    query: str
    query_id: int
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    latency_ms: float
    memory_used_mb: float
    cpu_percent: float
    retrieval_metrics: Dict[str, float]
    answer_quality_metrics: Dict[str, float]
    error: Optional[str] = None

@dataclass
class TechniqueSummary:
    """Summary statistics for a technique"""
    technique: str
    total_queries: int
    successful_queries: int
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    avg_memory_mb: float
    avg_cpu_percent: float
    avg_retrieval_precision: float
    avg_retrieval_recall: float
    avg_answer_quality: float
    error_rate: float

class ProductionBenchmarkRunner:
    """Enhanced benchmark runner for production scale testing"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.connection = None
        self.embedding_func = None
        self.llm_func = None
        
        # Performance monitoring
        self.start_time = time.time()
        self.results: List[BenchmarkResult] = []
        
    def setup_models(self, embedding_model: str = "intfloat/e5-base-v2"):
        """Setup embedding and LLM models"""
        logger.info("🔧 Setting up models for benchmarking...")
        
        try:
            # Setup embedding function
            self.embedding_func = get_embedding_func(model_name=embedding_model, mock=False)
            
            # Test embedding
            test_embedding = self.embedding_func(["test"])
            logger.info(f"✅ Embedding model loaded: {len(test_embedding[0])} dimensions")
            
            # Setup LLM function
            self.llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
            
            # Test LLM
            test_response = self.llm_func("Test prompt")
            logger.info("✅ LLM model loaded and tested")
            
        except Exception as e:
            logger.error(f"❌ Model setup failed: {e}")
            raise
    
    def setup_database(self):
        """Setup database connection"""
        logger.info("🔧 Setting up database connection...")
        
        try:
            self.connection = get_iris_connection()
            if not self.connection:
                raise Exception("Failed to establish database connection")
            
            # Test connection and get document count
            cursor = self.connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            doc_count = cursor.fetchone()[0]
            cursor.close()
            
            logger.info(f"✅ Database connected: {doc_count} documents with embeddings")
            
            if doc_count == 0:
                raise Exception("No documents with embeddings found in database")
                
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            raise
    
    def get_biomedical_queries(self, count: int = 50) -> List[str]:
        """Generate biomedical research queries for testing"""
        base_queries = [
            "What are the latest treatments for diabetes?",
            "How does machine learning help in drug discovery?",
            "What are the side effects of immunotherapy?",
            "How do genetic mutations cause cancer?",
            "What is the role of AI in medical diagnosis?",
            "How effective is CRISPR gene editing?",
            "What are biomarkers for Alzheimer's disease?",
            "How do vaccines work against viral infections?",
            "What causes antibiotic resistance?",
            "How is precision medicine changing treatment?",
            "What are the mechanisms of autoimmune diseases?",
            "How do stem cells contribute to regenerative medicine?",
            "What are the latest advances in cancer immunotherapy?",
            "How does the microbiome affect human health?",
            "What are the challenges in developing new antibiotics?",
            "How do epigenetic modifications influence disease?",
            "What is the role of inflammation in chronic diseases?",
            "How are organoids used in disease modeling?",
            "What are the applications of nanotechnology in medicine?",
            "How do protein folding disorders cause disease?",
            "What are the latest developments in gene therapy?",
            "How does aging affect the immune system?",
            "What are the mechanisms of drug resistance in cancer?",
            "How do environmental factors influence genetic expression?",
            "What are the challenges in personalized medicine?",
            "How do neural networks help in medical imaging?",
            "What are the latest advances in vaccine development?",
            "How does oxidative stress contribute to disease?",
            "What are the applications of machine learning in genomics?",
            "How do circadian rhythms affect health and disease?",
            "What are the mechanisms of cellular senescence?",
            "How do metabolic disorders affect organ function?",
            "What are the latest treatments for neurodegenerative diseases?",
            "How does the blood-brain barrier affect drug delivery?",
            "What are the applications of artificial intelligence in pathology?",
            "How do hormonal imbalances affect health?",
            "What are the mechanisms of tissue regeneration?",
            "How does stress affect the immune system?",
            "What are the latest advances in cardiac medicine?",
            "How do genetic variants affect drug metabolism?",
            "What are the challenges in developing cancer vaccines?",
            "How does nutrition influence gene expression?",
            "What are the mechanisms of cellular reprogramming?",
            "How do infectious diseases evolve and spread?",
            "What are the applications of robotics in surgery?",
            "How does exercise affect molecular pathways?",
            "What are the latest developments in organ transplantation?",
            "How do environmental toxins affect human health?",
            "What are the mechanisms of pain perception and management?",
            "How does the gut-brain axis influence behavior and cognition?"
        ]
        
        # Extend with variations if needed
        queries = base_queries[:count]
        
        # Add variations if we need more
        if len(queries) < count:
            variations = []
            for query in base_queries:
                variations.extend([
                    f"Recent research on {query.lower()}",
                    f"Clinical trials for {query.lower()}",
                    f"Molecular mechanisms of {query.lower()}"
                ])
            
            queries.extend(variations[:count - len(queries)])
        
        return queries[:count]
    
    def monitor_system_resources(self) -> Dict[str, float]:
        """Monitor current system resources"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            "memory_used_mb": memory.used / (1024 * 1024),
            "memory_percent": memory.percent,
            "cpu_percent": cpu_percent
        }
    
    def run_single_query_benchmark(self, technique: str, pipeline, query: str, query_id: int) -> BenchmarkResult:
        """Run benchmark for a single query"""
        logger.debug(f"Running {technique} benchmark for query {query_id}: {query[:50]}...")
        
        # Monitor resources before
        resources_before = self.monitor_system_resources()
        
        start_time = time.time()
        error = None
        answer = ""
        retrieved_documents = []
        retrieval_metrics = {}
        answer_quality_metrics = {}
        
        try:
            # Run the pipeline
            result = pipeline.run(query)
            
            # Extract results
            answer = result.get("answer", "")
            retrieved_documents = result.get("retrieved_documents", [])
            
            # Calculate retrieval metrics
            if retrieved_documents:
                retrieval_metrics = calculate_retrieval_metrics(
                    retrieved_documents=retrieved_documents,
                    query=query,
                    ground_truth_docs=[]  # Would need ground truth for proper evaluation
                )
            
            # Calculate answer quality metrics
            if answer:
                answer_quality_metrics = calculate_answer_quality_metrics(
                    answer=answer,
                    query=query,
                    context_documents=retrieved_documents
                )
            
        except Exception as e:
            error = str(e)
            logger.error(f"Error in {technique} for query {query_id}: {e}")
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Monitor resources after
        resources_after = self.monitor_system_resources()
        
        return BenchmarkResult(
            technique=technique,
            query=query,
            query_id=query_id,
            answer=answer,
            retrieved_documents=retrieved_documents,
            latency_ms=latency_ms,
            memory_used_mb=resources_after["memory_used_mb"],
            cpu_percent=resources_after["cpu_percent"],
            retrieval_metrics=retrieval_metrics,
            answer_quality_metrics=answer_quality_metrics,
            error=error
        )
    
    def run_technique_benchmark(self, technique: str, queries: List[str]) -> List[BenchmarkResult]:
        """Run benchmark for a specific technique"""
        logger.info(f"🔍 Running {technique} benchmark with {len(queries)} queries...")
        
        # Initialize pipeline
        pipeline = None
        try:
            if technique == "basic_rag":
                pipeline = BasicRAGPipeline(
                    iris_connector=self.connection,
                    embedding_func=self.embedding_func,
                    llm_func=self.llm_func
                )
            elif technique == "graphrag":
                pipeline = GraphRAGPipeline(
                    iris_connector=self.connection,
                    embedding_func=self.embedding_func,
                    llm_func=self.llm_func
                )
            elif technique == "hyde":
                pipeline = HyDEPipeline(
                    iris_connector=self.connection,
                    embedding_func=self.embedding_func,
                    llm_func=self.llm_func
                )
            elif technique == "crag":
                pipeline = CRAGPipeline(
                    iris_connector=self.connection,
                    embedding_func=self.embedding_func,
                    llm_func=self.llm_func
                )
            else:
                raise ValueError(f"Unknown technique: {technique}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {technique} pipeline: {e}")
            return []
        
        # Run benchmarks
        results = []
        for i, query in enumerate(queries):
            result = self.run_single_query_benchmark(technique, pipeline, query, i)
            results.append(result)
            
            # Log progress
            if (i + 1) % 10 == 0:
                successful = len([r for r in results if r.error is None])
                avg_latency = np.mean([r.latency_ms for r in results if r.error is None])
                logger.info(f"Progress: {i+1}/{len(queries)} queries, {successful} successful, {avg_latency:.1f}ms avg latency")
            
            # Memory cleanup
            if (i + 1) % 20 == 0:
                gc.collect()
        
        logger.info(f"✅ {technique} benchmark complete: {len([r for r in results if r.error is None])}/{len(results)} successful")
        return results
    
    def calculate_technique_summary(self, results: List[BenchmarkResult]) -> TechniqueSummary:
        """Calculate summary statistics for a technique"""
        if not results:
            return TechniqueSummary(
                technique="unknown",
                total_queries=0,
                successful_queries=0,
                avg_latency_ms=0,
                median_latency_ms=0,
                p95_latency_ms=0,
                avg_memory_mb=0,
                avg_cpu_percent=0,
                avg_retrieval_precision=0,
                avg_retrieval_recall=0,
                avg_answer_quality=0,
                error_rate=1.0
            )
        
        successful_results = [r for r in results if r.error is None]
        
        if not successful_results:
            return TechniqueSummary(
                technique=results[0].technique,
                total_queries=len(results),
                successful_queries=0,
                avg_latency_ms=0,
                median_latency_ms=0,
                p95_latency_ms=0,
                avg_memory_mb=0,
                avg_cpu_percent=0,
                avg_retrieval_precision=0,
                avg_retrieval_recall=0,
                avg_answer_quality=0,
                error_rate=1.0
            )
        
        latencies = [r.latency_ms for r in successful_results]
        memories = [r.memory_used_mb for r in successful_results]
        cpu_percents = [r.cpu_percent for r in successful_results]
        
        # Calculate retrieval metrics
        precisions = []
        recalls = []
        for r in successful_results:
            if r.retrieval_metrics:
                precisions.append(r.retrieval_metrics.get("precision", 0))
                recalls.append(r.retrieval_metrics.get("recall", 0))
        
        # Calculate answer quality metrics
        answer_qualities = []
        for r in successful_results:
            if r.answer_quality_metrics:
                answer_qualities.append(r.answer_quality_metrics.get("overall_quality", 0))
        
        return TechniqueSummary(
            technique=results[0].technique,
            total_queries=len(results),
            successful_queries=len(successful_results),
            avg_latency_ms=np.mean(latencies),
            median_latency_ms=np.median(latencies),
            p95_latency_ms=np.percentile(latencies, 95),
            avg_memory_mb=np.mean(memories),
            avg_cpu_percent=np.mean(cpu_percents),
            avg_retrieval_precision=np.mean(precisions) if precisions else 0,
            avg_retrieval_recall=np.mean(recalls) if recalls else 0,
            avg_answer_quality=np.mean(answer_qualities) if answer_qualities else 0,
            error_rate=1 - (len(successful_results) / len(results))
        )
    
    def create_visualizations(self, summaries: List[TechniqueSummary]):
        """Create benchmark visualization charts"""
        logger.info("📊 Creating benchmark visualizations...")
        
        if not summaries:
            logger.warning("No summaries to visualize")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Production Scale RAG Benchmark Results', fontsize=16, fontweight='bold')
        
        techniques = [s.technique for s in summaries]
        
        # 1. Latency comparison
        latencies = [s.avg_latency_ms for s in summaries]
        axes[0, 0].bar(techniques, latencies, color='skyblue')
        axes[0, 0].set_title('Average Latency (ms)')
        axes[0, 0].set_ylabel('Latency (ms)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Memory usage comparison
        memories = [s.avg_memory_mb for s in summaries]
        axes[0, 1].bar(techniques, memories, color='lightcoral')
        axes[0, 1].set_title('Average Memory Usage (MB)')
        axes[0, 1].set_ylabel('Memory (MB)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Success rate comparison
        success_rates = [(1 - s.error_rate) * 100 for s in summaries]
        axes[0, 2].bar(techniques, success_rates, color='lightgreen')
        axes[0, 2].set_title('Success Rate (%)')
        axes[0, 2].set_ylabel('Success Rate (%)')
        axes[0, 2].set_ylim(0, 100)
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Retrieval precision comparison
        precisions = [s.avg_retrieval_precision for s in summaries]
        axes[1, 0].bar(techniques, precisions, color='gold')
        axes[1, 0].set_title('Average Retrieval Precision')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Answer quality comparison
        qualities = [s.avg_answer_quality for s in summaries]
        axes[1, 1].bar(techniques, qualities, color='mediumpurple')
        axes[1, 1].set_title('Average Answer Quality')
        axes[1, 1].set_ylabel('Quality Score')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Latency distribution (box plot)
        if len(self.results) > 0:
            latency_data = []
            technique_labels = []
            for technique in techniques:
                technique_results = [r for r in self.results if r.technique == technique and r.error is None]
                if technique_results:
                    latency_data.append([r.latency_ms for r in technique_results])
                    technique_labels.append(technique)
            
            if latency_data:
                axes[1, 2].boxplot(latency_data, labels=technique_labels)
                axes[1, 2].set_title('Latency Distribution')
                axes[1, 2].set_ylabel('Latency (ms)')
                axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.output_dir / f"benchmark_results_{int(time.time())}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualizations saved to {plot_file}")
        
        plt.close()
    
    def save_results(self, summaries: List[TechniqueSummary]):
        """Save benchmark results to files"""
        timestamp = int(time.time())
        
        # Save detailed results
        detailed_results = []
        for result in self.results:
            detailed_results.append(asdict(result))
        
        detailed_file = self.output_dir / f"detailed_results_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Save summaries
        summary_data = []
        for summary in summaries:
            summary_data.append(asdict(summary))
        
        summary_file = self.output_dir / f"summary_results_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(summary_data)
        csv_file = self.output_dir / f"summary_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        logger.info(f"📁 Results saved:")
        logger.info(f"  Detailed: {detailed_file}")
        logger.info(f"  Summary: {summary_file}")
        logger.info(f"  CSV: {csv_file}")
    
    def run_full_benchmark(self, techniques: List[str], num_queries: int = 50):
        """Run full benchmark suite"""
        logger.info("🚀 Starting Production Scale RAG Benchmark")
        logger.info("=" * 80)
        
        try:
            # Setup
            self.setup_models()
            self.setup_database()
            
            # Generate queries
            queries = self.get_biomedical_queries(num_queries)
            logger.info(f"📝 Generated {len(queries)} biomedical queries")
            
            # Run benchmarks for each technique
            all_summaries = []
            
            for technique in techniques:
                logger.info(f"\n🔍 Benchmarking {technique}...")
                
                try:
                    results = self.run_technique_benchmark(technique, queries)
                    self.results.extend(results)
                    
                    summary = self.calculate_technique_summary(results)
                    all_summaries.append(summary)
                    
                    logger.info(f"✅ {technique} complete: {summary.successful_queries}/{summary.total_queries} successful")
                    logger.info(f"   Avg latency: {summary.avg_latency_ms:.1f}ms")
                    logger.info(f"   Error rate: {summary.error_rate:.1%}")
                    
                except Exception as e:
                    logger.error(f"❌ {technique} benchmark failed: {e}")
                    continue
            
            # Create visualizations and save results
            if all_summaries:
                self.create_visualizations(all_summaries)
                self.save_results(all_summaries)
                
                # Print final summary
                logger.info("\n" + "=" * 80)
                logger.info("🎉 Benchmark Complete!")
                logger.info(f"⏱️  Total time: {(time.time() - self.start_time)/60:.1f} minutes")
                logger.info(f"📊 Techniques tested: {len(all_summaries)}")
                logger.info(f"📝 Queries per technique: {num_queries}")
                
                logger.info("\n📈 SUMMARY RESULTS:")
                for summary in all_summaries:
                    logger.info(f"  {summary.technique}:")
                    logger.info(f"    Success rate: {(1-summary.error_rate):.1%}")
                    logger.info(f"    Avg latency: {summary.avg_latency_ms:.1f}ms")
                    logger.info(f"    P95 latency: {summary.p95_latency_ms:.1f}ms")
                    logger.info(f"    Avg memory: {summary.avg_memory_mb:.1f}MB")
                
                return True
            else:
                logger.error("❌ No successful benchmarks completed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            return False
        
        finally:
            # Cleanup
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Production Scale RAG Benchmarking")
    parser.add_argument("--techniques", type=str, default="basic_rag,graphrag",
                       help="Comma-separated list of techniques to benchmark")
    parser.add_argument("--queries", type=int, default=50,
                       help="Number of queries to test per technique")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--embedding-model", type=str, default="intfloat/e5-base-v2",
                       help="Embedding model to use")
    parser.add_argument("--full-benchmark", action="store_true",
                       help="Run full benchmark with all available techniques")
    
    args = parser.parse_args()
    
    # Parse techniques
    if args.full_benchmark:
        techniques = ["basic_rag", "graphrag", "hyde", "crag"]
    else:
        techniques = [t.strip() for t in args.techniques.split(",")]
    
    logger.info(f"Enhanced RAG Benchmark Runner")
    logger.info(f"Techniques: {techniques}")
    logger.info(f"Queries per technique: {args.queries}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Run benchmark
    runner = ProductionBenchmarkRunner(args.output_dir)
    success = runner.run_full_benchmark(techniques, args.queries)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()