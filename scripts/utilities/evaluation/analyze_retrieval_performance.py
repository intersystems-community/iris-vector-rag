#!/usr/bin/env python3
"""
Retrieval Performance Analysis - Non-LLM Components Only

This script analyzes the retrieval-only performance of each RAG pipeline,
isolating the non-LLM portions to understand pure retrieval efficiency.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.validation.factory import ValidatedPipelineFactory
from iris_rag.config.manager import ConfigurationManager
from common.iris_dbapi_connector import IRISDBAPIConnector
from common.embedding_utils import get_embedding_function


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval-only performance"""
    pipeline_name: str
    avg_retrieval_time: float
    std_retrieval_time: float
    avg_documents_retrieved: float
    retrieval_operations: List[str]  # List of operations performed
    complexity_score: int  # 1-5 scale of computational complexity


def print_flush(message: str):
    """Print with immediate flush for real-time output."""
    print(message, flush=True)
    sys.stdout.flush()


class RetrievalPerformanceAnalyzer:
    """Analyzes retrieval-only performance across RAG pipelines"""
    
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.iris_connector = IRISDBAPIConnector()
        self.embedding_func = get_embedding_function()
        self.pipelines = {}
        self.test_queries = [
            "What are the effects of metformin on type 2 diabetes?",
            "How does SGLT2 inhibition affect kidney function?",
            "What is the mechanism of action of GLP-1 receptor agonists?",
            "What are the cardiovascular benefits of SGLT2 inhibitors?",
            "How do statins prevent cardiovascular disease?"
        ]
        
    def initialize_pipelines(self):
        """Initialize all pipelines for testing"""
        print_flush("🔧 Initializing pipelines for retrieval analysis...")
        
        pipeline_names = ['basic', 'hyde', 'crag', 'colbert', 'noderag', 'graphrag', 'hybrid_ifind']
        factory = ValidatedPipelineFactory(
            iris_connector=self.iris_connector,
            embedding_func=self.embedding_func,
            llm_func=None,  # We don't need LLM for retrieval testing
            config_manager=self.config_manager,
            auto_setup=True
        )
        
        for name in pipeline_names:
            try:
                print_flush(f"  📋 Initializing {name}...")
                pipeline = factory.create_pipeline(name)
                self.pipelines[name] = pipeline
                print_flush(f"  ✅ {name} initialized")
            except Exception as e:
                print_flush(f"  ❌ {name} failed: {e}")
                
    def analyze_pipeline_retrieval_complexity(self, pipeline_name: str) -> Tuple[List[str], int]:
        """Analyze the computational complexity of each pipeline's retrieval"""
        
        complexity_analysis = {
            'basic': {
                'operations': ['Query Embedding', 'Vector Similarity Search', 'Top-K Selection'],
                'complexity': 2  # Simple vector search
            },
            'hyde': {
                'operations': ['LLM Hypothetical Doc Generation', 'Doc Embedding', 'Vector Similarity Search', 'Top-K Selection'],
                'complexity': 4  # LLM generation + vector search
            },
            'crag': {
                'operations': ['Initial Retrieval', 'Relevance Assessment', 'Corrective Actions', 'Knowledge Base Expansion', 'Re-ranking'],
                'complexity': 5  # Most complex with multiple retrieval rounds
            },
            'colbert': {
                'operations': ['Query Token Embedding', 'Token-level MaxSim Operations', 'Late Interaction Scoring'],
                'complexity': 5  # Token-level operations are expensive (when working)
            },
            'noderag': {
                'operations': ['Initial Node Search', 'Graph Traversal', 'Node Content Retrieval', 'Multi-hop Reasoning'],
                'complexity': 4  # Graph operations
            },
            'graphrag': {
                'operations': ['Entity Extraction', 'Graph-based Retrieval', 'Entity Relationship Traversal', 'Vector Fallback'],
                'complexity': 4  # Entity + graph operations
            },
            'hybrid_ifind': {
                'operations': ['Vector Search', 'IFind Text Search', 'Result Fusion', 'Hybrid Ranking'],
                'complexity': 3  # Dual search methods
            }
        }
        
        info = complexity_analysis.get(pipeline_name, {'operations': ['Unknown'], 'complexity': 1})
        return info['operations'], info['complexity']
    
    def measure_retrieval_only_performance(self, pipeline_name: str, num_iterations: int = 5) -> RetrievalMetrics:
        """Measure retrieval performance without LLM generation"""
        print_flush(f"📊 Measuring retrieval performance for {pipeline_name}...")
        
        pipeline = self.pipelines[pipeline_name]
        retrieval_times = []
        documents_retrieved = []
        
        operations, complexity = self.analyze_pipeline_retrieval_complexity(pipeline_name)
        
        for i in range(num_iterations):
            for query in self.test_queries:
                try:
                    start_time = time.time()
                    
                    # Call retrieval-only methods based on pipeline type
                    if pipeline_name == 'basic':
                        docs = pipeline.query(query, top_k=10)
                    elif pipeline_name == 'hyde':
                        # For HyDE, we need to measure the hypothetical doc generation + retrieval
                        # This includes LLM call for hypothetical doc, but that's part of HyDE's retrieval
                        docs = pipeline.query(query, top_k=10)
                    elif pipeline_name == 'crag':
                        docs = pipeline.query(query, top_k=10)
                    elif pipeline_name == 'colbert':
                        docs = pipeline.query(query, top_k=10)
                    elif pipeline_name == 'noderag':
                        docs = pipeline.retrieve_documents(query, top_k=10)
                    elif pipeline_name == 'graphrag':
                        docs = pipeline.query(query, top_k=10)
                    elif pipeline_name == 'hybrid_ifind':
                        docs = pipeline.query(query, top_k=10)
                    else:
                        docs = []
                    
                    retrieval_time = time.time() - start_time
                    retrieval_times.append(retrieval_time)
                    documents_retrieved.append(len(docs) if docs else 0)
                    
                    print_flush(f"    Query {i+1}: {retrieval_time:.3f}s, {len(docs) if docs else 0} docs")
                    
                except Exception as e:
                    print_flush(f"    ❌ Query {i+1} failed: {e}")
                    retrieval_times.append(float('inf'))
                    documents_retrieved.append(0)
        
        # Filter out failed queries
        valid_times = [t for t in retrieval_times if t != float('inf')]
        
        if not valid_times:
            print_flush(f"  ❌ All queries failed for {pipeline_name}")
            return RetrievalMetrics(
                pipeline_name=pipeline_name,
                avg_retrieval_time=float('inf'),
                std_retrieval_time=0,
                avg_documents_retrieved=0,
                retrieval_operations=operations,
                complexity_score=complexity
            )
        
        return RetrievalMetrics(
            pipeline_name=pipeline_name,
            avg_retrieval_time=statistics.mean(valid_times),
            std_retrieval_time=statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
            avg_documents_retrieved=statistics.mean(documents_retrieved),
            retrieval_operations=operations,
            complexity_score=complexity
        )
    
    def run_comprehensive_retrieval_analysis(self) -> Dict[str, RetrievalMetrics]:
        """Run comprehensive retrieval analysis across all pipelines"""
        print_flush("🚀 Starting Comprehensive Retrieval Performance Analysis")
        print_flush("=" * 80)
        
        self.initialize_pipelines()
        
        results = {}
        for pipeline_name in self.pipelines.keys():
            print_flush(f"\n📊 Analyzing {pipeline_name.upper()} Retrieval Performance")
            print_flush("-" * 50)
            
            metrics = self.measure_retrieval_only_performance(pipeline_name)
            results[pipeline_name] = metrics
            
            print_flush(f"  ⏱️  Avg Retrieval Time: {metrics.avg_retrieval_time:.3f}s")
            print_flush(f"  📄 Avg Documents Retrieved: {metrics.avg_documents_retrieved:.1f}")
            print_flush(f"  🔧 Complexity Score: {metrics.complexity_score}/5")
            print_flush(f"  🔄 Operations: {', '.join(metrics.retrieval_operations)}")
        
        return results
    
    def generate_retrieval_performance_report(self, results: Dict[str, RetrievalMetrics]):
        """Generate detailed retrieval performance report"""
        print_flush("\n" + "=" * 80)
        print_flush("📊 RETRIEVAL-ONLY PERFORMANCE ANALYSIS REPORT")
        print_flush("=" * 80)
        
        # Sort by retrieval time
        sorted_results = sorted(results.items(), key=lambda x: x[1].avg_retrieval_time)
        
        print_flush("\n🏆 RETRIEVAL SPEED RANKING (Fastest to Slowest):")
        print_flush("-" * 60)
        
        for rank, (name, metrics) in enumerate(sorted_results, 1):
            if metrics.avg_retrieval_time == float('inf'):
                status = "❌ FAILED"
                time_str = "N/A"
            else:
                status = "✅"
                time_str = f"{metrics.avg_retrieval_time:.3f}s"
            
            print_flush(f"{rank}. {status} {name.upper():<15} {time_str:<10} (Complexity: {metrics.complexity_score}/5)")
        
        print_flush("\n🔧 COMPLEXITY vs PERFORMANCE ANALYSIS:")
        print_flush("-" * 60)
        
        for name, metrics in sorted_results:
            if metrics.avg_retrieval_time != float('inf'):
                efficiency = metrics.complexity_score / metrics.avg_retrieval_time
                print_flush(f"{name.upper():<15} Complexity: {metrics.complexity_score}/5, Time: {metrics.avg_retrieval_time:.3f}s, Efficiency: {efficiency:.2f}")
        
        print_flush("\n🔄 RETRIEVAL OPERATIONS BREAKDOWN:")
        print_flush("-" * 60)
        
        for name, metrics in results.items():
            print_flush(f"\n{name.upper()}:")
            for i, op in enumerate(metrics.retrieval_operations, 1):
                print_flush(f"  {i}. {op}")
        
        # Save results to JSON
        output_file = f"retrieval_performance_analysis_{int(time.time())}.json"
        output_data = {
            name: {
                'avg_retrieval_time': metrics.avg_retrieval_time,
                'std_retrieval_time': metrics.std_retrieval_time,
                'avg_documents_retrieved': metrics.avg_documents_retrieved,
                'retrieval_operations': metrics.retrieval_operations,
                'complexity_score': metrics.complexity_score
            }
            for name, metrics in results.items()
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print_flush(f"\n💾 Results saved to: {output_file}")


def main():
    """Main execution function"""
    analyzer = RetrievalPerformanceAnalyzer()
    results = analyzer.run_comprehensive_retrieval_analysis()
    analyzer.generate_retrieval_performance_report(results)


if __name__ == "__main__":
    main()