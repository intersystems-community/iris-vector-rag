"""
Comprehensive before/after benchmark for reranking pipeline performance.

This script tests:
1. Current reranking pipeline performance (baseline)
2. Performance with different optimizations
3. Quality metrics (relevance scores)
4. Edge cases (few candidates, many candidates)
"""

import time
import json
import statistics
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Set up paths for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.models import Document


class RerankerBenchmark:
    """Benchmark suite for testing reranker performance improvements."""
    
    def __init__(self):
        self.results = {
            "baseline": {},
            "optimized": {},
            "comparison": {}
        }
        
        # Test queries with different characteristics
        self.test_queries = [
            "What is InterSystems IRIS?",
            "How does vector search work?", 
            "What are the benefits of RAG?",
            "Explain database performance optimization",
            "What is machine learning?"
        ]
        
        # Test scenarios
        self.test_scenarios = [
            {"name": "small_candidates", "top_k": 3, "rerank_factor": 1.5},  # ~4-5 candidates
            {"name": "normal_candidates", "top_k": 5, "rerank_factor": 2},   # ~10 candidates  
            {"name": "large_candidates", "top_k": 10, "rerank_factor": 3},  # ~30 candidates
            {"name": "max_candidates", "top_k": 20, "rerank_factor": 5}     # ~100 candidates (capped)
        ]
    
    def setup_pipelines(self):
        """Initialize baseline and test pipelines."""
        print("🔧 Setting up test pipelines...")
        
        # Shared managers
        self.connection_manager = ConnectionManager()
        self.config_manager = ConfigurationManager()
        
        # Baseline: Current reranking pipeline
        self.baseline_pipeline = BasicRAGRerankingPipeline(
            self.connection_manager, 
            self.config_manager
        )
        
        # Comparison: Basic pipeline (no reranking)
        self.basic_pipeline = BasicRAGPipeline(
            self.connection_manager,
            self.config_manager
        )
        
        print("✅ Pipelines initialized")
    
    def load_test_data(self):
        """Load test documents."""
        print("📄 Loading test documents...")
        
        # Load standard test documents
        test_docs = [
            Document(
                page_content="InterSystems IRIS is a multi-model database that supports SQL, JSON, and object data models. It is used in high-performance transactional systems.",
                metadata={"source": "./data/test_txt_docs/1.txt", "filename": "1.txt"}
            ),
            Document(
                page_content="Vector search uses mathematical representations to find semantically similar content in large datasets.",
                metadata={"source": "./data/test_txt_docs/2.txt", "filename": "2.txt"}
            ),
            Document(
                page_content="Retrieval-Augmented Generation (RAG) combines document retrieval with LLM-based generation to produce grounded answers.",
                metadata={"source": "./data/test_txt_docs/3.txt", "filename": "3.txt"}
            ),
            Document(
                page_content="Database performance can be optimized through proper indexing, query optimization, and hardware scaling.",
                metadata={"source": "./data/test_txt_docs/4.txt", "filename": "4.txt"}
            ),
            Document(
                page_content="Machine learning enables computers to learn and make decisions from data without explicit programming.",
                metadata={"source": "./data/test_txt_docs/5.txt", "filename": "5.txt"}
            ),
            Document(
                page_content="The InterSystems IRIS database provides embedded analytics, interoperability, and horizontal scalability.",
                metadata={"source": "./data/test_txt_docs/6.txt", "filename": "6.txt"}
            ),
            Document(
                page_content="Natural language processing helps computers understand and generate human language.",
                metadata={"source": "./data/test_txt_docs/7.txt", "filename": "7.txt"}
            ),
            Document(
                page_content="Cloud computing provides scalable access to computing resources over the internet.",
                metadata={"source": "./data/test_txt_docs/8.txt", "filename": "8.txt"}
            ),
            Document(
                page_content="Artificial intelligence encompasses machine learning, deep learning, and cognitive computing.",
                metadata={"source": "./data/test_txt_docs/9.txt", "filename": "9.txt"}
            ),
            Document(
                page_content="Data warehousing involves collecting, storing, and managing large amounts of data for analysis.",
                metadata={"source": "./data/test_txt_docs/10.txt", "filename": "10.txt"}
            )
        ]
        
        # Load documents into pipelines
        self.baseline_pipeline.load_documents("", documents=test_docs)
        self.basic_pipeline.load_documents("", documents=test_docs)
        
        print(f"✅ Loaded {len(test_docs)} test documents")
    
    def run_performance_test(self, pipeline, pipeline_name: str, scenario: Dict, query: str) -> Dict[str, Any]:
        """Run a single performance test."""
        print(f"  🔄 Testing {pipeline_name} - {scenario['name']} - {query[:30]}...")
        
        # Multiple runs for statistical accuracy
        times = []
        results = []
        
        for run in range(3):  # 3 runs for average
            start_time = time.time()
            
            try:
                # Extract scenario parameters, avoiding top_k duplication
                scenario_params = {k: v for k, v in scenario.items() if k not in ['name', 'top_k']}
                
                result = pipeline.query(
                    query, 
                    top_k=scenario['top_k'],
                    **scenario_params
                )
                end_time = time.time()
                
                execution_time = end_time - start_time
                times.append(execution_time)
                results.append(result)
                
            except Exception as e:
                print(f"    ❌ Error in run {run}: {e}")
                times.append(float('inf'))
                results.append({"error": str(e)})
        
        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        
        if valid_times:
            avg_time = statistics.mean(valid_times)
            min_time = min(valid_times)
            max_time = max(valid_times)
            std_dev = statistics.stdev(valid_times) if len(valid_times) > 1 else 0
        else:
            avg_time = min_time = max_time = std_dev = float('inf')
        
        # Analyze results
        if results and "retrieved_documents" in results[0]:
            num_docs = len(results[0]["retrieved_documents"])
            reranked = results[0].get("metadata", {}).get("reranked", False)
        else:
            num_docs = 0
            reranked = False
        
        return {
            "scenario": scenario['name'],
            "query": query[:50] + "..." if len(query) > 50 else query,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "std_dev": std_dev,
            "num_documents": num_docs,
            "reranked": reranked,
            "success_rate": len(valid_times) / len(times),
            "raw_times": times
        }
    
    def run_baseline_benchmark(self):
        """Run baseline performance tests."""
        print("\n📊 Running BASELINE tests (current reranking pipeline)...")
        
        baseline_results = []
        
        for scenario in self.test_scenarios:
            for query in self.test_queries:
                result = self.run_performance_test(
                    self.baseline_pipeline, 
                    "Baseline Rerank", 
                    scenario, 
                    query
                )
                baseline_results.append(result)
        
        self.results["baseline"] = {
            "total_tests": len(baseline_results),
            "avg_time": statistics.mean([r["avg_time"] for r in baseline_results if r["avg_time"] != float('inf')]),
            "results": baseline_results
        }
        
        print(f"✅ Baseline: {self.results['baseline']['total_tests']} tests, avg time: {self.results['baseline']['avg_time']:.3f}s")
    
    def run_comparison_benchmark(self):
        """Run comparison with basic pipeline (no reranking)."""
        print("\n📊 Running COMPARISON tests (basic pipeline, no reranking)...")
        
        comparison_results = []
        
        for scenario in self.test_scenarios:
            for query in self.test_queries:
                result = self.run_performance_test(
                    self.basic_pipeline,
                    "Basic (No Rerank)",
                    scenario,
                    query
                )
                comparison_results.append(result)
        
        self.results["comparison"] = {
            "total_tests": len(comparison_results),
            "avg_time": statistics.mean([r["avg_time"] for r in comparison_results if r["avg_time"] != float('inf')]),
            "results": comparison_results
        }
        
        print(f"✅ Comparison: {self.results['comparison']['total_tests']} tests, avg time: {self.results['comparison']['avg_time']:.3f}s")
    
    def analyze_edge_cases(self):
        """Test edge cases that might reveal issues."""
        print("\n🔍 Testing edge cases...")
        
        edge_cases = []
        
        # Test case: Very few candidates (should still rerank)
        print("  Testing: Few candidates scenario")
        result = self.run_performance_test(
            self.baseline_pipeline,
            "Edge Case",
            {"name": "few_candidates", "top_k": 8, "rerank_factor": 1.1},  # ~8-9 candidates
            "What is InterSystems IRIS?"
        )
        edge_cases.append(result)
        
        # Test case: Requesting more than available
        print("  Testing: More requested than available")
        result = self.run_performance_test(
            self.baseline_pipeline,
            "Edge Case", 
            {"name": "more_than_available", "top_k": 50, "rerank_factor": 1},  # Want 50, only have 10
            "Machine learning applications"
        )
        edge_cases.append(result)
        
        self.results["edge_cases"] = edge_cases
        print(f"✅ Edge cases: {len(edge_cases)} tests completed")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print("\n📈 Generating performance report...")
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_summary": {
                "baseline_avg_time": self.results["baseline"]["avg_time"],
                "comparison_avg_time": self.results["comparison"]["avg_time"],
                "reranking_overhead": self.results["baseline"]["avg_time"] - self.results["comparison"]["avg_time"],
                "overhead_percentage": ((self.results["baseline"]["avg_time"] - self.results["comparison"]["avg_time"]) / self.results["comparison"]["avg_time"]) * 100
            },
            "detailed_results": self.results,
            "recommendations": []
        }
        
        # Add recommendations based on results
        overhead = report["test_summary"]["reranking_overhead"]
        overhead_pct = report["test_summary"]["overhead_percentage"]
        
        if overhead > 1.0:
            report["recommendations"].append("⚠️  HIGH OVERHEAD: Reranking adds >1s per query - implement model caching")
        
        if overhead_pct > 200:
            report["recommendations"].append("⚠️  EXCESSIVE OVERHEAD: >200% time increase - optimize immediately")
        
        if overhead < 0.1:
            report["recommendations"].append("✅ LOW OVERHEAD: Reranking cost is minimal")
        
        # Check for edge case issues
        edge_results = self.results.get("edge_cases", [])
        for edge in edge_results:
            if not edge.get("reranked", False) and edge["num_documents"] > 1:
                report["recommendations"].append(f"🔧 EDGE CASE: {edge['scenario']} didn't rerank {edge['num_documents']} documents")
        
        return report
    
    def save_results(self, report: Dict[str, Any]):
        """Save benchmark results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"rerank_benchmark_{timestamp}.json"
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filepath}")
        return filepath
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("🚀 Starting comprehensive reranking benchmark...")
        
        try:
            self.setup_pipelines()
            self.load_test_data()
            self.run_baseline_benchmark()
            self.run_comparison_benchmark()
            self.analyze_edge_cases()
            
            report = self.generate_report()
            filepath = self.save_results(report)
            
            # Print summary
            print("\n" + "="*60)
            print("📊 BENCHMARK RESULTS SUMMARY")
            print("="*60)
            print(f"Baseline (Rerank):    {report['test_summary']['baseline_avg_time']:.3f}s avg")
            print(f"Comparison (No Rerank): {report['test_summary']['comparison_avg_time']:.3f}s avg")
            print(f"Reranking Overhead:    {report['test_summary']['reranking_overhead']:.3f}s ({report['test_summary']['overhead_percentage']:.1f}%)")
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"  {rec}")
            print(f"\nFull results: {filepath}")
            print("="*60)
            
            return report
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            raise


if __name__ == "__main__":
    benchmark = RerankerBenchmark()
    report = benchmark.run_full_benchmark()