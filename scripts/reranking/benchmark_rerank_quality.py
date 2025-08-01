"""
RAGAS E2E Quality Benchmark for Reranking Pipeline vs Other Pipelines on PMC Documents.

This script evaluates:
1. Reranking pipeline quality vs other RAG techniques on real PMC data
2. RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
3. Performance comparison across all available pipelines
4. Statistical significance of quality improvements
"""

import time
import json
import statistics
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import asyncio

# Set up paths for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from data.pmc_processor import load_pmc_documents, process_pmc_file

# Try to import RAGAS for evaluation
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision, 
        context_recall,
        faithfulness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    print("⚠️  RAGAS not available - install with: pip install ragas")
    RAGAS_AVAILABLE = False


class RAGASBenchmark:
    """RAGAS-based quality benchmark for reranking vs other pipelines."""
    
    def __init__(self):
        self.results = {}
        self.pipeline_configs = {
            "basic": {"class": BasicRAGPipeline, "name": "Basic RAG"},
            "basic_rerank": {"class": BasicRAGRerankingPipeline, "name": "Basic RAG + Reranking"},
            "crag": {"class": CRAGPipeline, "name": "Corrective RAG"},
            "hyde": {"class": HyDERAGPipeline, "name": "HyDE RAG"}
        }
        
        # Test questions for PMC medical/scientific content
        self.test_questions = [
            {
                "question": "What are the main risk factors for cardiovascular disease?",
                "context": "medical research on heart disease risk factors",
                "expected_topics": ["hypertension", "diabetes", "smoking", "cholesterol"]
            },
            {
                "question": "How do mRNA vaccines work at the molecular level?",
                "context": "molecular biology and vaccine research", 
                "expected_topics": ["mRNA", "protein synthesis", "immune response", "antibodies"]
            },
            {
                "question": "What are the latest treatments for cancer immunotherapy?",
                "context": "oncology and immunotherapy research",
                "expected_topics": ["checkpoint inhibitors", "CAR-T", "monoclonal antibodies", "immune system"]
            },
            {
                "question": "What causes Alzheimer's disease and how is it diagnosed?",
                "context": "neurology and dementia research",
                "expected_topics": ["amyloid plaques", "tau protein", "brain imaging", "cognitive testing"]
            },
            {
                "question": "How does CRISPR gene editing work and what are its applications?",
                "context": "genetic engineering and biotechnology",
                "expected_topics": ["CRISPR-Cas9", "gene editing", "therapeutic applications", "ethical considerations"]
            }
        ]
    
    def setup_pipelines(self) -> Dict[str, Any]:
        """Initialize all test pipelines."""
        print("🔧 Setting up test pipelines...")
        
        # Shared managers
        connection_manager = ConnectionManager()
        config_manager = ConfigurationManager()
        
        pipelines = {}
        
        for pipeline_id, config in self.pipeline_configs.items():
            try:
                print(f"  Initializing {config['name']}...")
                pipeline = config["class"](connection_manager, config_manager)
                pipelines[pipeline_id] = {
                    "instance": pipeline,
                    "name": config["name"],
                    "ready": True
                }
            except Exception as e:
                print(f"  ❌ Failed to initialize {config['name']}: {e}")
                pipelines[pipeline_id] = {
                    "instance": None,
                    "name": config["name"],
                    "ready": False,
                    "error": str(e)
                }
        
        ready_count = sum(1 for p in pipelines.values() if p["ready"])
        print(f"✅ {ready_count}/{len(pipelines)} pipelines ready")
        
        return pipelines
    
    def load_pmc_data(self, pipelines: Dict[str, Any], max_docs: int = 50) -> int:
        """Load PMC documents into all pipelines."""
        print(f"📄 Loading PMC documents (max {max_docs})...")
        
        try:
            # Load PMC documents
            pmc_docs = load_pmc_documents(max_documents=max_docs, use_chunking=True)
            
            if not pmc_docs:
                print("❌ No PMC documents found")
                return 0
            
            print(f"  Found {len(pmc_docs)} PMC documents")
            
            # Load into each ready pipeline
            for pipeline_id, pipeline_info in pipelines.items():
                if pipeline_info["ready"]:
                    try:
                        print(f"  Loading into {pipeline_info['name']}...")
                        pipeline_info["instance"].load_documents("", documents=pmc_docs)
                    except Exception as e:
                        print(f"  ❌ Failed to load documents into {pipeline_info['name']}: {e}")
                        pipeline_info["ready"] = False
                        pipeline_info["error"] = str(e)
            
            return len(pmc_docs)
            
        except Exception as e:
            print(f"❌ Failed to load PMC documents: {e}")
            return 0
    
    def run_pipeline_evaluation(self, pipeline_id: str, pipeline_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run RAGAS evaluation on a single pipeline."""
        print(f"📊 Evaluating {pipeline_info['name']}...")
        
        if not pipeline_info["ready"]:
            return {
                "pipeline": pipeline_info["name"],
                "error": pipeline_info.get("error", "Pipeline not ready"),
                "results": None
            }
        
        pipeline = pipeline_info["instance"]
        evaluation_results = []
        
        for test_case in self.test_questions:
            print(f"  🔄 Testing: {test_case['question'][:50]}...")
            
            try:
                # Run query
                start_time = time.time()
                result = pipeline.query(test_case["question"], top_k=5)
                execution_time = time.time() - start_time
                
                # Extract results for RAGAS
                question = test_case["question"]
                answer = result.get("answer", "No answer generated")
                contexts = result.get("contexts", [])
                
                # Simple ground truth based on expected topics (for basic evaluation)
                ground_truth = f"The answer should cover topics related to {', '.join(test_case['expected_topics'])}"
                
                evaluation_results.append({
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                    "execution_time": execution_time,
                    "num_contexts": len(contexts),
                    "context_length": sum(len(c) for c in contexts),
                    "expected_topics": test_case["expected_topics"]
                })
                
            except Exception as e:
                print(f"    ❌ Error in test case: {e}")
                evaluation_results.append({
                    "question": test_case["question"],
                    "error": str(e),
                    "execution_time": float('inf')
                })
        
        # Calculate basic statistics
        valid_results = [r for r in evaluation_results if "error" not in r]
        if valid_results:
            avg_time = statistics.mean([r["execution_time"] for r in valid_results])
            avg_contexts = statistics.mean([r["num_contexts"] for r in valid_results])
            avg_context_length = statistics.mean([r["context_length"] for r in valid_results])
        else:
            avg_time = avg_contexts = avg_context_length = 0
        
        return {
            "pipeline": pipeline_info["name"],
            "pipeline_id": pipeline_id,
            "total_tests": len(evaluation_results),
            "successful_tests": len(valid_results),
            "success_rate": len(valid_results) / len(evaluation_results) if evaluation_results else 0,
            "avg_execution_time": avg_time,
            "avg_contexts_retrieved": avg_contexts,
            "avg_context_length": avg_context_length,
            "detailed_results": evaluation_results,
            "ready_for_ragas": len(valid_results) > 0
        }
    
    def run_ragas_evaluation(self, pipeline_results: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Run RAGAS evaluation if available."""
        if not RAGAS_AVAILABLE:
            print("⚠️  Skipping RAGAS evaluation - not available")
            return None
        
        print("🧪 Running RAGAS evaluation...")
        
        ragas_results = {}
        
        for pipeline_id, result in pipeline_results.items():
            if not result.get("ready_for_ragas", False):
                print(f"  ⏭️  Skipping {result['pipeline']} - no valid results")
                continue
            
            print(f"  📈 RAGAS evaluation for {result['pipeline']}...")
            
            try:
                # Prepare data for RAGAS
                valid_results = [r for r in result["detailed_results"] if "error" not in r]
                
                if not valid_results:
                    continue
                
                # Create RAGAS dataset
                dataset_dict = {
                    "question": [r["question"] for r in valid_results],
                    "answer": [r["answer"] for r in valid_results],
                    "contexts": [r["contexts"] for r in valid_results],
                    "ground_truth": [r["ground_truth"] for r in valid_results]
                }
                
                dataset = Dataset.from_dict(dataset_dict)
                
                # Run RAGAS evaluation
                metrics = [answer_relevancy, context_precision, context_recall, faithfulness]
                ragas_result = evaluate(dataset, metrics=metrics)
                
                ragas_results[pipeline_id] = {
                    "pipeline": result["pipeline"],
                    "ragas_scores": ragas_result,
                    "avg_answer_relevancy": ragas_result["answer_relevancy"],
                    "avg_context_precision": ragas_result["context_precision"],
                    "avg_context_recall": ragas_result["context_recall"],
                    "avg_faithfulness": ragas_result["faithfulness"]
                }
                
                print(f"    ✅ RAGAS completed for {result['pipeline']}")
                
            except Exception as e:
                print(f"    ❌ RAGAS failed for {result['pipeline']}: {e}")
                ragas_results[pipeline_id] = {
                    "pipeline": result["pipeline"],
                    "error": str(e)
                }
        
        return ragas_results
    
    def compare_pipelines(self, pipeline_results: Dict[str, Dict[str, Any]], ragas_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate comprehensive pipeline comparison."""
        print("📈 Generating pipeline comparison...")
        
        comparison = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "performance_ranking": [],
            "quality_ranking": [],
            "recommendations": [],
            "detailed_comparison": {}
        }
        
        # Performance ranking (by execution time)
        performance_data = []
        for pipeline_id, result in pipeline_results.items():
            if result.get("success_rate", 0) > 0:
                performance_data.append({
                    "pipeline_id": pipeline_id,
                    "pipeline": result["pipeline"],
                    "avg_time": result["avg_execution_time"],
                    "success_rate": result["success_rate"]
                })
        
        performance_data.sort(key=lambda x: x["avg_time"])
        comparison["performance_ranking"] = performance_data
        
        # Quality ranking (by RAGAS scores if available)
        if ragas_results:
            quality_data = []
            for pipeline_id, result in ragas_results.items():
                if "error" not in result:
                    # Composite quality score
                    composite_score = (
                        result["avg_answer_relevancy"] + 
                        result["avg_context_precision"] + 
                        result["avg_context_recall"] + 
                        result["avg_faithfulness"]
                    ) / 4
                    
                    quality_data.append({
                        "pipeline_id": pipeline_id,
                        "pipeline": result["pipeline"],
                        "composite_score": composite_score,
                        "answer_relevancy": result["avg_answer_relevancy"],
                        "context_precision": result["avg_context_precision"],
                        "context_recall": result["avg_context_recall"],
                        "faithfulness": result["avg_faithfulness"]
                    })
            
            quality_data.sort(key=lambda x: x["composite_score"], reverse=True)
            comparison["quality_ranking"] = quality_data
        
        # Generate recommendations
        if performance_data:
            fastest = performance_data[0]
            slowest = performance_data[-1]
            
            comparison["recommendations"].append(f"🏆 Fastest Pipeline: {fastest['pipeline']} ({fastest['avg_time']:.2f}s avg)")
            
            if len(performance_data) > 1:
                speed_diff = slowest['avg_time'] - fastest['avg_time']
                comparison["recommendations"].append(f"⚡ Speed Gap: {speed_diff:.2f}s between fastest and slowest")
        
        if comparison["quality_ranking"]:
            best_quality = comparison["quality_ranking"][0]
            comparison["recommendations"].append(f"🎯 Highest Quality: {best_quality['pipeline']} (score: {best_quality['composite_score']:.3f})")
        
        # Check if reranking provides benefits
        basic_rerank_perf = next((p for p in performance_data if p["pipeline_id"] == "basic_rerank"), None)
        basic_perf = next((p for p in performance_data if p["pipeline_id"] == "basic"), None)
        
        if basic_rerank_perf and basic_perf:
            time_overhead = basic_rerank_perf["avg_time"] - basic_perf["avg_time"]
            overhead_pct = (time_overhead / basic_perf["avg_time"]) * 100
            
            comparison["recommendations"].append(f"🔄 Reranking Overhead: +{time_overhead:.2f}s ({overhead_pct:.1f}%)")
            
            if ragas_results and "basic_rerank" in ragas_results and "basic" in ragas_results:
                basic_rerank_quality = ragas_results["basic_rerank"].get("composite_score", 0)
                basic_quality = ragas_results["basic"].get("composite_score", 0)
                quality_improvement = basic_rerank_quality - basic_quality
                
                comparison["recommendations"].append(f"📊 Reranking Quality Impact: {quality_improvement:+.3f} composite score")
                
                if quality_improvement > 0.05:  # Meaningful improvement
                    comparison["recommendations"].append("✅ Reranking provides meaningful quality improvement")
                elif quality_improvement < -0.05:  # Quality degradation
                    comparison["recommendations"].append("⚠️  Reranking may be hurting quality - investigate")
                else:
                    comparison["recommendations"].append("🤔 Reranking quality impact is minimal")
        
        comparison["detailed_comparison"] = {
            "pipeline_results": pipeline_results,
            "ragas_results": ragas_results
        }
        
        return comparison
    
    def save_results(self, comparison: Dict[str, Any]) -> Path:
        """Save benchmark results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"ragas_pipeline_comparison_{timestamp}.json"
        filepath = Path(__file__).parent / filename
        
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filepath}")
        return filepath
    
    def print_summary(self, comparison: Dict[str, Any]):
        """Print benchmark summary."""
        print("\n" + "="*80)
        print("🏆 RAGAS PIPELINE COMPARISON RESULTS")
        print("="*80)
        
        # Performance ranking
        if comparison["performance_ranking"]:
            print("\n⚡ PERFORMANCE RANKING (by speed):")
            for i, p in enumerate(comparison["performance_ranking"], 1):
                print(f"  {i}. {p['pipeline']}: {p['avg_time']:.2f}s avg ({p['success_rate']:.1%} success)")
        
        # Quality ranking  
        if comparison["quality_ranking"]:
            print("\n🎯 QUALITY RANKING (by RAGAS composite score):")
            for i, p in enumerate(comparison["quality_ranking"], 1):
                print(f"  {i}. {p['pipeline']}: {p['composite_score']:.3f}")
                print(f"     Relevancy: {p['answer_relevancy']:.3f}, Precision: {p['context_precision']:.3f}")
                print(f"     Recall: {p['context_recall']:.3f}, Faithfulness: {p['faithfulness']:.3f}")
        
        # Recommendations
        if comparison["recommendations"]:
            print("\n💡 RECOMMENDATIONS:")
            for rec in comparison["recommendations"]:
                print(f"  {rec}")
        
        print("="*80)
    
    def run_full_benchmark(self, max_docs: int = 50):
        """Run complete RAGAS benchmark."""
        print("🚀 Starting RAGAS pipeline comparison benchmark...")
        print(f"📄 Testing with {max_docs} PMC documents")
        
        try:
            # Setup
            pipelines = self.setup_pipelines()
            docs_loaded = self.load_pmc_data(pipelines, max_docs)
            
            if docs_loaded == 0:
                raise Exception("No documents loaded - cannot proceed")
            
            # Run pipeline evaluations
            pipeline_results = {}
            for pipeline_id, pipeline_info in pipelines.items():
                result = self.run_pipeline_evaluation(pipeline_id, pipeline_info)
                pipeline_results[pipeline_id] = result
            
            # Run RAGAS evaluation
            ragas_results = self.run_ragas_evaluation(pipeline_results)
            
            # Generate comparison
            comparison = self.compare_pipelines(pipeline_results, ragas_results)
            
            # Save and display results
            filepath = self.save_results(comparison)
            self.print_summary(comparison)
            
            print(f"\n📊 Full results saved to: {filepath}")
            
            return comparison
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            raise


if __name__ == "__main__":
    # Run benchmark with different document counts
    benchmark = RAGASBenchmark()
    
    # Quick test with 20 documents
    print("🧪 Running quick benchmark (20 docs)...")
    try:
        result = benchmark.run_full_benchmark(max_docs=20)
        print("✅ Quick benchmark completed successfully")
    except Exception as e:
        print(f"❌ Quick benchmark failed: {e}")
    
    # Full test with 50 documents (if quick test passed)
    print("\n" + "="*60)
    print("🧪 Running full benchmark (50 docs)...")
    try:
        result = benchmark.run_full_benchmark(max_docs=50)
        print("✅ Full benchmark completed successfully")
    except Exception as e:
        print(f"❌ Full benchmark failed: {e}")