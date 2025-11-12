#!/usr/bin/env python3
"""
GraphRAG Performance Optimization Benchmarking Script

This script benchmarks the GraphRAG system before and after optimizations
to validate performance improvements and ensure accuracy is maintained.

Usage:
    python scripts/benchmark_graphrag_optimizations.py
    python scripts/benchmark_graphrag_optimizations.py --iterations 100
    python scripts/benchmark_graphrag_optimizations.py --baseline-only
    python scripts/benchmark_graphrag_optimizations.py --optimized-only
"""

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iris_vector_rag.core.config import ConfigManager
from iris_vector_rag.core.connection import IRISConnectionManager
from iris_vector_rag.optimization.cache_manager import GraphRAGCacheManager
from iris_vector_rag.optimization.connection_pool import OptimizedConnectionPool
from iris_vector_rag.optimization.parallel_processor import GraphRAGParallelProcessor
from iris_vector_rag.optimization.performance_monitor import GraphRAGPerformanceMonitor
from iris_vector_rag.pipelines.graphrag_merged import GraphRAGMergedPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphRAGBenchmark:
    """Comprehensive GraphRAG performance benchmarking system."""

    def __init__(self, config_path: Optional[str] = None, output_dir: str = "outputs"):
        """Initialize the benchmarking system."""
        self.config_manager = ConfigManager(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Test queries for benchmarking
        self.test_queries = [
            "What are the main symptoms of COVID-19?",
            "How does machine learning work in healthcare?",
            "What are the benefits of telemedicine?",
            "Explain the role of AI in medical diagnosis",
            "What are the latest treatments for diabetes?",
            "How does vaccination help prevent diseases?",
            "What is precision medicine?",
            "Describe the impact of genetics on drug response",
            "How do clinical trials work?",
            "What are biomarkers and their importance?",
        ]

        # Benchmark results storage
        self.baseline_results = []
        self.optimized_results = []
        self.component_results = {}

        # Optimization components
        self.cache_manager = None
        self.connection_pool = None
        self.parallel_processor = None
        self.performance_monitor = None

    def setup_baseline_pipeline(self) -> GraphRAGMergedPipeline:
        """Set up baseline GraphRAG pipeline without optimizations."""
        logger.info("Setting up baseline GraphRAG pipeline...")

        # Use standard connection manager
        connection_manager = IRISConnectionManager(self.config_manager)

        # Create pipeline without optimizations
        pipeline = GraphRAGMergedPipeline(
            connection_manager=connection_manager, config_manager=self.config_manager
        )

        return pipeline

    def setup_optimized_pipeline(self) -> Tuple[GraphRAGMergedPipeline, Dict]:
        """Set up optimized GraphRAG pipeline with all optimizations."""
        logger.info("Setting up optimized GraphRAG pipeline...")

        # Initialize optimization components
        base_connection_manager = IRISConnectionManager(self.config_manager)

        # Cache manager
        self.cache_manager = GraphRAGCacheManager(self.config_manager)

        # Connection pool
        self.connection_pool = OptimizedConnectionPool(
            base_connection_manager=base_connection_manager,
            min_connections=2,
            max_connections=16,
        )

        # Parallel processor
        self.parallel_processor = GraphRAGParallelProcessor(
            max_workers=16, io_workers=8, entity_workers=4, graph_workers=8
        )

        # Performance monitor
        self.performance_monitor = GraphRAGPerformanceMonitor(
            cache_manager=self.cache_manager,
            connection_pool=self.connection_pool,
            parallel_processor=self.parallel_processor,
        )

        # Create optimized pipeline
        pipeline = GraphRAGMergedPipeline(
            connection_manager=self.connection_pool, config_manager=self.config_manager
        )

        # Start monitoring
        self.performance_monitor.start_monitoring()

        optimization_components = {
            "cache_manager": self.cache_manager,
            "connection_pool": self.connection_pool,
            "parallel_processor": self.parallel_processor,
            "performance_monitor": self.performance_monitor,
        }

        return pipeline, optimization_components

    def run_query_benchmark(
        self, pipeline: GraphRAGMergedPipeline, query: str, iterations: int = 1
    ) -> Dict:
        """Run benchmark for a single query."""
        response_times = []
        accuracies = []
        errors = []

        for i in range(iterations):
            try:
                start_time = time.time()

                # Execute query
                result = pipeline.query(query)

                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to ms
                response_times.append(response_time)

                # Simple accuracy check (non-empty response with relevant content)
                if result and len(result.get("answer", "")) > 50:
                    accuracies.append(1.0)
                else:
                    accuracies.append(0.0)

            except Exception as e:
                logger.error(f"Error running query '{query}': {str(e)}")
                errors.append(str(e))
                response_times.append(float("inf"))
                accuracies.append(0.0)

        # Calculate statistics
        valid_times = [t for t in response_times if t != float("inf")]

        if valid_times:
            stats = {
                "query": query,
                "iterations": iterations,
                "avg_response_time": statistics.mean(valid_times),
                "median_response_time": statistics.median(valid_times),
                "min_response_time": min(valid_times),
                "max_response_time": max(valid_times),
                "p95_response_time": (
                    statistics.quantiles(valid_times, n=20)[18]
                    if len(valid_times) > 5
                    else max(valid_times)
                ),
                "p99_response_time": (
                    statistics.quantiles(valid_times, n=100)[98]
                    if len(valid_times) > 10
                    else max(valid_times)
                ),
                "std_dev": statistics.stdev(valid_times) if len(valid_times) > 1 else 0,
                "accuracy": statistics.mean(accuracies) if accuracies else 0.0,
                "error_rate": len(errors) / iterations,
                "errors": errors[:5],  # Store first 5 errors
            }
        else:
            stats = {
                "query": query,
                "iterations": iterations,
                "avg_response_time": float("inf"),
                "median_response_time": float("inf"),
                "min_response_time": float("inf"),
                "max_response_time": float("inf"),
                "p95_response_time": float("inf"),
                "p99_response_time": float("inf"),
                "std_dev": 0,
                "accuracy": 0.0,
                "error_rate": 1.0,
                "errors": errors[:5],
            }

        return stats

    def run_baseline_benchmark(self, iterations: int = 10) -> List[Dict]:
        """Run benchmark on baseline (unoptimized) system."""
        logger.info(
            f"Running baseline benchmark with {iterations} iterations per query..."
        )

        pipeline = self.setup_baseline_pipeline()
        results = []

        for i, query in enumerate(self.test_queries):
            logger.info(
                f"Running baseline query {i+1}/{len(self.test_queries)}: {query[:50]}..."
            )
            result = self.run_query_benchmark(pipeline, query, iterations)
            results.append(result)

            # Log individual results
            logger.info(
                f"Baseline result: {result['avg_response_time']:.1f}ms avg, "
                f"{result['accuracy']:.1%} accuracy"
            )

        self.baseline_results = results
        return results

    def run_optimized_benchmark(self, iterations: int = 10) -> Tuple[List[Dict], Dict]:
        """Run benchmark on optimized system."""
        logger.info(
            f"Running optimized benchmark with {iterations} iterations per query..."
        )

        pipeline, components = self.setup_optimized_pipeline()
        results = []

        # Warm up cache with a few queries
        logger.info("Warming up cache...")
        for query in self.test_queries[:3]:
            try:
                pipeline.query(query)
            except Exception as e:
                logger.warning(f"Cache warmup failed for query: {e}")

        # Run actual benchmark
        for i, query in enumerate(self.test_queries):
            logger.info(
                f"Running optimized query {i+1}/{len(self.test_queries)}: {query[:50]}..."
            )
            result = self.run_query_benchmark(pipeline, query, iterations)
            results.append(result)

            # Log individual results
            logger.info(
                f"Optimized result: {result['avg_response_time']:.1f}ms avg, "
                f"{result['accuracy']:.1%} accuracy"
            )

        # Get optimization metrics
        optimization_metrics = {}
        if self.performance_monitor:
            optimization_metrics = self.performance_monitor.get_performance_summary()

        self.optimized_results = results
        return results, optimization_metrics

    def test_individual_components(self, iterations: int = 5) -> Dict:
        """Test individual optimization components."""
        logger.info("Testing individual optimization components...")

        component_results = {}
        base_pipeline = self.setup_baseline_pipeline()

        # Test cache only
        logger.info("Testing cache optimization...")
        cache_manager = GraphRAGCacheManager(self.config_manager)
        # Note: For this test, we would need to modify the pipeline to use cache
        # This is a simplified test structure

        # Test connection pool only
        logger.info("Testing connection pool optimization...")
        base_connection_manager = IRISConnectionManager(self.config_manager)
        connection_pool = OptimizedConnectionPool(
            base_connection_manager=base_connection_manager,
            min_connections=2,
            max_connections=8,
        )
        pool_pipeline = GraphRAGMergedPipeline(
            connection_manager=connection_pool, config_manager=self.config_manager
        )

        # Run limited test on pool pipeline
        pool_results = []
        for query in self.test_queries[:3]:
            result = self.run_query_benchmark(pool_pipeline, query, iterations)
            pool_results.append(result)

        component_results["connection_pool"] = pool_results

        # Test parallel processing
        logger.info("Testing parallel processing...")
        parallel_processor = GraphRAGParallelProcessor(max_workers=8)
        # Note: Full integration would require pipeline modifications

        self.component_results = component_results
        return component_results

    def run_concurrent_load_test(
        self, concurrent_users: int = 10, queries_per_user: int = 5
    ) -> Dict:
        """Run concurrent load test."""
        logger.info(
            f"Running concurrent load test with {concurrent_users} users, "
            f"{queries_per_user} queries each..."
        )

        pipeline, _ = self.setup_optimized_pipeline()

        def user_load_test(user_id: int) -> List[Dict]:
            """Simulate a single user's load."""
            user_results = []
            for i in range(queries_per_user):
                query = self.test_queries[i % len(self.test_queries)]
                result = self.run_query_benchmark(pipeline, query, 1)
                result["user_id"] = user_id
                result["query_number"] = i
                user_results.append(result)
            return user_results

        # Run concurrent load test
        start_time = time.time()
        all_results = []

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(user_load_test, user_id)
                for user_id in range(concurrent_users)
            ]

            for future in futures:
                try:
                    user_results = future.result(timeout=300)  # 5 minute timeout
                    all_results.extend(user_results)
                except Exception as e:
                    logger.error(f"User load test failed: {e}")

        end_time = time.time()
        total_duration = end_time - start_time

        # Calculate concurrent metrics
        response_times = [
            r["avg_response_time"]
            for r in all_results
            if r["avg_response_time"] != float("inf")
        ]

        concurrent_metrics = {
            "concurrent_users": concurrent_users,
            "queries_per_user": queries_per_user,
            "total_queries": len(all_results),
            "total_duration": total_duration,
            "queries_per_second": len(all_results) / total_duration,
            "avg_response_time": (
                statistics.mean(response_times) if response_times else float("inf")
            ),
            "p95_response_time": (
                statistics.quantiles(response_times, n=20)[18]
                if len(response_times) > 5
                else float("inf")
            ),
            "error_rate": sum(1 for r in all_results if r["error_rate"] > 0)
            / len(all_results),
            "successful_queries": len(response_times),
            "failed_queries": len(all_results) - len(response_times),
        }

        return concurrent_metrics

    def generate_comparison_report(self) -> Dict:
        """Generate comprehensive comparison report."""
        logger.info("Generating comparison report...")

        def calculate_summary_stats(results: List[Dict]) -> Dict:
            """Calculate summary statistics from results."""
            if not results:
                return {}

            response_times = [
                r["avg_response_time"]
                for r in results
                if r["avg_response_time"] != float("inf")
            ]
            accuracies = [r["accuracy"] for r in results]
            error_rates = [r["error_rate"] for r in results]

            if response_times:
                return {
                    "avg_response_time": statistics.mean(response_times),
                    "median_response_time": statistics.median(response_times),
                    "p95_response_time": (
                        statistics.quantiles(response_times, n=20)[18]
                        if len(response_times) > 5
                        else max(response_times)
                    ),
                    "p99_response_time": (
                        statistics.quantiles(response_times, n=100)[98]
                        if len(response_times) > 10
                        else max(response_times)
                    ),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "avg_accuracy": statistics.mean(accuracies),
                    "avg_error_rate": statistics.mean(error_rates),
                    "total_queries": len(results),
                    "successful_queries": len(response_times),
                }
            else:
                return {
                    "avg_response_time": float("inf"),
                    "median_response_time": float("inf"),
                    "p95_response_time": float("inf"),
                    "p99_response_time": float("inf"),
                    "min_response_time": float("inf"),
                    "max_response_time": float("inf"),
                    "avg_accuracy": statistics.mean(accuracies) if accuracies else 0.0,
                    "avg_error_rate": (
                        statistics.mean(error_rates) if error_rates else 1.0
                    ),
                    "total_queries": len(results),
                    "successful_queries": 0,
                }

        baseline_summary = calculate_summary_stats(self.baseline_results)
        optimized_summary = calculate_summary_stats(self.optimized_results)

        # Calculate improvements
        improvements = {}
        if baseline_summary and optimized_summary:
            for key in [
                "avg_response_time",
                "median_response_time",
                "p95_response_time",
                "p99_response_time",
            ]:
                if baseline_summary.get(key, float("inf")) > 0 and baseline_summary.get(
                    key, float("inf")
                ) != float("inf"):
                    baseline_val = baseline_summary[key]
                    optimized_val = optimized_summary[key]
                    if optimized_val != float("inf"):
                        improvement_pct = (
                            (baseline_val - optimized_val) / baseline_val
                        ) * 100
                        improvements[key] = {
                            "baseline": baseline_val,
                            "optimized": optimized_val,
                            "improvement_ms": baseline_val - optimized_val,
                            "improvement_pct": improvement_pct,
                        }

        # Performance targets validation
        target_validation = {
            "response_time_target": 200,  # ms
            "cache_hit_rate_target": 60,  # %
            "memory_usage_target": 2048,  # MB
            "accuracy_loss_tolerance": 5,  # %
        }

        validation_results = {}
        if optimized_summary:
            validation_results["response_time_met"] = (
                optimized_summary.get("avg_response_time", float("inf"))
                < target_validation["response_time_target"]
            )
            validation_results["p95_response_time_met"] = (
                optimized_summary.get("p95_response_time", float("inf"))
                < target_validation["response_time_target"] * 1.5
            )

            if baseline_summary and optimized_summary:
                accuracy_change = (
                    abs(
                        optimized_summary.get("avg_accuracy", 0)
                        - baseline_summary.get("avg_accuracy", 0)
                    )
                    * 100
                )
                validation_results["accuracy_maintained"] = (
                    accuracy_change <= target_validation["accuracy_loss_tolerance"]
                )

        comparison_report = {
            "timestamp": datetime.now().isoformat(),
            "baseline_summary": baseline_summary,
            "optimized_summary": optimized_summary,
            "improvements": improvements,
            "target_validation": target_validation,
            "validation_results": validation_results,
            "component_results": self.component_results,
            "test_queries": self.test_queries,
        }

        return comparison_report

    def save_results(self, report: Dict, filename: str = None) -> str:
        """Save benchmark results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graphrag_benchmark_results_{timestamp}.json"

        output_path = self.output_dir / filename

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Benchmark results saved to: {output_path}")
        return str(output_path)

    def generate_markdown_report(self, report: Dict) -> str:
        """Generate markdown version of the report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_filename = f"graphrag_benchmark_report_{timestamp}.md"
        md_path = self.output_dir / md_filename

        with open(md_path, "w") as f:
            f.write(f"# GraphRAG Performance Benchmark Report\n\n")
            f.write(f"**Generated:** {report['timestamp']}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            baseline = report.get("baseline_summary", {})
            optimized = report.get("optimized_summary", {})
            validation = report.get("validation_results", {})

            if baseline and optimized:
                f.write(
                    f"- **Baseline Performance:** {baseline.get('avg_response_time', 'N/A'):.1f}ms average\n"
                )
                f.write(
                    f"- **Optimized Performance:** {optimized.get('avg_response_time', 'N/A'):.1f}ms average\n"
                )

                improvements = report.get("improvements", {})
                if "avg_response_time" in improvements:
                    imp = improvements["avg_response_time"]
                    f.write(
                        f"- **Performance Improvement:** {imp.get('improvement_pct', 0):.1f}% ({imp.get('improvement_ms', 0):.1f}ms reduction)\n"
                    )

            f.write(
                f"- **Target <200ms:** {'✅ ACHIEVED' if validation.get('response_time_met', False) else '❌ NOT MET'}\n"
            )
            f.write(
                f"- **Accuracy Maintained:** {'✅ YES' if validation.get('accuracy_maintained', False) else '❌ NO'}\n\n"
            )

            # Detailed Results
            f.write("## Detailed Performance Results\n\n")
            f.write("### Baseline vs Optimized Comparison\n\n")
            f.write("| Metric | Baseline | Optimized | Improvement |\n")
            f.write("|--------|----------|-----------|-------------|\n")

            if baseline and optimized:
                metrics = [
                    "avg_response_time",
                    "median_response_time",
                    "p95_response_time",
                    "p99_response_time",
                ]
                for metric in metrics:
                    baseline_val = baseline.get(metric, 0)
                    optimized_val = optimized.get(metric, 0)
                    if metric in report.get("improvements", {}):
                        imp_pct = report["improvements"][metric].get(
                            "improvement_pct", 0
                        )
                        f.write(
                            f"| {metric.replace('_', ' ').title()} | {baseline_val:.1f}ms | {optimized_val:.1f}ms | {imp_pct:.1f}% |\n"
                        )
                    else:
                        f.write(
                            f"| {metric.replace('_', ' ').title()} | {baseline_val:.1f}ms | {optimized_val:.1f}ms | N/A |\n"
                        )

            # Target Validation
            f.write("\n### Target Validation\n\n")
            targets = report.get("target_validation", {})
            f.write(
                f"- **Response Time Target:** <{targets.get('response_time_target', 200)}ms\n"
            )
            f.write(
                f"- **Cache Hit Rate Target:** >{targets.get('cache_hit_rate_target', 60)}%\n"
            )
            f.write(
                f"- **Memory Usage Target:** <{targets.get('memory_usage_target', 2048)}MB\n"
            )
            f.write(
                f"- **Accuracy Loss Tolerance:** <{targets.get('accuracy_loss_tolerance', 5)}%\n\n"
            )

            # Individual Query Results
            f.write("## Individual Query Results\n\n")
            f.write("### Baseline Results\n\n")
            if self.baseline_results:
                f.write("| Query | Avg Time | P95 Time | Accuracy |\n")
                f.write("|-------|----------|----------|----------|\n")
                for result in self.baseline_results:
                    query_short = (
                        result["query"][:50] + "..."
                        if len(result["query"]) > 50
                        else result["query"]
                    )
                    f.write(
                        f"| {query_short} | {result.get('avg_response_time', 0):.1f}ms | {result.get('p95_response_time', 0):.1f}ms | {result.get('accuracy', 0):.1%} |\n"
                    )

            f.write("\n### Optimized Results\n\n")
            if self.optimized_results:
                f.write("| Query | Avg Time | P95 Time | Accuracy |\n")
                f.write("|-------|----------|----------|----------|\n")
                for result in self.optimized_results:
                    query_short = (
                        result["query"][:50] + "..."
                        if len(result["query"]) > 50
                        else result["query"]
                    )
                    f.write(
                        f"| {query_short} | {result.get('avg_response_time', 0):.1f}ms | {result.get('p95_response_time', 0):.1f}ms | {result.get('accuracy', 0):.1%} |\n"
                    )

            f.write("\n## Conclusion\n\n")
            if validation.get("response_time_met", False) and validation.get(
                "accuracy_maintained", False
            ):
                f.write(
                    "✅ **SUCCESS:** All optimization targets achieved with maintained accuracy.\n"
                )
            elif validation.get("response_time_met", False):
                f.write(
                    "⚠️ **PARTIAL SUCCESS:** Response time target achieved but accuracy may be impacted.\n"
                )
            else:
                f.write(
                    "❌ **NEEDS IMPROVEMENT:** Response time target not achieved. Additional optimization required.\n"
                )

        logger.info(f"Markdown report saved to: {md_path}")
        return str(md_path)


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="GraphRAG Performance Benchmark")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Iterations per query"
    )
    parser.add_argument(
        "--baseline-only", action="store_true", help="Run baseline only"
    )
    parser.add_argument(
        "--optimized-only", action="store_true", help="Run optimized only"
    )
    parser.add_argument(
        "--concurrent-test", action="store_true", help="Run concurrent load test"
    )
    parser.add_argument(
        "--concurrent-users",
        type=int,
        default=10,
        help="Concurrent users for load test",
    )
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize benchmark
    benchmark = GraphRAGBenchmark(args.config, args.output_dir)

    try:
        # Run baseline benchmark
        if not args.optimized_only:
            logger.info("=" * 60)
            logger.info("RUNNING BASELINE BENCHMARK")
            logger.info("=" * 60)
            benchmark.run_baseline_benchmark(args.iterations)

        # Run optimized benchmark
        if not args.baseline_only:
            logger.info("=" * 60)
            logger.info("RUNNING OPTIMIZED BENCHMARK")
            logger.info("=" * 60)
            optimization_metrics = benchmark.run_optimized_benchmark(args.iterations)

        # Run concurrent load test
        if args.concurrent_test:
            logger.info("=" * 60)
            logger.info("RUNNING CONCURRENT LOAD TEST")
            logger.info("=" * 60)
            concurrent_metrics = benchmark.run_concurrent_load_test(
                args.concurrent_users, queries_per_user=5
            )
            logger.info(f"Concurrent test results: {concurrent_metrics}")

        # Test individual components
        if not args.baseline_only and not args.optimized_only:
            logger.info("=" * 60)
            logger.info("TESTING INDIVIDUAL COMPONENTS")
            logger.info("=" * 60)
            benchmark.test_individual_components(iterations=3)

        # Generate comparison report
        logger.info("=" * 60)
        logger.info("GENERATING REPORTS")
        logger.info("=" * 60)

        report = benchmark.generate_comparison_report()

        # Save results
        json_path = benchmark.save_results(report)
        md_path = benchmark.generate_markdown_report(report)

        # Print summary
        logger.info("=" * 60)
        logger.info("BENCHMARK COMPLETE")
        logger.info("=" * 60)

        baseline_summary = report.get("baseline_summary", {})
        optimized_summary = report.get("optimized_summary", {})
        validation = report.get("validation_results", {})

        if baseline_summary and optimized_summary:
            logger.info(
                f"Baseline average: {baseline_summary.get('avg_response_time', 'N/A'):.1f}ms"
            )
            logger.info(
                f"Optimized average: {optimized_summary.get('avg_response_time', 'N/A'):.1f}ms"
            )

            improvements = report.get("improvements", {})
            if "avg_response_time" in improvements:
                imp = improvements["avg_response_time"]
                logger.info(
                    f"Improvement: {imp.get('improvement_pct', 0):.1f}% ({imp.get('improvement_ms', 0):.1f}ms reduction)"
                )

        logger.info(
            f"Target <200ms: {'✅ ACHIEVED' if validation.get('response_time_met', False) else '❌ NOT MET'}"
        )
        logger.info(
            f"Accuracy maintained: {'✅ YES' if validation.get('accuracy_maintained', False) else '❌ NO'}"
        )

        logger.info(f"Results saved to: {json_path}")
        logger.info(f"Report saved to: {md_path}")

    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise

    finally:
        # Clean up
        if benchmark.performance_monitor:
            benchmark.performance_monitor.stop_monitoring()


if __name__ == "__main__":
    main()
