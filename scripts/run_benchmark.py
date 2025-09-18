#!/usr/bin/env python3
"""
Simple Benchmark Runner

A streamlined script to run RAG pipeline benchmarks using the unified interface
that leverages the existing comprehensive evaluation framework.

Usage Examples:
    # Quick performance test with 50 queries
    python scripts/run_benchmark.py --quick
    
    # Full benchmark with all pipelines
    python scripts/run_benchmark.py --full
    
    # Custom benchmark
    python scripts/run_benchmark.py --num-queries 200 --pipelines BasicRAGPipeline CRAGPipeline
    
    # Performance only (fast)
    python scripts/run_benchmark.py --performance-only --num-queries 25
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_rag_benchmark import UnifiedRAGBenchmark, BenchmarkConfig

def setup_environment():
    """Setup environment and validate requirements."""
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("Warning: .env file not found. Some features may not work.")
        print("Please ensure environment variables are properly configured.")
    
    # Check if evaluation framework is available
    eval_framework_path = Path('evaluation_framework')
    if not eval_framework_path.exists():
        print("Error: evaluation_framework directory not found.")
        print("This script requires the existing evaluation infrastructure.")
        sys.exit(1)
    
    print("Environment validation passed ✓")

def run_quick_benchmark():
    """Run a quick benchmark for testing."""
    print("🚀 Running Quick Benchmark (50 queries, 2 pipelines)")
    
    config = BenchmarkConfig(
        experiment_name="quick_benchmark",
        num_queries=50,
        pipelines=['BasicRAGPipeline', 'CRAGPipeline'],
        output_dir="outputs/quick_benchmark"
    )
    
    benchmark = UnifiedRAGBenchmark(config)
    results = benchmark.run_performance_benchmark()
    
    print("✅ Quick benchmark completed!")
    return results

def run_full_benchmark():
    """Run full comprehensive benchmark."""
    print("🔥 Running Full Benchmark (100 queries, all pipelines)")
    
    config = BenchmarkConfig(
        experiment_name="full_benchmark", 
        num_queries=100,
        pipelines=['BasicRAGPipeline', 'CRAGPipeline', 'GraphRAGPipeline', 'BasicRAGRerankingPipeline'],
        output_dir="outputs/full_benchmark"
    )
    
    benchmark = UnifiedRAGBenchmark(config)
    results = benchmark.run_full_benchmark()
    
    print("✅ Full benchmark completed!")
    return results

def run_performance_only(num_queries=25):
    """Run performance-only benchmark (fast)."""
    print(f"⚡ Running Performance-Only Benchmark ({num_queries} queries)")
    
    config = BenchmarkConfig(
        experiment_name="performance_benchmark",
        num_queries=num_queries,
        pipelines=['BasicRAGPipeline', 'CRAGPipeline'],
        output_dir="outputs/performance_benchmark"
    )
    
    benchmark = UnifiedRAGBenchmark(config)
    results = benchmark.run_performance_benchmark()
    
    print("✅ Performance benchmark completed!")
    return results

def run_custom_benchmark(num_queries, pipelines):
    """Run custom benchmark with specified parameters."""
    print(f"⚙️  Running Custom Benchmark ({num_queries} queries, {len(pipelines)} pipelines)")
    
    config = BenchmarkConfig(
        experiment_name="custom_benchmark",
        num_queries=num_queries,
        pipelines=pipelines,
        output_dir="outputs/custom_benchmark"
    )
    
    benchmark = UnifiedRAGBenchmark(config)
    results = benchmark.run_performance_benchmark()
    
    print("✅ Custom benchmark completed!")
    return results

def print_results_summary(results):
    """Print a summary of benchmark results."""
    if 'performance' in results:
        print("\n📊 Performance Summary:")
        print("-" * 60)
        for pipeline, metrics in results['performance'].items():
            success_rate = metrics.get('success_rate', 0) * 100
            avg_latency = metrics.get('avg_latency_ms', 0)
            throughput = metrics.get('throughput_qps', 0)
            
            print(f"{pipeline:25} | Success: {success_rate:5.1f}% | "
                  f"Latency: {avg_latency:6.1f}ms | Throughput: {throughput:5.2f} QPS")
    
    if 'benchmark_duration' in results:
        duration = results['benchmark_duration']
        print(f"\n⏱️  Total benchmark time: {duration:.1f} seconds")

def main():
    """Main entry point with simple command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple RAG Pipeline Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --quick                    # Quick test (50 queries, 2 pipelines)
  %(prog)s --full                     # Full benchmark (100 queries, all pipelines)  
  %(prog)s --performance-only         # Fast performance test (25 queries)
  %(prog)s --num-queries 200          # Custom number of queries
  %(prog)s --pipelines BasicRAGPipeline CRAGPipeline  # Specific pipelines
        """
    )
    
    # Preset options
    parser.add_argument('--quick', action='store_true', help='Run quick benchmark (50 queries, 2 pipelines)')
    parser.add_argument('--full', action='store_true', help='Run full benchmark (100 queries, all pipelines)')
    parser.add_argument('--performance-only', action='store_true', help='Run performance-only benchmark (fast)')
    
    # Custom options
    parser.add_argument('--num-queries', type=int, default=100, help='Number of queries to test (default: 100)')
    parser.add_argument('--pipelines', nargs='+', 
                       default=['BasicRAGPipeline', 'CRAGPipeline', 'GraphRAGPipeline', 'BasicRAGRerankingPipeline'],
                       help='Pipelines to benchmark')
    parser.add_argument('--output-dir', default='outputs/benchmark', help='Output directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup environment
    setup_environment()
    
    try:
        # Run appropriate benchmark
        if args.quick:
            results = run_quick_benchmark()
        elif args.full:
            results = run_full_benchmark()
        elif args.performance_only:
            results = run_performance_only(args.num_queries)
        else:
            results = run_custom_benchmark(args.num_queries, args.pipelines)
        
        # Print summary
        print_results_summary(results)
        
        print(f"\n📁 Detailed results saved to: {args.output_dir}")
        print("💡 For advanced analysis, use the evaluation_framework tools directly")
        
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()