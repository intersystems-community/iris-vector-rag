#!/usr/bin/env python3
"""
Unified RAG Pipeline Benchmarking System

This script provides a clean, consolidated interface to the existing comprehensive
evaluation framework. It leverages the mature infrastructure already in place:
- evaluation_framework/evaluation_orchestrator.py
- evaluation_framework/real_production_evaluation.py  
- evaluation_framework/comparative_analysis_system.py
- evaluation_framework/ragas_metrics_framework.py

Rather than duplicating functionality, this provides a simplified interface
for common benchmarking tasks while maintaining access to the full power
of the underlying evaluation infrastructure.
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Add evaluation framework to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'evaluation_framework'))

# Import existing evaluation infrastructure
from evaluation_orchestrator import (
    EvaluationOrchestrator,
    EvaluationExperimentConfig,
    create_evaluation_orchestrator
)
from real_production_evaluation import RealProductionEvaluator
from comparative_analysis_system import (
    ComparativeAnalysisSystem,
    PipelineEvaluationConfig,
    create_comparative_analysis_system
)

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkConfig:
    """Simplified benchmark configuration leveraging existing infrastructure."""
    
    # Basic parameters
    experiment_name: str = "rag_pipeline_benchmark"
    num_queries: int = 100
    pipelines: List[str] = field(default_factory=lambda: [
        'BasicRAGPipeline', 'CRAGPipeline', 'GraphRAGPipeline', 'BasicRAGRerankingPipeline'
    ])
    
    # Performance measurement
    measure_latency: bool = True
    measure_throughput: bool = True
    measure_memory: bool = True
    
    # Output configuration
    output_dir: str = "outputs/unified_benchmark"
    generate_html_report: bool = True
    generate_comparison_plots: bool = True
    
    # Infrastructure settings
    use_real_infrastructure: bool = True
    max_workers: int = 4
    
    def to_evaluation_config(self) -> EvaluationExperimentConfig:
        """Convert to the existing evaluation framework configuration."""
        return EvaluationExperimentConfig(
            experiment_name=self.experiment_name,
            test_questions_per_pipeline=self.num_queries,
            pipelines_to_evaluate=self.pipelines,
            max_workers=self.max_workers,
            enable_caching=True,
            enable_checkpointing=True
        )
    
    def to_pipeline_config(self) -> PipelineEvaluationConfig:
        """Convert to the existing pipeline evaluation configuration."""
        return PipelineEvaluationConfig(
            questions_per_pipeline_test=self.num_queries,
            max_workers=self.max_workers,
            output_dir=self.output_dir,
            generate_plots=self.generate_comparison_plots,
            generate_interactive_dashboard=self.generate_html_report
        )


class UnifiedRAGBenchmark:
    """
    Unified benchmark interface leveraging existing evaluation infrastructure.
    
    This class provides a clean, simple interface to the comprehensive evaluation
    framework while avoiding duplication of existing functionality.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        self.start_time = None
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Unified RAG Benchmark initialized")
        logger.info(f"Pipelines to benchmark: {config.pipelines}")
        logger.info(f"Number of queries: {config.num_queries}")
        logger.info(f"Output directory: {config.output_dir}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """
        Run performance benchmark using existing real production evaluator.
        
        Returns:
            Benchmark results with performance metrics
        """
        logger.info("Starting performance benchmark using real production infrastructure")
        self.start_time = time.time()
        
        try:
            # Use existing real production evaluator
            evaluator = RealProductionEvaluator()
            
            # Run evaluation with performance measurement
            performance_results = {}
            
            for pipeline_name in self.config.pipelines:
                if pipeline_name in evaluator.pipelines:
                    logger.info(f"Benchmarking {pipeline_name}")
                    
                    # Measure pipeline performance
                    pipeline_metrics = self._measure_pipeline_performance(
                        evaluator.pipelines[pipeline_name],
                        pipeline_name
                    )
                    performance_results[pipeline_name] = pipeline_metrics
                else:
                    logger.warning(f"Pipeline {pipeline_name} not available")
            
            self.results['performance'] = performance_results
            self.results['benchmark_duration'] = time.time() - self.start_time
            
            logger.info("Performance benchmark completed")
            return self.results
            
        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            raise
    
    def run_quality_benchmark(self) -> Dict[str, Any]:
        """
        Run quality benchmark using existing comparative analysis system.
        
        Returns:
            Quality evaluation results
        """
        logger.info("Starting quality benchmark using comparative analysis system")
        
        try:
            # Use existing comparative analysis system
            pipeline_config = self.config.to_pipeline_config()
            analysis_system = create_comparative_analysis_system(pipeline_config)
            
            # Run comprehensive evaluation
            quality_results = analysis_system.run_comprehensive_evaluation()
            
            self.results['quality'] = quality_results
            
            logger.info("Quality benchmark completed")
            return self.results
            
        except Exception as e:
            logger.error(f"Quality benchmark failed: {e}")
            raise
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run both performance and quality benchmarks.
        
        Returns:
            Complete benchmark results
        """
        logger.info("Starting full benchmark (performance + quality)")
        
        # Run performance benchmark
        self.run_performance_benchmark()
        
        # Run quality benchmark
        self.run_quality_benchmark()
        
        # Generate consolidated report
        self._generate_consolidated_report()
        
        return self.results
    
    def _measure_pipeline_performance(self, pipeline, pipeline_name: str) -> Dict[str, Any]:
        """Measure performance metrics for a single pipeline."""
        metrics = {
            'pipeline_name': pipeline_name,
            'latency_measurements': [],
            'memory_measurements': [],
            'success_count': 0,
            'total_queries': 0
        }
        
        # Sample queries for testing
        test_queries = [
            "What are the symptoms of diabetes?",
            "How is cancer diagnosed?",
            "What causes heart disease?",
            "What are the side effects of chemotherapy?",
            "How is blood pressure measured?"
        ]
        
        # Limit to configured number of queries
        queries_to_test = test_queries[:min(len(test_queries), self.config.num_queries)]
        
        for i, query in enumerate(queries_to_test):
            try:
                metrics['total_queries'] += 1
                
                # Measure latency
                if self.config.measure_latency:
                    start_time = time.time()
                    result = pipeline.query(query)
                    end_time = time.time()
                    
                    latency = (end_time - start_time) * 1000  # Convert to milliseconds
                    metrics['latency_measurements'].append(latency)
                    
                    if result and hasattr(result, 'answer') and result.answer:
                        metrics['success_count'] += 1
                
                # Memory measurement would require process monitoring
                if self.config.measure_memory:
                    # Placeholder for memory measurement
                    metrics['memory_measurements'].append(0.0)
                
                if i % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(queries_to_test)} queries for {pipeline_name}")
                    
            except Exception as e:
                logger.warning(f"Query failed for {pipeline_name}: {e}")
                continue
        
        # Calculate summary statistics
        if metrics['latency_measurements']:
            metrics['avg_latency_ms'] = sum(metrics['latency_measurements']) / len(metrics['latency_measurements'])
            metrics['p95_latency_ms'] = sorted(metrics['latency_measurements'])[int(0.95 * len(metrics['latency_measurements']))]
            metrics['p99_latency_ms'] = sorted(metrics['latency_measurements'])[int(0.99 * len(metrics['latency_measurements']))]
        
        metrics['success_rate'] = metrics['success_count'] / metrics['total_queries'] if metrics['total_queries'] > 0 else 0.0
        
        if self.config.measure_throughput and metrics['total_queries'] > 0:
            total_time = sum(metrics['latency_measurements']) / 1000  # Convert to seconds
            metrics['throughput_qps'] = metrics['total_queries'] / total_time if total_time > 0 else 0.0
        
        return metrics
    
    def _generate_consolidated_report(self):
        """Generate consolidated HTML and JSON reports."""
        logger.info("Generating consolidated reports")
        
        # Save JSON results
        results_file = self.output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Generate HTML report if requested
        if self.config.generate_html_report:
            self._generate_html_report(results_file)
        
        logger.info(f"Reports saved to {self.output_dir}")
    
    def _generate_html_report(self, results_file: Path):
        """Generate simple HTML report."""
        html_file = self.output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Pipeline Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .pipeline {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px 15px 10px 0; }}
                .metric-value {{ font-weight: bold; color: #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RAG Pipeline Benchmark Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Configuration: {self.config.num_queries} queries across {len(self.config.pipelines)} pipelines</p>
            </div>
            
            <div class="section">
                <h2>Performance Results</h2>
                {self._generate_performance_html()}
            </div>
            
            <div class="section">
                <h2>Quality Results</h2>
                <p>Detailed quality metrics available in: <a href="{results_file.name}">{results_file.name}</a></p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>Complete benchmark results and raw data available in the JSON file.</p>
                <p>For detailed analysis, use the existing evaluation framework tools in the evaluation_framework/ directory.</p>
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    def _generate_performance_html(self) -> str:
        """Generate HTML for performance results."""
        if 'performance' not in self.results:
            return "<p>No performance results available</p>"
        
        html = ""
        for pipeline_name, metrics in self.results['performance'].items():
            html += f"""
            <div class="pipeline">
                <h3>{pipeline_name}</h3>
                <div class="metric">Success Rate: <span class="metric-value">{metrics.get('success_rate', 0):.2%}</span></div>
                <div class="metric">Avg Latency: <span class="metric-value">{metrics.get('avg_latency_ms', 0):.1f}ms</span></div>
                <div class="metric">P95 Latency: <span class="metric-value">{metrics.get('p95_latency_ms', 0):.1f}ms</span></div>
                <div class="metric">Throughput: <span class="metric-value">{metrics.get('throughput_qps', 0):.2f} QPS</span></div>
            </div>
            """
        return html


def main():
    """Main entry point for unified benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified RAG Pipeline Benchmark")
    parser.add_argument('--experiment-name', default='rag_benchmark', help='Experiment name')
    parser.add_argument('--num-queries', type=int, default=100, help='Number of queries to test')
    parser.add_argument('--pipelines', nargs='+', 
                       default=['BasicRAGPipeline', 'CRAGPipeline', 'GraphRAGPipeline', 'BasicRAGRerankingPipeline'],
                       help='Pipelines to benchmark')
    parser.add_argument('--output-dir', default='outputs/unified_benchmark', help='Output directory')
    parser.add_argument('--benchmark-type', choices=['performance', 'quality', 'full'], default='full',
                       help='Type of benchmark to run')
    parser.add_argument('--no-html', action='store_true', help='Skip HTML report generation')
    
    args = parser.parse_args()
    
    # Create configuration
    config = BenchmarkConfig(
        experiment_name=args.experiment_name,
        num_queries=args.num_queries,
        pipelines=args.pipelines,
        output_dir=args.output_dir,
        generate_html_report=not args.no_html
    )
    
    # Run benchmark
    benchmark = UnifiedRAGBenchmark(config)
    
    try:
        if args.benchmark_type == 'performance':
            results = benchmark.run_performance_benchmark()
        elif args.benchmark_type == 'quality':
            results = benchmark.run_quality_benchmark()
        else:  # 'full'
            results = benchmark.run_full_benchmark()
        
        print(f"\nBenchmark completed successfully!")
        print(f"Results saved to: {config.output_dir}")
        
        if 'performance' in results:
            print("\nPerformance Summary:")
            for pipeline, metrics in results['performance'].items():
                print(f"  {pipeline}: {metrics.get('success_rate', 0):.1%} success, "
                      f"{metrics.get('avg_latency_ms', 0):.1f}ms avg latency")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()