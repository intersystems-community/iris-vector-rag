#!/usr/bin/env python
# demo_benchmark_analysis.py
"""
Demonstration script showing how to use the comparative analysis package
to analyze and visualize RAG benchmark results.
"""

import json
import os
from typing import Dict, Any
from datetime import datetime

from eval.comparative import (
    calculate_technique_comparison,
    generate_comparison_chart,
    generate_comparative_bar_chart,
    generate_combined_report,
    REFERENCE_BENCHMARKS
)

def load_benchmark_results(results_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load benchmark results from a JSON file.
    
    Args:
        results_file: Path to JSON file containing benchmark results
        
    Returns:
        Dictionary mapping technique names to their benchmark results
    """
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        # Generate sample results for demonstration
        return generate_sample_results()
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return results

def generate_sample_results() -> Dict[str, Dict[str, Any]]:
    """
    Generate sample benchmark results for demonstration.
    
    Returns:
        Dictionary mapping technique names to their benchmark results
    """
    print("Generating sample benchmark results for demonstration")
    
    # Sample benchmark results with metrics
    return {
        "basic_rag": {
            "pipeline": "basic_rag",
            "timestamp": datetime.now().isoformat(),
            "queries_run": 100,
            "metrics": {
                "context_recall": 0.72,
                "answer_faithfulness": 0.80,
                "answer_relevance": 0.75,
                "latency_p50": 105,
                "latency_p95": 150,
                "throughput_qps": 18.5
            },
            "query_results": [
                {
                    "query": "What are the effects of metformin on type 2 diabetes?",
                    "answer": "Metformin helps manage type 2 diabetes by reducing glucose production in the liver and increasing insulin sensitivity.",
                    "latency_ms": 120
                },
                {
                    "query": "How does SGLT2 inhibition affect kidney function?",
                    "answer": "SGLT2 inhibitors protect kidney function by reducing hyperfiltration and decreasing albuminuria.",
                    "latency_ms": 150
                }
            ]
        },
        "hyde": {
            "pipeline": "hyde",
            "timestamp": datetime.now().isoformat(),
            "queries_run": 100,
            "metrics": {
                "context_recall": 0.78,
                "answer_faithfulness": 0.82,
                "answer_relevance": 0.80,
                "latency_p50": 120,
                "latency_p95": 180,
                "throughput_qps": 15.2
            },
            "query_results": [
                {
                    "query": "What are the effects of metformin on type 2 diabetes?",
                    "answer": "Metformin, a first-line treatment for type 2 diabetes, reduces hepatic glucose production and improves insulin sensitivity in peripheral tissues.",
                    "latency_ms": 130
                },
                {
                    "query": "How does SGLT2 inhibition affect kidney function?",
                    "answer": "SGLT2 inhibitors have renoprotective effects, reducing glomerular hyperfiltration and decreasing albuminuria in patients with diabetic kidney disease.",
                    "latency_ms": 160
                }
            ]
        },
        "colbert": {
            "pipeline": "colbert",
            "timestamp": datetime.now().isoformat(),
            "queries_run": 100,
            "metrics": {
                "context_recall": 0.85,
                "answer_faithfulness": 0.87,
                "answer_relevance": 0.83,
                "latency_p50": 150,
                "latency_p95": 220,
                "throughput_qps": 12.8
            },
            "query_results": [
                {
                    "query": "What are the effects of metformin on type 2 diabetes?",
                    "answer": "Metformin reduces insulin resistance and hepatic glucose production, key mechanisms in treating type 2 diabetes. It also has beneficial effects on lipid metabolism and cardiovascular outcomes.",
                    "latency_ms": 155
                },
                {
                    "query": "How does SGLT2 inhibition affect kidney function?",
                    "answer": "SGLT2 inhibitors reduce intraglomerular pressure through tubuloglomerular feedback mechanisms, decreasing albuminuria and slowing the progression of diabetic kidney disease. Recent studies show they also reduce major adverse kidney events in patients with chronic kidney disease.",
                    "latency_ms": 170
                }
            ]
        }
    }

def main():
    """Main function to demonstrate the comparative analysis workflow."""
    print("RAG Benchmark Analysis Demo")
    print("==========================\n")
    
    # 1. Load benchmark results
    results_file = "benchmark_results/sample_results.json"
    benchmarks = load_benchmark_results(results_file)
    
    # 2. Analyze comparative performance
    print("Calculating comparative metrics...")
    comparison = calculate_technique_comparison(benchmarks)
    
    # Display best techniques
    print("\nBest performing techniques:")
    print(f"- Best for retrieval quality: {comparison['best_technique']['retrieval_quality']}")
    print(f"- Best for answer quality: {comparison['best_technique']['answer_quality']}")
    print(f"- Best for performance: {comparison['best_technique']['performance']}")
    
    # 3. Generate visualizations
    print("\nGenerating visualization charts...")
    output_dir = os.path.join("benchmark_results", f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for visualization
    metrics = {tech: bench["metrics"] for tech, bench in benchmarks.items()}
    
    # Generate radar chart
    radar_path = generate_comparison_chart(
        metrics=metrics,
        chart_type="radar",
        output_path=os.path.join(output_dir, "radar_comparison.png")
    )
    print(f"- Created radar chart: {radar_path}")
    
    # Generate bar chart for a specific metric
    metric = "context_recall"
    bar_path = generate_comparison_chart(
        metrics=metrics,
        chart_type="bar",
        metric=metric,
        output_path=os.path.join(output_dir, f"bar_{metric}.png")
    )
    print(f"- Created bar chart for {metric}: {bar_path}")
    
    # 4. Compare with published benchmarks
    print("\nComparing with published benchmarks...")
    dataset_type = "multihop"  # This is just for demonstration
    
    # Extract our metrics for comparison
    # In a real scenario, you'd calculate the benchmark-specific metrics first
    our_metrics = {
        "basic_rag": {
            "answer_f1": 0.65,
            "supporting_facts_f1": 0.68,
            "joint_f1": 0.51
        },
        "colbert": {
            "answer_f1": 0.70,
            "supporting_facts_f1": 0.73,
            "joint_f1": 0.58
        },
        "graphrag": {
            "answer_f1": 0.77,
            "supporting_facts_f1": 0.82,
            "joint_f1": 0.67
        }
    }
    
    # Get reference benchmarks for comparison
    if dataset_type in REFERENCE_BENCHMARKS:
        reference_metrics = REFERENCE_BENCHMARKS[dataset_type]
        
        # Generate comparative bar chart for a specific metric
        benchmark_metric = "joint_f1"
        comp_path = generate_comparative_bar_chart(
            our_results={tech: metrics[benchmark_metric] for tech, metrics in our_metrics.items()},
            reference_results={tech: metrics[benchmark_metric] for tech, metrics in reference_metrics.items()},
            metric=benchmark_metric,
            output_path=os.path.join(output_dir, f"comp_{benchmark_metric}.png")
        )
        print(f"- Created comparison with published benchmarks: {comp_path}")
    else:
        print(f"No reference benchmarks available for {dataset_type}")
    
    # 5. Generate comprehensive report
    print("\nGenerating comprehensive report...")
    # We'll enrich our benchmarks with additional data for the report
    for tech in benchmarks:
        if tech in our_metrics:
            benchmarks[tech]["benchmark_metrics"] = our_metrics[tech]
    
    report_paths = generate_combined_report(
        benchmarks=benchmarks,
        output_dir=output_dir,
        dataset_name=dataset_type
    )
    
    print("\nReport generated successfully:")
    print(f"- JSON report: {report_paths['json']}")
    print(f"- Markdown report: {report_paths['markdown']}")
    print(f"- Charts: {len(report_paths['charts'])} visualizations created")
    
    print("\nDemo completed. Check the output directory for generated files.")

if __name__ == "__main__":
    main()
