# eval/comparative/reporting.py
"""
Report generation functions for benchmark results.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from eval.metrics import calculate_benchmark_metrics
from eval.comparative.visualization import (
    generate_radar_chart, 
    generate_bar_chart, 
    generate_comparative_bar_chart
)
from eval.comparative.reference_data import REFERENCE_BENCHMARKS

def generate_combined_report(benchmarks: Dict[str, Dict[str, Any]], 
                            output_dir: str = None,
                            dataset_name: str = "medical") -> Dict[str, str]:
    """
    Generate a comprehensive comparative report in multiple formats.
    
    Args:
        benchmarks: Dictionary mapping technique names to their benchmark results
        output_dir: Directory to save report files
        dataset_name: Type of dataset used (medical, multihop, bioasq)
        
    Returns:
        Dictionary mapping report types to their file paths
    """
    if not benchmarks:
        raise ValueError("Benchmarks dictionary is empty")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join("benchmark_results", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result paths
    report_paths = {
        "json": os.path.join(output_dir, "benchmark_results.json"),
        "markdown": os.path.join(output_dir, "benchmark_report.md"),
        "charts": []
    }
    
    # 1. Calculate comparisons
    from eval.comparative.analysis import calculate_technique_comparison
    comparison = calculate_technique_comparison(benchmarks)
    
    # 2. Extract metrics for visualization
    all_metrics = {}
    for tech, bench in benchmarks.items():
        if "metrics" in bench:
            all_metrics[tech] = bench["metrics"]
    
    # 3. Calculate benchmark-specific metrics if we have the right dataset
    benchmark_metrics = {}
    
    if dataset_name in ["multihop", "bioasq"]:
        for tech_name, tech_results in benchmarks.items():
            # Calculate benchmark-specific metrics
            if "query_results" in tech_results and tech_results["query_results"]:
                # Extract queries from results
                queries = [{"query": res["query"], "type": res.get("type", "")} 
                          for res in tech_results["query_results"] if isinstance(res, dict)]
                
                # Calculate metrics
                tech_benchmark_metrics = calculate_benchmark_metrics(
                    tech_results["query_results"], queries, dataset_name)
                
                # Add to overall metrics
                benchmark_metrics[tech_name] = tech_benchmark_metrics
    
    # 4. Generate visualizations
    
    # Radar chart for overall comparison
    try:
        # Create normalized metrics for radar chart
        from eval.metrics import normalize_metrics
        normalized_metrics = {}
        for tech, metrics in all_metrics.items():
            normalized_metrics[tech] = normalize_metrics(metrics, invert_latency=True, scale_to_unit=True)
        
        radar_path = generate_radar_chart(
            normalized_metrics, 
            os.path.join(output_dir, "radar_comparison.png")
        )
        report_paths["charts"].append({"type": "radar", "path": radar_path})
    except Exception as e:
        print(f"Error generating radar chart: {e}")
    
    # Bar charts for individual metrics
    metric_categories = {
        "retrieval": ["context_recall", "precision_at_5"],
        "answer": ["answer_faithfulness", "answer_relevance"],
        "performance": ["throughput_qps", "latency_p50", "latency_p95"]
    }
    
    for category, metrics_list in metric_categories.items():
        for metric in metrics_list:
            # Check if this metric exists in our data
            if any(metric in metrics for metrics in all_metrics.values()):
                try:
                    # Determine if lower is better
                    lower_is_better = any(perf in metric for perf in ["latency", "p50", "p95", "p99"])
                    
                    bar_path = generate_bar_chart(
                        all_metrics, 
                        metric, 
                        os.path.join(output_dir, f"bar_{metric}.png"),
                        lower_is_better=lower_is_better
                    )
                    report_paths["charts"].append({"type": "bar", "metric": metric, "path": bar_path})
                except Exception as e:
                    print(f"Error generating bar chart for {metric}: {e}")
    
    # Generate comparison charts with reference benchmarks
    if dataset_name in REFERENCE_BENCHMARKS and benchmark_metrics:
        reference_data = REFERENCE_BENCHMARKS[dataset_name]
        
        # For each metric in the reference benchmarks
        for metric in reference_data[list(reference_data.keys())[0]].keys():
            # Extract our metrics
            our_metrics = {tech: bench_metrics.get(metric, 0) 
                         for tech, bench_metrics in benchmark_metrics.items() 
                         if metric in bench_metrics}
            
            # Extract reference metrics
            ref_metrics = {tech: bench_metrics.get(metric, 0) 
                         for tech, bench_metrics in reference_data.items() 
                         if metric in bench_metrics}
            
            if our_metrics and ref_metrics:
                try:
                    # Determine if lower is better
                    lower_is_better = any(perf in metric for perf in ["latency", "p50", "p95", "p99"])
                    
                    comp_path = generate_comparative_bar_chart(
                        our_metrics,
                        ref_metrics,
                        metric,
                        os.path.join(output_dir, f"comparison_{metric}.png"),
                        lower_is_better=lower_is_better
                    )
                    report_paths["charts"].append({
                        "type": "comparative", 
                        "metric": metric, 
                        "path": comp_path
                    })
                except Exception as e:
                    print(f"Error generating comparative chart for {metric}: {e}")
    
    # 5. Save JSON report with all raw data
    with open(report_paths["json"], 'w') as f:
        json.dump({
            "benchmarks": benchmarks,
            "comparison": comparison,
            "benchmark_metrics": benchmark_metrics,
            "generated_at": datetime.now().isoformat(),
            "charts": report_paths["charts"]
        }, f, indent=2)
    
    # 6. Save Markdown report
    with open(report_paths["markdown"], 'w') as f:
        # Title and introduction
        f.write("# RAG Techniques Benchmark Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary of techniques
        f.write("## Benchmark Summary\n\n")
        f.write("The following RAG techniques were benchmarked:\n\n")
        
        for tech in benchmarks.keys():
            f.write(f"- **{tech}**\n")
        
        f.write("\n")
        
        # Best performing techniques
        f.write("## Best Performing Techniques\n\n")
        
        if comparison["best_technique"]["retrieval_quality"]:
            f.write(f"- **Best for Retrieval Quality**: {comparison['best_technique']['retrieval_quality']}\n")
        
        if comparison["best_technique"]["answer_quality"]:
            f.write(f"- **Best for Answer Quality**: {comparison['best_technique']['answer_quality']}\n")
        
        if comparison["best_technique"]["performance"]:
            f.write(f"- **Best for Performance**: {comparison['best_technique']['performance']}\n")
        
        f.write("\n")
        
        # Key metrics
        f.write("## Key Metrics\n\n")
        
        # Group metrics by category
        metric_categories = {
            "Retrieval Quality": ["context_recall", "precision_at_5", "precision_at_10"],
            "Answer Quality": ["answer_faithfulness", "answer_relevance"],
            "Performance": ["latency_p50", "latency_p95", "throughput_qps"]
        }
        
        for category, category_metrics in metric_categories.items():
            f.write(f"### {category}\n\n")
            
            # Create a markdown table for metrics in this category
            f.write("| Technique | " + " | ".join([m.replace('_', ' ').title() for m in category_metrics]) + " |\n")
            f.write("| --- | " + " | ".join(["---" for _ in category_metrics]) + " |\n")
            
            for tech, tech_metrics in all_metrics.items():
                values = []
                for metric in category_metrics:
                    if metric in tech_metrics:
                        # Format based on metric type
                        if "latency" in metric or "p50" in metric or "p95" in metric:
                            values.append(f"{tech_metrics[metric]:.2f} ms")
                        elif "throughput" in metric or "qps" in metric:
                            values.append(f"{tech_metrics[metric]:.2f} q/s")
                        else:
                            values.append(f"{tech_metrics[metric]:.3f}")
                    else:
                        values.append("N/A")
                
                f.write(f"| {tech} | " + " | ".join(values) + " |\n")
            
            f.write("\n")
        
        # Benchmark comparisons if applicable
        if dataset_name in REFERENCE_BENCHMARKS and benchmark_metrics:
            f.write("## Comparison to Published Benchmarks\n\n")
            f.write(f"Our techniques were compared against published benchmarks for {dataset_name} datasets:\n\n")
            
            reference_data = REFERENCE_BENCHMARKS[dataset_name]
            reference_metrics = list(reference_data[list(reference_data.keys())[0]].keys())
            
            # Create a markdown table
            f.write("| Technique | " + " | ".join([m.replace('_', ' ').title() for m in reference_metrics]) + " |\n")
            f.write("| --- | " + " | ".join(["---" for _ in reference_metrics]) + " |\n")
            
            # First add our techniques
            for tech, tech_metrics in benchmark_metrics.items():
                values = []
                for metric in reference_metrics:
                    if metric in tech_metrics:
                        values.append(f"{tech_metrics[metric]:.3f}")
                    else:
                        values.append("N/A")
                
                f.write(f"| {tech} | " + " | ".join(values) + " |\n")
            
            # Then add reference techniques
            for tech, tech_metrics in reference_data.items():
                values = []
                for metric in reference_metrics:
                    if metric in tech_metrics:
                        values.append(f"{tech_metrics[metric]:.3f}")
                    else:
                        values.append("N/A")
                
                f.write(f"| Ref: {tech} | " + " | ".join(values) + " |\n")
            
            f.write("\n")
        
        # Charts
        f.write("## Visualizations\n\n")
        
        for chart in report_paths["charts"]:
            chart_type = chart.get("type")
            chart_path = chart.get("path")
            
            if chart_type == "radar":
                f.write("### Overall Comparison\n\n")
                f.write(f"![Radar Chart Comparison]({os.path.basename(chart_path)})\n\n")
            elif chart_type == "bar":
                metric = chart.get("metric", "").replace('_', ' ').title()
                f.write(f"### {metric} Comparison\n\n")
                f.write(f"![Bar Chart - {metric}]({os.path.basename(chart_path)})\n\n")
            elif chart_type == "comparative":
                metric = chart.get("metric", "").replace('_', ' ').title()
                f.write(f"### {metric} vs Published Benchmarks\n\n")
                f.write(f"![Comparison - {metric}]({os.path.basename(chart_path)})\n\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        
        # Get the overall best technique
        best_techniques = list(comparison["best_technique"].values())
        best_counts = {}
        for tech in best_techniques:
            if tech:
                best_counts[tech] = best_counts.get(tech, 0) + 1
        
        overall_best = max(best_counts.items(), key=lambda x: x[1])[0] if best_counts else None
        
        if overall_best:
            f.write(f"**{overall_best}** emerged as the overall best technique in our benchmarks, ")
            f.write(f"leading in {best_counts[overall_best]} out of 3 categories. ")
        
        f.write("For specific use cases, consider the following recommendations:\n\n")
        
        f.write("- **For retrieval-critical applications**: ")
        if comparison["best_technique"]["retrieval_quality"]:
            f.write(f"Use {comparison['best_technique']['retrieval_quality']}")
        else:
            f.write("No clear winner")
        f.write("\n")
        
        f.write("- **For answer quality focus**: ")
        if comparison["best_technique"]["answer_quality"]:
            f.write(f"Use {comparison['best_technique']['answer_quality']}")
        else:
            f.write("No clear winner")
        f.write("\n")
        
        f.write("- **For performance-critical systems**: ")
        if comparison["best_technique"]["performance"]:
            f.write(f"Use {comparison['best_technique']['performance']}")
        else:
            f.write("No clear winner")
        f.write("\n\n")
        
        # Final note
        f.write("Performance may vary with different datasets, configurations, and specific application requirements. ")
        f.write("These results should be used as guidelines for initial technique selection, ")
        f.write("with additional testing recommended for your specific use case.\n")
    
    return report_paths
