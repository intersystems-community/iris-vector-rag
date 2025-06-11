#!/usr/bin/env python3
"""
Generates a performance and RAGAS metrics report from TDD evaluation results.

This script parses JSON results produced by ComprehensiveRAGASEvaluationFramework
(or similar frameworks that output RAGASEvaluationResult and PipelinePerformanceMetrics)
and generates a summary report in Markdown format.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

# Attempt to import dataclasses from the project structure if needed
# This assumes the script might be run from the project root or similar context
import sys
import os
project_root_path = Path(__file__).resolve().parent.parent
if str(project_root_path) not in sys.path:
    sys.path.insert(0, str(project_root_path))

try:
    from eval.comprehensive_ragas_evaluation import PipelinePerformanceMetrics, RAGASEvaluationResult
except ImportError:
    # Define dummy classes if import fails, to allow script to run standalone for basic JSON
    print("Warning: Could not import RAGAS evaluation dataclasses. Using dummy definitions.")
    print("Ensure this script is run in an environment where project modules are accessible for full functionality.")

    class RAGASEvaluationResult:
        # Define minimal fields based on expected JSON structure
        def __init__(self, **kwargs):
            self.pipeline_name = kwargs.get("pipeline_name")
            self.query = kwargs.get("query")
            self.answer = kwargs.get("answer")
            self.contexts = kwargs.get("contexts", [])
            self.ground_truth = kwargs.get("ground_truth")
            self.response_time = kwargs.get("response_time")
            self.documents_retrieved = kwargs.get("documents_retrieved", 0)
            self.success = kwargs.get("success", False)
            self.error = kwargs.get("error")
            self.answer_relevancy = kwargs.get("answer_relevancy")
            self.context_precision = kwargs.get("context_precision")
            self.context_recall = kwargs.get("context_recall")
            self.faithfulness = kwargs.get("faithfulness")
            self.answer_similarity = kwargs.get("answer_similarity")
            self.answer_correctness = kwargs.get("answer_correctness")
            self.avg_similarity_score = kwargs.get("avg_similarity_score")
            self.answer_length = kwargs.get("answer_length", 0)
            self.iteration = kwargs.get("iteration", 0)


    class PipelinePerformanceMetrics:
        def __init__(self, **kwargs):
            self.pipeline_name = kwargs.get("pipeline_name")
            self.total_queries = kwargs.get("total_queries", 0)
            self.success_rate = kwargs.get("success_rate", 0.0)
            self.avg_response_time = kwargs.get("avg_response_time", 0.0)
            self.std_response_time = kwargs.get("std_response_time", 0.0)
            self.avg_documents_retrieved = kwargs.get("avg_documents_retrieved", 0.0)
            self.avg_answer_length = kwargs.get("avg_answer_length", 0.0)
            self.avg_answer_relevancy = kwargs.get("avg_answer_relevancy")
            self.avg_context_precision = kwargs.get("avg_context_precision")
            self.avg_context_recall = kwargs.get("avg_context_recall")
            self.avg_faithfulness = kwargs.get("avg_faithfulness")
            self.avg_answer_similarity = kwargs.get("avg_answer_similarity")
            self.avg_answer_correctness = kwargs.get("avg_answer_correctness")
            self.individual_results = [RAGASEvaluationResult(**res) for res in kwargs.get("individual_results", [])]


def load_results(json_file_path: Path) -> Dict[str, PipelinePerformanceMetrics]:
    """Loads evaluation results from a JSON file."""
    with open(json_file_path, 'r') as f:
        raw_data = json.load(f)

    # Deserialize into PipelinePerformanceMetrics objects
    # This assumes the top level of JSON is a dict of pipeline_name to metrics data
    parsed_results = {}
    for pipeline_name, metrics_data in raw_data.items():
        # If the data is already structured like PipelinePerformanceMetrics, pass it directly
        # This handles cases where the JSON might be directly from framework.save_results
        if isinstance(metrics_data, dict) and "pipeline_name" in metrics_data:
             # Reconstruct individual_results if they are dicts
            if "individual_results" in metrics_data and metrics_data["individual_results"]:
                if isinstance(metrics_data["individual_results"][0], dict): # check if needs reconstruction
                    metrics_data["individual_results"] = [
                        RAGASEvaluationResult(**res_data) for res_data in metrics_data["individual_results"]
                    ]
            parsed_results[pipeline_name] = PipelinePerformanceMetrics(**metrics_data)
        else:
            # This case might not be hit if JSON is well-formed from the framework
            print(f"Warning: Unexpected data structure for pipeline {pipeline_name}. Skipping.")
            continue
            
    return parsed_results

# New stub functions for TDD RAGAS report generation

def collect_tdd_ragas_results(json_file_path: Path) -> Dict[str, PipelinePerformanceMetrics]:
    """
    Collects TDD RAGAS results from a JSON file.
    
    Args:
        json_file_path: Path to the JSON results file
        
    Returns:
        Dict[str, PipelinePerformanceMetrics]: Parsed results keyed by pipeline name
    """
    print(f"Collecting TDD RAGAS results from {json_file_path}")
    
    if not json_file_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_file_path}")
    
    try:
        results = load_results(json_file_path)
        print(f"Successfully loaded results for {len(results)} pipelines")
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        raise

def analyze_performance_metrics(results: Dict[str, PipelinePerformanceMetrics]) -> Dict[str, Any]:
    """
    Analyzes general performance metrics from the results.
    
    Args:
        results: Pipeline performance metrics
        
    Returns:
        Dict[str, Any]: Performance analysis including response times, success rates, etc.
    """
    print("Analyzing performance metrics...")
    
    if not results:
        return {"error": "No results to analyze"}
    
    # Extract performance data
    performance_data = {}
    response_times = []
    success_rates = []
    documents_retrieved = []
    
    for pipeline_name, metrics in results.items():
        performance_data[pipeline_name] = {
            "avg_response_time": metrics.avg_response_time,
            "std_response_time": metrics.std_response_time,
            "success_rate": metrics.success_rate,
            "total_queries": metrics.total_queries,
            "avg_documents_retrieved": metrics.avg_documents_retrieved,
            "avg_answer_length": metrics.avg_answer_length
        }
        
        response_times.append(metrics.avg_response_time)
        success_rates.append(metrics.success_rate)
        documents_retrieved.append(metrics.avg_documents_retrieved)
    
    # Calculate aggregate statistics
    analysis = {
        "summary": f"Performance analysis for {len(results)} pipelines",
        "pipeline_count": len(results),
        "performance_by_pipeline": performance_data,
        "aggregate_statistics": {
            "avg_response_time": {
                "mean": sum(response_times) / len(response_times),
                "min": min(response_times),
                "max": max(response_times),
                "std": pd.Series(response_times).std() if len(response_times) > 1 else 0
            },
            "success_rate": {
                "mean": sum(success_rates) / len(success_rates),
                "min": min(success_rates),
                "max": max(success_rates)
            },
            "documents_retrieved": {
                "mean": sum(documents_retrieved) / len(documents_retrieved),
                "min": min(documents_retrieved),
                "max": max(documents_retrieved)
            }
        },
        "performance_ranking": sorted(
            performance_data.items(),
            key=lambda x: x[1]["avg_response_time"]
        )
    }
    
    return analysis

def analyze_ragas_metrics(results: Dict[str, PipelinePerformanceMetrics]) -> Dict[str, Any]:
    """
    Analyzes RAGAS specific metrics from the results.
    
    Args:
        results: Pipeline performance metrics
        
    Returns:
        Dict[str, Any]: RAGAS analysis including quality scores and distributions
    """
    print("Analyzing RAGAS metrics...")
    
    if not results:
        return {"error": "No results to analyze"}
    
    # Extract RAGAS data
    ragas_data = {}
    all_scores = {
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
        "faithfulness": []
    }
    
    for pipeline_name, metrics in results.items():
        pipeline_ragas = {
            "avg_answer_relevancy": metrics.avg_answer_relevancy,
            "avg_context_precision": metrics.avg_context_precision,
            "avg_context_recall": metrics.avg_context_recall,
            "avg_faithfulness": metrics.avg_faithfulness,
            "individual_scores": []
        }
        
        # Collect individual scores for distribution analysis
        for result in metrics.individual_results:
            if result.success:
                individual_scores = {
                    "answer_relevancy": result.answer_relevancy,
                    "context_precision": result.context_precision,
                    "context_recall": result.context_recall,
                    "faithfulness": result.faithfulness
                }
                pipeline_ragas["individual_scores"].append(individual_scores)
                
                # Add to aggregate collections
                for metric_name, score in individual_scores.items():
                    if score is not None:
                        all_scores[metric_name].append(score)
        
        ragas_data[pipeline_name] = pipeline_ragas
    
    # Calculate aggregate RAGAS statistics
    aggregate_ragas = {}
    for metric_name, scores in all_scores.items():
        if scores:
            aggregate_ragas[metric_name] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "std": pd.Series(scores).std() if len(scores) > 1 else 0,
                "count": len(scores)
            }
        else:
            aggregate_ragas[metric_name] = {
                "mean": 0, "min": 0, "max": 0, "std": 0, "count": 0
            }
    
    analysis = {
        "summary": f"RAGAS analysis for {len(results)} pipelines",
        "ragas_by_pipeline": ragas_data,
        "aggregate_ragas_statistics": aggregate_ragas,
        "quality_ranking": sorted(
            [(name, data["avg_faithfulness"]) for name, data in ragas_data.items()
             if data["avg_faithfulness"] is not None],
            key=lambda x: x[1], reverse=True
        ),
        "threshold_compliance": {
            pipeline_name: {
                "answer_relevancy": (data["avg_answer_relevancy"] or 0) >= 0.7,
                "context_precision": (data["avg_context_precision"] or 0) >= 0.6,
                "context_recall": (data["avg_context_recall"] or 0) >= 0.7,
                "faithfulness": (data["avg_faithfulness"] or 0) >= 0.8
            }
            for pipeline_name, data in ragas_data.items()
        }
    }
    
    return analysis

def analyze_scalability_trends(results: Dict[str, PipelinePerformanceMetrics]) -> Dict[str, Any]:
    """
    Analyzes scalability trends from the results.
    
    For single-file results, this provides pipeline comparison analysis.
    For multi-scale results, this would analyze trends across document counts.
    
    Args:
        results: Pipeline performance metrics
        
    Returns:
        Dict[str, Any]: Scalability analysis including trends and bottlenecks
    """
    print("Analyzing scalability trends...")
    
    if not results:
        return {"error": "No results to analyze"}
    
    # Since we're analyzing a single result file, focus on pipeline comparison
    # and identify potential scalability bottlenecks
    
    pipeline_analysis = {}
    response_time_variance = []
    
    for pipeline_name, metrics in results.items():
        # Analyze response time consistency (lower std = better scalability)
        response_time_consistency = (
            1 - (metrics.std_response_time / metrics.avg_response_time)
            if metrics.avg_response_time > 0 else 0
        )
        
        # Calculate efficiency score (success rate / response time)
        efficiency_score = (
            metrics.success_rate / metrics.avg_response_time
            if metrics.avg_response_time > 0 else 0
        )
        
        pipeline_analysis[pipeline_name] = {
            "avg_response_time": metrics.avg_response_time,
            "response_time_consistency": response_time_consistency,
            "efficiency_score": efficiency_score,
            "documents_per_second": (
                metrics.avg_documents_retrieved / metrics.avg_response_time
                if metrics.avg_response_time > 0 else 0
            ),
            "scalability_score": (response_time_consistency + efficiency_score) / 2
        }
        
        response_time_variance.append(metrics.std_response_time)
    
    # Identify best and worst performing pipelines for scalability
    scalability_ranking = sorted(
        pipeline_analysis.items(),
        key=lambda x: x[1]["scalability_score"],
        reverse=True
    )
    
    analysis = {
        "summary": f"Scalability analysis for {len(results)} pipelines",
        "pipeline_scalability": pipeline_analysis,
        "scalability_ranking": scalability_ranking,
        "bottleneck_analysis": {
            "highest_response_time": max(
                pipeline_analysis.items(),
                key=lambda x: x[1]["avg_response_time"]
            ),
            "lowest_consistency": min(
                pipeline_analysis.items(),
                key=lambda x: x[1]["response_time_consistency"]
            ),
            "most_efficient": max(
                pipeline_analysis.items(),
                key=lambda x: x[1]["efficiency_score"]
            )
        },
        "recommendations": [
            f"Best scalability: {scalability_ranking[0][0]}" if scalability_ranking else "No data",
            f"Needs optimization: {scalability_ranking[-1][0]}" if scalability_ranking else "No data",
            "Consider response time consistency for production deployment",
            "Monitor document retrieval efficiency under load"
        ]
    }
    
    return analysis

def generate_markdown_summary(
    performance_analysis: Dict[str, Any], 
    ragas_analysis: Dict[str, Any], 
    scalability_analysis: Dict[str, Any], 
    input_file_name: str
) -> str:
    """
    Generates a comprehensive Markdown summary from analyzed data.
    
    Args:
        performance_analysis: Performance metrics analysis
        ragas_analysis: RAGAS quality metrics analysis
        scalability_analysis: Scalability trends analysis
        input_file_name: Name of source data file
        
    Returns:
        str: Formatted Markdown report
    """
    print("Generating comprehensive Markdown summary...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    content = [
        f"# TDD RAGAS Performance Report",
        f"",
        f"**Generated:** {timestamp}  ",
        f"**Source Data:** `{input_file_name}`  ",
        f"**Report Type:** Comprehensive TDD+RAGAS Integration Analysis",
        f"",
        f"## Executive Summary",
        f"",
        _generate_executive_summary(performance_analysis, ragas_analysis, scalability_analysis),
        f"",
        f"## Performance Analysis",
        f"",
        _generate_performance_section(performance_analysis),
        f"",
        f"## RAGAS Quality Metrics",
        f"",
        _generate_ragas_section(ragas_analysis),
        f"",
        f"## Scalability Analysis",
        f"",
        _generate_scalability_section(scalability_analysis),
        f"",
        f"## Recommendations",
        f"",
        _generate_recommendations(performance_analysis, ragas_analysis, scalability_analysis),
        f"",
        f"## Detailed Data",
        f"",
        f"<details>",
        f"<summary>Click to expand raw analysis data</summary>",
        f"",
        f"### Performance Analysis Data",
        f"```json",
        json.dumps(performance_analysis, indent=2, default=str),
        f"```",
        f"",
        f"### RAGAS Analysis Data", 
        f"```json",
        json.dumps(ragas_analysis, indent=2, default=str),
        f"```",
        f"",
        f"### Scalability Analysis Data",
        f"```json", 
        json.dumps(scalability_analysis, indent=2, default=str),
        f"```",
        f"",
        f"</details>",
        f"",
        f"---",
        f"*Report generated by TDD RAGAS Performance Analysis Framework*"
    ]
    
    return "\n".join(content)

# Removed old generate_markdown_report function as its logic is replaced by generate_markdown_summary and analysis functions.

def main():
    parser = argparse.ArgumentParser(description="Generate TDD RAGAS Performance Report from JSON results.")
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the JSON results file from RAGAS evaluation."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/tdd_ragas_reports"), # Keep consistent output directory
        help="Directory to save the generated report. Default: reports/tdd_ragas_reports"
    )
    parser.add_argument(
        "--report-name",
        type=str,
        default="tdd_ragas_performance_report", # Keep consistent report name base
        help="Base name for the report file. Timestamp will be added. Default: tdd_ragas_performance_report"
    )

    args = parser.parse_args()

    if not args.input_file.exists() or not args.input_file.is_file():
        print(f"Error: Input file not found or is not a file: {args.input_file}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file_name = f"{args.report_name}_{timestamp}.md"
    report_file_path = args.output_dir / report_file_name

    print(f"Collecting results from: {args.input_file}")
    # Use the new collection function
    results_data = collect_tdd_ragas_results(args.input_file)

    if not results_data:
        print("No results collected. Exiting.")
        sys.exit(1)
    
    print("Analyzing performance metrics...")
    performance_analysis = analyze_performance_metrics(results_data)
    
    print("Analyzing RAGAS metrics...")
    ragas_analysis = analyze_ragas_metrics(results_data)
    
    print("Analyzing scalability trends...")
    scalability_analysis = analyze_scalability_trends(results_data) # This might need more context or multiple files in a real scenario

    print(f"Generating Markdown summary at: {report_file_path}")
    markdown_content = generate_markdown_summary(
        performance_analysis,
        ragas_analysis,
        scalability_analysis,
        args.input_file.name
    )
    
    with open(report_file_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"Markdown report generated successfully at: {report_file_path}")
    print("Report generation complete.")

if __name__ == "__main__":
    main()
def _generate_executive_summary(
    performance_analysis: Dict[str, Any],
    ragas_analysis: Dict[str, Any], 
    scalability_analysis: Dict[str, Any]
) -> str:
    """Generates executive summary section."""
    pipeline_count = performance_analysis.get("pipeline_count", 0)
    
    # Get best performing pipeline
    perf_ranking = performance_analysis.get("performance_ranking", [])
    best_perf_pipeline = perf_ranking[0][0] if perf_ranking else "Unknown"
    
    # Get highest quality pipeline
    quality_ranking = ragas_analysis.get("quality_ranking", [])
    best_quality_pipeline = quality_ranking[0][0] if quality_ranking else "Unknown"
    
    # Get most scalable pipeline
    scalability_ranking = scalability_analysis.get("scalability_ranking", [])
    most_scalable_pipeline = scalability_ranking[0][0] if scalability_ranking else "Unknown"
    
    summary = [
        f"This report analyzes the performance and quality metrics for **{pipeline_count} RAG pipelines** ",
        f"using the TDD+RAGAS integration framework.",
        f"",
        f"**Key Findings:**",
        f"- **Fastest Pipeline:** {best_perf_pipeline}",
        f"- **Highest Quality:** {best_quality_pipeline}",
        f"- **Most Scalable:** {most_scalable_pipeline}",
        f"",
        f"**Overall Assessment:** {'✅ All pipelines meet quality thresholds' if _all_pipelines_compliant(ragas_analysis) else '⚠️ Some pipelines below quality thresholds'}"
    ]
    
    return "\n".join(summary)

def _generate_performance_section(performance_analysis: Dict[str, Any]) -> str:
    """Generates performance analysis section."""
    if "error" in performance_analysis:
        return f"❌ **Error:** {performance_analysis['error']}"
    
    agg_stats = performance_analysis.get("aggregate_statistics", {})
    response_time_stats = agg_stats.get("avg_response_time", {})
    success_rate_stats = agg_stats.get("success_rate", {})
    
    content = [
        f"### Response Time Analysis",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Average Response Time | {response_time_stats.get('mean', 0):.3f}s |",
        f"| Fastest Pipeline | {response_time_stats.get('min', 0):.3f}s |",
        f"| Slowest Pipeline | {response_time_stats.get('max', 0):.3f}s |",
        f"| Standard Deviation | {response_time_stats.get('std', 0):.3f}s |",
        f"",
        f"### Success Rate Analysis",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Average Success Rate | {success_rate_stats.get('mean', 0):.1%} |",
        f"| Best Success Rate | {success_rate_stats.get('max', 0):.1%} |",
        f"| Worst Success Rate | {success_rate_stats.get('min', 0):.1%} |",
        f"",
        f"### Pipeline Performance Ranking",
        f"",
        f"| Rank | Pipeline | Avg Response Time | Success Rate |",
        f"|------|----------|-------------------|--------------|"
    ]
    
    # Add pipeline ranking
    perf_by_pipeline = performance_analysis.get("performance_by_pipeline", {})
    ranking = performance_analysis.get("performance_ranking", [])
    
    for i, (pipeline_name, _) in enumerate(ranking[:10], 1):  # Top 10
        pipeline_data = perf_by_pipeline.get(pipeline_name, {})
        response_time = pipeline_data.get("avg_response_time", 0)
        success_rate = pipeline_data.get("success_rate", 0)
        content.append(f"| {i} | {pipeline_name} | {response_time:.3f}s | {success_rate:.1%} |")
    
    return "\n".join(content)

def _generate_ragas_section(ragas_analysis: Dict[str, Any]) -> str:
    """Generates RAGAS quality metrics section."""
    if "error" in ragas_analysis:
        return f"❌ **Error:** {ragas_analysis['error']}"
    
    agg_stats = ragas_analysis.get("aggregate_ragas_statistics", {})
    
    content = [
        f"### RAGAS Quality Metrics Overview",
        f"",
        f"| Metric | Mean | Min | Max | Std Dev |",
        f"|--------|------|-----|-----|---------|"
    ]
    
    # Add aggregate statistics
    for metric_name, stats in agg_stats.items():
        mean_val = stats.get("mean", 0)
        min_val = stats.get("min", 0)
        max_val = stats.get("max", 0)
        std_val = stats.get("std", 0)
        
        content.append(f"| {metric_name.replace('_', ' ').title()} | {mean_val:.3f} | {min_val:.3f} | {max_val:.3f} | {std_val:.3f} |")
    
    content.extend([
        f"",
        f"### Quality Ranking by Faithfulness",
        f"",
        f"| Rank | Pipeline | Faithfulness Score |",
        f"|------|----------|-------------------|"
    ])
    
    # Add quality ranking
    quality_ranking = ragas_analysis.get("quality_ranking", [])
    for i, (pipeline_name, score) in enumerate(quality_ranking[:10], 1):
        content.append(f"| {i} | {pipeline_name} | {score:.3f} |")
    
    # Add threshold compliance
    content.extend([
        f"",
        f"### Threshold Compliance",
        f"",
        f"| Pipeline | Answer Relevancy | Context Precision | Context Recall | Faithfulness |",
        f"|----------|------------------|-------------------|----------------|--------------|"
    ])
    
    threshold_compliance = ragas_analysis.get("threshold_compliance", {})
    for pipeline_name, compliance in threshold_compliance.items():
        ar_status = "✅" if compliance.get("answer_relevancy", False) else "❌"
        cp_status = "✅" if compliance.get("context_precision", False) else "❌"
        cr_status = "✅" if compliance.get("context_recall", False) else "❌"
        f_status = "✅" if compliance.get("faithfulness", False) else "❌"
        
        content.append(f"| {pipeline_name} | {ar_status} | {cp_status} | {cr_status} | {f_status} |")
    
    return "\n".join(content)

def _generate_scalability_section(scalability_analysis: Dict[str, Any]) -> str:
    """Generates scalability analysis section."""
    if "error" in scalability_analysis:
        return f"❌ **Error:** {scalability_analysis['error']}"
    
    content = [
        f"### Scalability Ranking",
        f"",
        f"| Rank | Pipeline | Scalability Score | Efficiency Score | Consistency |",
        f"|------|----------|-------------------|------------------|-------------|"
    ]
    
    # Add scalability ranking
    scalability_ranking = scalability_analysis.get("scalability_ranking", [])
    for i, (pipeline_name, metrics) in enumerate(scalability_ranking[:10], 1):
        scalability_score = metrics.get("scalability_score", 0)
        efficiency_score = metrics.get("efficiency_score", 0)
        consistency = metrics.get("response_time_consistency", 0)
        
        content.append(f"| {i} | {pipeline_name} | {scalability_score:.3f} | {efficiency_score:.3f} | {consistency:.3f} |")
    
    # Add bottleneck analysis
    bottleneck_analysis = scalability_analysis.get("bottleneck_analysis", {})
    content.extend([
        f"",
        f"### Bottleneck Analysis",
        f"",
        f"- **Highest Response Time:** {bottleneck_analysis.get('highest_response_time', ['Unknown', {}])[0]}",
        f"- **Lowest Consistency:** {bottleneck_analysis.get('lowest_consistency', ['Unknown', {}])[0]}",
        f"- **Most Efficient:** {bottleneck_analysis.get('most_efficient', ['Unknown', {}])[0]}"
    ])
    
    return "\n".join(content)

def _generate_recommendations(
    performance_analysis: Dict[str, Any],
    ragas_analysis: Dict[str, Any],
    scalability_analysis: Dict[str, Any]
) -> str:
    """Generates recommendations section."""
    recommendations = []
    
    # Performance recommendations
    perf_ranking = performance_analysis.get("performance_ranking", [])
    if perf_ranking:
        fastest_pipeline = perf_ranking[0][0]
        recommendations.append(f"🚀 **Performance:** Consider {fastest_pipeline} for latency-critical applications")
    
    # Quality recommendations
    quality_ranking = ragas_analysis.get("quality_ranking", [])
    if quality_ranking:
        highest_quality = quality_ranking[0][0]
        recommendations.append(f"🎯 **Quality:** {highest_quality} provides the best answer quality")
    
    # Scalability recommendations
    scalability_ranking = scalability_analysis.get("scalability_ranking", [])
    if scalability_ranking:
        most_scalable = scalability_ranking[0][0]
        recommendations.append(f"📈 **Scalability:** {most_scalable} shows best scalability characteristics")
    
    # Compliance recommendations
    if not _all_pipelines_compliant(ragas_analysis):
        recommendations.append("⚠️ **Quality Improvement:** Some pipelines need optimization to meet RAGAS thresholds")
    
    # General recommendations
    recommendations.extend([
        "🔍 **Monitoring:** Implement continuous monitoring of response times and quality metrics",
        "🧪 **Testing:** Regular TDD+RAGAS evaluation should be part of CI/CD pipeline",
        "📊 **Optimization:** Focus on pipelines with low consistency scores for stability improvements"
    ])
    
    return "\n".join([f"- {rec}" for rec in recommendations])

def _all_pipelines_compliant(ragas_analysis: Dict[str, Any]) -> bool:
    """Checks if all pipelines meet RAGAS quality thresholds."""
    threshold_compliance = ragas_analysis.get("threshold_compliance", {})
    
    for pipeline_compliance in threshold_compliance.values():
        if not all(pipeline_compliance.values()):
            return False
    
    return True