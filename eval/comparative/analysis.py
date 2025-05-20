# eval/comparative/analysis.py
"""
Analysis functions for comparing RAG techniques.
"""

from typing import Dict, Any, List
import numpy as np

def calculate_technique_comparison(benchmarks: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comparative metrics between different RAG techniques.
    
    Args:
        benchmarks: Dictionary mapping technique names to their benchmark results
        
    Returns:
        Dictionary with comparative analysis
    """
    if not benchmarks:
        return {}
        
    # Initialize result structure
    result = {
        "rankings": {},
        "percentage_diff": {},
        "best_technique": {
            "retrieval_quality": None,
            "answer_quality": None,
            "performance": None
        }
    }
    
    # Get all unique metrics
    all_metrics = set()
    for tech, bench in benchmarks.items():
        if "metrics" in bench:
            all_metrics.update(bench["metrics"].keys())
    
    # Categorize metrics
    retrieval_metrics = [m for m in all_metrics if "recall" in m or "precision" in m]
    answer_metrics = [m for m in all_metrics if "answer" in m or "faithfulness" in m or "relevance" in m]
    performance_metrics = [m for m in all_metrics if any(
        perf in m for perf in ["latency", "throughput", "qps", "p50", "p95", "p99"])]
    
    # Calculate rankings for each metric
    for metric in all_metrics:
        techniques = []
        values = []
        
        for tech, bench in benchmarks.items():
            if "metrics" in bench and metric in bench["metrics"]:
                techniques.append(tech)
                values.append(bench["metrics"][metric])
        
        if not techniques:
            continue
        
        # Determine if lower is better for this metric (e.g., latency)
        lower_is_better = any(perf in metric for perf in ["latency", "p50", "p95", "p99"])
        
        # Sort techniques by metric value
        if lower_is_better:
            # For metrics where lower is better (e.g., latency), sort in ascending order
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        else:
            # For metrics where higher is better (e.g., recall), sort in descending order
            sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
        
        # Apply sorting
        ranked_techniques = [techniques[i] for i in sorted_indices]
        sorted_values = [values[i] for i in sorted_indices]
        
        # Store ranking
        result["rankings"][metric] = ranked_techniques
        
        # Calculate percentage differences
        result["percentage_diff"][metric] = {}
        if len(ranked_techniques) > 1:
            # Calculate pairwise percentage differences
            for i, tech1 in enumerate(ranked_techniques):
                for j, tech2 in enumerate(ranked_techniques):
                    if i != j:
                        val1 = benchmarks[tech1]["metrics"][metric]
                        val2 = benchmarks[tech2]["metrics"][metric]
                        
                        # Calculate percentage difference
                        if lower_is_better:
                            if val2 > 0:  # Avoid division by zero
                                # For latency, lower is better, so we invert the calculation
                                # (val2 - val1) / val2 * 100 shows how much faster tech1 is
                                pct_diff = (val2 - val1) / val2 * 100
                                if pct_diff > 0:
                                    # tech1 is better (lower value)
                                    key = f"{tech1}_vs_{tech2}"
                                    result["percentage_diff"][metric][key] = pct_diff
                        else:
                            if val2 > 0:  # Avoid division by zero
                                # For recall etc., higher is better
                                # (val1 - val2) / val2 * 100 shows how much better tech1 is
                                pct_diff = (val1 - val2) / val2 * 100
                                if pct_diff > 0:
                                    # tech1 is better (higher value)
                                    key = f"{tech1}_vs_{tech2}"
                                    result["percentage_diff"][metric][key] = pct_diff
    
    # Determine best technique for each category
    # For retrieval quality
    if retrieval_metrics:
        # Average rankings across all retrieval metrics
        tech_scores = {}
        for metric in retrieval_metrics:
            if metric in result["rankings"]:
                for i, tech in enumerate(result["rankings"][metric]):
                    if tech not in tech_scores:
                        tech_scores[tech] = 0
                    # Lower rank (position) is better
                    tech_scores[tech] += i
        
        if tech_scores:
            # Get the technique with the lowest average rank
            result["best_technique"]["retrieval_quality"] = min(tech_scores.items(), key=lambda x: x[1])[0]
    
    # For answer quality
    if answer_metrics:
        # Average rankings across all answer metrics
        tech_scores = {}
        for metric in answer_metrics:
            if metric in result["rankings"]:
                for i, tech in enumerate(result["rankings"][metric]):
                    if tech not in tech_scores:
                        tech_scores[tech] = 0
                    # Lower rank (position) is better
                    tech_scores[tech] += i
        
        if tech_scores:
            # Get the technique with the lowest average rank
            result["best_technique"]["answer_quality"] = min(tech_scores.items(), key=lambda x: x[1])[0]
    
    # For performance
    if performance_metrics:
        # Average rankings across all performance metrics
        tech_scores = {}
        for metric in performance_metrics:
            if metric in result["rankings"]:
                for i, tech in enumerate(result["rankings"][metric]):
                    if tech not in tech_scores:
                        tech_scores[tech] = 0
                    # Lower rank (position) is better
                    tech_scores[tech] += i
        
        if tech_scores:
            # Get the technique with the lowest average rank
            result["best_technique"]["performance"] = min(tech_scores.items(), key=lambda x: x[1])[0]
    
    return result

def calculate_statistical_significance(benchmarks: Dict[str, Dict[str, Any]], 
                                      metric: str, 
                                      alpha: float = 0.05) -> Dict[str, bool]:
    """
    Calculate whether differences between techniques are statistically significant.
    
    Args:
        benchmarks: Dictionary mapping technique names to their benchmark results
        metric: The metric to analyze for significance
        alpha: Significance level
        
    Returns:
        Dictionary mapping technique pairs to significance results
    """
    try:
        from scipy import stats
    except ImportError:
        print("Warning: scipy not found. Statistical significance calculations require scipy.")
        return {}
    
    if not benchmarks:
        return {}
    
    # Get all techniques that have query results and the specified metric
    valid_techniques = []
    for tech, bench in benchmarks.items():
        if "query_results" in bench and len(bench["query_results"]) > 0:
            # Check if at least some query results have this metric
            has_metric = any(metric in qr for qr in bench["query_results"] if isinstance(qr, dict))
            if has_metric:
                valid_techniques.append(tech)
    
    if len(valid_techniques) < 2:
        return {}  # Need at least two techniques to compare
    
    result = {}
    
    # Perform pairwise comparisons
    for i, tech1 in enumerate(valid_techniques):
        for j, tech2 in enumerate(valid_techniques):
            if i < j:  # Only compare each pair once
                # Extract metric values for each technique
                values1 = [qr.get(metric) for qr in benchmarks[tech1]["query_results"] 
                          if isinstance(qr, dict) and metric in qr]
                values2 = [qr.get(metric) for qr in benchmarks[tech2]["query_results"] 
                          if isinstance(qr, dict) and metric in qr]
                
                # Filter out None values
                values1 = [v for v in values1 if v is not None]
                values2 = [v for v in values2 if v is not None]
                
                if not values1 or not values2:
                    continue
                
                # Perform Mann-Whitney U test (non-parametric test that doesn't assume normal distribution)
                try:
                    u_stat, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
                    
                    # Store result
                    pair_key = f"{tech1}_vs_{tech2}"
                    result[pair_key] = p_value < alpha
                    
                except Exception as e:
                    # Fallback to t-test if Mann-Whitney fails
                    try:
                        t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
                        
                        # Store result
                        pair_key = f"{tech1}_vs_{tech2}"
                        result[pair_key] = p_value < alpha
                    except Exception as e2:
                        # If both tests fail, consider the difference not significant
                        pair_key = f"{tech1}_vs_{tech2}"
                        result[pair_key] = False
    
    return result
