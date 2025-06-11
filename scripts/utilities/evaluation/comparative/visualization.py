# eval/comparative/visualization.py
"""
Visualization functions for generating comparison charts and graphs.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid display issues
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime

def generate_comparison_chart(metrics: Dict[str, Dict[str, float]], 
                             chart_type: str = "radar", 
                             metric: str = None,
                             output_path: str = None) -> str:
    """
    Generate a chart visualizing technique differences.
    
    Args:
        metrics: Dictionary mapping technique names to their metrics
        chart_type: Type of chart to generate ('radar', 'bar', 'line')
        metric: The specific metric to visualize (required for 'bar' chart type)
        output_path: Path to save the chart image
        
    Returns:
        Path to the generated chart image
    """
    if not metrics:
        raise ValueError("Metrics dictionary is empty")
    
    if chart_type == "radar":
        return generate_radar_chart(metrics, output_path)
    elif chart_type == "bar":
        if not metric:
            raise ValueError("Bar chart requires a specific metric to visualize")
        # Determine if lower is better based on metric name
        lower_is_better = any(metric.startswith(prefix) for prefix in ['latency', 'p50', 'p95', 'p99'])
        return generate_bar_chart(metrics, metric, output_path, lower_is_better=lower_is_better)
    elif chart_type == "line":
        # Not yet implemented
        raise NotImplementedError("Line chart generation not yet implemented")
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

def generate_radar_chart(metrics: Dict[str, Dict[str, float]], output_path: str = None) -> str:
    """
    Generate a radar chart comparing techniques across metrics.
    
    Args:
        metrics: Dictionary mapping technique names to their metrics
        output_path: Path to save the chart image
        
    Returns:
        Path to the generated chart image
    """
    if not metrics:
        raise ValueError("Metrics dictionary is empty")
    
    # Create figure with polar coordinates
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Get common metrics across all techniques
    all_metrics = set()
    for tech_metrics in metrics.values():
        all_metrics.update(tech_metrics.keys())
    
    # Filter out any problematic metrics
    all_metrics = [m for m in sorted(all_metrics) if not m.startswith('_')]
    
    # Make sure we have metrics
    if not all_metrics:
        raise ValueError("No metrics found in the provided dictionary")
    
    # Number of metrics
    N = len(all_metrics)
    
    # Angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Colors for different techniques
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Plot each technique
    for i, (technique, tech_metrics) in enumerate(metrics.items()):
        # Get values for each metric, defaulting to 0 if metric not present
        values = [tech_metrics.get(metric, 0) for metric in all_metrics]
        values += values[:1]  # Close the loop
        
        # Use cyclic color selection
        color = colors[i % len(colors)]
        
        # Plot line and fill area
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=technique, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Add metric labels
    metric_labels = [m.replace('_', ' ').title() for m in all_metrics]
    plt.xticks(angles[:-1], metric_labels)
    
    # Set radial limits to be slightly larger than the max value
    max_value = max([max([metrics[t].get(m, 0) for m in all_metrics], default=1.0) for t in metrics.keys()], default=1.0)
    plt.ylim(0, max_value * 1.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title("RAG Techniques Comparison")
    
    # Create output directory if it doesn't exist
    if output_path is None:
        output_dir = "benchmark_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"radar_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_bar_chart(metrics: Dict[str, Dict[str, float]], 
                      metric: str, 
                      output_path: str = None,
                      lower_is_better: bool = False) -> str:
    """
    Generate a bar chart comparing techniques on a specific metric.
    
    Args:
        metrics: Dictionary mapping technique names to their metrics
        metric: The metric to visualize
        output_path: Path to save the chart image
        lower_is_better: Whether lower values are better for this metric (e.g. latency)
        
    Returns:
        Path to the generated chart image
    """
    if not metrics:
        raise ValueError("Metrics dictionary is empty")
        
    # Check if the specified metric exists in at least one technique
    metric_exists = any(metric in tech_metrics for tech_metrics in metrics.values())
    if not metric_exists:
        raise ValueError(f"Metric '{metric}' not found in any technique")
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract techniques and their values for the metric
    techniques = list(metrics.keys())
    values = [tech_metrics.get(metric, 0) for tech_metrics in [metrics[t] for t in techniques]]
    
    # Sort by performance if requested
    if lower_is_better:
        # For metrics where lower is better (e.g. latency), sort in ascending order
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
    else:
        # For metrics where higher is better (e.g. recall), sort in descending order
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    
    # Apply sorting
    techniques = [techniques[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    # Choose colors based on performance (green for best, yellow for middle, red for worst)
    colors = []
    for i in range(len(values)):
        if i == 0:  # Best
            colors.append('#2ca02c')  # Green
        elif i == len(values) - 1:  # Worst
            colors.append('#d62728')  # Red
        else:  # Middle performers
            colors.append('#ff7f0e')  # Orange
    
    # Create bars
    bars = ax.bar(techniques, values, color=colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        # Ensure the text is readable
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(values),
                f"{height:.2f}", ha='center', va='bottom')
    
    # Format metric name for display
    display_metric = metric.replace('_', ' ').title()
    
    # Add labels and title
    ax.set_ylabel(display_metric)
    ax.set_title(f"Comparison of {display_metric} across RAG Techniques")
    
    # Add performance indicator
    if lower_is_better:
        ax.text(0.02, 0.02, "Lower is better", transform=ax.transAxes, 
                fontsize=10, verticalalignment='bottom', color='gray')
    else:
        ax.text(0.02, 0.02, "Higher is better", transform=ax.transAxes, 
                fontsize=10, verticalalignment='bottom', color='gray')
    
    # Create output directory if it doesn't exist
    if output_path is None:
        output_dir = "benchmark_results"
        os.makedirs(output_dir, exist_ok=True)
        clean_metric = metric.replace(' ', '_').lower()
        output_path = os.path.join(output_dir, f"bar_chart_{clean_metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path

def generate_comparative_bar_chart(our_results: Dict[str, float], 
                                 reference_results: Dict[str, float],
                                 metric: str,
                                 output_path: str = None,
                                 lower_is_better: bool = False) -> str:
    """
    Generate a bar chart comparing our results with published benchmarks.
    
    Args:
        our_results: Dictionary mapping our techniques to their metric values
        reference_results: Dictionary mapping reference techniques to their metric values
        metric: The metric being compared
        output_path: Path to save the chart image
        lower_is_better: Whether lower values are better (e.g., latency)
        
    Returns:
        Path to the generated chart image
    """
    if not our_results or not reference_results:
        raise ValueError("Both our results and reference results must be provided")
    
    # Set up figure with larger size to accommodate more bars
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Combine our results and reference results
    all_techniques = list(our_results.keys()) + [f"Ref: {t}" for t in reference_results.keys()]
    all_values = list(our_results.values()) + list(reference_results.values())
    
    # Sort by performance
    if lower_is_better:
        # For metrics where lower is better, sort in ascending order
        sorted_indices = sorted(range(len(all_values)), key=lambda i: all_values[i])
    else:
        # For metrics where higher is better, sort in descending order
        sorted_indices = sorted(range(len(all_values)), key=lambda i: all_values[i], reverse=True)
    
    # Apply sorting
    all_techniques = [all_techniques[i] for i in sorted_indices]
    all_values = [all_values[i] for i in sorted_indices]
    
    # Assign colors based on whether it's our technique or reference
    colors = []
    for tech in all_techniques:
        if tech.startswith("Ref:"):
            colors.append('#9467bd')  # Purple for reference
        else:
            colors.append('#1f77b4')  # Blue for our techniques
    
    # Create bars
    bars = ax.bar(all_techniques, all_values, color=colors)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(all_values),
                f"{height:.2f}", ha='center', va='bottom')
    
    # Format metric name for display
    display_metric = metric.replace('_', ' ').title()
    
    # Add labels and title
    ax.set_ylabel(display_metric)
    ax.set_title(f"Comparison of {display_metric} with Published Benchmarks")
    
    # Add grid lines for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-labels if there are many techniques
    if len(all_techniques) > 5:
        plt.xticks(rotation=45, ha='right')
    
    # Add performance indicator
    if lower_is_better:
        ax.text(0.02, 0.02, "Lower is better", transform=ax.transAxes, 
                fontsize=10, verticalalignment='bottom', color='gray')
    else:
        ax.text(0.02, 0.02, "Higher is better", transform=ax.transAxes, 
                fontsize=10, verticalalignment='bottom', color='gray')
    
    # Create output directory if it doesn't exist
    if output_path is None:
        output_dir = "benchmark_results"
        os.makedirs(output_dir, exist_ok=True)
        clean_metric = metric.replace(' ', '_').lower()
        output_path = os.path.join(output_dir, f"comparative_chart_{clean_metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Save chart
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path
