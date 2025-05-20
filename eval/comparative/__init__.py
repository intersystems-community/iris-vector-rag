# eval/comparative/__init__.py
"""
Comparative analysis module for RAG technique benchmarking.
Provides tools for analyzing, visualizing and reporting on benchmark results.
"""

from eval.comparative.analysis import calculate_technique_comparison, calculate_statistical_significance
from eval.comparative.visualization import (
    generate_comparison_chart, 
    generate_radar_chart,
    generate_bar_chart,
    generate_comparative_bar_chart
)
from eval.comparative.reporting import generate_combined_report
from eval.comparative.reference_data import REFERENCE_BENCHMARKS

__all__ = [
    'calculate_technique_comparison',
    'calculate_statistical_significance',
    'generate_comparison_chart',
    'generate_radar_chart',
    'generate_bar_chart',
    'generate_comparative_bar_chart',
    'generate_combined_report',
    'REFERENCE_BENCHMARKS'
]
