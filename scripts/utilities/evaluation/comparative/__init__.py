# eval/comparative/__init__.py
"""
Comparative analysis module for RAG technique benchmarking.
Provides tools for analyzing, visualizing and reporting on benchmark results.
"""

from .analysis import calculate_technique_comparison, calculate_statistical_significance
from .visualization import (
    generate_comparison_chart,
    generate_radar_chart,
    generate_bar_chart,
    generate_comparative_bar_chart
)
from .reporting import generate_combined_report
from .reference_data import REFERENCE_BENCHMARKS

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
