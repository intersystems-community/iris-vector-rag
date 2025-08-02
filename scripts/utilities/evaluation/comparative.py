# eval/comparative.py
"""
Wrapper module for backward compatibility.
This module has been reorganized into a package structure for better maintainability.
New code should import directly from the eval.comparative package.
"""

# Re-export everything from the package
from scripts.utilities.evaluation.comparative import (
    calculate_technique_comparison,
    calculate_statistical_significance,
    generate_comparison_chart,
    generate_radar_chart,
    generate_bar_chart,
    generate_comparative_bar_chart,
    generate_combined_report,
    REFERENCE_BENCHMARKS
)

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
