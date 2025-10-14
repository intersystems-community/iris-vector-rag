"""
Biomedical RAG Pipeline Evaluation Framework

A comprehensive, statistically rigorous evaluation system for comparing
biomedical RAG pipelines with empirical evidence and actionable insights.
"""

__version__ = "1.0.0"
__author__ = "RAG Templates Team"

# Import warnings configuration
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Core components
try:
    from .biomedical_question_generator import (
        QuestionGenerationConfig,
        create_biomedical_question_generator,
    )
except ImportError as e:
    print(f"Warning: Question generator not available - {e}")

try:
    from .ragas_metrics_framework import create_biomedical_ragas_framework
except ImportError as e:
    print(f"Warning: RAGAS framework not available - {e}")

try:
    from .statistical_evaluation_methodology import create_statistical_framework
except ImportError as e:
    print(f"Warning: Statistical framework not available - {e}")

try:
    from .comparative_analysis_system import create_comparative_analysis_system
except ImportError as e:
    print(f"Warning: Comparative analysis not available - {e}")

try:
    from .pmc_data_pipeline import PMCProcessingConfig, create_pmc_data_pipeline
except ImportError as e:
    print(f"Warning: PMC pipeline not available - {e}")

try:
    from .evaluation_orchestrator import EvaluationOrchestrator
except ImportError as e:
    print(f"Warning: Evaluation orchestrator not available - {e}")

try:
    from .empirical_reporting import create_empirical_reporting_framework
except ImportError as e:
    print(f"Warning: Empirical reporting not available - {e}")

__all__ = [
    "create_biomedical_question_generator",
    "QuestionGenerationConfig",
    "create_biomedical_ragas_framework",
    "create_statistical_framework",
    "create_comparative_analysis_system",
    "create_pmc_data_pipeline",
    "PMCProcessingConfig",
    "EvaluationOrchestrator",
    "create_empirical_reporting_framework",
]
