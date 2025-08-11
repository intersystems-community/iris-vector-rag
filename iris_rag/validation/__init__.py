"""
Pre-condition validation module for iris_rag pipelines.

This module provides validation infrastructure to ensure pipelines have
all required data and dependencies before execution.
"""

from .requirements import PipelineRequirements, BasicRAGRequirements, ColBERTRequirements
from .validator import PreConditionValidator
from .orchestrator import SetupOrchestrator
from .factory import ValidatedPipelineFactory
from .embedding_validator import EmbeddingValidator, EmbeddingQualityIssues

__all__ = [
    "PipelineRequirements",
    "BasicRAGRequirements",
    "ColBERTRequirements",
    "PreConditionValidator",
    "SetupOrchestrator",
    "ValidatedPipelineFactory",
    "EmbeddingValidator",
    "EmbeddingQualityIssues",
]
