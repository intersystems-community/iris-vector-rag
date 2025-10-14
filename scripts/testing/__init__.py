"""
Example Testing Framework for rag-templates.

This module provides comprehensive testing infrastructure for examples and demos,
including mock providers, validation suites, and automated test execution.
"""

from .example_runner import ExampleTestResult, ExampleTestRunner
from .mock_providers import MockDataProvider, MockLLMProvider
from .validation_suite import ValidationResult, ValidationSuite

__all__ = [
    "MockLLMProvider",
    "MockDataProvider",
    "ExampleTestRunner",
    "ExampleTestResult",
    "ValidationSuite",
    "ValidationResult",
]
