"""
E2E Test Package for RAG Templates

This package contains true end-to-end tests that validate the complete RAG pipeline
from document ingestion to query response generation. All tests in this package
follow strict E2E principles:

- NO MOCKS or stubs anywhere in the test chain
- MUST use real IRIS database connections
- MUST use realistic PMC biomedical data  
- MUST test complete workflows end-to-end
- MUST validate actual behavior and performance

Test Coverage Areas:
- Core Framework Architecture (iris_rag/core/base.py, iris_rag/core/models.py)
- Vector Store IRIS Implementation (iris_rag/storage/vector_store_iris.py)
- Configuration Management (iris_rag/config/manager.py)

Based on patterns from evaluation_framework/true_e2e_evaluation.py
"""

__version__ = "1.0.0"
__author__ = "RAG Templates E2E Testing Framework"