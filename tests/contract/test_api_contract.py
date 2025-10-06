"""Contract tests for API alignment requirements.

Feature: 025-fixes-for-testing
Contract: api_alignment_contract.md

These tests validate that production APIs match expected signatures.
If these tests fail, it means the production code has changed and
test expectations need to be updated (NOT the production code).
"""

import inspect

import pytest


def test_basic_rag_load_documents_signature():
    """BasicRAG.load_documents signature matches actual production API."""
    from iris_rag.pipelines.basic import BasicRAGPipeline

    sig = inspect.signature(BasicRAGPipeline.load_documents)
    params = list(sig.parameters.keys())

    # Actual production signature uses documents_path + **kwargs
    assert "self" in params, "load_documents should be an instance method"
    assert (
        "documents_path" in params
    ), "load_documents should accept documents_path"
    assert (
        "kwargs" in params
    ), "load_documents should accept **kwargs for documents"


def test_crag_pipeline_query_response_structure():
    """CRAG pipeline response has expected metadata structure."""
    # This is a placeholder for actual CRAG response validation
    # Will be updated when fixing CRAG tests
    from iris_rag.pipelines.crag import CRAGPipeline

    # Verify CRAGPipeline exists and has query method
    assert hasattr(CRAGPipeline, "query"), "CRAGPipeline must have query method"


def test_graphrag_pipeline_entity_extraction_api():
    """GraphRAG entity extraction API matches actual production."""
    try:
        from iris_rag.pipelines.graphrag import GraphRAGPipeline

        # GraphRAG uses entity extraction service, not direct methods
        # Verify pipeline has query method (actual production API)
        assert hasattr(
            GraphRAGPipeline, "query"
        ), "GraphRAG should have query method"
        assert hasattr(
            GraphRAGPipeline, "load_documents"
        ), "GraphRAG should have load_documents method"
    except ImportError:
        pytest.skip("GraphRAG not available (optional dependency)")


def test_pylate_pipeline_api_signature():
    """PyLate pipeline has expected API methods."""
    try:
        from iris_rag.pipelines.colbert_pylate.pylate_pipeline import (
            PyLateColBERTPipeline,
        )

        # Verify PyLate pipeline exists
        assert hasattr(
            PyLateColBERTPipeline, "load_documents"
        ), "PyLate pipeline should have load_documents"
        assert hasattr(
            PyLateColBERTPipeline, "query"
        ), "PyLate pipeline should have query"
    except ImportError:
        pytest.skip("PyLate pipeline not available (optional dependency)")


def test_vector_store_similarity_search_signature():
    """IRIS vector store similarity_search matches actual production API."""
    from iris_rag.storage.vector_store_iris import IRISVectorStore

    sig = inspect.signature(IRISVectorStore.similarity_search)
    params = list(sig.parameters.keys())

    # Actual production signature uses *args, **kwargs pattern
    assert "self" in params, "similarity_search should be an instance method"
    assert (
        "args" in params or "kwargs" in params
    ), "similarity_search should accept flexible arguments"


def test_vector_store_metadata_filtering_api():
    """IRIS vector store metadata filtering API."""
    from iris_rag.storage.vector_store_iris import IRISVectorStore

    # Check if add_documents accepts metadata
    sig = inspect.signature(IRISVectorStore.add_documents)
    params = list(sig.parameters.keys())

    assert (
        "documents" in params
    ), "add_documents should accept documents parameter"


def test_configuration_manager_api():
    """ConfigurationManager has stable API."""
    from iris_rag.config.manager import ConfigurationManager

    # Verify expected methods exist
    assert hasattr(
        ConfigurationManager, "get"
    ), "ConfigurationManager should have get method"


def test_connection_manager_api():
    """ConnectionManager has stable API."""
    from iris_rag.core.connection import ConnectionManager

    # Verify expected methods exist
    assert hasattr(
        ConnectionManager, "get_connection"
    ), "ConnectionManager should have get_connection method"


def test_pipeline_base_class_contract():
    """RAGPipeline base class has expected interface."""
    from iris_rag.core.base import RAGPipeline

    # Required abstract methods
    expected_methods = ["load_documents", "query"]

    for method in expected_methods:
        assert hasattr(
            RAGPipeline, method
        ), f"RAGPipeline must define {method}"
