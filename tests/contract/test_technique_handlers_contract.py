"""
Contract tests for Python Technique Handlers.

These tests verify that TechniqueHandler implementations satisfy the
ITechniqueHandler interface contract for each of the 6 RAG pipelines.
Following TDD principles, these tests MUST FAIL initially.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest
from typing import Dict, Any


class TestTechniqueHandlerInterface:
    """Contract tests for ITechniqueHandler interface."""

    def test_technique_handlers_module_exists(self):
        """Verify iris_rag.mcp.technique_handlers module can be imported."""
        try:
            import iris_vector_rag.mcp.technique_handlers
            assert True
        except ImportError as e:
            pytest.fail(f"TechniqueHandlers module not found: {e}")

    def test_technique_handler_registry_exists(self):
        """Verify TechniqueHandlerRegistry class exists."""
        from iris_vector_rag.mcp import technique_handlers
        assert hasattr(technique_handlers, 'TechniqueHandlerRegistry'), \
            "TechniqueHandlerRegistry class not found"

    def test_registry_implements_required_methods(self):
        """Verify TechniqueHandlerRegistry implements required methods."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        required_methods = [
            'register_handler',
            'get_handler',
            'list_techniques',
            'get_all_handlers'
        ]

        for method_name in required_methods:
            assert hasattr(TechniqueHandlerRegistry, method_name), \
                f"TechniqueHandlerRegistry missing method: {method_name}"


class TestTechniqueHandlerRegistry:
    """Test TechniqueHandlerRegistry functionality."""

    def test_registry_initialization(self):
        """Verify TechniqueHandlerRegistry can be initialized."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        assert registry is not None

    def test_registry_list_techniques_returns_6_pipelines(self):
        """Verify registry lists all 6 RAG pipelines."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        techniques = registry.list_techniques()

        expected_pipelines = [
            'basic',
            'basic_rerank',
            'crag',
            'graphrag',
            'pylate_colbert',
            'iris_global_graphrag'
        ]

        assert isinstance(techniques, list), "list_techniques must return list"
        assert len(techniques) == 6, f"Expected 6 techniques, got {len(techniques)}"

        for pipeline in expected_pipelines:
            assert pipeline in techniques, \
                f"Expected pipeline '{pipeline}' not in techniques list"

    def test_registry_get_handler_returns_handler_object(self):
        """Verify get_handler returns handler objects."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()

        # Get handler for basic pipeline
        handler = registry.get_handler('basic')
        assert handler is not None, "get_handler('basic') returned None"

        # Verify handler has required methods
        assert hasattr(handler, 'execute'), "Handler missing 'execute' method"
        assert hasattr(handler, 'validate_params'), "Handler missing 'validate_params' method"
        assert hasattr(handler, 'health_check'), "Handler missing 'health_check' method"

    def test_registry_get_handler_invalid_technique_raises(self):
        """Verify get_handler raises KeyError for invalid technique."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()

        with pytest.raises(KeyError):
            registry.get_handler('invalid_technique_name')


class TestBasicRAGHandler:
    """Contract tests for BasicRAG technique handler."""

    @pytest.mark.asyncio
    async def test_basic_handler_execute(self):
        """Verify BasicRAG handler execute method."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler('basic')

        result = await handler.execute(
            query='What are the symptoms of diabetes?',
            params={'top_k': 5}
        )

        # Verify response structure
        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'retrieved_documents' in result
        assert 'sources' in result
        assert 'metadata' in result
        assert 'performance' in result

        # Verify performance metrics
        assert 'execution_time_ms' in result['performance']
        assert 'retrieval_time_ms' in result['performance']
        assert 'generation_time_ms' in result['performance']
        assert 'tokens_used' in result['performance']

    def test_basic_handler_validate_params(self):
        """Verify BasicRAG handler parameter validation."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler('basic')

        # Valid parameters
        validated = handler.validate_params({'top_k': 5})
        assert validated['top_k'] == 5

        # Default parameters
        validated_default = handler.validate_params({})
        assert 'top_k' in validated_default
        assert validated_default['top_k'] == 5  # default value

        # Invalid parameters (top_k out of range)
        from iris_vector_rag.mcp.validation import ValidationError
        with pytest.raises(ValidationError):
            handler.validate_params({'top_k': 100})  # max is 50

    @pytest.mark.asyncio
    async def test_basic_handler_health_check(self):
        """Verify BasicRAG handler health check."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler('basic')

        health = await handler.health_check()

        assert isinstance(health, dict)
        assert 'status' in health
        assert health['status'] in ['healthy', 'degraded', 'unavailable']
        assert 'last_success' in health
        assert 'error_rate' in health


class TestCRAGHandler:
    """Contract tests for CRAG technique handler."""

    @pytest.mark.asyncio
    async def test_crag_handler_execute_with_correction(self):
        """Verify CRAG handler execute method with correction strategy."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler('crag')

        result = await handler.execute(
            query='What is diabetes?',
            params={
                'top_k': 5,
                'confidence_threshold': 0.8,
                'correction_strategy': 'rewrite'
            }
        )

        # Verify standard response structure
        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'retrieved_documents' in result
        assert 'metadata' in result

        # Verify CRAG-specific metadata
        assert 'confidence_score' in result['metadata'] or \
               'correction_applied' in result['metadata']

    def test_crag_handler_validate_params_confidence_threshold(self):
        """Verify CRAG handler validates confidence_threshold parameter."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler('crag')

        # Valid confidence threshold
        validated = handler.validate_params({'confidence_threshold': 0.8})
        assert validated['confidence_threshold'] == 0.8

        # Invalid confidence threshold (out of range)
        from iris_vector_rag.mcp.validation import ValidationError
        with pytest.raises(ValidationError):
            handler.validate_params({'confidence_threshold': 1.5})  # max is 1.0


class TestGraphRAGHandler:
    """Contract tests for HybridGraphRAG technique handler."""

    @pytest.mark.asyncio
    async def test_graphrag_handler_execute_hybrid_search(self):
        """Verify GraphRAG handler execute method with hybrid search."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler('graphrag')

        result = await handler.execute(
            query='How does diabetes affect the heart?',
            params={
                'top_k': 5,
                'search_method': 'hybrid',
                'graph_traversal_depth': 2,
                'rrf_k': 60
            }
        )

        # Verify standard response
        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'retrieved_documents' in result

        # Verify GraphRAG-specific metadata
        assert 'metadata' in result
        metadata = result['metadata']
        # May include search_method, graph_traversal_depth, rrf_score

    def test_graphrag_handler_validate_params_search_method(self):
        """Verify GraphRAG handler validates search_method parameter."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler('graphrag')

        # Valid search methods
        for method in ['hybrid', 'vector', 'text', 'graph', 'rrf']:
            validated = handler.validate_params({'search_method': method})
            assert validated['search_method'] == method

        # Invalid search method
        from iris_vector_rag.mcp.validation import ValidationError
        with pytest.raises(ValidationError):
            handler.validate_params({'search_method': 'invalid_method'})


class TestPyLateColBERTHandler:
    """Contract tests for PyLateColBERT technique handler."""

    @pytest.mark.asyncio
    async def test_pylate_colbert_handler_execute(self):
        """Verify PyLateColBERT handler execute method."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler('pylate_colbert')

        result = await handler.execute(
            query='What is diabetes?',
            params={
                'top_k': 5,
                'interaction_threshold': 0.5
            }
        )

        assert isinstance(result, dict)
        assert 'answer' in result
        assert 'retrieved_documents' in result

    def test_pylate_colbert_handler_validate_params(self):
        """Verify PyLateColBERT handler validates interaction_threshold."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler('pylate_colbert')

        # Valid interaction threshold
        validated = handler.validate_params({'interaction_threshold': 0.5})
        assert validated['interaction_threshold'] == 0.5

        # Invalid interaction threshold
        from iris_vector_rag.mcp.validation import ValidationError
        with pytest.raises(ValidationError):
            handler.validate_params({'interaction_threshold': 1.5})


class TestAllHandlersCommon:
    """Common contract tests for all 6 technique handlers."""

    @pytest.mark.parametrize('technique', [
        'basic',
        'basic_rerank',
        'crag',
        'graphrag',
        'pylate_colbert',
        'iris_global_graphrag'
    ])
    def test_handler_has_required_interface_methods(self, technique):
        """Verify all handlers implement ITechniqueHandler interface."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler(technique)

        # Verify required methods exist
        assert hasattr(handler, 'execute'), \
            f"{technique} handler missing 'execute' method"
        assert hasattr(handler, 'validate_params'), \
            f"{technique} handler missing 'validate_params' method"
        assert hasattr(handler, 'health_check'), \
            f"{technique} handler missing 'health_check' method"

        # Verify methods are callable
        assert callable(handler.execute)
        assert callable(handler.validate_params)
        assert callable(handler.health_check)

    @pytest.mark.parametrize('technique', [
        'basic',
        'basic_rerank',
        'crag',
        'graphrag',
        'pylate_colbert',
        'iris_global_graphrag'
    ])
    def test_handler_validate_params_returns_dict(self, technique):
        """Verify all handlers' validate_params returns dict."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler(technique)

        validated = handler.validate_params({})
        assert isinstance(validated, dict), \
            f"{technique} handler validate_params must return dict"

    @pytest.mark.parametrize('technique', [
        'basic',
        'basic_rerank',
        'crag',
        'graphrag',
        'pylate_colbert',
        'iris_global_graphrag'
    ])
    @pytest.mark.asyncio
    async def test_handler_health_check_returns_status(self, technique):
        """Verify all handlers' health_check returns status."""
        from iris_vector_rag.mcp.technique_handlers import TechniqueHandlerRegistry

        registry = TechniqueHandlerRegistry()
        handler = registry.get_handler(technique)

        health = await handler.health_check()
        assert isinstance(health, dict)
        assert 'status' in health
        assert health['status'] in ['healthy', 'degraded', 'unavailable']
