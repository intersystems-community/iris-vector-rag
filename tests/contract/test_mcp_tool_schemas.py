"""
Contract tests for MCP Tool Schemas.

These tests verify that all 8 MCP tool schemas (6 RAG pipelines + 2 utility tools)
are correctly defined and validated. Following TDD principles, these tests MUST FAIL initially.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest
import json
from pathlib import Path


class TestMCPToolSchemaFile:
    """Test MCP tool schema JSON file structure."""

    def test_schema_file_exists(self):
        """Verify mcp_tool_schema.json file exists in contracts directory."""
        schema_path = Path(__file__).parent.parent.parent / 'specs' / '043-complete-mcp-tools' / 'contracts' / 'mcp_tool_schema.json'
        assert schema_path.exists(), f"MCP tool schema file not found at {schema_path}"

    def test_schema_file_is_valid_json(self):
        """Verify schema file contains valid JSON."""
        schema_path = Path(__file__).parent.parent.parent / 'specs' / '043-complete-mcp-tools' / 'contracts' / 'mcp_tool_schema.json'

        with open(schema_path, 'r') as f:
            schema_data = json.load(f)

        assert isinstance(schema_data, dict)
        assert 'tools' in schema_data
        assert isinstance(schema_data['tools'], list)

    def test_schema_contains_8_tools(self):
        """Verify schema defines 8 tools (6 RAG + 2 utility)."""
        schema_path = Path(__file__).parent.parent.parent / 'specs' / '043-complete-mcp-tools' / 'contracts' / 'mcp_tool_schema.json'

        with open(schema_path, 'r') as f:
            schema_data = json.load(f)

        assert len(schema_data['tools']) == 8, \
            f"Expected 8 tools, found {len(schema_data['tools'])}"


class TestMCPToolSchemaModule:
    """Test MCP tool schema Python module."""

    def test_tool_schemas_module_exists(self):
        """Verify iris_rag.mcp.tool_schemas module can be imported."""
        try:
            import iris_vector_rag.mcp.tool_schemas
            assert True
        except ImportError as e:
            pytest.fail(f"tool_schemas module not found: {e}")

    def test_tool_schemas_module_has_get_schema_function(self):
        """Verify module provides get_schema function."""
        from iris_vector_rag.mcp import tool_schemas
        assert hasattr(tool_schemas, 'get_schema'), \
            "tool_schemas module missing get_schema function"

    def test_tool_schemas_module_has_get_all_schemas_function(self):
        """Verify module provides get_all_schemas function."""
        from iris_vector_rag.mcp import tool_schemas
        assert hasattr(tool_schemas, 'get_all_schemas'), \
            "tool_schemas module missing get_all_schemas function"

    def test_tool_schemas_module_has_validate_params_function(self):
        """Verify module provides validate_params function."""
        from iris_vector_rag.mcp import tool_schemas
        assert hasattr(tool_schemas, 'validate_params'), \
            "tool_schemas module missing validate_params function"


class TestRAGToolSchemas:
    """Test individual RAG pipeline tool schemas."""

    @pytest.mark.parametrize('tool_name,expected_description_keywords', [
        ('rag_basic', ['vector', 'similarity', 'search']),
        ('rag_basic_rerank', ['rerank', 'cross-encoder']),
        ('rag_crag', ['corrective', 'self-evaluation']),
        ('rag_graphrag', ['hybrid', 'graph', 'knowledge']),
        ('rag_pylate_colbert', ['colbert', 'late', 'interaction']),
        ('rag_iris_global_graphrag', ['global', 'academic', 'papers'])
    ])
    def test_rag_tool_schema_exists(self, tool_name, expected_description_keywords):
        """Verify RAG tool schema exists with correct structure."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema(tool_name)
        assert schema is not None, f"Schema for {tool_name} not found"

        # Verify schema structure
        assert 'name' in schema
        assert schema['name'] == tool_name
        assert 'description' in schema
        assert 'inputSchema' in schema

        # Verify description contains expected keywords
        description_lower = schema['description'].lower()
        for keyword in expected_description_keywords:
            assert keyword.lower() in description_lower, \
                f"Expected keyword '{keyword}' not in {tool_name} description"

    @pytest.mark.parametrize('tool_name', [
        'rag_basic',
        'rag_basic_rerank',
        'rag_crag',
        'rag_graphrag',
        'rag_pylate_colbert',
        'rag_iris_global_graphrag'
    ])
    def test_rag_tool_requires_query_parameter(self, tool_name):
        """Verify all RAG tools require 'query' parameter."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema(tool_name)
        input_schema = schema['inputSchema']

        # Verify query parameter exists
        assert 'properties' in input_schema
        assert 'query' in input_schema['properties']

        # Verify query is required
        assert 'required' in input_schema
        assert 'query' in input_schema['required']

        # Verify query is string type
        query_prop = input_schema['properties']['query']
        assert query_prop['type'] == 'string'

    @pytest.mark.parametrize('tool_name', [
        'rag_basic',
        'rag_basic_rerank',
        'rag_crag',
        'rag_graphrag',
        'rag_pylate_colbert',
        'rag_iris_global_graphrag'
    ])
    def test_rag_tool_has_top_k_parameter(self, tool_name):
        """Verify all RAG tools have 'top_k' parameter with defaults."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema(tool_name)
        input_schema = schema['inputSchema']

        assert 'properties' in input_schema
        assert 'top_k' in input_schema['properties']

        top_k_prop = input_schema['properties']['top_k']
        assert top_k_prop['type'] == 'integer'
        assert 'default' in top_k_prop
        assert 'minimum' in top_k_prop
        assert 'maximum' in top_k_prop
        assert top_k_prop['minimum'] >= 1
        assert top_k_prop['maximum'] <= 100


class TestBasicRAGSchema:
    """Test BasicRAG tool schema specifics."""

    def test_basic_rag_schema_parameters(self):
        """Verify BasicRAG schema has correct parameters."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema('rag_basic')
        props = schema['inputSchema']['properties']

        # Expected parameters
        assert 'query' in props
        assert 'top_k' in props
        assert 'include_sources' in props
        assert 'include_metadata' in props

        # Verify boolean parameters have correct type
        assert props['include_sources']['type'] == 'boolean'
        assert props['include_metadata']['type'] == 'boolean'

        # Verify default values
        assert props['include_sources']['default'] is True
        assert props['include_metadata']['default'] is True


class TestCRAGSchema:
    """Test CRAG tool schema specifics."""

    def test_crag_schema_has_confidence_threshold(self):
        """Verify CRAG schema has confidence_threshold parameter."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema('rag_crag')
        props = schema['inputSchema']['properties']

        assert 'confidence_threshold' in props
        threshold_prop = props['confidence_threshold']

        assert threshold_prop['type'] == 'number'
        assert 'default' in threshold_prop
        assert threshold_prop['default'] == 0.8
        assert 'minimum' in threshold_prop
        assert 'maximum' in threshold_prop
        assert threshold_prop['minimum'] == 0.0
        assert threshold_prop['maximum'] == 1.0

    def test_crag_schema_has_correction_strategy(self):
        """Verify CRAG schema has correction_strategy parameter."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema('rag_crag')
        props = schema['inputSchema']['properties']

        assert 'correction_strategy' in props
        strategy_prop = props['correction_strategy']

        assert strategy_prop['type'] == 'string'
        assert 'enum' in strategy_prop
        assert 'rewrite' in strategy_prop['enum']
        assert 'web_search' in strategy_prop['enum']


class TestGraphRAGSchema:
    """Test HybridGraphRAG tool schema specifics."""

    def test_graphrag_schema_has_search_method(self):
        """Verify GraphRAG schema has search_method parameter."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema('rag_graphrag')
        props = schema['inputSchema']['properties']

        assert 'search_method' in props
        method_prop = props['search_method']

        assert method_prop['type'] == 'string'
        assert 'enum' in method_prop
        expected_methods = ['hybrid', 'vector', 'text', 'graph', 'rrf']
        for method in expected_methods:
            assert method in method_prop['enum']

    def test_graphrag_schema_has_graph_traversal_depth(self):
        """Verify GraphRAG schema has graph_traversal_depth parameter."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema('rag_graphrag')
        props = schema['inputSchema']['properties']

        assert 'graph_traversal_depth' in props
        depth_prop = props['graph_traversal_depth']

        assert depth_prop['type'] == 'integer'
        assert 'default' in depth_prop
        assert 'minimum' in depth_prop
        assert 'maximum' in depth_prop
        assert depth_prop['minimum'] == 1
        assert depth_prop['maximum'] == 5


class TestUtilityToolSchemas:
    """Test utility tool schemas (health_check, metrics)."""

    def test_health_check_tool_schema(self):
        """Verify rag_health_check tool schema."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema('rag_health_check')
        assert schema is not None

        # Verify description mentions health/status
        assert 'health' in schema['description'].lower() or \
               'status' in schema['description'].lower()

        # Verify parameters
        props = schema['inputSchema']['properties']
        assert 'include_details' in props
        assert 'include_performance_metrics' in props

        # Verify boolean parameters
        assert props['include_details']['type'] == 'boolean'
        assert props['include_performance_metrics']['type'] == 'boolean'

    def test_metrics_tool_schema(self):
        """Verify rag_metrics tool schema."""
        from iris_vector_rag.mcp.tool_schemas import get_schema

        schema = get_schema('rag_metrics')
        assert schema is not None

        # Verify description mentions metrics/performance
        assert 'metric' in schema['description'].lower() or \
               'performance' in schema['description'].lower()

        # Verify parameters
        props = schema['inputSchema']['properties']
        assert 'time_range' in props
        assert 'technique_filter' in props

        # Verify time_range enum
        time_range_prop = props['time_range']
        assert time_range_prop['type'] == 'string'
        assert 'enum' in time_range_prop
        expected_ranges = ['5m', '15m', '1h', '6h', '24h', '7d']
        for range_val in expected_ranges:
            assert range_val in time_range_prop['enum']


class TestSchemaValidation:
    """Test parameter validation against schemas."""

    def test_validate_params_accepts_valid_parameters(self):
        """Verify validate_params accepts valid parameters."""
        from iris_vector_rag.mcp.tool_schemas import validate_params

        # Valid parameters for basic RAG
        validated = validate_params('rag_basic', {
            'query': 'What is diabetes?',
            'top_k': 5
        })

        assert validated is not None
        assert validated['query'] == 'What is diabetes?'
        assert validated['top_k'] == 5

    def test_validate_params_applies_defaults(self):
        """Verify validate_params applies default values."""
        from iris_vector_rag.mcp.tool_schemas import validate_params

        # Minimal parameters (only required fields)
        validated = validate_params('rag_basic', {
            'query': 'test query'
        })

        # Should have defaults applied
        assert 'top_k' in validated
        assert validated['top_k'] == 5  # default value
        assert 'include_sources' in validated
        assert validated['include_sources'] is True  # default value

    def test_validate_params_rejects_invalid_range(self):
        """Verify validate_params rejects out-of-range values."""
        from iris_vector_rag.mcp.tool_schemas import validate_params
        from iris_vector_rag.mcp.validation import ValidationError

        # top_k exceeds maximum
        with pytest.raises(ValidationError):
            validate_params('rag_basic', {
                'query': 'test',
                'top_k': 100  # max is 50
            })

    def test_validate_params_rejects_invalid_enum(self):
        """Verify validate_params rejects invalid enum values."""
        from iris_vector_rag.mcp.tool_schemas import validate_params
        from iris_vector_rag.mcp.validation import ValidationError

        # Invalid search_method
        with pytest.raises(ValidationError):
            validate_params('rag_graphrag', {
                'query': 'test',
                'search_method': 'invalid_method'
            })

    def test_validate_params_rejects_missing_required(self):
        """Verify validate_params rejects missing required parameters."""
        from iris_vector_rag.mcp.tool_schemas import validate_params
        from iris_vector_rag.mcp.validation import ValidationError

        # Missing required 'query' parameter
        with pytest.raises(ValidationError):
            validate_params('rag_basic', {
                'top_k': 5
            })


class TestGetAllSchemas:
    """Test get_all_schemas function."""

    def test_get_all_schemas_returns_8_tools(self):
        """Verify get_all_schemas returns all 8 tool schemas."""
        from iris_vector_rag.mcp.tool_schemas import get_all_schemas

        all_schemas = get_all_schemas()
        assert isinstance(all_schemas, dict)
        assert len(all_schemas) == 8

        # Verify all expected tools are present
        expected_tools = [
            'rag_basic',
            'rag_basic_rerank',
            'rag_crag',
            'rag_graphrag',
            'rag_pylate_colbert',
            'rag_iris_global_graphrag',
            'rag_health_check',
            'rag_metrics'
        ]

        for tool_name in expected_tools:
            assert tool_name in all_schemas

    def test_get_all_schemas_returns_complete_schemas(self):
        """Verify get_all_schemas returns complete schema objects."""
        from iris_vector_rag.mcp.tool_schemas import get_all_schemas

        all_schemas = get_all_schemas()

        for tool_name, schema in all_schemas.items():
            assert 'name' in schema
            assert 'description' in schema
            assert 'inputSchema' in schema
            assert schema['name'] == tool_name
