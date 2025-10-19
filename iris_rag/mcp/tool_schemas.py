"""
MCP Tool Schema Definitions.

This module provides access to MCP tool schemas for all 6 RAG pipelines
and 2 utility tools, loaded from the contract schema file.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


# Load schemas from contract file
_SCHEMA_PATH = Path(__file__).parent.parent.parent / 'specs' / '043-complete-mcp-tools' / 'contracts' / 'mcp_tool_schema.json'

with open(_SCHEMA_PATH, 'r') as f:
    _SCHEMA_DATA = json.load(f)

# Index schemas by tool name for fast lookup
_SCHEMAS_BY_NAME: Dict[str, Dict[str, Any]] = {
    tool['name']: tool
    for tool in _SCHEMA_DATA['tools']
}


def get_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Get schema for a specific MCP tool.

    Args:
        tool_name: Tool name (e.g., "rag_basic", "rag_crag")

    Returns:
        Tool schema dict or None if not found
    """
    return _SCHEMAS_BY_NAME.get(tool_name)


def get_all_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Get all MCP tool schemas.

    Returns:
        Dict mapping tool names to schema objects
    """
    return _SCHEMAS_BY_NAME.copy()


def validate_params(tool_name: str, params: Dict[str, Any], skip_query: bool = False) -> Dict[str, Any]:
    """
    Validate parameters against tool schema and apply defaults.

    Args:
        tool_name: Tool name
        params: Parameters to validate
        skip_query: If True, skip validation of 'query' parameter (used when query is passed separately)

    Returns:
        Validated parameters with defaults applied

    Raises:
        ValidationError: If parameters are invalid
    """
    from iris_rag.mcp.validation import ValidationError

    schema = get_schema(tool_name)
    if schema is None:
        raise ValidationError('tool_name', tool_name, f"Unknown tool: {tool_name}")

    input_schema = schema['inputSchema']
    properties = input_schema.get('properties', {})
    required_fields = input_schema.get('required', [])

    # Start with copy of provided params
    validated = params.copy()

    # Check required fields
    for field in required_fields:
        if skip_query and field == 'query':
            # Skip query validation - it's passed separately to execute()
            continue
        if field not in validated:
            raise ValidationError(field, None, f"Required parameter '{field}' missing")

    # Apply defaults and validate each parameter
    for param_name, param_schema in properties.items():
        if param_name not in validated:
            # Apply default if available
            if 'default' in param_schema:
                validated[param_name] = param_schema['default']
        else:
            # Validate provided value
            value = validated[param_name]
            param_type = param_schema.get('type')

            # Type validation
            if param_type == 'integer' and not isinstance(value, int):
                raise ValidationError(param_name, value,
                                     f"Parameter '{param_name}' must be integer")

            if param_type == 'number' and not isinstance(value, (int, float)):
                raise ValidationError(param_name, value,
                                     f"Parameter '{param_name}' must be number")

            if param_type == 'string' and not isinstance(value, str):
                raise ValidationError(param_name, value,
                                     f"Parameter '{param_name}' must be string")

            if param_type == 'boolean' and not isinstance(value, bool):
                raise ValidationError(param_name, value,
                                     f"Parameter '{param_name}' must be boolean")

            if param_type == 'array' and not isinstance(value, list):
                raise ValidationError(param_name, value,
                                     f"Parameter '{param_name}' must be array")

            # Range validation for numbers
            if param_type in ('integer', 'number'):
                if 'minimum' in param_schema and value < param_schema['minimum']:
                    raise ValidationError(param_name, value,
                                         f"Parameter '{param_name}' must be >= {param_schema['minimum']}")

                if 'maximum' in param_schema and value > param_schema['maximum']:
                    raise ValidationError(param_name, value,
                                         f"Parameter '{param_name}' must be <= {param_schema['maximum']}")

            # Enum validation
            if 'enum' in param_schema:
                if value not in param_schema['enum']:
                    valid_values = ', '.join(str(v) for v in param_schema['enum'])
                    raise ValidationError(param_name, value,
                                         f"Parameter '{param_name}' must be one of: {valid_values}")

            # String length validation
            if param_type == 'string':
                if 'minLength' in param_schema and len(value) < param_schema['minLength']:
                    raise ValidationError(param_name, value,
                                         f"Parameter '{param_name}' must be at least {param_schema['minLength']} characters")

                if 'maxLength' in param_schema and len(value) > param_schema['maxLength']:
                    raise ValidationError(param_name, value,
                                         f"Parameter '{param_name}' must be at most {param_schema['maxLength']} characters")

    return validated


def get_tool_names() -> list[str]:
    """
    Get list of all tool names.

    Returns:
        List of tool names
    """
    return list(_SCHEMAS_BY_NAME.keys())
