"""Iris-LLM MCP handler implementation."""

from __future__ import annotations

from typing import Any, Dict, List

from iris_vector_rag.mcp.validation import ValidationError


ToolSet = Any


class IrisLLMToolHandler:
    """Wrap an iris-llm ``ToolSet`` so MCP can present its tools."""

    def __init__(self, toolset: ToolSet) -> None:
        self.toolset = toolset
        self._tools: Dict[str, Any] = {}
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._load_tool_definitions()

    def _load_tool_definitions(self) -> None:
        candidates = self._discover_tools()
        for tool in candidates:
            name = self._extract_metadata(tool, ('name', 'tool_name', 'id'))
            if not name:
                continue
            self._tools[name] = tool
            self._schemas[name] = self._build_schema(tool)

    def _discover_tools(self) -> List[Any]:
        tools: List[Any] = []
        for attr in ('tools', 'get_tools', 'list_tools'):
            target = getattr(self.toolset, attr, None)
            if target is None:
                continue
            candidate = target() if callable(target) else target
            if candidate is None:
                continue
            if isinstance(candidate, dict):
                tools.extend(candidate.values())
            elif isinstance(candidate, (list, tuple)):
                tools.extend(candidate)
            else:
                tools.append(candidate)
        return tools

    def _extract_metadata(self, subject: Any, names: tuple[str, ...]) -> Any:
        for name in names:
            if isinstance(subject, dict):
                value = subject.get(name)
            else:
                value = getattr(subject, name, None)
            if value:
                return value
        return None

    def _build_schema(self, tool: Any) -> Dict[str, Any]:
        schema = self._extract_metadata(tool, (
            'input_schema',
            'inputSchema',
            'schema',
            'tool_schema',
            'toolSchema'
        ))
        if not isinstance(schema, dict):
            schema = {}
        return {
            'type': schema.get('type', 'object'),
            'properties': schema.get('properties', {}),
            'required': schema.get('required', [])
        }

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return MCP-compatible schema definitions for each tool."""
        tool_defs: List[Dict[str, Any]] = []
        for name, tool in self._tools.items():
            description = self._extract_metadata(tool, ('description', 'summary', 'details')) or ''
            schema = self._schemas.get(name, {'type': 'object', 'properties': {}, 'required': []})
            tool_defs.append({
                'name': name,
                'description': description,
                'inputSchema': schema
            })
        return tool_defs

    def has_tool(self, tool_name: str) -> bool:
        """Return True if the toolset exposes the requested tool."""
        return tool_name in self._tools

    def validate_params(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate required fields declared by the tool's schema."""
        schema = self._schemas.get(tool_name)
        if not schema:
            return params
        for required in schema.get('required', []):
            if required not in params:
                raise ValidationError(
                    required,
                    params.get(required),
                    f"Tool '{tool_name}' missing required parameter '{required}'"
                )
        return params

    def execute(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Run the selected iris-llm tool."""
        if tool_name not in self._tools:
            raise KeyError(f"Unknown iris-llm tool: {tool_name}")
        tool = self._tools[tool_name]
        if hasattr(self.toolset, 'execute'):
            return self.toolset.execute(tool_name, params)
        for attr in ('execute', 'run', '__call__'):
            method = getattr(tool, attr, None)
            if callable(method):
                return method(**params)
        if callable(tool):
            return tool(**params)
        raise RuntimeError(f"Cannot execute iris-llm tool '{tool_name}'")


def register_iris_toolset(toolset: ToolSet) -> IrisLLMToolHandler:
    """Construct an MCP handler around an iris-llm ToolSet."""
    return IrisLLMToolHandler(toolset)
