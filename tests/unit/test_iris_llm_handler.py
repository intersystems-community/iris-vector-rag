import pytest
from unittest.mock import MagicMock
from iris_vector_rag.mcp.iris_llm_handler import IrisLLMToolHandler
from iris_vector_rag.mcp.validation import ValidationError

def test_handler_initialization():
    mock_toolset = MagicMock()
    mock_toolset.get_tools.return_value = []
    # Configure mock to avoid auto-discovery noise
    type(mock_toolset).tools = MagicMock(return_value=None)
    type(mock_toolset).list_tools = MagicMock(return_value=None)
    
    handler = IrisLLMToolHandler(mock_toolset)
    assert handler.toolset == mock_toolset

def test_get_tools_discovery():
    mock_toolset = MagicMock()
    mock_tool = {
        "name": "my_tool",
        "description": "my description",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}
    }
    mock_toolset.get_tools.return_value = [mock_tool]
    type(mock_toolset).tools = MagicMock(return_value=None)
    type(mock_toolset).list_tools = MagicMock(return_value=None)
    
    handler = IrisLLMToolHandler(mock_toolset)
    tools = handler.get_tools()
    
    assert len(tools) == 1
    assert tools[0]["name"] == "my_tool"
    assert tools[0]["description"] == "my description"
    assert tools[0]["inputSchema"]["type"] == "object"

def test_execute_via_toolset():
    mock_toolset = MagicMock()
    mock_tool = {"name": "my_tool"}
    mock_toolset.get_tools.return_value = [mock_tool]
    type(mock_toolset).tools = MagicMock(return_value=None)
    type(mock_toolset).list_tools = MagicMock(return_value=None)
    
    mock_toolset.execute.return_value = "result"
    
    handler = IrisLLMToolHandler(mock_toolset)
    result = handler.execute("my_tool", {"arg": 1})
    
    assert result == "result"
    mock_toolset.execute.assert_called_once_with("my_tool", {"arg": 1})

def test_execute_callable_tool():
    # Use a real class for the tool to avoid MagicMock hasattr issues
    class MyTool:
        def __init__(self):
            self.name = "callable_tool"
            self.description = "desc"
            self.input_schema = {"type": "object"}
            self.called_with = None
        def __call__(self, **kwargs):
            self.called_with = kwargs
            return "callable_result"
    
    tool = MyTool()
    
    # Create a toolset without 'execute'
    class MockToolSet:
        def get_tools(self):
            return [tool]
    
    mock_toolset = MockToolSet()
    
    handler = IrisLLMToolHandler(mock_toolset)
    result = handler.execute("callable_tool", {"x": 10})
    
    assert result == "callable_result"
    assert tool.called_with == {"x": 10}

def test_validate_params():
    mock_tool = {
        "name": "val_tool",
        "input_schema": {"required": ["important"]}
    }
    mock_toolset = MagicMock()
    mock_toolset.get_tools.return_value = [mock_tool]
    type(mock_toolset).tools = MagicMock(return_value=None)
    type(mock_toolset).list_tools = MagicMock(return_value=None)
    
    handler = IrisLLMToolHandler(mock_toolset)
    
    # Valid
    assert handler.validate_params("val_tool", {"important": "here"}) == {"important": "here"}
    
    # Invalid
    with pytest.raises(ValidationError):
        handler.validate_params("val_tool", {"not_important": "there"})

def test_has_tool():
    mock_toolset = MagicMock()
    mock_toolset.get_tools.return_value = [{"name": "exists"}]
    type(mock_toolset).tools = MagicMock(return_value=None)
    type(mock_toolset).list_tools = MagicMock(return_value=None)
    
    handler = IrisLLMToolHandler(mock_toolset)
    assert handler.has_tool("exists") is True
    assert handler.has_tool("does_not_exist") is False
