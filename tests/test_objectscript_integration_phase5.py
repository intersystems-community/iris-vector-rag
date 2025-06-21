"""
Test ObjectScript Integration Phase 5: Library Consumption Framework Parity

Tests the extended ObjectScript bridge with Simple and Standard API integration
while maintaining backward compatibility.
"""

import pytest
import json
import sys
import os

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestObjectScriptSimpleAPIIntegration:
    """Test Simple API integration for ObjectScript."""
    
    def test_invoke_simple_rag_zero_config(self):
        """Test zero-config Simple API invocation from ObjectScript."""
        from objectscript.python_bridge import invoke_simple_rag
        
        query = "What is machine learning?"
        result_json = invoke_simple_rag(query)
        
        # Parse result
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True
        assert "result" in result
        assert "timestamp" in result
        
        # Verify answer content
        answer = result["result"]
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert "machine learning" in answer.lower() or "error" in answer.lower()
    
    def test_add_documents_simple_api(self):
        """Test adding documents via Simple API from ObjectScript."""
        from objectscript.python_bridge import add_documents_simple
        
        documents = ["Document 1 about AI", "Document 2 about ML"]
        documents_json = json.dumps(documents)
        
        result_json = add_documents_simple(documents_json)
        result = json.loads(result_json)
        
        # Verify success
        assert result["success"] is True
        assert "result" in result
    
    def test_configure_simple_rag(self):
        """Test configuring Simple API from ObjectScript."""
        from objectscript.python_bridge import configure_simple_rag
        
        config = {
            "embeddings:model": "all-MiniLM-L6-v2",
            "pipelines:basic:chunk_size": 500
        }
        config_json = json.dumps(config)
        
        result_json = configure_simple_rag(config_json)
        result = json.loads(result_json)
        
        # Verify success
        assert result["success"] is True
    
    def test_get_simple_rag_status(self):
        """Test getting Simple API status from ObjectScript."""
        from objectscript.python_bridge import get_simple_rag_status
        
        result_json = get_simple_rag_status()
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True
        assert "result" in result
        
        status = result["result"]
        assert "initialized" in status
        assert "document_count" in status


class TestObjectScriptStandardAPIIntegration:
    """Test Standard API integration for ObjectScript."""
    
    def test_invoke_configurable_rag_basic(self):
        """Test basic configurable RAG invocation from ObjectScript."""
        from objectscript.python_bridge import invoke_configurable_rag
        
        query = "What is artificial intelligence?"
        config = {"technique": "basic"}
        config_json = json.dumps(config)
        
        result_json = invoke_configurable_rag(query, config_json)
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True
        assert "result" in result
        
        # Verify answer content
        answer = result["result"]
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_invoke_configurable_rag_with_options(self):
        """Test configurable RAG with advanced options from ObjectScript."""
        from objectscript.python_bridge import invoke_configurable_rag_with_options
        
        query = "What is deep learning?"
        config = {
            "technique": "basic",
            "llm_provider": "openai",
            "technique_config": {
                "top_k": 3
            }
        }
        options = {
            "include_sources": True,
            "return_dict": True
        }
        
        config_json = json.dumps(config)
        options_json = json.dumps(options)
        
        result_json = invoke_configurable_rag_with_options(query, config_json, options_json)
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True
        assert "result" in result
        
        # Verify enhanced result format
        enhanced_result = result["result"]
        assert "answer" in enhanced_result
        assert "sources" in enhanced_result
        assert "metadata" in enhanced_result
        assert "technique" in enhanced_result
    
    def test_get_available_techniques(self):
        """Test getting available techniques from ObjectScript."""
        from objectscript.python_bridge import get_available_techniques
        
        result_json = get_available_techniques()
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True
        assert "result" in result
        
        techniques = result["result"]
        assert isinstance(techniques, list)
        assert "basic" in techniques
    
    def test_switch_technique(self):
        """Test switching RAG technique from ObjectScript."""
        from objectscript.python_bridge import switch_technique
        
        technique = "basic"
        config = {"chunk_size": 800}
        config_json = json.dumps(config)
        
        result_json = switch_technique(technique, config_json)
        result = json.loads(result_json)
        
        # Verify success
        assert result["success"] is True
    
    def test_get_technique_info(self):
        """Test getting technique information from ObjectScript."""
        from objectscript.python_bridge import get_technique_info
        
        technique = "basic"
        result_json = get_technique_info(technique)
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True
        assert "result" in result
        
        info = result["result"]
        assert isinstance(info, dict)


class TestObjectScriptConfigurationBridge:
    """Test configuration bridge functions for ObjectScript."""
    
    def test_load_configuration(self):
        """Test loading configuration from file for ObjectScript."""
        from objectscript.python_bridge import load_configuration
        
        # Test with non-existent file (should handle gracefully)
        config_path = "non_existent_config.yaml"
        result_json = load_configuration(config_path)
        result = json.loads(result_json)
        
        # Should handle missing file gracefully
        assert "success" in result
        assert "error" in result or result["success"] is True
    
    def test_validate_configuration(self):
        """Test validating configuration from ObjectScript."""
        from objectscript.python_bridge import validate_configuration
        
        config = {
            "database": {
                "iris": {
                    "host": "localhost",
                    "port": 1972
                }
            }
        }
        config_json = json.dumps(config)
        
        result_json = validate_configuration(config_json)
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True or "error" in result
    
    def test_get_default_configuration(self):
        """Test getting default configuration for ObjectScript."""
        from objectscript.python_bridge import get_default_configuration
        
        result_json = get_default_configuration()
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True
        assert "result" in result
        
        config = result["result"]
        assert isinstance(config, dict)
        assert "database" in config
        assert "embeddings" in config


class TestObjectScriptBackwardCompatibility:
    """Test backward compatibility with existing ObjectScript functions."""
    
    def test_existing_basic_rag_still_works(self):
        """Test that existing basic RAG function still works."""
        from objectscript.python_bridge import invoke_basic_rag
        
        query = "What is AI?"
        config = {
            "embedding_func": None,
            "llm_func": None
        }
        config_json = json.dumps(config)
        
        result_json = invoke_basic_rag(query, config_json)
        result = json.loads(result_json)
        
        # Should work or fail gracefully
        assert "success" in result
        assert "timestamp" in result
    
    def test_existing_health_check_still_works(self):
        """Test that existing health check function still works."""
        from objectscript.python_bridge import health_check
        
        result_json = health_check()
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True
        assert "result" in result
        
        health = result["result"]
        assert "status" in health
        assert "components" in health
    
    def test_existing_get_available_pipelines_still_works(self):
        """Test that existing pipeline listing function still works."""
        from objectscript.python_bridge import get_available_pipelines
        
        result_json = get_available_pipelines()
        result = json.loads(result_json)
        
        # Verify structure
        assert result["success"] is True
        assert "result" in result
        
        pipelines = result["result"]
        assert isinstance(pipelines, dict)


class TestObjectScriptErrorHandling:
    """Test error handling in ObjectScript integration."""
    
    def test_simple_api_error_handling(self):
        """Test error handling in Simple API functions."""
        from objectscript.python_bridge import invoke_simple_rag
        
        # Test with empty query
        result_json = invoke_simple_rag("")
        result = json.loads(result_json)
        
        # Should handle gracefully
        assert "success" in result
        assert "timestamp" in result
    
    def test_standard_api_error_handling(self):
        """Test error handling in Standard API functions."""
        from objectscript.python_bridge import invoke_configurable_rag
        
        query = "Test query"
        invalid_config = "invalid json"
        
        result_json = invoke_configurable_rag(query, invalid_config)
        result = json.loads(result_json)
        
        # Should handle invalid JSON gracefully
        assert result["success"] is False
        assert "error" in result
    
    def test_configuration_error_handling(self):
        """Test error handling in configuration functions."""
        from objectscript.python_bridge import validate_configuration
        
        invalid_config = "not json"
        result_json = validate_configuration(invalid_config)
        result = json.loads(result_json)
        
        # Should handle invalid JSON gracefully
        assert result["success"] is False
        assert "error" in result


class TestObjectScriptJSONResponseFormat:
    """Test JSON response format consistency."""
    
    def test_all_functions_return_valid_json(self):
        """Test that all ObjectScript functions return valid JSON."""
        from objectscript import python_bridge
        
        # Get all callable functions from the module
        functions_to_test = [
            'health_check',
            'get_available_pipelines',
            'get_default_configuration'
        ]
        
        for func_name in functions_to_test:
            if hasattr(python_bridge, func_name):
                func = getattr(python_bridge, func_name)
                if callable(func):
                    try:
                        result_json = func()
                        # Should be valid JSON
                        result = json.loads(result_json)
                        assert isinstance(result, dict)
                        assert "success" in result
                        assert "timestamp" in result
                    except Exception as e:
                        pytest.fail(f"Function {func_name} failed: {e}")
    
    def test_response_format_consistency(self):
        """Test that all responses follow the same format."""
        from objectscript.python_bridge import health_check, get_default_configuration
        
        functions = [health_check, get_default_configuration]
        
        for func in functions:
            result_json = func()
            result = json.loads(result_json)
            
            # All responses should have these fields
            required_fields = ["success", "timestamp"]
            for field in required_fields:
                assert field in result, f"Missing {field} in {func.__name__} response"
            
            # Success responses should have result
            if result["success"]:
                assert "result" in result
            else:
                assert "error" in result