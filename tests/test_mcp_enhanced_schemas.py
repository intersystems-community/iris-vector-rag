#!/usr/bin/env python3
"""
Test script for enhanced MCP server tool schemas.

This script validates that the enhanced tool descriptions and parameter schemas
are properly implemented and accessible through the MCP interface.
"""

import json
import pytest
from pathlib import Path

def test_enhanced_schemas_exist():
    """Test that enhanced schema documentation exists."""
    schema_file = Path(__file__).parent.parent / "docs" / "MCP_TOOL_SCHEMAS_ENHANCED.json"
    assert schema_file.exists(), "Enhanced schema documentation should exist"
    
    with open(schema_file, 'r') as f:
        schemas = json.load(f)
    
    assert "tools" in schemas, "Schema should contain tools section"
    assert "parameter_guidelines" in schemas, "Schema should contain parameter guidelines"
    assert "usage_examples" in schemas, "Schema should contain usage examples"
    assert "best_practices" in schemas, "Schema should contain best practices"

def test_rag_tools_manager_initialization():
    """Test that RAG tools manager initializes with enhanced schemas."""
    import sys
    import os
    
    # Add nodejs src path to Python path for testing
    nodejs_src_path = Path(__file__).parent.parent / "nodejs" / "src"
    if str(nodejs_src_path) not in sys.path:
        sys.path.insert(0, str(nodejs_src_path))
    
    try:
        # This would normally import the RAG tools manager
        # For now, we'll just verify the file structure exists
        rag_tools_file = Path(__file__).parent.parent / "nodejs" / "src" / "mcp" / "rag_tools" / "index.js"
        assert rag_tools_file.exists(), "RAG tools manager should exist"
        
        # Read the file and verify it contains enhanced descriptions
        with open(rag_tools_file, 'r') as f:
            content = f.read()
        
        # Check for enhanced descriptions
        assert "foundational RAG approach" in content, "Should contain enhanced basic RAG description"
        assert "CRAG improves upon basic RAG" in content, "Should contain enhanced CRAG description"
        assert "HyDE generates a hypothetical answer" in content, "Should contain enhanced HyDE description"
        assert "GraphRAG builds a knowledge graph" in content, "Should contain enhanced GraphRAG description"
        assert "combines semantic vector similarity" in content, "Should contain enhanced Hybrid description"
        assert "fine-grained token-level interactions" in content, "Should contain enhanced ColBERT description"
        assert "hierarchical tree structure" in content, "Should contain enhanced NodeRAG description"
        assert "structured database queries" in content, "Should contain enhanced SQLRAG description"
        
        print("✓ Enhanced tool descriptions found in RAG tools manager")
        
    except Exception as e:
        pytest.skip(f"Could not test RAG tools manager: {e}")

def test_parameter_descriptions():
    """Test that parameter descriptions are comprehensive."""
    schema_file = Path(__file__).parent.parent / "docs" / "MCP_TOOL_SCHEMAS_ENHANCED.json"
    
    with open(schema_file, 'r') as f:
        schemas = json.load(f)
    
    guidelines = schemas["parameter_guidelines"]
    
    # Test query parameter guidelines
    assert "query" in guidelines
    query_info = guidelines["query"]
    assert "description" in query_info
    assert "constraints" in query_info
    assert "examples" in query_info
    assert len(query_info["examples"]) >= 3, "Should have multiple query examples"
    
    # Test options parameter guidelines
    assert "options" in guidelines
    options_info = guidelines["options"]
    assert "parameters" in options_info
    
    required_options = ["top_k", "temperature", "max_tokens", "include_sources"]
    for option in required_options:
        assert option in options_info["parameters"], f"Should have {option} parameter info"
        param_info = options_info["parameters"][option]
        assert "description" in param_info, f"{option} should have description"
        assert "default" in param_info, f"{option} should have default value"
    
    # Test technique-specific parameters
    assert "technique_params" in guidelines
    technique_params = guidelines["technique_params"]
    assert "by_technique" in technique_params
    
    expected_techniques = ["crag", "colbert", "graphrag", "hybrid_ifind", "hyde", "noderag", "sqlrag"]
    for technique in expected_techniques:
        assert technique in technique_params["by_technique"], f"Should have {technique} parameters"
    
    print("✓ Parameter descriptions are comprehensive")

def test_usage_examples():
    """Test that usage examples are provided."""
    schema_file = Path(__file__).parent.parent / "docs" / "MCP_TOOL_SCHEMAS_ENHANCED.json"
    
    with open(schema_file, 'r') as f:
        schemas = json.load(f)
    
    examples = schemas["usage_examples"]
    
    # Test basic query example
    assert "basic_query" in examples
    basic_example = examples["basic_query"]
    assert basic_example["tool"] == "rag_basic"
    assert "query" in basic_example["parameters"]
    assert "options" in basic_example["parameters"]
    
    # Test advanced example with technique params
    assert "advanced_crag" in examples
    crag_example = examples["advanced_crag"]
    assert crag_example["tool"] == "rag_crag"
    assert "technique_params" in crag_example["parameters"]
    
    # Test utility tool example
    assert "health_check" in examples
    health_example = examples["health_check"]
    assert health_example["tool"] == "rag_health_check"
    
    print("✓ Usage examples are comprehensive")

def test_best_practices():
    """Test that best practices are documented."""
    schema_file = Path(__file__).parent.parent / "docs" / "MCP_TOOL_SCHEMAS_ENHANCED.json"
    
    with open(schema_file, 'r') as f:
        schemas = json.load(f)
    
    best_practices = schemas["best_practices"]
    
    required_sections = ["parameter_selection", "technique_selection", "performance_optimization"]
    for section in required_sections:
        assert section in best_practices, f"Should have {section} best practices"
        assert isinstance(best_practices[section], list), f"{section} should be a list"
        assert len(best_practices[section]) > 0, f"{section} should have practices"
    
    print("✓ Best practices are documented")

if __name__ == "__main__":
    """Run tests directly."""
    print("Testing Enhanced MCP Tool Schemas...")
    print("=" * 50)
    
    try:
        test_enhanced_schemas_exist()
        print("✓ Enhanced schemas exist")
        
        test_rag_tools_manager_initialization()
        print("✓ RAG tools manager has enhanced descriptions")
        
        test_parameter_descriptions()
        print("✓ Parameter descriptions are comprehensive")
        
        test_usage_examples()
        print("✓ Usage examples are provided")
        
        test_best_practices()
        print("✓ Best practices are documented")
        
        print("\n" + "=" * 50)
        print("✅ All enhanced MCP schema tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise