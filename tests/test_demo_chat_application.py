#!/usr/bin/env python3
"""
TDD tests for demo chat application.

This test suite drives the development of a comprehensive demo chat application
that showcases all rag-templates capabilities including:
- Simple API usage
- Standard API configuration  
- Enterprise features
- Framework migration paths
- IRIS existing data integration
- ObjectScript integration
- MCP server functionality
"""

import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestDemoChatApplicationCore:
    """Test core chat application functionality."""
    
    def test_chat_app_initialization(self):
        """Test that chat application initializes with proper configuration."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        assert app is not None
        assert hasattr(app, 'rag_simple')
        assert hasattr(app, 'rag_standard') 
        assert hasattr(app, 'rag_enterprise')
        assert hasattr(app, 'conversation_history')
        assert app.conversation_history == []
    
    def test_simple_api_chat(self):
        """Test simple API chat functionality."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Test simple chat
        response = app.chat_simple("What is machine learning?")
        
        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Check conversation history
        assert len(app.conversation_history) == 1
        assert app.conversation_history[0]['mode'] == 'simple'
        assert app.conversation_history[0]['query'] == "What is machine learning?"
        assert app.conversation_history[0]['response'] == response
    
    def test_standard_api_chat(self):
        """Test standard API chat with configuration."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Test with different techniques
        techniques = ['basic', 'hyde', 'crag']
        for technique in techniques:
            response = app.chat_standard(
                "What is deep learning?", 
                technique=technique,
                max_results=3
            )
            
            assert response is not None
            assert isinstance(response, dict)
            assert 'answer' in response
            assert 'technique' in response
            assert response['technique'] == technique
    
    def test_enterprise_api_chat(self):
        """Test enterprise API chat with advanced features."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Test enterprise features
        response = app.chat_enterprise(
            "Analyze the relationship between AI and healthcare",
            technique='graphrag',
            include_sources=True,
            confidence_threshold=0.8
        )
        
        assert response is not None
        assert isinstance(response, dict)
        assert 'answer' in response
        assert 'sources' in response
        assert 'confidence' in response
        assert 'technique' in response
    
    def test_conversation_history_management(self):
        """Test conversation history tracking and retrieval."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Have multiple conversations
        app.chat_simple("What is AI?")
        app.chat_standard("What is ML?", technique='basic')
        app.chat_enterprise("What is DL?", technique='hyde')
        
        # Check history
        history = app.get_conversation_history()
        assert len(history) == 3
        
        # Check history filtering
        simple_history = app.get_conversation_history(mode='simple')
        assert len(simple_history) == 1
        assert simple_history[0]['mode'] == 'simple'
        
        # Clear history
        app.clear_conversation_history()
        assert len(app.conversation_history) == 0


class TestDemoChatApplicationDataIntegration:
    """Test chat application with data integration features."""
    
    def test_document_loading_demo(self):
        """Test document loading and querying demo."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Test document loading
        sample_docs = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand text."
        ]
        
        result = app.load_sample_documents(sample_docs)
        
        assert result is True
        assert app.document_count > 0
        
        # Test querying loaded documents
        response = app.chat_simple("What is machine learning?")
        assert "machine learning" in response.lower()
    
    def test_directory_loading_demo(self):
        """Test loading documents from directory."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Create temporary directory with sample files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample files
            (temp_path / "doc1.txt").write_text("Artificial intelligence overview")
            (temp_path / "doc2.txt").write_text("Machine learning fundamentals")
            (temp_path / "doc3.md").write_text("# Deep Learning\nAdvanced neural networks")
            
            # Test directory loading
            result = app.load_documents_from_directory(str(temp_path))
            
            assert result is True
            assert app.document_count >= 3
    
    def test_iris_existing_data_demo(self):
        """Test IRIS existing data integration demo."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Mock IRIS connection and existing data
        mock_iris_config = {
            "existing_tables": {
                "Hospital.Patient": {
                    "content_fields": ["FirstName", "LastName", "Diagnosis"],
                    "id_field": "PatientID",
                    "template": "Patient {FirstName} {LastName}: {Diagnosis}"
                }
            }
        }
        
        result = app.configure_iris_integration(mock_iris_config)
        
        assert result is True
        assert app.iris_integration_enabled is True
        
        # Test query with IRIS integration
        response = app.chat_enterprise(
            "Show me diabetes patients",
            use_iris_data=True
        )
        
        assert response is not None
        assert isinstance(response, dict)


class TestDemoChatApplicationMigrationPaths:
    """Test migration path demonstrations."""
    
    def test_langchain_migration_demo(self):
        """Test LangChain migration demonstration."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Demo LangChain migration
        migration_demo = app.demonstrate_langchain_migration(
            "What is the difference between AI and ML?"
        )
        
        assert migration_demo is not None
        assert isinstance(migration_demo, dict)
        assert 'before_code' in migration_demo
        assert 'after_code' in migration_demo
        assert 'performance_comparison' in migration_demo
        assert 'lines_of_code_reduction' in migration_demo
        
        # Should show significant reduction
        assert migration_demo['lines_of_code_reduction'] > 80  # >80% reduction
    
    def test_llamaindex_migration_demo(self):
        """Test LlamaIndex migration demonstration."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        migration_demo = app.demonstrate_llamaindex_migration(
            "Explain neural networks"
        )
        
        assert migration_demo is not None
        assert isinstance(migration_demo, dict)
        assert 'before_code' in migration_demo
        assert 'after_code' in migration_demo
        assert 'setup_time_improvement' in migration_demo
        
        # Should show significant time improvement
        assert migration_demo['setup_time_improvement'] > 10  # >10x faster
    
    def test_custom_rag_migration_demo(self):
        """Test custom RAG migration demonstration."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        migration_demo = app.demonstrate_custom_rag_migration(
            "How does vector search work?"
        )
        
        assert migration_demo is not None
        assert isinstance(migration_demo, dict)
        assert 'complexity_reduction' in migration_demo
        assert migration_demo['complexity_reduction'] > 90  # >90% reduction


class TestDemoChatApplicationObjectScriptIntegration:
    """Test ObjectScript and embedded Python integration."""
    
    def test_objectscript_bridge_demo(self):
        """Test ObjectScript bridge demonstration."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Test ObjectScript integration demo
        objectscript_demo = app.demonstrate_objectscript_integration(
            "Patient lookup for diabetes care"
        )
        
        assert objectscript_demo is not None
        assert isinstance(objectscript_demo, dict)
        assert 'objectscript_code' in objectscript_demo
        assert 'python_bridge' in objectscript_demo
        assert 'performance_benefits' in objectscript_demo
    
    def test_embedded_python_demo(self):
        """Test embedded Python demonstration."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        embedded_demo = app.demonstrate_embedded_python(
            "Analyze patient treatment outcomes"
        )
        
        assert embedded_demo is not None
        assert isinstance(embedded_demo, dict)
        assert 'embedded_code' in embedded_demo
        assert 'performance_metrics' in embedded_demo
        assert 'iris_sql_integration' in embedded_demo
    
    def test_wsgi_deployment_demo(self):
        """Test IRIS WSGI deployment demonstration."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        wsgi_demo = app.demonstrate_wsgi_deployment()
        
        assert wsgi_demo is not None
        assert isinstance(wsgi_demo, dict)
        assert 'flask_app_code' in wsgi_demo
        assert 'deployment_config' in wsgi_demo
        assert 'performance_comparison' in wsgi_demo
        
        # Should show 2x performance improvement
        performance = wsgi_demo['performance_comparison']
        assert 'gunicorn_baseline' in performance
        assert 'iris_wsgi_improvement' in performance
        assert performance['iris_wsgi_improvement'] >= 2.0


class TestDemoChatApplicationMCPIntegration:
    """Test MCP server integration."""
    
    def test_mcp_server_initialization(self):
        """Test MCP server initialization."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        mcp_server = app.initialize_mcp_server()
        
        assert mcp_server is not None
        assert hasattr(mcp_server, 'list_tools')
        assert hasattr(mcp_server, 'call_tool')
        
        # Check available tools
        tools = mcp_server.list_tools()
        assert len(tools) > 0
        
        # Should have RAG technique tools
        tool_names = [tool['name'] for tool in tools]
        assert 'rag_query_basic' in tool_names
        assert 'rag_query_colbert' in tool_names
        assert 'rag_query_hyde' in tool_names
    
    def test_mcp_tool_execution(self):
        """Test MCP tool execution."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        mcp_server = app.initialize_mcp_server()
        
        # Test basic RAG tool
        result = mcp_server.call_tool(
            'rag_query_basic',
            {'query': 'What is artificial intelligence?'}
        )
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'content' in result
        assert len(result['content']) > 0
    
    def test_mcp_document_management(self):
        """Test MCP document management tools."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        mcp_server = app.initialize_mcp_server()
        
        # Test document addition tool
        result = mcp_server.call_tool(
            'add_documents',
            {
                'documents': [
                    'Sample document about AI',
                    'Another document about ML'
                ]
            }
        )
        
        assert result is not None
        assert result.get('success') is True
        
        # Test document count tool
        count_result = mcp_server.call_tool('get_document_count', {})
        assert count_result is not None
        assert count_result.get('count', 0) >= 2


class TestDemoChatApplicationPerformance:
    """Test performance demonstration features."""
    
    def test_technique_performance_comparison(self):
        """Test RAG technique performance comparison."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        query = "Compare machine learning algorithms"
        comparison = app.compare_technique_performance(query)
        
        assert comparison is not None
        assert isinstance(comparison, dict)
        
        # Should test multiple techniques
        techniques = ['basic', 'hyde', 'crag', 'colbert']
        for technique in techniques:
            assert technique in comparison
            result = comparison[technique]
            assert 'execution_time' in result
            assert 'answer_quality' in result
            assert 'answer' in result
    
    def test_scalability_demonstration(self):
        """Test scalability demonstration with multiple documents."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Test with different document counts
        doc_counts = [10, 100, 500]
        scalability_results = app.demonstrate_scalability(doc_counts)
        
        assert scalability_results is not None
        assert isinstance(scalability_results, dict)
        
        for count in doc_counts:
            assert str(count) in scalability_results
            result = scalability_results[str(count)]
            assert 'load_time' in result
            assert 'query_time' in result
            assert 'memory_usage' in result


class TestDemoChatApplicationUserInterface:
    """Test user interface components."""
    
    def test_cli_interface(self):
        """Test command-line interface."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Test CLI command processing
        cli_response = app.process_cli_command("simple", "What is AI?")
        assert cli_response is not None
        
        cli_response = app.process_cli_command("standard", "What is ML?", technique="hyde")
        assert cli_response is not None
        
        cli_response = app.process_cli_command("enterprise", "What is DL?", technique="graphrag")
        assert cli_response is not None
    
    def test_web_interface_endpoints(self):
        """Test web interface endpoints."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        web_app = app.create_web_interface()
        
        assert web_app is not None
        assert hasattr(web_app, 'test_client')
        
        # Test with Flask test client
        with web_app.test_client() as client:
            # Test chat endpoint
            response = client.post('/chat', json={
                'query': 'What is machine learning?',
                'mode': 'simple'
            })
            assert response.status_code == 200
            
            # Test migration demo endpoint
            response = client.get('/demo/migration/langchain')
            assert response.status_code == 200
            
            # Test technique comparison endpoint
            response = client.post('/demo/compare', json={
                'query': 'Explain neural networks'
            })
            assert response.status_code == 200


class TestDemoChatApplicationDocumentation:
    """Test documentation and help features."""
    
    def test_technique_documentation(self):
        """Test technique documentation generation."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        # Test documentation for each technique
        techniques = ['basic', 'hyde', 'crag', 'colbert', 'graphrag', 'hybrid_ifind', 'noderag', 'sql_rag']
        
        for technique in techniques:
            docs = app.get_technique_documentation(technique)
            assert docs is not None
            assert isinstance(docs, dict)
            assert 'name' in docs
            assert 'description' in docs
            assert 'use_cases' in docs
            assert 'example_code' in docs
    
    def test_migration_guide_generation(self):
        """Test migration guide generation."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        frameworks = ['langchain', 'llamaindex', 'custom']
        
        for framework in frameworks:
            guide = app.generate_migration_guide(framework)
            assert guide is not None
            assert isinstance(guide, dict)
            assert 'framework' in guide
            assert 'before_example' in guide
            assert 'after_example' in guide
            assert 'benefits' in guide
    
    def test_interactive_tutorial(self):
        """Test interactive tutorial system."""
        from examples.demo_chat_app import DemoChatApp
        
        app = DemoChatApp()
        
        tutorial = app.start_interactive_tutorial()
        assert tutorial is not None
        assert hasattr(tutorial, 'current_step')
        assert hasattr(tutorial, 'total_steps')
        assert tutorial.total_steps > 0
        
        # Test tutorial progression
        step1 = tutorial.get_current_step()
        assert step1 is not None
        
        next_step = tutorial.advance_step()
        assert next_step is not None
        assert tutorial.current_step > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])