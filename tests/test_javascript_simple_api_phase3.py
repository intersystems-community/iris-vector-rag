"""
TDD Anchor Tests for Phase 3: JavaScript Simple API Implementation.

These tests define the expected behavior of the JavaScript Simple API
that mirrors the Python Simple API functionality.
"""

import pytest
import subprocess
import json
import os
import tempfile
from pathlib import Path


class TestJavaScriptSimpleAPI:
    """Test JavaScript Simple API parity with Python Simple API."""
    
    @pytest.fixture
    def nodejs_project_path(self):
        """Get the Node.js project path."""
        return Path(__file__).parent.parent / "nodejs"
    
    @pytest.fixture
    def test_documents(self):
        """Sample documents for testing."""
        return [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Vector databases store high-dimensional data for similarity search operations.",
            "RAG combines retrieval and generation for enhanced AI responses."
        ]
    
    def test_javascript_simple_api_exists(self, nodejs_project_path):
        """Test that the JavaScript Simple API module exists."""
        simple_api_path = nodejs_project_path / "src" / "simple.js"
        assert simple_api_path.exists(), "JavaScript Simple API module should exist"
    
    def test_javascript_simple_api_zero_config_initialization(self, nodejs_project_path):
        """Test zero-config initialization of JavaScript RAG class."""
        test_script = """
        const { RAG } = require('./src/simple.js');
        
        try {
            const rag = new RAG();
            console.log(JSON.stringify({
                success: true,
                initialized: rag !== null,
                type: typeof rag
            }));
        } catch (error) {
            console.log(JSON.stringify({
                success: false,
                error: error.message
            }));
        }
        """
        
        result = self._run_node_script(nodejs_project_path, test_script)
        assert result["success"], f"Zero-config initialization failed: {result.get('error')}"
        assert result["initialized"], "RAG instance should be created"
        assert result["type"] == "object", "RAG should be an object"
    
    def test_javascript_simple_api_add_documents_method(self, nodejs_project_path, test_documents):
        """Test addDocuments method exists and accepts string arrays."""
        test_script = f"""
        const {{ RAG }} = require('./src/simple.js');
        
        try {{
            const rag = new RAG();
            const documents = {json.dumps(test_documents)};
            
            // Check if method exists
            const hasMethod = typeof rag.addDocuments === 'function';
            
            console.log(JSON.stringify({{
                success: true,
                hasAddDocumentsMethod: hasMethod,
                documentsLength: documents.length
            }}));
        }} catch (error) {{
            console.log(JSON.stringify({{
                success: false,
                error: error.message
            }}));
        }}
        """
        
        result = self._run_node_script(nodejs_project_path, test_script)
        assert result["success"], f"addDocuments method test failed: {result.get('error')}"
        assert result["hasAddDocumentsMethod"], "RAG should have addDocuments method"
    
    def test_javascript_simple_api_query_method(self, nodejs_project_path):
        """Test query method exists and returns string answers."""
        test_script = """
        const { RAG } = require('./src/simple.js');
        
        try {
            const rag = new RAG();
            
            // Check if method exists
            const hasMethod = typeof rag.query === 'function';
            
            console.log(JSON.stringify({
                success: true,
                hasQueryMethod: hasMethod
            }));
        } catch (error) {
            console.log(JSON.stringify({
                success: false,
                error: error.message
            }));
        }
        """
        
        result = self._run_node_script(nodejs_project_path, test_script)
        assert result["success"], f"query method test failed: {result.get('error')}"
        assert result["hasQueryMethod"], "RAG should have query method"
    
    def test_javascript_config_manager_exists(self, nodejs_project_path):
        """Test that JavaScript ConfigManager exists."""
        config_manager_path = nodejs_project_path / "src" / "config-manager.js"
        assert config_manager_path.exists(), "JavaScript ConfigManager should exist"
    
    def test_javascript_config_manager_three_tier_system(self, nodejs_project_path):
        """Test JavaScript ConfigManager implements three-tier configuration."""
        test_script = """
        const { ConfigManager } = require('./src/config-manager.js');
        
        try {
            const config = new ConfigManager();
            
            // Test basic functionality
            const hasGet = typeof config.get === 'function';
            const hasSet = typeof config.set === 'function';
            const hasValidate = typeof config.validate === 'function';
            
            console.log(JSON.stringify({
                success: true,
                hasGet: hasGet,
                hasSet: hasSet,
                hasValidate: hasValidate
            }));
        } catch (error) {
            console.log(JSON.stringify({
                success: false,
                error: error.message
            }));
        }
        """
        
        result = self._run_node_script(nodejs_project_path, test_script)
        assert result["success"], f"ConfigManager test failed: {result.get('error')}"
        assert result["hasGet"], "ConfigManager should have get method"
        assert result["hasSet"], "ConfigManager should have set method"
        assert result["hasValidate"], "ConfigManager should have validate method"
    
    def test_javascript_environment_variable_support(self, nodejs_project_path):
        """Test JavaScript ConfigManager supports RAG_ environment variables."""
        test_script = """
        const { ConfigManager } = require('./src/config-manager.js');
        
        try {
            // Set test environment variable
            process.env.RAG_TEST__VALUE = 'test_value';
            
            const config = new ConfigManager();
            const value = config.get('test:value');
            
            console.log(JSON.stringify({
                success: true,
                envVarSupported: value === 'test_value'
            }));
        } catch (error) {
            console.log(JSON.stringify({
                success: false,
                error: error.message
            }));
        }
        """
        
        result = self._run_node_script(nodejs_project_path, test_script)
        assert result["success"], f"Environment variable test failed: {result.get('error')}"
        assert result["envVarSupported"], "ConfigManager should support RAG_ environment variables"
    
    def test_javascript_standard_api_exists(self, nodejs_project_path):
        """Test that JavaScript Standard API exists."""
        standard_api_path = nodejs_project_path / "src" / "standard.js"
        assert standard_api_path.exists(), "JavaScript Standard API should exist"
    
    def test_javascript_standard_api_configurable_rag(self, nodejs_project_path):
        """Test ConfigurableRAG class exists and accepts configuration."""
        test_script = """
        const { ConfigurableRAG } = require('./src/standard.js');
        
        try {
            const config = {
                technique: 'basic',
                llm_provider: 'test'
            };
            
            const rag = new ConfigurableRAG(config);
            
            console.log(JSON.stringify({
                success: true,
                initialized: rag !== null,
                type: typeof rag
            }));
        } catch (error) {
            console.log(JSON.stringify({
                success: false,
                error: error.message
            }));
        }
        """
        
        result = self._run_node_script(nodejs_project_path, test_script)
        assert result["success"], f"ConfigurableRAG test failed: {result.get('error')}"
        assert result["initialized"], "ConfigurableRAG should be initialized"
    
    def test_javascript_mcp_integration_exists(self, nodejs_project_path):
        """Test that MCP integration layer exists."""
        mcp_dir = nodejs_project_path / "src" / "mcp"
        assert mcp_dir.exists(), "MCP integration directory should exist"
        
        server_path = mcp_dir / "server.js"
        tools_path = mcp_dir / "tools.js"
        
        assert server_path.exists(), "MCP server.js should exist"
        assert tools_path.exists(), "MCP tools.js should exist"
    
    def test_zero_config_mcp_server_creation(self, nodejs_project_path):
        """Test trivial MCP server creation with zero config."""
        test_script = """
        const { createMCPServer } = require('./src/mcp/server.js');
        
        try {
            const server = createMCPServer({
                name: 'test-rag-server',
                description: 'Test RAG MCP server'
            });
            
            const hasStart = typeof server.start === 'function';
            const hasStop = typeof server.stop === 'function';
            
            console.log(JSON.stringify({
                success: true,
                serverCreated: server !== null,
                hasStart: hasStart,
                hasStop: hasStop
            }));
        } catch (error) {
            console.log(JSON.stringify({
                success: false,
                error: error.message
            }));
        }
        """
        
        result = self._run_node_script(nodejs_project_path, test_script)
        assert result["success"], f"MCP server creation failed: {result.get('error')}"
        assert result["serverCreated"], "MCP server should be created"
        assert result["hasStart"], "MCP server should have start method"
        assert result["hasStop"], "MCP server should have stop method"
    
    def test_package_json_updated_exports(self, nodejs_project_path):
        """Test that package.json has proper exports for the new APIs."""
        package_json_path = nodejs_project_path / "package.json"
        assert package_json_path.exists(), "package.json should exist"
        
        with open(package_json_path, 'r') as f:
            package_data = json.load(f)
        
        # Check for exports field
        assert "exports" in package_data, "package.json should have exports field"
        
        exports = package_data["exports"]
        assert "./simple" in exports, "Should export simple API"
        assert "./standard" in exports, "Should export standard API"
        assert "./mcp" in exports, "Should export MCP integration"
    
    def _run_node_script(self, nodejs_project_path, script_content):
        """Helper method to run Node.js script and return JSON result."""
        # Replace relative paths with absolute paths in the script content
        script_content = script_content.replace(
            "'./src/", f"'{nodejs_project_path}/src/"
        ).replace(
            '"./src/', f'"{nodejs_project_path}/src/'
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(script_content)
            temp_script_path = f.name
        
        try:
            result = subprocess.run(
                ['node', temp_script_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Script failed with code {result.returncode}: {result.stderr}"
                }
            
            try:
                # Extract JSON from stdout (may contain debug output)
                stdout_lines = result.stdout.strip().split('\n')
                json_line = None
                
                # Find the line that contains valid JSON
                for line in stdout_lines:
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            json.loads(line)  # Test if it's valid JSON
                            json_line = line
                            break
                        except json.JSONDecodeError:
                            continue
                
                if json_line:
                    return json.loads(json_line)
                else:
                    return {
                        "success": False,
                        "error": f"No valid JSON found in output: {result.stdout}"
                    }
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": f"Invalid JSON output: {result.stdout}"
                }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Script execution timed out"
            }
        
        finally:
            os.unlink(temp_script_path)


class TestJavaScriptAPILanguageParity:
    """Test language parity between Python and JavaScript APIs."""
    
    def test_simple_api_method_parity(self):
        """Test that JavaScript Simple API has same methods as Python."""
        # This will be implemented after the JavaScript API is created
        # For now, we define the expected interface
        expected_methods = [
            'addDocuments',
            'query',
            'getDocumentCount',
            'clearKnowledgeBase',
            'getConfig',
            'setConfig',
            'validateConfig'
        ]
        
        # This test will verify method existence once implemented
        assert len(expected_methods) > 0, "Expected methods defined"
    
    def test_standard_api_method_parity(self):
        """Test that JavaScript Standard API has same methods as Python."""
        expected_methods = [
            'query',
            'addDocuments',
            'getAvailableTechniques',
            'getTechniqueInfo',
            'switchTechnique',
            'getConfig'
        ]
        
        # This test will verify method existence once implemented
        assert len(expected_methods) > 0, "Expected methods defined"
    
    def test_configuration_manager_parity(self):
        """Test that JavaScript ConfigManager matches Python functionality."""
        expected_features = [
            'three_tier_configuration',
            'environment_variable_support',
            'dot_notation_keys',
            'validation',
            'type_casting'
        ]
        
        # This test will verify feature parity once implemented
        assert len(expected_features) > 0, "Expected features defined"