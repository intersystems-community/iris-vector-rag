"""
Python wrapper for RAG Tools Manager.

This module provides a Python interface to the Node.js RAG tools manager,
enabling integration with the MCP server test suite and Python-based
components while leveraging the Node.js MCP infrastructure.
"""

import json
import subprocess
import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGToolsManager:
    """
    Python wrapper for Node.js RAG Tools Manager.
    
    Provides a Python interface to the Node.js MCP tools while maintaining
    compatibility with the existing test infrastructure and Python components.
    """
    
    def __init__(self):
        """Initialize the RAG tools manager."""
        self.logger = logging.getLogger(__name__)
        self.node_script_path = None
        self.initialized = False
        
        # Initialize paths
        self._initialize_paths()
        
        # Initialize tools
        self._initialize_tools()
    
    def _initialize_paths(self):
        """Initialize file paths for Node.js integration."""
        # Determine project root (assuming we're in nodejs/src/mcp/)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent.parent
        
        # Path to Node.js RAG tools script
        self.node_script_path = project_root / 'nodejs' / 'src' / 'mcp' / 'rag_tools' / 'index.js'
        
        self.logger.info(f"Initialized paths - Node.js script: {self.node_script_path}")
    
    def _initialize_tools(self):
        """Initialize the tools registry."""
        try:
            # For now, we'll define the tools statically
            # In production, this could query the Node.js script
            self.tools = {
                'rag_basic': {
                    'name': 'rag_basic',
                    'description': 'Execute basic RAG with vector similarity search',
                    'technique': 'basic'
                },
                'rag_crag': {
                    'name': 'rag_crag',
                    'description': 'Execute Corrective RAG with retrieval quality evaluation',
                    'technique': 'crag'
                },
                'rag_hyde': {
                    'name': 'rag_hyde',
                    'description': 'Execute HyDE RAG with hypothetical document embeddings',
                    'technique': 'hyde'
                },
                'rag_graphrag': {
                    'name': 'rag_graphrag',
                    'description': 'Execute Graph RAG with entity relationship traversal',
                    'technique': 'graphrag'
                },
                'rag_hybrid_ifind': {
                    'name': 'rag_hybrid_ifind',
                    'description': 'Execute Hybrid iFind RAG combining vector and keyword search',
                    'technique': 'hybrid_ifind'
                },
                'rag_colbert': {
                    'name': 'rag_colbert',
                    'description': 'Execute ColBERT RAG with late interaction retrieval',
                    'technique': 'colbert'
                },
                'rag_noderag': {
                    'name': 'rag_noderag',
                    'description': 'Execute Node RAG with hierarchical document structure',
                    'technique': 'noderag'
                },
                'rag_sqlrag': {
                    'name': 'rag_sqlrag',
                    'description': 'Execute SQL RAG with database-driven retrieval',
                    'technique': 'sqlrag'
                },
                'rag_health_check': {
                    'name': 'rag_health_check',
                    'description': 'Check health status of RAG system components',
                    'technique': 'health'
                },
                'rag_metrics': {
                    'name': 'rag_metrics',
                    'description': 'Get performance metrics for RAG techniques',
                    'technique': 'metrics'
                }
            }
            
            self.initialized = True
            self.logger.info(f"Initialized {len(self.tools)} RAG tools")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tools: {e}")
    
    def create_technique_tools(self) -> List[Dict[str, Any]]:
        """Create technique tools for MCP server registration."""
        tools = []
        
        for tool_name, tool_info in self.tools.items():
            tool_schema = {
                'name': tool_name,
                'description': tool_info['description'],
                'inputSchema': self._generate_tool_schema(tool_info['technique'])
            }
            tools.append(tool_schema)
        
        return tools
    
    def _generate_tool_schema(self, technique: str) -> Dict[str, Any]:
        """Generate input schema for a technique."""
        base_schema = {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'User query text',
                    'minLength': 1,
                    'maxLength': 2048
                },
                'options': {
                    'type': 'object',
                    'properties': {
                        'top_k': {
                            'type': 'integer',
                            'minimum': 1,
                            'maximum': 50,
                            'default': 5,
                            'description': 'Number of documents to retrieve'
                        },
                        'temperature': {
                            'type': 'number',
                            'minimum': 0.0,
                            'maximum': 2.0,
                            'default': 0.7,
                            'description': 'LLM generation temperature'
                        },
                        'include_sources': {
                            'type': 'boolean',
                            'default': True,
                            'description': 'Include source documents in response'
                        }
                    }
                }
            },
            'required': ['query']
        }
        
        # Add technique-specific parameters
        if technique == 'crag':
            base_schema['properties']['technique_params'] = {
                'type': 'object',
                'properties': {
                    'confidence_threshold': {
                        'type': 'number',
                        'minimum': 0.0,
                        'maximum': 1.0,
                        'default': 0.8,
                        'description': 'Threshold for retrieval confidence'
                    },
                    'correction_strategy': {
                        'type': 'string',
                        'enum': ['rewrite', 'expand', 'filter'],
                        'default': 'rewrite',
                        'description': 'Strategy for correcting poor retrievals'
                    }
                }
            }
        elif technique == 'colbert':
            base_schema['properties']['technique_params'] = {
                'type': 'object',
                'properties': {
                    'max_query_length': {
                        'type': 'integer',
                        'minimum': 32,
                        'maximum': 512,
                        'default': 256,
                        'description': 'Maximum query length in tokens'
                    },
                    'interaction_threshold': {
                        'type': 'number',
                        'minimum': 0.0,
                        'maximum': 1.0,
                        'default': 0.5,
                        'description': 'Token interaction threshold'
                    }
                }
            }
        
        return base_schema
    
    def validate_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters for a specific tool."""
        if tool_name not in self.tools:
            return {
                'valid': False,
                'error': f'Unknown tool: {tool_name}'
            }
        
        # Basic validation
        if 'query' not in parameters:
            return {
                'valid': False,
                'error': 'Missing required parameter: query'
            }
        
        query = parameters['query']
        if not isinstance(query, str) or len(query) == 0:
            return {
                'valid': False,
                'error': 'Query must be a non-empty string'
            }
        
        if len(query) > 2048:
            return {
                'valid': False,
                'error': 'Query too long (max 2048 characters)'
            }
        
        # Validate options if present
        if 'options' in parameters:
            options = parameters['options']
            if not isinstance(options, dict):
                return {
                    'valid': False,
                    'error': 'Options must be a dictionary'
                }
            
            if 'top_k' in options:
                top_k = options['top_k']
                if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
                    return {
                        'valid': False,
                        'error': 'top_k must be an integer between 1 and 50'
                    }
        
        return {'valid': True}
    
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for all tools."""
        schemas = {}
        
        for tool_name, tool_info in self.tools.items():
            schemas[tool_name] = self._generate_tool_schema(tool_info['technique'])
        
        return schemas
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via Node.js bridge."""
        try:
            # Validate parameters
            validation = self.validate_parameters(tool_name, parameters)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['error']
                }
            
            # For now, return mock responses
            # In production, this would call the Node.js script
            technique = self.tools[tool_name]['technique']
            
            if technique == 'health':
                return await self._mock_health_check(parameters)
            elif technique == 'metrics':
                return await self._mock_metrics(parameters)
            else:
                return await self._mock_technique_execution(technique, parameters)
                
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _mock_technique_execution(self, technique: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock technique execution for testing."""
        query = parameters['query']
        options = parameters.get('options', {})
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        return {
            'success': True,
            'result': {
                'technique': technique,
                'query': query,
                'answer': f'Mock answer for {technique} technique: {query}',
                'retrieved_documents': [
                    {
                        'content': f'Mock document content for {query}',
                        'source': 'mock_source',
                        'score': 0.85
                    }
                ],
                'metadata': {
                    'execution_mode': 'mock',
                    'options': options,
                    'technique_params': parameters.get('technique_params', {})
                },
                'performance': {
                    'response_time_ms': 100,
                    'technique': technique,
                    'timestamp': '2025-01-01T00:00:00Z'
                }
            }
        }
    
    async def _mock_health_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock health check for testing."""
        include_details = parameters.get('include_details', False)
        
        result = {
            'success': True,
            'result': {
                'status': 'healthy',
                'techniques_status': {
                    technique: 'available' for technique in ['basic', 'crag', 'hyde', 'graphrag', 'hybrid_ifind', 'colbert', 'noderag', 'sqlrag']
                },
                'database_connection': 'connected',
                'performance_metrics': {
                    'memory_usage_mb': 256,
                    'cpu_usage_percent': 15,
                    'active_connections': 1
                }
            }
        }
        
        if include_details:
            result['result']['details'] = {
                'uptime_seconds': 3600,
                'total_requests': 100,
                'error_rate': 0.01
            }
        
        return result
    
    async def _mock_metrics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mock metrics collection for testing."""
        time_range = parameters.get('time_range', '1h')
        technique_filter = parameters.get('technique_filter', [])
        
        return {
            'success': True,
            'result': {
                'time_range': time_range,
                'query_count': 50,
                'average_response_time': 250.5,
                'technique_usage': {
                    'rag_basic': 20,
                    'rag_crag': 15,
                    'rag_hyde': 10,
                    'rag_colbert': 5
                },
                'error_rate': 0.02,
                'performance_summary': {
                    'fastest_technique': 'rag_basic',
                    'slowest_technique': 'rag_colbert',
                    'most_used': 'rag_basic'
                }
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            'initialized': self.initialized,
            'total_tools': len(self.tools),
            'node_script_path': str(self.node_script_path),
            'available_techniques': [tool for tool in self.tools.keys() if tool.startswith('rag_') and not tool.endswith('_health_check') and not tool.endswith('_metrics')]
        }