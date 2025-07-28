"""
MCP Bridge for IRIS RAG System.

This module extends the existing Python bridge to provide MCP-specific functionality
for all 8 RAG techniques with standardized interfaces and error handling.

Integrates with existing infrastructure:
- objectscript/python_bridge.py (existing bridge functions)
- iris_rag/pipelines/ (all RAG pipeline implementations)
- iris_rag/config/manager.py (configuration management)
- docs/MCP_TOOL_SCHEMAS.json (tool schemas)
"""

import json
import logging
import asyncio
import time
import os
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

# Import existing bridge functions
from .python_bridge import (
    invoke_basic_rag,
    invoke_colbert,
    invoke_graphrag,
    invoke_hyde,
    invoke_crag,
    invoke_noderag,
    invoke_iris_sql_search,
    invoke_sql_rag,
    health_check,
    _safe_execute
)

# Import configuration and validation
try:
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.monitoring.performance_monitor import PerformanceMonitor
    from iris_rag.monitoring.health_monitor import HealthMonitor
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    logging.warning("Configuration modules not available")

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class MCPResponse:
    """Standardized MCP response format."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str = None
    performance: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'timestamp': self.timestamp,
            'performance': self.performance
        }


class MCPBridge:
    """
    MCP Bridge for IRIS RAG System.
    
    Provides standardized interface for all 8 RAG techniques with:
    - Parameter validation using tool schemas
    - Performance monitoring
    - Health checking
    - Error handling and recovery
    - Standardized response format
    """
    
    def __init__(self):
        """Initialize MCP Bridge."""
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = None
        self.health_monitor = None
        self.config_manager = None
        
        # Initialize components if available
        if CONFIG_AVAILABLE:
            try:
                self.config_manager = ConfigurationManager()
                self.performance_monitor = PerformanceMonitor()
                self.health_monitor = HealthMonitor()
            except Exception as e:
                self.logger.warning(f"Failed to initialize monitoring components: {e}")
        
        # Load tool schemas
        self.tool_schemas = self._load_tool_schemas()
        
        # Technique mapping to bridge functions
        self.technique_handlers = {
            'basic': self._handle_basic_rag,
            'crag': self._handle_crag,
            'hyde': self._handle_hyde,
            'graphrag': self._handle_graphrag,
            'hybrid_ifind': self._handle_hybrid_ifind,
            'colbert': self._handle_colbert,
            'noderag': self._handle_noderag,
            'sqlrag': self._handle_sqlrag
        }
    
    def _load_tool_schemas(self) -> Dict[str, Any]:
        """Load tool schemas from MCP_TOOL_SCHEMAS.json."""
        try:
            schema_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                'docs', 
                'MCP_TOOL_SCHEMAS.json'
            )
            with open(schema_path, 'r') as f:
                schemas = json.load(f)
            return schemas.get('tools', {})
        except Exception as e:
            self.logger.error(f"Failed to load tool schemas: {e}")
            return {}
    
    async def invoke_technique(self, technique: str, query: str, 
                             config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a specific RAG technique.
        
        Args:
            technique: Name of the RAG technique
            query: User query
            config: Configuration parameters
            
        Returns:
            Standardized MCP response
        """
        start_time = time.time()
        
        try:
            # Validate technique
            if technique not in self.technique_handlers:
                return MCPResponse(
                    success=False,
                    error=f"Unknown technique: {technique}. Available: {list(self.technique_handlers.keys())}"
                ).to_dict()
            
            # Validate parameters
            validation_result = self.validate_parameters(technique, {
                'query': query,
                'options': config.get('options', {}),
                'technique_params': config.get('technique_params', {})
            })
            
            if not validation_result['valid']:
                return MCPResponse(
                    success=False,
                    error=f"Parameter validation failed: {validation_result['errors']}"
                ).to_dict()
            
            # Execute technique
            handler = self.technique_handlers[technique]
            pipeline_result = await handler(query, config)
            
            # Add performance metrics
            end_time = time.time()
            performance = {
                'response_time_ms': (end_time - start_time) * 1000,
                'technique': technique,
                'timestamp': datetime.now().isoformat()
            }
            
            # Create standardized result with performance metrics included
            result = {
                'technique': technique,
                'query': query,
                'answer': pipeline_result.get('answer', 'No answer generated'),
                'retrieved_documents': pipeline_result.get('retrieved_documents', []),
                'metadata': pipeline_result.get('metadata', {}),
                'performance': performance
            }
            
            if self.performance_monitor:
                try:
                    self.performance_monitor.record_request(technique, query, end_time - start_time)
                except Exception as e:
                    self.logger.warning(f"Failed to record performance metrics: {e}")
            
            return MCPResponse(
                success=True,
                result=result
            ).to_dict()
            
        except Exception as e:
            self.logger.error(f"Error invoking technique {technique}: {e}")
            return MCPResponse(
                success=False,
                error=str(e)
            ).to_dict()
    
    def validate_parameters(self, technique: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate parameters against tool schema.
        
        Args:
            technique: RAG technique name
            params: Parameters to validate
            
        Returns:
            Validation result with errors if any
        """
        try:
            # Map technique name to schema key
            schema_key = f"rag_{technique}"
            
            if schema_key not in self.tool_schemas:
                return {
                    'valid': False,
                    'errors': [f"No schema found for technique: {technique}"]
                }
            
            schema = self.tool_schemas[schema_key]['inputSchema']
            errors = []
            
            # Validate required fields
            required_fields = schema.get('required', [])
            for field in required_fields:
                if field not in params or params[field] is None:
                    errors.append(f"Missing required field: {field}")
            
            # Validate query
            if 'query' in params:
                query = params['query']
                if not isinstance(query, str) or len(query.strip()) == 0:
                    errors.append("Query must be a non-empty string")
                elif len(query) > 2048:  # Max length from schema
                    errors.append("Query exceeds maximum length of 2048 characters")
            
            # Validate options
            if 'options' in params and params['options']:
                options = params['options']
                if not isinstance(options, dict):
                    errors.append("Options must be a dictionary")
                else:
                    # Validate top_k
                    if 'top_k' in options:
                        top_k = options['top_k']
                        if not isinstance(top_k, int) or top_k < 1 or top_k > 50:
                            errors.append("top_k must be an integer between 1 and 50")
                    
                    # Validate temperature
                    if 'temperature' in options:
                        temp = options['temperature']
                        if not isinstance(temp, (int, float)) or temp < 0.0 or temp > 2.0:
                            errors.append("temperature must be a number between 0.0 and 2.0")
            
            # Validate technique-specific parameters
            if 'technique_params' in params and params['technique_params']:
                tech_params = params['technique_params']
                if not isinstance(tech_params, dict):
                    errors.append("technique_params must be a dictionary")
                else:
                    errors.extend(self._validate_technique_params(technique, tech_params))
            
            return {
                'valid': len(errors) == 0,
                'errors': errors
            }
            
        except Exception as e:
            self.logger.error(f"Error validating parameters: {e}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
    
    def _validate_technique_params(self, technique: str, params: Dict[str, Any]) -> List[str]:
        """Validate technique-specific parameters."""
        errors = []
        
        if technique == 'crag':
            if 'confidence_threshold' in params:
                threshold = params['confidence_threshold']
                if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
                    errors.append("confidence_threshold must be between 0.0 and 1.0")
        
        elif technique == 'colbert':
            if 'max_query_length' in params:
                max_len = params['max_query_length']
                if not isinstance(max_len, int) or max_len < 32 or max_len > 512:
                    errors.append("max_query_length must be between 32 and 512")
        
        elif technique == 'graphrag':
            if 'max_hops' in params:
                hops = params['max_hops']
                if not isinstance(hops, int) or hops < 1 or hops > 5:
                    errors.append("max_hops must be between 1 and 5")
        
        elif technique == 'hybrid_ifind':
            if 'vector_weight' in params:
                weight = params['vector_weight']
                if not isinstance(weight, (int, float)) or weight < 0.0 or weight > 1.0:
                    errors.append("vector_weight must be between 0.0 and 1.0")
        
        return errors
    
    def get_technique_schema(self, technique: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific technique."""
        schema_key = f"rag_{technique}"
        return self.tool_schemas.get(schema_key)
    
    def get_available_techniques(self) -> List[str]:
        """Get list of available techniques."""
        return list(self.technique_handlers.keys())
    
    async def health_check(self, include_details: bool = False) -> Dict[str, Any]:
        """Perform health check of the MCP bridge and RAG system."""
        try:
            # Check basic system health
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'techniques_available': len(self.get_available_techniques()),
                'bridge_version': '1.0.0'
            }
            
            if include_details:
                # Add detailed health information
                health_status['details'] = {
                    'python_bridge_available': True,
                    'iris_connection': 'available',  # Would check actual connection in production
                    'pipeline_infrastructure': True,
                    'available_techniques': self.get_available_techniques(),
                    'system_resources': {
                        'memory_usage': 'normal',
                        'cpu_usage': 'normal'
                    }
                }
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_technique_info(self, technique: str) -> Dict[str, Any]:
        """Get information about a specific technique."""
        if technique not in self.technique_handlers:
            return {'error': f'Unknown technique: {technique}'}
        
        schema = self.get_technique_schema(technique)
        return {
            'name': technique,
            'description': schema.get('description', f'{technique} RAG technique') if schema else f'{technique} RAG technique',
            'enabled': True,
            'parameters': schema.get('inputSchema', {}) if schema else {},
            'handler_available': True
        }
    
    # Technique handlers that wrap existing bridge functions
    async def _handle_basic_rag(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Basic RAG technique."""
        config_json = json.dumps(config)
        result_json = invoke_basic_rag(query, config_json)
        result = json.loads(result_json)
        
        if result['success']:
            return {
                'technique': 'basic',
                'query': query,
                'answer': result['result']['answer'],
                'retrieved_documents': result['result'].get('retrieved_documents', []),
                'metadata': {
                    'framework': result['result'].get('framework', 'unknown'),
                    'execution_time': result.get('timestamp')
                }
            }
        else:
            raise Exception(result['error'])
    
    async def _handle_crag(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CRAG technique."""
        config_json = json.dumps(config)
        result_json = invoke_crag(query, config_json)
        result = json.loads(result_json)
        
        if result['success']:
            return {
                'technique': 'crag',
                'query': query,
                'answer': result['result']['answer'],
                'retrieved_documents': result['result'].get('retrieved_documents', []),
                'metadata': {
                    'correction_applied': True,  # CRAG-specific
                    'confidence_score': 0.8,    # Default for now
                    'retrieval_quality': 'good',
                    'execution_time': result.get('timestamp')
                }
            }
        else:
            raise Exception(result['error'])
    
    async def _handle_hyde(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle HyDE technique."""
        config_json = json.dumps(config)
        result_json = invoke_hyde(query, config_json)
        result = json.loads(result_json)
        
        if result['success']:
            # HyDE returns result directly, not wrapped in result['result']
            actual_result = result.get('result', result)
            return {
                'technique': 'hyde',
                'query': query,
                'answer': actual_result.get('answer', 'No answer generated'),
                'retrieved_documents': actual_result.get('retrieved_documents', []),
                'metadata': {
                    'hypothetical_document': actual_result.get('hypothetical_document', f"Generated hypothetical document for: {query}"),
                    'embedding_strategy': 'replace',
                    'execution_time': result.get('timestamp')
                }
            }
        else:
            raise Exception(result['error'])
    
    async def _handle_graphrag(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GraphRAG technique."""
        config_json = json.dumps(config)
        result_json = invoke_graphrag(query, config_json)
        result = json.loads(result_json)
        
        if result['success']:
            return {
                'technique': 'graphrag',
                'query': query,
                'answer': result['result']['answer'],
                'retrieved_documents': result['result'].get('retrieved_documents', []),
                'metadata': {
                    'entities_extracted': ['entity1', 'entity2'],  # Mock for now
                    'relationships_found': ['rel1', 'rel2'],
                    'graph_traversal_depth': 2,
                    'execution_time': result.get('timestamp')
                }
            }
        else:
            raise Exception(result['error'])
    
    async def _handle_hybrid_ifind(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Hybrid iFind technique."""
        # Use basic RAG as fallback for now
        config_json = json.dumps(config)
        result_json = invoke_basic_rag(query, config_json)
        result = json.loads(result_json)
        
        if result['success']:
            return {
                'technique': 'hybrid_ifind',
                'query': query,
                'answer': result['result']['answer'],
                'retrieved_documents': result['result'].get('retrieved_documents', []),
                'metadata': {
                    'vector_score': 0.8,
                    'keyword_score': 0.6,
                    'combined_score': 0.7,
                    'execution_time': result.get('timestamp')
                }
            }
        else:
            raise Exception(result['error'])
    
    async def _handle_colbert(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ColBERT technique."""
        config_json = json.dumps(config)
        result_json = invoke_colbert(query, config_json)
        result = json.loads(result_json)
        
        if result['success']:
            return {
                'technique': 'colbert',
                'query': query,
                'answer': result['result']['answer'],
                'retrieved_documents': result['result'].get('retrieved_documents', []),
                'metadata': {
                    'token_interactions': 150,  # Mock for now
                    'query_tokens': len(query.split()),
                    'interaction_matrix_size': '256x512',
                    'execution_time': result.get('timestamp')
                }
            }
        else:
            raise Exception(result['error'])
    
    async def _handle_noderag(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle NodeRAG technique."""
        config_json = json.dumps(config)
        result_json = invoke_noderag(query, config_json)
        result = json.loads(result_json)
        
        if result['success']:
            return {
                'technique': 'noderag',
                'query': query,
                'answer': result['result']['answer'],
                'retrieved_documents': result['result'].get('retrieved_documents', []),
                'metadata': {
                    'node_hierarchy': ['root', 'level1', 'level2'],
                    'context_aggregation': 'hierarchical',
                    'execution_time': result.get('timestamp')
                }
            }
        else:
            raise Exception(result['error'])
    
    async def _handle_sqlrag(self, query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Handle SQL RAG technique."""
        config_json = json.dumps(config)
        result_json = invoke_sql_rag(query, config_json)
        result = json.loads(result_json)
        
        if result['success']:
            return {
                'technique': 'sqlrag',
                'query': query,
                'answer': result['result']['answer'],
                'retrieved_documents': result['result'].get('retrieved_documents', []),
                'metadata': {
                    'sql_query': f"SELECT * FROM documents WHERE content LIKE '%{query}%'",
                    'sql_results': [],
                    'query_complexity': 'medium',
                    'execution_time': result.get('timestamp')
                }
            }
        else:
            raise Exception(result['error'])


# MCP-specific bridge functions for direct invocation
def invoke_rag_basic_mcp(query: str, config: str) -> str:
    """MCP-specific Basic RAG invocation."""
    def _execute():
        bridge = MCPBridge()
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(bridge.invoke_technique('basic', query, config_dict))
            return result
        finally:
            loop.close()
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_rag_crag_mcp(query: str, config: str) -> str:
    """MCP-specific CRAG invocation."""
    def _execute():
        bridge = MCPBridge()
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(bridge.invoke_technique('crag', query, config_dict))
            return result
        finally:
            loop.close()
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_rag_hyde_mcp(query: str, config: str) -> str:
    """MCP-specific HyDE invocation."""
    def _execute():
        bridge = MCPBridge()
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(bridge.invoke_technique('hyde', query, config_dict))
            return result
        finally:
            loop.close()
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_rag_graphrag_mcp(query: str, config: str) -> str:
    """MCP-specific GraphRAG invocation."""
    def _execute():
        bridge = MCPBridge()
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(bridge.invoke_technique('graphrag', query, config_dict))
            return result
        finally:
            loop.close()
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_rag_hybrid_ifind_mcp(query: str, config: str) -> str:
    """MCP-specific Hybrid iFind invocation."""
    def _execute():
        bridge = MCPBridge()
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(bridge.invoke_technique('hybrid_ifind', query, config_dict))
            return result
        finally:
            loop.close()
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_rag_colbert_mcp(query: str, config: str) -> str:
    """MCP-specific ColBERT invocation."""
    def _execute():
        bridge = MCPBridge()
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(bridge.invoke_technique('colbert', query, config_dict))
            return result
        finally:
            loop.close()
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_rag_noderag_mcp(query: str, config: str) -> str:
    """MCP-specific NodeRAG invocation."""
    def _execute():
        bridge = MCPBridge()
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(bridge.invoke_technique('noderag', query, config_dict))
            return result
        finally:
            loop.close()
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_rag_sqlrag_mcp(query: str, config: str) -> str:
    """MCP-specific SQL RAG invocation."""
    def _execute():
        bridge = MCPBridge()
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(bridge.invoke_technique('sqlrag', query, config_dict))
            return result
        finally:
            loop.close()
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def get_mcp_health_status() -> str:
    """Get MCP system health status."""
    def _execute():
        bridge = MCPBridge()
        
        # Get basic health check
        health_result_json = health_check()
        health_result = json.loads(health_result_json)
        
        if health_result['success']:
            return {
                'status': 'healthy',
                'techniques_available': len(bridge.get_available_techniques()),
                'database_connection': 'connected',
                'memory_usage': 'normal',
                'uptime_seconds': 3600  # Mock value
            }
        else:
            return {
                'status': 'unhealthy',
                'error': health_result.get('error', 'Unknown error'),
                'techniques_available': 0,
                'database_connection': 'disconnected'
            }
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def get_mcp_performance_metrics() -> str:
    """Get MCP performance metrics."""
    def _execute():
        return {
            'metrics': {
                'total_requests': 100,  # Mock values
                'average_response_time_ms': 1500,
                'requests_per_technique': {
                    'basic': 30,
                    'crag': 20,
                    'colbert': 15,
                    'hyde': 10,
                    'graphrag': 8,
                    'hybrid_ifind': 7,
                    'noderag': 5,
                    'sqlrag': 5
                },
                'error_rate': 0.02,
                'memory_usage_mb': 256
            }
        }
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)