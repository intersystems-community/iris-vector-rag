"""
Technique Handler Registry for MCP Server.

This module provides the registry and handlers for all 8 RAG techniques,
integrating with the existing pipeline implementations and providing
standardized interfaces for the MCP server.

Leverages existing infrastructure:
- iris_rag/pipelines/ (all RAG pipeline implementations)
- iris_rag/pipelines/factory.py (pipeline factory)
- iris_rag/pipelines/registry.py (pipeline registry)
"""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

# Import existing pipeline infrastructure
try:
    from iris_rag.pipelines.factory import PipelineFactory
    from iris_rag.pipelines.registry import PipelineRegistry
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.config.pipeline_config_service import PipelineConfigService
    from iris_rag.utils.module_loader import ModuleLoader
    PIPELINE_INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    PIPELINE_INFRASTRUCTURE_AVAILABLE = False
    logging.warning("Pipeline infrastructure not available")

# Import connection management
try:
    from common.iris_connection_manager import get_iris_connection
    CONNECTION_AVAILABLE = True
except ImportError:
    CONNECTION_AVAILABLE = False
    logging.warning("Connection management not available")

logger = logging.getLogger(__name__)


@dataclass
class TechniqueInfo:
    """Information about a RAG technique."""
    name: str
    description: str
    pipeline_class: str
    module_path: str
    enabled: bool = True
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class TechniqueHandlerRegistry:
    """
    Registry for RAG technique handlers.
    
    Manages the registration and execution of all 8 RAG techniques,
    providing a standardized interface for the MCP server while
    leveraging the existing pipeline infrastructure.
    """
    
    def __init__(self):
        """Initialize the technique handler registry."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.config_manager = None
        self.pipeline_factory = None
        self.pipeline_registry = None
        self.connection_manager = None
        
        # Registry state
        self.techniques = {}
        self.handlers = {}
        self.initialized = False
        
        # Initialize infrastructure
        self._initialize_infrastructure()
        
        # Register techniques
        self._register_techniques()
    
    def _initialize_infrastructure(self):
        """Initialize pipeline infrastructure components."""
        try:
            if PIPELINE_INFRASTRUCTURE_AVAILABLE:
                self.config_manager = ConfigurationManager()
                
                # Initialize pipeline components
                config_service = PipelineConfigService()
                module_loader = ModuleLoader()
                
                # Framework dependencies
                framework_dependencies = {
                    'config_manager': self.config_manager
                }
                
                if CONNECTION_AVAILABLE:
                    self.connection_manager = type('ConnectionManager', (), {
                        'get_connection': lambda: get_iris_connection()
                    })()
                    framework_dependencies['connection_manager'] = self.connection_manager
                
                # Initialize factory and registry
                self.pipeline_factory = PipelineFactory(
                    config_service, 
                    module_loader, 
                    framework_dependencies
                )
                self.pipeline_registry = PipelineRegistry(self.pipeline_factory)
                
                self.logger.info("Pipeline infrastructure initialized successfully")
                self.initialized = True
            else:
                self.logger.warning("Pipeline infrastructure not available - using fallback mode")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline infrastructure: {e}")
    
    def _register_techniques(self):
        """Register all available RAG techniques."""
        # Define technique configurations
        technique_configs = [
            TechniqueInfo(
                name="basic",
                description="Standard retrieval-augmented generation with vector similarity search",
                pipeline_class="BasicRAGPipeline",
                module_path="iris_rag.pipelines.basic"
            ),
            TechniqueInfo(
                name="crag",
                description="Corrective RAG with retrieval quality evaluation and correction",
                pipeline_class="CRAGPipeline",
                module_path="iris_rag.pipelines.crag"
            ),
            TechniqueInfo(
                name="hyde",
                description="Hypothetical Document Embeddings for improved retrieval",
                pipeline_class="HyDERAGPipeline",
                module_path="iris_rag.pipelines.hyde"
            ),
            TechniqueInfo(
                name="graphrag",
                description="Graph-based retrieval using entity relationships",
                pipeline_class="GraphRAGPipeline",
                module_path="iris_rag.pipelines.graphrag"
            ),
            TechniqueInfo(
                name="hybrid_ifind",
                description="Hybrid search combining vector similarity and keyword matching",
                pipeline_class="HybridIFindPipeline",
                module_path="iris_rag.pipelines.hybrid_ifind"
            ),
            TechniqueInfo(
                name="colbert",
                description="Late interaction retrieval with token-level matching",
                pipeline_class="ColBERTRAGPipeline",
                module_path="iris_rag.pipelines.colbert.pipeline"
            ),
            TechniqueInfo(
                name="noderag",
                description="Node-based retrieval with hierarchical document structure",
                pipeline_class="NodeRAGPipeline",
                module_path="iris_rag.pipelines.noderag"
            ),
            TechniqueInfo(
                name="sqlrag",
                description="SQL-based retrieval and generation",
                pipeline_class="SQLRAGPipeline",
                module_path="iris_rag.pipelines.sql_rag"
            )
        ]
        
        # Register each technique
        for technique_info in technique_configs:
            self.techniques[technique_info.name] = technique_info
            self.handlers[technique_info.name] = self._create_handler(technique_info)
        
        self.logger.info(f"Registered {len(self.techniques)} techniques: {list(self.techniques.keys())}")
    
    def _create_handler(self, technique_info: TechniqueInfo) -> Callable:
        """Create a handler function for a specific technique."""
        async def handler(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
            return await self._execute_technique(technique_info, query, config)
        
        return handler
    
    async def _execute_technique(self, technique_info: TechniqueInfo, 
                                query: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific RAG technique."""
        start_time = time.time()
        
        try:
            # Validate inputs
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            
            # Prepare configuration
            options = config.get('options', {})
            technique_params = config.get('technique_params', {})
            
            # Execute technique
            if self.initialized and self.pipeline_factory:
                # Use pipeline infrastructure
                result = await self._execute_with_pipeline(technique_info, query, options, technique_params)
            else:
                # Use fallback implementation
                result = await self._execute_fallback(technique_info, query, options, technique_params)
            
            # Add execution metadata
            execution_time = time.time() - start_time
            result['performance'] = {
                'execution_time_ms': execution_time * 1000,
                'technique': technique_info.name,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing technique {technique_info.name}: {e}")
            raise
    
    async def _execute_with_pipeline(self, technique_info: TechniqueInfo, 
                                   query: str, options: Dict[str, Any], 
                                   technique_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute technique using pipeline infrastructure."""
        try:
            # Get or create pipeline instance
            pipeline = self.pipeline_registry.get_pipeline(technique_info.name)
            
            if not pipeline:
                # Try to create pipeline
                pipeline = self.pipeline_factory.create_pipeline(technique_info.name)
                if pipeline:
                    # Register for future use
                    self.pipeline_registry._pipelines[technique_info.name] = pipeline
            
            if not pipeline:
                raise Exception(f"Failed to create pipeline for technique: {technique_info.name}")
            
            # Execute pipeline
            # Note: Different pipelines may have different interfaces
            # We'll use a standardized approach
            if hasattr(pipeline, 'execute'):
                result = await self._async_execute(pipeline.execute, query, options)
            elif hasattr(pipeline, 'run'):
                result = await self._async_execute(pipeline.run, query, options)
            else:
                raise Exception(f"Pipeline {technique_info.name} has no execute or run method")
            
            # Standardize result format
            return self._standardize_result(result, technique_info.name, query)
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed for {technique_info.name}: {e}")
            # Fall back to basic implementation
            return await self._execute_fallback(technique_info, query, options, technique_params)
    
    async def _execute_fallback(self, technique_info: TechniqueInfo, 
                              query: str, options: Dict[str, Any], 
                              technique_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute technique using fallback implementation."""
        # Mock implementation for now - in production this would use the bridge functions
        return {
            'technique': technique_info.name,
            'query': query,
            'answer': f"Mock answer for {technique_info.name} technique: {query}",
            'retrieved_documents': [
                {
                    'content': f"Mock document content for {query}",
                    'source': 'mock_source',
                    'score': 0.85
                }
            ],
            'metadata': {
                'execution_mode': 'fallback',
                'options': options,
                'technique_params': technique_params
            }
        }
    
    async def _async_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function asynchronously, handling both sync and async functions."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def _standardize_result(self, result: Any, technique: str, query: str) -> Dict[str, Any]:
        """Standardize pipeline result to common format."""
        if isinstance(result, dict):
            # Already in dictionary format
            standardized = {
                'technique': technique,
                'query': query,
                'answer': result.get('answer', ''),
                'retrieved_documents': result.get('retrieved_documents', []),
                'metadata': result.get('metadata', {})
            }
            
            # Ensure answer is present
            if not standardized['answer'] and 'result' in result:
                standardized['answer'] = str(result['result'])
            
            return standardized
        else:
            # Convert other formats
            return {
                'technique': technique,
                'query': query,
                'answer': str(result),
                'retrieved_documents': [],
                'metadata': {'raw_result': result}
            }
    
    def register_handlers(self):
        """Register all technique handlers (for compatibility)."""
        # Already done in __init__, but provided for interface compatibility
        pass
    
    def get_handler(self, technique: str) -> Optional[Callable]:
        """Get handler for a specific technique."""
        return self.handlers.get(technique)
    
    def list_techniques(self) -> List[str]:
        """List all available techniques."""
        return list(self.techniques.keys())
    
    def get_technique_info(self, technique: str) -> Optional[TechniqueInfo]:
        """Get information about a specific technique."""
        return self.techniques.get(technique)
    
    def is_technique_available(self, technique: str) -> bool:
        """Check if a technique is available."""
        technique_info = self.techniques.get(technique)
        return technique_info is not None and technique_info.enabled
    
    def get_technique_schema(self, technique: str) -> Dict[str, Any]:
        """Get schema for a specific technique."""
        technique_info = self.techniques.get(technique)
        if not technique_info:
            return {}
        
        # Base schema for all techniques
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
                        'max_tokens': {
                            'type': 'integer',
                            'minimum': 50,
                            'maximum': 4096,
                            'default': 1024,
                            'description': 'Maximum tokens in response'
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
        elif technique == 'graphrag':
            base_schema['properties']['technique_params'] = {
                'type': 'object',
                'properties': {
                    'max_hops': {
                        'type': 'integer',
                        'minimum': 1,
                        'maximum': 5,
                        'default': 2,
                        'description': 'Maximum graph traversal hops'
                    },
                    'entity_threshold': {
                        'type': 'number',
                        'minimum': 0.0,
                        'maximum': 1.0,
                        'default': 0.7,
                        'description': 'Entity extraction confidence threshold'
                    }
                }
            }
        elif technique == 'hybrid_ifind':
            base_schema['properties']['technique_params'] = {
                'type': 'object',
                'properties': {
                    'vector_weight': {
                        'type': 'number',
                        'minimum': 0.0,
                        'maximum': 1.0,
                        'default': 0.7,
                        'description': 'Weight for vector similarity scores'
                    },
                    'keyword_weight': {
                        'type': 'number',
                        'minimum': 0.0,
                        'maximum': 1.0,
                        'default': 0.3,
                        'description': 'Weight for keyword matching scores'
                    }
                }
            }
        
        return {
            'name': f'rag_{technique}',
            'description': technique_info.description,
            'inputSchema': base_schema
        }
    
    async def execute_technique(self, technique: str, query: str, 
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific technique."""
        if technique not in self.handlers:
            raise ValueError(f"Unknown technique: {technique}")
        
        handler = self.handlers[technique]
        return await handler(query, config)
    
    def get_registry_status(self) -> Dict[str, Any]:
        """Get status of the technique registry."""
        return {
            'initialized': self.initialized,
            'total_techniques': len(self.techniques),
            'enabled_techniques': len([t for t in self.techniques.values() if t.enabled]),
            'pipeline_infrastructure_available': PIPELINE_INFRASTRUCTURE_AVAILABLE,
            'connection_available': CONNECTION_AVAILABLE,
            'techniques': {
                name: {
                    'enabled': info.enabled,
                    'description': info.description,
                    'pipeline_class': info.pipeline_class
                }
                for name, info in self.techniques.items()
            }
        }