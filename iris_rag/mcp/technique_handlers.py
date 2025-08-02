"""
Technique Handlers Registry for MCP Integration

This module provides the TechniqueHandlerRegistry for managing RAG technique
handlers in the MCP system. Implements minimal functionality to satisfy
test requirements following TDD principles.

GREEN PHASE: Minimal implementation to make tests pass.
"""

from typing import Dict, List, Any, Optional, Callable


class TechniqueHandlerRegistry:
    """
    Registry for managing RAG technique handlers.
    
    Provides registration, retrieval, and management of technique handlers
    for the MCP system.
    """
    
    def __init__(self):
        """Initialize the technique handler registry."""
        self.handlers = {}
        self.technique_metadata = {}
        
        # Register default techniques for GREEN phase
        self._register_default_techniques()
    
    def _register_default_techniques(self):
        """Register default technique handlers for GREEN phase."""
        default_techniques = [
            'basic', 'crag', 'hyde', 'graphrag',
            'hybrid_ifind', 'colbert', 'noderag', 'sqlrag'
        ]
        
        for technique in default_techniques:
            self.register_technique(
                technique,
                self._create_mock_handler(technique),
                {
                    'name': technique,
                    'description': f'{technique.upper()} RAG technique',
                    'version': '1.0.0',
                    'enabled': True,
                    'parameters': {
                        'query': {'type': 'string', 'required': True},
                        'top_k': {'type': 'integer', 'default': 5},
                        'temperature': {'type': 'float', 'default': 0.7}
                    }
                }
            )
    
    def _create_mock_handler(self, technique: str) -> Callable:
        """
        Create a mock handler function for a technique.
        
        Args:
            technique: Name of the technique
            
        Returns:
            Mock handler function
        """
        def mock_handler(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
            """Mock handler implementation for GREEN phase."""
            return {
                'success': True,
                'technique': technique,
                'query': query,
                'answer': f'Mock answer from {technique} technique',
                'retrieved_documents': [],
                'metadata': {
                    'execution_time_ms': 100,
                    'technique_specific': f'{technique}_data'
                }
            }
        
        return mock_handler
    
    def register_technique(self, name: str, handler: Callable, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a technique handler.
        
        Args:
            name: Name of the technique
            handler: Handler function for the technique
            metadata: Optional metadata for the technique
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if not callable(handler):
                return False
            
            self.handlers[name] = handler
            self.technique_metadata[name] = metadata or {}
            return True
        except Exception:
            return False
    
    def unregister_technique(self, name: str) -> bool:
        """
        Unregister a technique handler.
        
        Args:
            name: Name of the technique to unregister
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            if name in self.handlers:
                del self.handlers[name]
            if name in self.technique_metadata:
                del self.technique_metadata[name]
            return True
        except Exception:
            return False
    
    def get_handler(self, name: str) -> Optional[Callable]:
        """
        Get a technique handler by name.
        
        Args:
            name: Name of the technique
            
        Returns:
            Handler function if found, None otherwise
        """
        return self.handlers.get(name)
    
    def list_techniques(self) -> List[str]:
        """
        List all registered technique names.
        
        Returns:
            List of technique names
        """
        return list(self.handlers.keys())
    
    def get_technique_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a technique.
        
        Args:
            name: Name of the technique
            
        Returns:
            Metadata dictionary if found, None otherwise
        """
        return self.technique_metadata.get(name)
    
    def is_technique_registered(self, name: str) -> bool:
        """
        Check if a technique is registered.
        
        Args:
            name: Name of the technique
            
        Returns:
            True if technique is registered, False otherwise
        """
        return name in self.handlers
    
    def get_enabled_techniques(self) -> List[str]:
        """
        Get list of enabled technique names.
        
        Returns:
            List of enabled technique names
        """
        enabled = []
        for name, metadata in self.technique_metadata.items():
            if metadata.get('enabled', True):
                enabled.append(name)
        return enabled
    
    def enable_technique(self, name: str) -> bool:
        """
        Enable a technique.
        
        Args:
            name: Name of the technique
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.technique_metadata:
            self.technique_metadata[name]['enabled'] = True
            return True
        return False
    
    def disable_technique(self, name: str) -> bool:
        """
        Disable a technique.
        
        Args:
            name: Name of the technique
            
        Returns:
            True if successful, False otherwise
        """
        if name in self.technique_metadata:
            self.technique_metadata[name]['enabled'] = False
            return True
        return False
    
    def execute_technique(self, name: str, query: str, 
                         config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a technique handler.
        
        Args:
            name: Name of the technique
            query: Query string
            config: Configuration dictionary
            
        Returns:
            Result dictionary
        """
        try:
            handler = self.get_handler(name)
            if not handler:
                return {
                    'success': False,
                    'error': f'Technique {name} not found'
                }
            
            metadata = self.get_technique_metadata(name)
            if metadata and not metadata.get('enabled', True):
                return {
                    'success': False,
                    'error': f'Technique {name} is disabled'
                }
            
            return handler(query, config)
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_technique_config(self, name: str, 
                                 config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration for a technique.
        
        Args:
            name: Name of the technique
            config: Configuration to validate
            
        Returns:
            Validation result with valid flag and errors
        """
        errors = []
        
        metadata = self.get_technique_metadata(name)
        if not metadata:
            errors.append(f'Technique {name} not found')
            return {'valid': False, 'errors': errors}
        
        parameters = metadata.get('parameters', {})
        
        # Basic validation for GREEN phase
        for param_name, param_info in parameters.items():
            if param_info.get('required', False) and param_name not in config:
                errors.append(f'Required parameter {param_name} is missing')
            
            if param_name in config:
                param_type = param_info.get('type')
                param_value = config[param_name]
                
                if param_type == 'string' and not isinstance(param_value, str):
                    errors.append(f'Parameter {param_name} must be a string')
                elif param_type == 'integer' and not isinstance(param_value, int):
                    errors.append(f'Parameter {param_name} must be an integer')
                elif param_type == 'float' and not isinstance(param_value, (int, float)):
                    errors.append(f'Parameter {param_name} must be a number')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary containing registry statistics
        """
        enabled_count = len(self.get_enabled_techniques())
        
        return {
            'total_techniques': len(self.handlers),
            'enabled_techniques': enabled_count,
            'disabled_techniques': len(self.handlers) - enabled_count,
            'technique_names': self.list_techniques(),
            'registry_size_bytes': len(str(self.handlers)) + len(str(self.technique_metadata))
        }