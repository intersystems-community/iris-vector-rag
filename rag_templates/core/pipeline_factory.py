"""
Pipeline Factory for RAG Templates Library Consumption Framework.

This module provides dynamic pipeline creation with dependency injection,
component lifecycle management, and configuration validation.
"""

import logging
import importlib
from typing import Dict, Any, Optional, Type, List
from .config_manager import ConfigurationManager
from .technique_registry import TechniqueRegistry
from .errors import ConfigurationError, InitializationError

logger = logging.getLogger(__name__)


class PipelineFactory:
    """
    Factory for creating RAG pipelines with dependency injection.
    
    This factory provides:
    1. Dynamic pipeline creation with technique selection
    2. Dependency injection for components (LLM, embeddings, vector store)
    3. Component lifecycle management
    4. Configuration validation and auto-setup
    """
    
    def __init__(self, config_manager: ConfigurationManager, 
                 technique_registry: Optional[TechniqueRegistry] = None):
        """
        Initialize the Pipeline Factory.
        
        Args:
            config_manager: Configuration manager instance
            technique_registry: Optional technique registry (creates default if None)
        """
        self.config_manager = config_manager
        self.technique_registry = technique_registry or TechniqueRegistry()
        self._component_cache: Dict[str, Any] = {}
        
        logger.info("Pipeline factory initialized")
    
    def create_pipeline(self, technique_name: str, 
                       config_overrides: Optional[Dict[str, Any]] = None):
        """
        Create a pipeline instance for the specified technique.
        
        Args:
            technique_name: Name of the RAG technique
            config_overrides: Optional configuration overrides
            
        Returns:
            Initialized pipeline instance
            
        Raises:
            ConfigurationError: If technique is not found or invalid
            InitializationError: If pipeline creation fails
        """
        try:
            # Validate technique exists and is available
            if not self.technique_registry.is_technique_available(technique_name):
                available_techniques = self.technique_registry.get_enabled_techniques()
                raise ConfigurationError(
                    f"Technique '{technique_name}' is not available. "
                    f"Available techniques: {', '.join(available_techniques)}",
                    details={"technique": technique_name, "available": available_techniques}
                )
            
            # Get technique information
            technique_info = self.technique_registry.get_technique_info(technique_name)
            if not technique_info:
                raise ConfigurationError(
                    f"Technique '{technique_name}' not found in registry",
                    details={"technique": technique_name}
                )
            
            # Validate configuration
            pipeline_config = self._prepare_pipeline_config(technique_info, config_overrides)
            
            # Create pipeline instance
            pipeline_class = self._load_pipeline_class(technique_info)
            pipeline_instance = self._instantiate_pipeline(
                pipeline_class, 
                technique_info, 
                pipeline_config
            )
            
            logger.info(f"Created pipeline for technique: {technique_name}")
            return pipeline_instance
            
        except Exception as e:
            if isinstance(e, (ConfigurationError, InitializationError)):
                raise
            
            raise InitializationError(
                f"Failed to create pipeline for technique '{technique_name}': {str(e)}",
                component="PipelineFactory",
                details={"technique": technique_name, "error": str(e)}
            ) from e
    
    def _prepare_pipeline_config(self, technique_info: Dict[str, Any], 
                                config_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare configuration for pipeline creation.
        
        Args:
            technique_info: Technique metadata
            config_overrides: Optional configuration overrides
            
        Returns:
            Merged configuration dictionary
        """
        # Start with technique defaults
        config = technique_info.get("params", {}).copy()
        
        # Apply global configuration
        technique_name = technique_info["name"]
        global_config = self.config_manager.get_pipeline_config(technique_name)
        if global_config:
            config.update({k: v for k, v in global_config.items() if v is not None})
        
        # Apply overrides
        if config_overrides:
            config.update(config_overrides)
        
        # Validate configuration
        self.technique_registry.validate_technique_config(technique_name, config)
        
        return config
    
    def _load_pipeline_class(self, technique_info: Dict[str, Any]) -> Type:
        """
        Load the pipeline class for the technique.
        
        Args:
            technique_info: Technique metadata
            
        Returns:
            Pipeline class
            
        Raises:
            InitializationError: If class cannot be loaded
        """
        module_path = technique_info["module"]
        class_name = technique_info["class"]
        
        try:
            module = importlib.import_module(module_path)
            pipeline_class = getattr(module, class_name)
            
            logger.debug(f"Loaded pipeline class: {module_path}.{class_name}")
            return pipeline_class
            
        except ImportError as e:
            raise InitializationError(
                f"Failed to import pipeline module '{module_path}': {str(e)}",
                component="PipelineFactory",
                details={"module": module_path, "class": class_name}
            ) from e
        except AttributeError as e:
            raise InitializationError(
                f"Pipeline class '{class_name}' not found in module '{module_path}': {str(e)}",
                component="PipelineFactory",
                details={"module": module_path, "class": class_name}
            ) from e
    
    def _instantiate_pipeline(self, pipeline_class: Type, 
                             technique_info: Dict[str, Any],
                             config: Dict[str, Any]):
        """
        Instantiate the pipeline with dependency injection.
        
        Args:
            pipeline_class: Pipeline class to instantiate
            technique_info: Technique metadata
            config: Pipeline configuration
            
        Returns:
            Initialized pipeline instance
        """
        # Get or create required components
        connection_manager = self._get_connection_manager()
        llm_func = self._get_llm_function()
        
        # Create pipeline instance with dependency injection
        try:
            # Try different constructor signatures
            pipeline_instance = self._try_pipeline_constructors(
                pipeline_class,
                connection_manager=connection_manager,
                config_manager=self.config_manager,
                llm_func=llm_func,
                config=config
            )
            
            return pipeline_instance
            
        except Exception as e:
            raise InitializationError(
                f"Failed to instantiate pipeline: {str(e)}",
                component="PipelineFactory",
                details={"class": pipeline_class.__name__, "error": str(e)}
            ) from e
    
    def _try_pipeline_constructors(self, pipeline_class: Type, **kwargs):
        """
        Try different constructor signatures for pipeline instantiation.
        
        Args:
            pipeline_class: Pipeline class to instantiate
            **kwargs: Available components for injection
            
        Returns:
            Initialized pipeline instance
        """
        # Common constructor patterns
        constructor_patterns = [
            # Pattern 1: connection_manager, config_manager, llm_func
            lambda: pipeline_class(
                connection_manager=kwargs["connection_manager"],
                config_manager=kwargs["config_manager"],
                llm_func=kwargs["llm_func"]
            ),
            # Pattern 2: connection_manager, config_manager
            lambda: pipeline_class(
                connection_manager=kwargs["connection_manager"],
                config_manager=kwargs["config_manager"]
            ),
            # Pattern 3: config_manager only
            lambda: pipeline_class(
                config_manager=kwargs["config_manager"]
            ),
            # Pattern 4: no arguments (default constructor)
            lambda: pipeline_class()
        ]
        
        last_error = None
        for pattern in constructor_patterns:
            try:
                instance = pattern()
                
                # Try to inject LLM function if instance has the attribute
                if hasattr(instance, 'llm_func') and kwargs.get("llm_func"):
                    instance.llm_func = kwargs["llm_func"]
                
                return instance
                
            except TypeError as e:
                last_error = e
                continue
        
        # If all patterns failed, raise the last error
        raise last_error or Exception("No suitable constructor found")
    
    def _get_connection_manager(self):
        """Get or create connection manager."""
        if "connection_manager" not in self._component_cache:
            try:
                from common.iris_connection_manager import get_iris_connection
                self._component_cache["connection_manager"] = get_iris_connection()
            except ImportError:
                logger.warning("ConnectionManager not available, using mock")
                # Create a mock connection manager for testing
                from unittest.mock import MagicMock
                self._component_cache["connection_manager"] = MagicMock()
        
        return self._component_cache["connection_manager"]
    
    def _get_llm_function(self):
        """Get or create LLM function."""
        if "llm_func" not in self._component_cache:
            try:
                from common.utils import get_llm_func
                self._component_cache["llm_func"] = get_llm_func()
            except ImportError:
                logger.warning("LLM function not available, using mock")
                # Create a mock LLM function for testing
                self._component_cache["llm_func"] = lambda x: f"Mock response to: {x}"
        
        return self._component_cache["llm_func"]
    
    def _get_embedding_function(self):
        """Get or create embedding function."""
        if "embedding_func" not in self._component_cache:
            try:
                from common.utils import get_embedding_func
                self._component_cache["embedding_func"] = get_embedding_func()
            except ImportError:
                logger.warning("Embedding function not available, using mock")
                # Create a mock embedding function for testing
                import numpy as np
                self._component_cache["embedding_func"] = lambda x: np.random.rand(384).tolist()
        
        return self._component_cache["embedding_func"]
    
    def clear_cache(self) -> None:
        """Clear the component cache."""
        self._component_cache.clear()
        logger.debug("Component cache cleared")
    
    def get_available_techniques(self) -> List[str]:
        """
        Get list of available techniques.
        
        Returns:
            List of available technique names
        """
        return self.technique_registry.get_enabled_techniques()
    
    def validate_technique(self, technique_name: str) -> bool:
        """
        Validate that a technique can be created.
        
        Args:
            technique_name: Name of the technique to validate
            
        Returns:
            True if technique can be created
        """
        try:
            return self.technique_registry.is_technique_available(technique_name)
        except Exception:
            return False