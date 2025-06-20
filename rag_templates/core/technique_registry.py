"""
Technique Registry for RAG Templates Library Consumption Framework.

This module provides dynamic technique registration and discovery capabilities,
allowing the framework to manage available RAG techniques and their metadata.
"""

import logging
import yaml
import os
from typing import Dict, List, Any, Optional
from .errors import ConfigurationError, ValidationError

logger = logging.getLogger(__name__)


class TechniqueRegistry:
    """
    Registry for managing RAG technique discovery and metadata.
    
    This registry provides:
    1. Dynamic technique registration and discovery
    2. Technique metadata and requirements management
    3. Validation of technique dependencies
    4. Support for custom technique plugins
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Technique Registry.
        
        Args:
            config_path: Optional path to technique configuration file
        """
        self._techniques: Dict[str, Dict[str, Any]] = {}
        self._config_path = config_path or "config/pipelines.yaml"
        
        # Load built-in techniques
        self._load_builtin_techniques()
        
        # Load techniques from configuration file
        self._load_techniques_from_config()
        
        logger.info(f"Technique registry initialized with {len(self._techniques)} techniques")
    
    def _load_builtin_techniques(self) -> None:
        """Load built-in RAG techniques."""
        builtin_techniques = [
            {
                "name": "basic",
                "module": "iris_rag.pipelines.basic",
                "class": "BasicRAGPipeline",
                "enabled": True,
                "description": "Basic RAG pipeline with vector similarity search",
                "params": {
                    "top_k": 5,
                    "chunk_size": 1000,
                    "similarity_threshold": 0.7
                },
                "requirements": ["iris_rag", "sentence_transformers"]
            },
            {
                "name": "colbert",
                "module": "iris_rag.pipelines.colbert",
                "class": "ColBERTRAGPipeline",
                "enabled": True,
                "description": "ColBERT-based RAG with late interaction",
                "params": {
                    "top_k": 10,
                    "max_query_length": 512,
                    "doc_maxlen": 180
                },
                "requirements": ["iris_rag", "colbert"]
            },
            {
                "name": "hyde",
                "module": "iris_rag.pipelines.hyde",
                "class": "HyDERAGPipeline",
                "enabled": True,
                "description": "Hypothetical Document Embeddings RAG",
                "params": {
                    "top_k": 5,
                    "use_hypothetical_doc": True,
                    "temperature": 0.1
                },
                "requirements": ["iris_rag"]
            },
            {
                "name": "crag",
                "module": "iris_rag.pipelines.crag",
                "class": "CRAGPipeline",
                "enabled": True,
                "description": "Corrective RAG with confidence scoring",
                "params": {
                    "top_k": 5,
                    "confidence_threshold": 0.8,
                    "use_web_search": False
                },
                "requirements": ["iris_rag"]
            },
            {
                "name": "noderag",
                "module": "iris_rag.pipelines.noderag",
                "class": "NodeRAGPipeline",
                "enabled": True,
                "description": "Node-based RAG with hierarchical retrieval",
                "params": {
                    "top_k": 5,
                    "node_chunk_size": 512,
                    "overlap": 50
                },
                "requirements": ["iris_rag"]
            },
            {
                "name": "graphrag",
                "module": "iris_rag.pipelines.graphrag",
                "class": "GraphRAGPipeline",
                "enabled": True,
                "description": "Graph-based RAG with community detection",
                "params": {
                    "top_k": 5,
                    "community_level": 2,
                    "use_global_search": True
                },
                "requirements": ["iris_rag", "networkx"]
            },
            {
                "name": "hybrid_ifind",
                "module": "iris_rag.pipelines.hybrid_ifind",
                "class": "HybridIFindRAGPipeline",
                "enabled": True,
                "description": "Hybrid RAG combining vector search and iFind",
                "params": {
                    "top_k": 5,
                    "ifind_weight": 0.3,
                    "vector_weight": 0.7
                },
                "requirements": ["iris_rag"]
            }
        ]
        
        for technique in builtin_techniques:
            self._techniques[technique["name"]] = technique
        
        logger.debug(f"Loaded {len(builtin_techniques)} built-in techniques")
    
    def _load_techniques_from_config(self) -> None:
        """Load techniques from configuration file."""
        if not os.path.exists(self._config_path):
            logger.debug(f"Configuration file not found: {self._config_path}")
            return
        
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            pipelines = config.get("pipelines", [])
            for pipeline in pipelines:
                if isinstance(pipeline, dict) and "name" in pipeline:
                    # Convert config format to registry format
                    technique = {
                        "name": pipeline["name"].lower(),
                        "module": pipeline.get("module", ""),
                        "class": pipeline.get("class", ""),
                        "enabled": pipeline.get("enabled", True),
                        "description": pipeline.get("description", ""),
                        "params": pipeline.get("params", {}),
                        "requirements": pipeline.get("requirements", [])
                    }
                    
                    # Override built-in techniques if they exist
                    self._techniques[technique["name"]] = technique
            
            logger.info(f"Loaded techniques from config: {self._config_path}")
            
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse technique config: {e}")
        except Exception as e:
            logger.warning(f"Failed to load technique config: {e}")
    
    def list_techniques(self) -> List[Dict[str, Any]]:
        """
        List all registered techniques.
        
        Returns:
            List of technique dictionaries with metadata
        """
        return [technique.copy() for technique in self._techniques.values()]
    
    def get_technique_info(self, technique_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific technique.
        
        Args:
            technique_name: Name of the technique
            
        Returns:
            Technique information dictionary or None if not found
        """
        technique = self._techniques.get(technique_name.lower())
        return technique.copy() if technique else None
    
    def is_technique_available(self, technique_name: str) -> bool:
        """
        Check if a technique is available and enabled.
        
        Args:
            technique_name: Name of the technique
            
        Returns:
            True if technique is available and enabled
        """
        technique = self._techniques.get(technique_name.lower())
        if not technique:
            return False
        
        if not technique.get("enabled", True):
            return False
        
        # Check if required modules can be imported
        try:
            module_path = technique.get("module", "")
            if module_path:
                __import__(module_path)
            return True
        except ImportError:
            logger.debug(f"Technique {technique_name} not available: missing dependencies")
            return False
    
    def register_technique(self, technique_info: Dict[str, Any]) -> None:
        """
        Register a custom technique.
        
        Args:
            technique_info: Dictionary containing technique metadata
            
        Raises:
            ValidationError: If technique info is invalid
        """
        # Validate required fields
        required_fields = ["name", "module", "class"]
        for field in required_fields:
            if field not in technique_info:
                raise ValidationError(
                    f"Missing required field '{field}' in technique registration",
                    field=field
                )
        
        technique_name = technique_info["name"].lower()
        
        # Set defaults
        technique = {
            "name": technique_name,
            "module": technique_info["module"],
            "class": technique_info["class"],
            "enabled": technique_info.get("enabled", True),
            "description": technique_info.get("description", "Custom technique"),
            "params": technique_info.get("params", {}),
            "requirements": technique_info.get("requirements", [])
        }
        
        self._techniques[technique_name] = technique
        logger.info(f"Registered custom technique: {technique_name}")
    
    def unregister_technique(self, technique_name: str) -> bool:
        """
        Unregister a technique.
        
        Args:
            technique_name: Name of the technique to unregister
            
        Returns:
            True if technique was unregistered, False if not found
        """
        technique_name = technique_name.lower()
        if technique_name in self._techniques:
            del self._techniques[technique_name]
            logger.info(f"Unregistered technique: {technique_name}")
            return True
        return False
    
    def get_enabled_techniques(self) -> List[str]:
        """
        Get list of enabled technique names.
        
        Returns:
            List of enabled technique names
        """
        return [
            name for name, info in self._techniques.items()
            if info.get("enabled", True)
        ]
    
    def validate_technique_config(self, technique_name: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a specific technique.
        
        Args:
            technique_name: Name of the technique
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValidationError: If configuration is invalid
        """
        technique = self._techniques.get(technique_name.lower())
        if not technique:
            raise ValidationError(
                f"Unknown technique: {technique_name}",
                field="technique_name"
            )
        
        # Basic validation - check if config keys match expected params
        expected_params = technique.get("params", {})
        for key, value in config.items():
            if key in expected_params:
                expected_type = type(expected_params[key])
                if expected_type != type(None) and not isinstance(value, expected_type):
                    logger.warning(
                        f"Type mismatch for {technique_name}.{key}: "
                        f"expected {expected_type}, got {type(value)}"
                    )
        
        return True
    
    def get_technique_requirements(self, technique_name: str) -> List[str]:
        """
        Get requirements for a specific technique.
        
        Args:
            technique_name: Name of the technique
            
        Returns:
            List of required packages/modules
        """
        technique = self._techniques.get(technique_name.lower())
        return technique.get("requirements", []) if technique else []