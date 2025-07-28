"""
Configuration schema validation framework for Quick Start system.

This module provides JSON schema validation for configuration templates,
ensuring that resolved configurations meet structural and business requirements.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema
from jsonschema import Draft7Validator, ValidationError as JsonSchemaValidationError
from referencing import Registry, Resource

from .interfaces import ValidationError


logger = logging.getLogger(__name__)


class ConfigurationSchemaValidator:
    """
    JSON schema validator for configuration templates.
    
    Provides validation of configuration structures against JSON schemas,
    with support for profile-specific constraints and custom validation rules.
    """
    
    def __init__(self, schema_dir: Optional[Path] = None):
        """
        Initialize the schema validator.
        
        Args:
            schema_dir: Directory containing JSON schema files.
                       Defaults to quick_start/config/schemas/
        """
        if schema_dir is None:
            schema_dir = Path(__file__).parent / "schemas"
        
        self.schema_dir = Path(schema_dir)
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._validator_cache: Dict[str, Draft7Validator] = {}
        self._registry: Optional[Registry] = None
        
        # Ensure schema directory exists
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Initialized schema validator with schema_dir: {self.schema_dir}")
    
    def load_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Load a JSON schema by name.
        
        Args:
            schema_name: Name of the schema file (without .json extension)
            
        Returns:
            Dictionary containing the JSON schema
            
        Raises:
            ValidationError: If schema file doesn't exist or is invalid
        """
        # Check cache first
        if schema_name in self._schema_cache:
            return self._schema_cache[schema_name]
        
        schema_path = self.schema_dir / f"{schema_name}.json"
        
        if not schema_path.exists():
            raise ValidationError(f"Schema not found: {schema_name}")
        
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            # Validate the schema itself
            Draft7Validator.check_schema(schema)
            
            # Cache the schema
            self._schema_cache[schema_name] = schema
            
            logger.debug(f"Loaded schema: {schema_name}")
            return schema
            
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in schema {schema_name}: {e}")
        except jsonschema.SchemaError as e:
            raise ValidationError(f"Invalid JSON schema {schema_name}: {e}")
        except Exception as e:
            raise ValidationError(f"Error loading schema {schema_name}: {e}")
    
    def _build_registry(self) -> Registry:
        """
        Build a registry containing all schemas for reference resolution.
        
        Returns:
            Registry with all available schemas
        """
        if self._registry is not None:
            return self._registry
        
        registry = Registry()
        
        # Load all schema files in the directory
        for schema_file in self.schema_dir.glob("*.json"):
            schema_name = schema_file.stem
            try:
                schema = self.load_schema(schema_name)
                # Create a resource with the schema content
                resource = Resource.from_contents(schema)
                # Register with the schema filename as the URI
                registry = registry.with_resource(f"{schema_name}.json", resource)
            except Exception as e:
                logger.warning(f"Failed to load schema {schema_name} for registry: {e}")
        
        self._registry = registry
        return registry
    
    def get_validator(self, schema_name: str) -> Draft7Validator:
        """
        Get a JSON schema validator for the specified schema.
        
        Args:
            schema_name: Name of the schema
            
        Returns:
            Draft7Validator instance for the schema
        """
        # Check cache first
        if schema_name in self._validator_cache:
            return self._validator_cache[schema_name]
        
        schema = self.load_schema(schema_name)
        registry = self._build_registry()
        
        # Create validator with registry for reference resolution
        validator = Draft7Validator(schema, registry=registry)
        
        # Cache the validator
        self._validator_cache[schema_name] = validator
        
        return validator
    
    def validate_configuration(
        self, 
        config: Dict[str, Any], 
        schema_name: str = "base_config",
        profile: Optional[str] = None
    ) -> None:
        """
        Validate a configuration against a JSON schema.
        
        Args:
            config: Configuration dictionary to validate
            schema_name: Name of the schema to validate against
            profile: Optional profile name for profile-specific validation
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            validator = self.get_validator(schema_name)
            
            # Perform basic JSON schema validation
            errors = list(validator.iter_errors(config))
            
            if errors:
                error_messages = []
                for error in errors:
                    path = " -> ".join(str(p) for p in error.absolute_path)
                    if path:
                        error_messages.append(f"At '{path}': {error.message}")
                    else:
                        error_messages.append(f"Root level: {error.message}")
                
                raise ValidationError(
                    f"Configuration validation failed:\n" + 
                    "\n".join(f"  - {msg}" for msg in error_messages)
                )
            
            # Perform profile-specific validation if profile is provided
            if profile:
                self._validate_profile_constraints(config, profile)
            
            # Perform custom validation rules
            self._validate_custom_rules(config)
            
            logger.debug(f"Configuration validation passed for schema: {schema_name}")
            
        except JsonSchemaValidationError as e:
            raise ValidationError(f"JSON schema validation error: {e.message}")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Validation error: {e}")
    
    def _validate_profile_constraints(self, config: Dict[str, Any], profile: str) -> None:
        """
        Validate profile-specific constraints.
        
        Args:
            config: Configuration to validate
            profile: Profile name
            
        Raises:
            ValidationError: If profile constraints are violated
        """
        # Profile-specific validation rules
        profile_rules = {
            "quick_start_minimal": {
                "max_sample_documents": 50,
                "required_mcp_tools": ["rag_basic", "rag_hyde", "rag_health_check"]
            },
            "quick_start_standard": {
                "max_sample_documents": 500,
                "required_mcp_tools": ["rag_basic", "rag_hyde", "rag_crag", "rag_hybrid_ifind"]
            },
            "quick_start_extended": {
                "max_sample_documents": 5000,
                "required_mcp_tools": ["rag_basic", "rag_hyde", "rag_crag", "rag_hybrid_ifind", "rag_graphrag"]
            }
        }
        
        if profile not in profile_rules:
            return  # No specific rules for this profile
        
        rules = profile_rules[profile]
        
        # Check sample document count
        sample_docs = config.get("sample_data", {}).get("document_count", 0)
        max_docs = rules.get("max_sample_documents", float('inf'))
        
        if sample_docs > max_docs:
            raise ValidationError(
                f"Profile '{profile}' allows maximum {max_docs} sample documents, "
                f"but configuration specifies {sample_docs}"
            )
        
        # Check required MCP tools
        mcp_tools = config.get("mcp_server", {}).get("enabled_tools", [])
        required_tools = rules.get("required_mcp_tools", [])
        
        missing_tools = set(required_tools) - set(mcp_tools)
        if missing_tools:
            raise ValidationError(
                f"Profile '{profile}' requires MCP tools: {required_tools}, "
                f"but missing: {list(missing_tools)}"
            )
    
    def _validate_custom_rules(self, config: Dict[str, Any]) -> None:
        """
        Validate custom business rules.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValidationError: If custom rules are violated
        """
        # Custom rule: Vector dimensions must be consistent
        vector_config = config.get("vector_index", {})
        embedding_config = config.get("embeddings", {})
        
        vector_dim = vector_config.get("dimension")
        embedding_dim = embedding_config.get("dimension")
        
        if vector_dim and embedding_dim and vector_dim != embedding_dim:
            raise ValidationError(
                f"Vector index dimension ({vector_dim}) must match "
                f"embedding dimension ({embedding_dim})"
            )
        
        # Custom rule: Database connection parameters
        db_config = config.get("database", {}).get("iris", {})
        if db_config:
            host = db_config.get("host")
            port = db_config.get("port")
            
            if host and not isinstance(host, str):
                raise ValidationError("Database host must be a string")
            
            if port and (not isinstance(port, int) or port <= 0 or port > 65535):
                raise ValidationError("Database port must be a valid integer between 1 and 65535")
        
        # Custom rule: MCP server configuration consistency
        mcp_config = config.get("mcp_server", {})
        if mcp_config.get("enabled", False):
            enabled_tools = mcp_config.get("enabled_tools", [])
            if not enabled_tools:
                raise ValidationError("MCP server is enabled but no tools are specified")
    
    def validate_schema_version(self, config: Dict[str, Any], expected_version: str = "1.0") -> None:
        """
        Validate schema version compatibility.
        
        Args:
            config: Configuration to validate
            expected_version: Expected schema version
            
        Raises:
            ValidationError: If version is incompatible
        """
        config_version = config.get("schema_version", "1.0")
        
        if config_version != expected_version:
            raise ValidationError(
                f"Schema version mismatch: expected {expected_version}, "
                f"got {config_version}"
            )
    
    def get_validation_errors(
        self, 
        config: Dict[str, Any], 
        schema_name: str = "base_config"
    ) -> List[str]:
        """
        Get detailed validation errors without raising exceptions.
        
        Args:
            config: Configuration to validate
            schema_name: Name of the schema to validate against
            
        Returns:
            List of validation error messages
        """
        try:
            self.validate_configuration(config, schema_name)
            return []
        except ValidationError as e:
            return [str(e)]
        except Exception as e:
            return [f"Unexpected validation error: {e}"]