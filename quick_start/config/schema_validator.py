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
        profile: Optional[str] = None,
        is_template: bool = False
    ) -> bool:
        """
        Validate a configuration against a JSON schema.
        
        Args:
            config: Configuration dictionary to validate
            schema_name: Name of the schema to validate against
            profile: Optional profile name for profile-specific validation
            is_template: If True, validates as a template (allows partial configs)
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # For templates, use lenient validation
            if is_template or "extends" in config:
                return self._validate_template(config, schema_name, profile)
            
            # For complete configurations, use full validation
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
            
            # Only perform custom validation for complete configurations (base_config schema)
            if schema_name == "base_config":
                self._validate_custom_rules(config)
                # Validate schema version compatibility
                self._validate_schema_version_compatibility(config)
            
            # Perform profile-specific validation only for profile schemas
            if profile and schema_name.startswith("quick_start_"):
                self._validate_profile_constraints(config, profile)
            
            logger.debug(f"Configuration validation passed for schema: {schema_name}")
            return True
            
        except JsonSchemaValidationError as e:
            raise ValidationError(f"JSON schema validation error: {e.message}")
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Validation error: {e}")
    
    def _validate_template(
        self,
        config: Dict[str, Any],
        schema_name: str,
        profile: Optional[str] = None
    ) -> bool:
        """
        Validate a template configuration with lenient rules.
        
        Templates are partial configurations that will be merged later,
        so we only validate the structure of provided fields.
        
        Args:
            config: Template configuration dictionary
            schema_name: Name of the schema to validate against
            profile: Optional profile name
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # For templates, we validate only the provided fields
            # Remove 'extends' field for validation as it's not part of the schema
            config_copy = config.copy()
            config_copy.pop('extends', None)
            
            # Validate specific fields that are present
            if 'sample_data' in config_copy:
                self._validate_sample_data_fields(config_copy['sample_data'])
            
            if 'mcp_server' in config_copy:
                self._validate_mcp_server_fields(config_copy['mcp_server'])
            
            if 'metadata' in config_copy:
                self._validate_metadata_fields(config_copy['metadata'])
            
            # Apply profile-specific constraints if this is a profile template
            # Extract profile from metadata if not provided explicitly
            effective_profile = profile
            if not effective_profile and 'metadata' in config_copy and 'profile' in config_copy['metadata']:
                effective_profile = config_copy['metadata']['profile']
            
            if effective_profile and effective_profile.startswith("quick_start_"):
                self._validate_profile_template_constraints(config_copy, effective_profile)
            
            logger.debug(f"Template validation passed for schema: {schema_name}")
            return True
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Template validation error: {e}")
    
    def _validate_sample_data_fields(self, sample_data: Dict[str, Any]) -> None:
        """Validate sample_data fields in templates."""
        if 'source' in sample_data:
            # Allow both schema values and legacy test values
            valid_sources = ['pmc', 'synthetic', 'custom', 'pmc_sample']
            if sample_data['source'] not in valid_sources:
                raise ValidationError(f"Invalid sample_data.source: {sample_data['source']}. Must be one of {valid_sources}")
        
        if 'document_count' in sample_data:
            count = sample_data['document_count']
            if not isinstance(count, int) or count < 1 or count > 10000:
                raise ValidationError(f"Invalid sample_data.document_count: {count}. Must be integer between 1 and 10000")
    
    def _validate_mcp_server_fields(self, mcp_server: Dict[str, Any]) -> None:
        """Validate mcp_server fields in templates."""
        if 'tools' in mcp_server:
            valid_tools = ['basic', 'crag', 'hyde', 'graphrag', 'hybrid_ifind', 'colbert', 'noderag', 'sqlrag', 'health_check', 'list_techniques', 'performance_metrics']
            tools = mcp_server['tools']
            if not isinstance(tools, list):
                raise ValidationError("mcp_server.tools must be a list")
            for tool in tools:
                if tool not in valid_tools:
                    raise ValidationError(f"Invalid tool: {tool}. Must be one of {valid_tools}")
        
        if 'port' in mcp_server:
            port = mcp_server['port']
            if not isinstance(port, int) or port < 1024 or port > 65535:
                raise ValidationError(f"Invalid mcp_server.port: {port}. Must be integer between 1024 and 65535")
    
    def _validate_metadata_fields(self, metadata: Dict[str, Any]) -> None:
        """Validate metadata fields in templates."""
        # Metadata validation is lenient for templates
        # We only check format if version/schema_version are provided
        if 'version' in metadata:
            version = metadata['version']
            if not isinstance(version, str):
                raise ValidationError(f"metadata.version must be a string, got {type(version).__name__}")
        
        if 'schema_version' in metadata:
            schema_version = metadata['schema_version']
            if not isinstance(schema_version, str):
                raise ValidationError(f"metadata.schema_version must be a string, got {type(schema_version).__name__}")
    
    def _validate_profile_template_constraints(self, config: Dict[str, Any], profile: str) -> None:
        """Validate profile-specific constraints for templates."""
        # Extract profile from metadata if not provided directly
        actual_profile = profile
        if not actual_profile and 'metadata' in config and 'profile' in config['metadata']:
            actual_profile = config['metadata']['profile']
        
        if actual_profile == "quick_start_minimal":
            # Check document count constraint for minimal profile
            if 'sample_data' in config and 'document_count' in config['sample_data']:
                count = config['sample_data']['document_count']
                if count > 50:
                    raise ValidationError(f"Minimal profile document_count must be <= 50, got {count}")
            
            # Check tool constraints for minimal profile
            if 'mcp_server' in config and 'tools' in config['mcp_server']:
                tools = config['mcp_server']['tools']
                allowed_tools = ['basic', 'health_check', 'list_techniques']
                for tool in tools:
                    if tool not in allowed_tools:
                        raise ValidationError(f"Minimal profile only allows tools: {allowed_tools}, got {tool}")
        
        elif actual_profile == "quick_start_standard":
            # Check document count constraint for standard profile
            if 'sample_data' in config and 'document_count' in config['sample_data']:
                count = config['sample_data']['document_count']
                if count > 500:
                    raise ValidationError(f"Standard profile document_count must be <= 500, got {count}")
        
        elif actual_profile == "quick_start_extended":
            # Check document count constraint for extended profile
            if 'sample_data' in config and 'document_count' in config['sample_data']:
                count = config['sample_data']['document_count']
                if count > 2000:
                    raise ValidationError(f"Extended profile document_count must be <= 2000, got {count}")
    
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
        mcp_tools = config.get("mcp_server", {}).get("tools", [])
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
            tools = mcp_config.get("tools", [])
            if not tools:
                raise ValidationError("MCP server is enabled but no tools are specified")
    
    def _validate_schema_version_compatibility(self, config: Dict[str, Any]) -> None:
        """
        Validate schema version compatibility.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValidationError: If version is incompatible
        """
        # Current supported schema version
        supported_version = "2024.1"
        
        # Get schema version from metadata section
        metadata = config.get("metadata", {})
        config_version = metadata.get("schema_version")
        
        if config_version is None:
            # If no schema_version is provided, this will be caught by JSON schema validation
            return
        
        if config_version != supported_version:
            raise ValidationError(
                f"Unsupported schema_version: {config_version}. "
                f"Supported version: {supported_version}"
            )

    def validate_schema_version(self, config: Dict[str, Any], expected_version: str = "2024.1") -> None:
        """
        Validate schema version compatibility (public method for backward compatibility).
        
        Args:
            config: Configuration to validate
            expected_version: Expected schema version
            
        Raises:
            ValidationError: If version is incompatible
        """
        # Get schema version from metadata section or root level for backward compatibility
        metadata = config.get("metadata", {})
        config_version = metadata.get("schema_version") or config.get("schema_version", "2024.1")
        
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