"""
Integration Factory for Quick Start Configuration Templates System.

This module provides a factory pattern for automatically selecting and creating
the appropriate integration adapter based on the target configuration manager type.
It simplifies the integration process by providing a single entry point for all
Quick Start template integrations.

Classes:
    IntegrationFactory: Factory for creating integration adapters
    IntegrationRequest: Data class for integration requests
    IntegrationResult: Data class for integration results

Usage:
    from quick_start.config.integration_factory import IntegrationFactory
    
    # Simple integration
    factory = IntegrationFactory()
    result = factory.integrate_template("basic_rag", "iris_rag")
    
    # Advanced integration with options
    result = factory.integrate_template(
        template_name="advanced_rag",
        target_manager="rag_templates",
        options={"validate_schema": True, "include_profiles": True}
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from .integration_adapters import (
    IrisRagConfigManagerAdapter,
    RagTemplatesConfigManagerAdapter,
    TemplateInheritanceAdapter,
    EnvironmentVariableIntegrationAdapter,
    SchemaValidationIntegrationAdapter,
    PipelineCompatibilityAdapter,
    ProfileSystemIntegrationAdapter,
    CrossLanguageCompatibilityAdapter,
    ConfigurationRoundTripAdapter,
    ErrorHandlingIntegrationAdapter
)

logger = logging.getLogger(__name__)


@dataclass
class IntegrationRequest:
    """Data class representing an integration request."""
    template_name: str
    target_manager: str
    options: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    profiles: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the integration request."""
        if not self.template_name:
            raise ValueError("Template name is required")
        if not self.target_manager:
            raise ValueError("Target manager is required")
        
        # Validate target manager type
        valid_managers = ["iris_rag", "rag_templates"]
        if self.target_manager not in valid_managers:
            raise ValueError(f"Target manager must be one of: {valid_managers}")


@dataclass
class IntegrationResult:
    """Data class representing an integration result."""
    success: bool
    template_name: str
    target_manager: str
    converted_config: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the result."""
        self.metadata[key] = value


class IntegrationFactory:
    """
    Factory for creating and managing Quick Start configuration integrations.
    
    This factory provides a unified interface for integrating Quick Start templates
    with existing configuration managers. It automatically selects the appropriate
    adapter based on the target manager type and handles the complete integration
    workflow including validation, conversion, and error handling.
    """
    
    def __init__(self):
        """Initialize the integration factory."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._adapters = self._initialize_adapters()
        self._manager_adapters = self._initialize_manager_adapters()
    
    def _initialize_adapters(self) -> Dict[str, Any]:
        """Initialize all available integration adapters."""
        return {
            "template_inheritance": TemplateInheritanceAdapter(),
            "environment_variables": EnvironmentVariableIntegrationAdapter(),
            "schema_validation": SchemaValidationIntegrationAdapter(),
            "pipeline_compatibility": PipelineCompatibilityAdapter(),
            "profile_system": ProfileSystemIntegrationAdapter(),
            "cross_language": CrossLanguageCompatibilityAdapter(),
            "round_trip": ConfigurationRoundTripAdapter(),
            "error_handling": ErrorHandlingIntegrationAdapter()
        }
    
    def _initialize_manager_adapters(self) -> Dict[str, Any]:
        """Initialize configuration manager specific adapters."""
        return {
            "iris_rag": IrisRagConfigManagerAdapter(),
            "rag_templates": RagTemplatesConfigManagerAdapter()
        }
    
    def integrate_template(
        self,
        template_name: str,
        target_manager: str,
        options: Optional[Dict[str, Any]] = None,
        environment_variables: Optional[Dict[str, Any]] = None,
        validation_rules: Optional[Dict[str, Any]] = None,
        profiles: Optional[List[str]] = None
    ) -> IntegrationResult:
        """
        Integrate a Quick Start template with a target configuration manager.
        
        Args:
            template_name: Name of the Quick Start template to integrate
            target_manager: Target configuration manager ("iris_rag" or "rag_templates")
            options: Optional integration options
            environment_variables: Optional environment variable overrides
            validation_rules: Optional custom validation rules
            profiles: Optional list of profiles to integrate
        
        Returns:
            IntegrationResult: Result of the integration process
        """
        # Create integration request
        request = IntegrationRequest(
            template_name=template_name,
            target_manager=target_manager,
            options=options or {},
            environment_variables=environment_variables or {},
            validation_rules=validation_rules or {},
            profiles=profiles or []
        )
        
        self.logger.info(f"Starting integration of template '{template_name}' with '{target_manager}' manager")
        
        # Create result object
        result = IntegrationResult(
            success=True,
            template_name=template_name,
            target_manager=target_manager
        )
        
        try:
            # Step 1: Load and validate template
            template_config = self._load_template(request, result)
            if not result.success:
                return result
            
            # Step 2: Apply template inheritance if needed
            if request.options.get("flatten_inheritance", True):
                template_config = self._apply_inheritance(template_config, request, result)
            
            # Step 3: Apply environment variables
            if request.environment_variables:
                template_config = self._apply_environment_variables(
                    template_config, request, result
                )
            
            # Step 4: Convert to target manager format
            converted_config = self._convert_to_target_format(
                template_config, request, result
            )
            if not result.success:
                return result
            
            result.converted_config = converted_config
            
            # Step 5: Validate converted configuration
            if request.options.get("validate_schema", True):
                self._validate_configuration(converted_config, request, result)
            
            # Step 6: Ensure pipeline compatibility
            if request.options.get("ensure_compatibility", True):
                self._ensure_pipeline_compatibility(converted_config, request, result)
            
            # Step 7: Integrate profiles if specified
            if request.profiles:
                self._integrate_profiles(converted_config, request, result)
            
            # Step 8: Ensure cross-language compatibility if needed
            if request.options.get("cross_language", False):
                self._ensure_cross_language_compatibility(converted_config, request, result)
            
            # Step 9: Test round-trip conversion if requested
            if request.options.get("test_round_trip", False):
                self._test_round_trip_conversion(converted_config, request, result)
            
            # Add final metadata
            result.add_metadata("integration_steps_completed", 9)
            result.add_metadata("adapter_used", target_manager)
            result.add_metadata("template_loaded", template_name)
            
            self.logger.info(f"Successfully integrated template '{template_name}' with '{target_manager}' manager")
            
        except Exception as e:
            self.logger.error(f"Integration failed: {str(e)}")
            result.add_error(f"Integration failed: {str(e)}")
            
            # Use error handling adapter for structured error reporting
            error_result = self._adapters["error_handling"].handle_integration_error(
                target_manager, str(e), {"template_name": template_name}
            )
            result.add_metadata("error_details", error_result)
        
        return result
    
    def _load_template(self, request: IntegrationRequest, result: IntegrationResult) -> Dict[str, Any]:
        """Load the Quick Start template configuration."""
        try:
            # For now, return a mock template - in real implementation,
            # this would load from the template system
            template_config = {
                "database": {
                    "iris": {
                        "host": "localhost",
                        "port": 1972,
                        "namespace": "USER",
                        "username": "demo",
                        "password": "demo"
                    }
                },
                "embeddings": {
                    "provider": "openai",
                    "model": "text-embedding-ada-002",
                    "dimensions": 1536
                },
                "llm": {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7
                }
            }
            
            result.add_metadata("template_loaded_from", "mock_system")
            return template_config
            
        except Exception as e:
            result.add_error(f"Failed to load template '{request.template_name}': {str(e)}")
            return {}
    
    def _apply_inheritance(
        self, 
        config: Dict[str, Any], 
        request: IntegrationRequest, 
        result: IntegrationResult
    ) -> Dict[str, Any]:
        """Apply template inheritance flattening."""
        try:
            adapter_result = self._adapters["template_inheritance"].flatten_inheritance_chain(
                config, request.target_manager
            )
            result.add_metadata("inheritance_applied", True)
            return adapter_result.get("flattened_config", config)
        except Exception as e:
            result.add_warning(f"Failed to apply inheritance: {str(e)}")
            return config
    
    def _apply_environment_variables(
        self, 
        config: Dict[str, Any], 
        request: IntegrationRequest, 
        result: IntegrationResult
    ) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        try:
            adapter_result = self._adapters["environment_variables"].integrate_environment_variables(
                config, request.target_manager, request.environment_variables
            )
            result.add_metadata("environment_variables_applied", True)
            return adapter_result.get("integrated_config", config)
        except Exception as e:
            result.add_warning(f"Failed to apply environment variables: {str(e)}")
            return config
    
    def _convert_to_target_format(
        self, 
        config: Dict[str, Any], 
        request: IntegrationRequest, 
        result: IntegrationResult
    ) -> Dict[str, Any]:
        """Convert configuration to target manager format."""
        try:
            adapter = self._manager_adapters[request.target_manager]
            converted_config = adapter.convert_from_quick_start(config)
            result.add_metadata("conversion_successful", True)
            return converted_config
        except Exception as e:
            result.add_error(f"Failed to convert to {request.target_manager} format: {str(e)}")
            return {}
    
    def _validate_configuration(
        self, 
        config: Dict[str, Any], 
        request: IntegrationRequest, 
        result: IntegrationResult
    ):
        """Validate the converted configuration."""
        try:
            validation_result = self._adapters["schema_validation"].validate_configuration(
                config, request.target_manager, request.validation_rules
            )
            result.validation_results = validation_result
            
            if validation_result.get("errors"):
                for error in validation_result["errors"]:
                    result.add_error(f"Validation error: {error}")
            
            result.add_metadata("validation_completed", True)
        except Exception as e:
            result.add_warning(f"Validation failed: {str(e)}")
    
    def _ensure_pipeline_compatibility(
        self, 
        config: Dict[str, Any], 
        request: IntegrationRequest, 
        result: IntegrationResult
    ):
        """Ensure pipeline compatibility."""
        try:
            compatibility_result = self._adapters["pipeline_compatibility"].ensure_compatibility(
                config, request.target_manager
            )
            result.add_metadata("pipeline_compatibility", compatibility_result)
        except Exception as e:
            result.add_warning(f"Pipeline compatibility check failed: {str(e)}")
    
    def _integrate_profiles(
        self, 
        config: Dict[str, Any], 
        request: IntegrationRequest, 
        result: IntegrationResult
    ):
        """Integrate specified profiles."""
        try:
            for profile in request.profiles:
                profile_result = self._adapters["profile_system"].integrate_profile(
                    profile, config, request.target_manager
                )
                result.add_metadata(f"profile_{profile}_integrated", profile_result)
        except Exception as e:
            result.add_warning(f"Profile integration failed: {str(e)}")
    
    def _ensure_cross_language_compatibility(
        self, 
        config: Dict[str, Any], 
        request: IntegrationRequest, 
        result: IntegrationResult
    ):
        """Ensure cross-language compatibility."""
        try:
            languages = request.options.get("target_languages", ["python"])
            for language in languages:
                compatibility_result = self._adapters["cross_language"].ensure_compatibility(
                    config, language
                )
                result.add_metadata(f"cross_language_{language}", compatibility_result)
        except Exception as e:
            result.add_warning(f"Cross-language compatibility failed: {str(e)}")
    
    def _test_round_trip_conversion(
        self, 
        config: Dict[str, Any], 
        request: IntegrationRequest, 
        result: IntegrationResult
    ):
        """Test round-trip conversion."""
        try:
            round_trip_result = self._adapters["round_trip"].test_round_trip_conversion(
                config, request.target_manager
            )
            result.add_metadata("round_trip_test", round_trip_result)
        except Exception as e:
            result.add_warning(f"Round-trip test failed: {str(e)}")
    
    def list_available_adapters(self) -> Dict[str, List[str]]:
        """List all available integration adapters."""
        return {
            "manager_adapters": list(self._manager_adapters.keys()),
            "integration_adapters": list(self._adapters.keys())
        }
    
    def get_adapter_info(self, adapter_name: str) -> Dict[str, Any]:
        """Get information about a specific adapter."""
        if adapter_name in self._manager_adapters:
            adapter = self._manager_adapters[adapter_name]
            return {
                "type": "manager_adapter",
                "name": adapter_name,
                "class": adapter.__class__.__name__,
                "description": adapter.__class__.__doc__ or "No description available"
            }
        elif adapter_name in self._adapters:
            adapter = self._adapters[adapter_name]
            return {
                "type": "integration_adapter",
                "name": adapter_name,
                "class": adapter.__class__.__name__,
                "description": adapter.__class__.__doc__ or "No description available"
            }
        else:
            raise ValueError(f"Adapter '{adapter_name}' not found")
    
    def validate_integration_request(self, request: IntegrationRequest) -> List[str]:
        """Validate an integration request and return any issues."""
        issues = []
        
        # Check if target manager adapter exists
        if request.target_manager not in self._manager_adapters:
            issues.append(f"No adapter available for manager '{request.target_manager}'")
        
        # Check if template name is valid (basic validation)
        if not request.template_name.strip():
            issues.append("Template name cannot be empty")
        
        # Validate options
        valid_options = {
            "flatten_inheritance", "validate_schema", "ensure_compatibility",
            "cross_language", "test_round_trip", "target_languages"
        }
        invalid_options = set(request.options.keys()) - valid_options
        if invalid_options:
            issues.append(f"Invalid options: {list(invalid_options)}")
        
        return issues


# Convenience functions for common integration patterns
def integrate_basic_template(template_name: str, target_manager: str) -> IntegrationResult:
    """
    Integrate a basic Quick Start template with minimal options.
    
    Args:
        template_name: Name of the template to integrate
        target_manager: Target configuration manager
    
    Returns:
        IntegrationResult: Result of the integration
    """
    factory = IntegrationFactory()
    return factory.integrate_template(template_name, target_manager)


def integrate_with_validation(
    template_name: str, 
    target_manager: str, 
    validation_rules: Optional[Dict[str, Any]] = None
) -> IntegrationResult:
    """
    Integrate a template with comprehensive validation.
    
    Args:
        template_name: Name of the template to integrate
        target_manager: Target configuration manager
        validation_rules: Optional custom validation rules
    
    Returns:
        IntegrationResult: Result of the integration
    """
    factory = IntegrationFactory()
    return factory.integrate_template(
        template_name=template_name,
        target_manager=target_manager,
        options={"validate_schema": True, "ensure_compatibility": True},
        validation_rules=validation_rules or {}
    )


def integrate_with_profiles(
    template_name: str, 
    target_manager: str, 
    profiles: List[str]
) -> IntegrationResult:
    """
    Integrate a template with specific profiles.
    
    Args:
        template_name: Name of the template to integrate
        target_manager: Target configuration manager
        profiles: List of profiles to integrate
    
    Returns:
        IntegrationResult: Result of the integration
    """
    factory = IntegrationFactory()
    return factory.integrate_template(
        template_name=template_name,
        target_manager=target_manager,
        profiles=profiles,
        options={"validate_schema": True}
    )