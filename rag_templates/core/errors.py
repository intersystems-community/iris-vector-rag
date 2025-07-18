"""
Error handling system for RAG Templates Library Consumption Framework.

This module provides a hierarchy of custom exceptions with helpful error messages
and suggestions for common failures, along with fallback strategies.
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class RAGFrameworkError(Exception):
    """
    Base exception for all RAG Framework errors.
    
    This is the root exception class that all other framework-specific
    exceptions inherit from. It provides common functionality for error
    handling and logging.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, 
                 suggestion: Optional[str] = None):
        """
        Initialize the RAG Framework error.
        
        Args:
            message: The error message
            details: Optional dictionary with additional error details
            suggestion: Optional suggestion for resolving the error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        
        # Log the error
        logger.error(f"RAGFrameworkError: {message}")
        if details:
            logger.error(f"Error details: {details}")
        if suggestion:
            logger.info(f"Suggestion: {suggestion}")
    
    def __str__(self) -> str:
        """Return a formatted error message with suggestion if available."""
        base_message = self.message
        if self.suggestion:
            base_message += f"\n\nSuggestion: {self.suggestion}"
        return base_message


class ConfigurationError(RAGFrameworkError):
    """
    Exception raised for configuration-related errors.
    
    This exception is raised when there are issues with configuration
    loading, validation, or missing required configuration values.
    """
    
    def __init__(self, message: str, config_key: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the configuration error.
        
        Args:
            message: The error message
            config_key: The configuration key that caused the error
            details: Optional dictionary with additional error details
        """
        # Generate helpful suggestions based on the error type
        suggestion = self._generate_suggestion(message, config_key)
        
        # Add config key to details if provided
        if config_key:
            details = details or {}
            details["config_key"] = config_key
        
        super().__init__(message, details, suggestion)
        self.config_key = config_key
    
    def _generate_suggestion(self, message: str, config_key: Optional[str]) -> str:
        """Generate helpful suggestions based on the error."""
        suggestions = []
        
        if config_key:
            suggestions.append(f"Check the configuration for key: {config_key}")
            
            # Specific suggestions based on config key patterns
            if "database" in config_key.lower():
                suggestions.append("Ensure database connection parameters are set correctly")
                suggestions.append("Check environment variables: RAG_DATABASE__IRIS__HOST, RAG_DATABASE__IRIS__PORT, etc.")
            elif "embedding" in config_key.lower():
                suggestions.append("Verify embedding model configuration")
                suggestions.append("Check if the embedding model is available")
            elif "llm" in config_key.lower():
                suggestions.append("Verify LLM configuration and API keys")
        
        if "missing" in message.lower() or "not found" in message.lower():
            suggestions.append("Set the required configuration value in config.yaml or environment variables")
            suggestions.append("Use environment variables with RAG_ prefix (e.g., RAG_DATABASE__IRIS__HOST)")
        
        if "validation" in message.lower():
            suggestions.append("Check the configuration format and data types")
            suggestions.append("Refer to the configuration schema documentation")
        
        return " | ".join(suggestions) if suggestions else "Check the configuration documentation"


class ConnectionError(RAGFrameworkError):
    """
    Exception raised for database or service connection errors.
    
    This exception is raised when there are issues connecting to
    the IRIS database or other external services.
    """
    
    def __init__(self, message: str, service: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the connection error.
        
        Args:
            message: The error message
            service: The service that failed to connect
            details: Optional dictionary with additional error details
        """
        suggestion = self._generate_connection_suggestion(service)
        
        if service:
            details = details or {}
            details["service"] = service
        
        super().__init__(message, details, suggestion)
        self.service = service
    
    def _generate_connection_suggestion(self, service: Optional[str]) -> str:
        """Generate connection-specific suggestions."""
        suggestions = [
            "Check network connectivity",
            "Verify service is running and accessible"
        ]
        
        if service and "iris" in service.lower():
            suggestions.extend([
                "Ensure IRIS database is running",
                "Check database connection parameters (host, port, namespace)",
                "Verify database credentials"
            ])
        
        return " | ".join(suggestions)


class InitializationError(RAGFrameworkError):
    """
    Exception raised for initialization failures.
    
    This exception is raised when there are issues during the
    initialization of RAG components or pipelines.
    """
    
    def __init__(self, message: str, component: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the initialization error.
        
        Args:
            message: The error message
            component: The component that failed to initialize
            details: Optional dictionary with additional error details
        """
        suggestion = self._generate_initialization_suggestion(component)
        
        if component:
            details = details or {}
            details["component"] = component
        
        super().__init__(message, details, suggestion)
        self.component = component
    
    def _generate_initialization_suggestion(self, component: Optional[str]) -> str:
        """Generate initialization-specific suggestions."""
        suggestions = [
            "Check component dependencies",
            "Verify configuration is complete"
        ]
        
        if component:
            suggestions.append(f"Review {component} initialization requirements")
        
        return " | ".join(suggestions)


class ValidationError(RAGFrameworkError):
    """
    Exception raised for data validation errors.
    
    This exception is raised when input data or configuration
    fails validation checks.
    """
    
    def __init__(self, message: str, field: Optional[str] = None,
                 expected_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize the validation error.
        
        Args:
            message: The error message
            field: The field that failed validation
            expected_type: The expected data type
            details: Optional dictionary with additional error details
        """
        suggestion = self._generate_validation_suggestion(field, expected_type)
        
        if field or expected_type:
            details = details or {}
            if field:
                details["field"] = field
            if expected_type:
                details["expected_type"] = expected_type
        
        super().__init__(message, details, suggestion)
        self.field = field
        self.expected_type = expected_type
    
    def _generate_validation_suggestion(self, field: Optional[str], 
                                      expected_type: Optional[str]) -> str:
        """Generate validation-specific suggestions."""
        suggestions = ["Check input data format and types"]
        
        if field:
            suggestions.append(f"Verify the '{field}' field")
        
        if expected_type:
            suggestions.append(f"Expected type: {expected_type}")
        
        return " | ".join(suggestions)


# Convenience function for handling common error scenarios
def handle_configuration_fallback(config_manager, key: str, default_value: Any,
                                error_message: Optional[str] = None) -> Any:
    """
    Handle configuration fallback with proper error handling.
    
    Args:
        config_manager: The configuration manager instance
        key: The configuration key to retrieve
        default_value: The fallback value to use
        error_message: Optional custom error message
        
    Returns:
        The configuration value or fallback value
        
    Raises:
        ConfigurationError: If the fallback strategy fails
    """
    try:
        value = config_manager.get(key)
        if value is None:
            logger.warning(f"Configuration key '{key}' not found, using fallback: {default_value}")
            return default_value
        return value
    except Exception as e:
        error_msg = error_message or f"Failed to retrieve configuration for key: {key}"
        raise ConfigurationError(
            error_msg,
            config_key=key,
            details={"original_error": str(e), "fallback_value": default_value}
        ) from e