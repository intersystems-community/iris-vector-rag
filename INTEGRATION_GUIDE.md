# Quick Start Configuration Templates Integration Guide

This guide provides comprehensive documentation for integrating Quick Start configuration templates with existing ConfigurationManager implementations in the RAG templates project.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Integration Components](#integration-components)
4. [Usage Examples](#usage-examples)
5. [Configuration Manager Integration](#configuration-manager-integration)
6. [Advanced Integration Patterns](#advanced-integration-patterns)
7. [Error Handling](#error-handling)
8. [Testing](#testing)
9. [Migration Guide](#migration-guide)
10. [Troubleshooting](#troubleshooting)

## Overview

The Quick Start Configuration Templates Integration System provides a seamless bridge between the Quick Start template system and existing configuration managers (`iris_rag` and `rag_templates`). This integration enables:

- **Zero-configuration startup** using Quick Start templates
- **Automatic format conversion** between template and manager formats
- **Environment variable integration** with proper precedence
- **Schema validation** and compatibility checking
- **Cross-language compatibility** for JavaScript and Python
- **Round-trip conversion** testing and validation

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Integration System                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │ Integration     │    │     Integration Adapters        │ │
│  │ Factory         │◄──►│                                 │ │
│  │                 │    │ • IrisRagConfigManagerAdapter   │ │
│  │ • Auto-routing  │    │ • RagTemplatesConfigManager     │ │
│  │ • Validation    │    │ • TemplateInheritanceAdapter    │ │
│  │ • Error handling│    │ • EnvironmentVariableAdapter    │ │
│  └─────────────────┘    │ • SchemaValidationAdapter       │ │
│                         │ • PipelineCompatibilityAdapter  │ │
│                         │ • ProfileSystemAdapter          │ │
│                         │ • CrossLanguageAdapter          │ │
│                         │ • ConfigurationRoundTripAdapter │ │
│                         │ • ErrorHandlingAdapter          │ │
│                         └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Configuration Managers                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐              ┌─────────────────────┐   │
│  │ iris_rag        │              │ rag_templates       │   │
│  │ Configuration   │              │ Configuration       │   │
│  │ Manager         │              │ Manager             │   │
│  │                 │              │                     │   │
│  │ + load_quick_   │              │ + load_quick_       │   │
│  │   start_template│              │   start_template    │   │
│  │ + list_quick_   │              │ + list_quick_       │   │
│  │   start_templates│             │   start_templates   │   │
│  │ + validate_     │              │ + validate_         │   │
│  │   integration   │              │   integration       │   │
│  └─────────────────┘              └─────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Integration Flow

1. **Template Loading**: Quick Start templates are loaded from the template system
2. **Format Conversion**: Templates are converted to target manager format
3. **Validation**: Converted configurations are validated against schemas
4. **Environment Integration**: Environment variables are applied with proper precedence
5. **Manager Integration**: Configurations are merged into existing managers
6. **Compatibility Checking**: Cross-language and pipeline compatibility is verified

## Integration Components

### Integration Factory

The `IntegrationFactory` is the main entry point for all integration operations:

```python
from quick_start.config.integration_factory import IntegrationFactory

# Create factory instance
factory = IntegrationFactory()

# Basic integration
result = factory.integrate_template("basic_rag", "iris_rag")

# Advanced integration with options
result = factory.integrate_template(
    template_name="advanced_rag",
    target_manager="rag_templates",
    options={
        "validate_schema": True,
        "ensure_compatibility": True,
        "cross_language": True
    },
    environment_variables={"RAG_DATABASE__HOST": "production-db"},
    validation_rules={"strict_mode": True}
)
```

### Integration Adapters

#### Core Manager Adapters

- **`IrisRagConfigManagerAdapter`**: Converts templates to iris_rag format
- **`RagTemplatesConfigManagerAdapter`**: Converts templates to rag_templates format

#### Specialized Integration Adapters

- **`TemplateInheritanceAdapter`**: Flattens template inheritance chains
- **`EnvironmentVariableIntegrationAdapter`**: Handles environment variable integration
- **`SchemaValidationIntegrationAdapter`**: Validates configurations against schemas
- **`PipelineCompatibilityAdapter`**: Ensures pipeline compatibility
- **`ProfileSystemIntegrationAdapter`**: Integrates configuration profiles
- **`CrossLanguageCompatibilityAdapter`**: Ensures cross-language compatibility
- **`ConfigurationRoundTripAdapter`**: Tests round-trip conversion
- **`ErrorHandlingIntegrationAdapter`**: Provides structured error handling

## Usage Examples

### Basic Integration

#### Using Integration Factory Directly

```python
from quick_start.config.integration_factory import (
    IntegrationFactory, 
    integrate_basic_template
)

# Simple integration using convenience function
result = integrate_basic_template("basic_rag", "iris_rag")

if result.success:
    print("Integration successful!")
    print(f"Converted config: {result.converted_config}")
else:
    print(f"Integration failed: {result.errors}")
```

#### Using Configuration Managers

```python
# iris_rag ConfigurationManager
from iris_rag.config.manager import ConfigurationManager

manager = ConfigurationManager()
config = manager.load_quick_start_template("basic_rag")
print(f"Database host: {manager.get('database:iris:host')}")

# rag_templates ConfigurationManager
from rag_templates.core.config_manager import ConfigurationManager

manager = ConfigurationManager()
config = manager.load_quick_start_template("advanced_rag")
print(f"Embedding model: {manager.get('embeddings:model')}")
```

### Advanced Integration with Validation

```python
from quick_start.config.integration_factory import integrate_with_validation

# Integration with comprehensive validation
result = integrate_with_validation(
    template_name="production_rag",
    target_manager="iris_rag",
    validation_rules={
        "require_database_credentials": True,
        "validate_embedding_dimensions": True,
        "check_llm_configuration": True
    }
)

if result.success:
    print("Validation passed!")
    print(f"Validation results: {result.validation_results}")
else:
    print(f"Validation failed: {result.errors}")
```

### Profile-Based Integration

```python
from quick_start.config.integration_factory import integrate_with_profiles

# Integration with specific profiles
result = integrate_with_profiles(
    template_name="enterprise_rag",
    target_manager="rag_templates",
    profiles=["production", "high_performance", "security_enhanced"]
)

if result.success:
    print("Profile integration successful!")
    for profile in result.metadata.get("integrated_profiles", []):
        print(f"Applied profile: {profile}")
```

### Environment Variable Integration

```python
from quick_start.config.integration_factory import IntegrationFactory

factory = IntegrationFactory()

# Integration with environment variable overrides
result = factory.integrate_template(
    template_name="cloud_rag",
    target_manager="iris_rag",
    environment_variables={
        "RAG_DATABASE__IRIS__HOST": "cloud-db.example.com",
        "RAG_DATABASE__IRIS__PORT": "1972",
        "RAG_EMBEDDINGS__MODEL": "text-embedding-ada-002",
        "RAG_LLM__PROVIDER": "openai",
        "RAG_LLM__API_KEY": "${OPENAI_API_KEY}"  # Reference to actual env var
    },
    options={"validate_schema": True}
)
```

## Configuration Manager Integration

### iris_rag ConfigurationManager

The `iris_rag.config.manager.ConfigurationManager` has been enhanced with Quick Start template support:

#### New Methods

```python
# Load Quick Start template
config = manager.load_quick_start_template(
    template_name="basic_rag",
    options={"validate_schema": True},
    environment_variables={"RAG_DATABASE__HOST": "localhost"},
    validation_rules={"strict_mode": False}
)

# List available templates
templates = manager.list_quick_start_templates()
print(f"Available adapters: {templates['available_adapters']}")

# Validate integration without applying
validation = manager.validate_quick_start_integration("advanced_rag")
if validation["valid"]:
    print("Integration would succeed")
else:
    print(f"Integration issues: {validation['issues']}")
```

#### Integration Example

```python
from iris_rag.config.manager import ConfigurationManager

# Initialize with Quick Start template
manager = ConfigurationManager()

# Load template with environment overrides
config = manager.load_quick_start_template(
    template_name="colbert_rag",
    options={
        "validate_schema": True,
        "ensure_compatibility": True
    },
    environment_variables={
        "RAG_DATABASE__IRIS__USERNAME": "production_user",
        "RAG_DATABASE__IRIS__PASSWORD": "secure_password"
    }
)

# Use existing manager methods
db_config = manager.get_database_config()
embedding_config = manager.get_embedding_config()
vector_config = manager.get_vector_index_config()
```

### rag_templates ConfigurationManager

The `rag_templates.core.config_manager.ConfigurationManager` has been enhanced with similar functionality:

#### Enhanced load_config Method

```python
from rag_templates.core.config_manager import ConfigurationManager

# Initialize manager
manager = ConfigurationManager()

# Load configuration from multiple sources with precedence
config = manager.load_config(
    quick_start_template="enterprise_rag",  # Lowest precedence
    config_file="production.yaml",          # Medium precedence
    config_dict={                           # Highest precedence
        "database": {
            "iris": {
                "host": "override-host"
            }
        }
    }
)

# Environment variables have final precedence
```

#### Integration Methods

```python
# Load Quick Start template
config = manager.load_quick_start_template("basic_rag")

# List available templates
templates = manager.list_quick_start_templates()

# Validate integration
validation = manager.validate_quick_start_integration("advanced_rag")
```

## Advanced Integration Patterns

### Custom Integration Workflow

```python
from quick_start.config.integration_factory import IntegrationFactory

class CustomIntegrationWorkflow:
    def __init__(self):
        self.factory = IntegrationFactory()
    
    def integrate_with_fallback(self, primary_template, fallback_template, target_manager):
        """Integrate with fallback template if primary fails."""
        
        # Try primary template
        result = self.factory.integrate_template(primary_template, target_manager)
        
        if result.success:
            return result
        
        # Fallback to secondary template
        print(f"Primary template '{primary_template}' failed, trying fallback...")
        fallback_result = self.factory.integrate_template(fallback_template, target_manager)
        
        # Combine metadata
        fallback_result.metadata["fallback_used"] = True
        fallback_result.metadata["primary_template_errors"] = result.errors
        
        return fallback_result
    
    def validate_before_integration(self, template_name, target_manager):
        """Validate integration before applying."""
        
        # Create validation request
        from quick_start.config.integration_factory import IntegrationRequest
        
        request = IntegrationRequest(
            template_name=template_name,
            target_manager=target_manager
        )
        
        # Validate request
        issues = self.factory.validate_integration_request(request)
        
        if issues:
            raise ValueError(f"Integration validation failed: {issues}")
        
        # Proceed with integration
        return self.factory.integrate_template(template_name, target_manager)

# Usage
workflow = CustomIntegrationWorkflow()
result = workflow.integrate_with_fallback("advanced_rag", "basic_rag", "iris_rag")
```

### Multi-Manager Integration

```python
def integrate_across_managers(template_name):
    """Integrate the same template across multiple managers."""
    
    factory = IntegrationFactory()
    results = {}
    
    managers = ["iris_rag", "rag_templates"]
    
    for manager_type in managers:
        try:
            result = factory.integrate_template(template_name, manager_type)
            results[manager_type] = {
                "success": result.success,
                "config": result.converted_config,
                "errors": result.errors
            }
        except Exception as e:
            results[manager_type] = {
                "success": False,
                "config": None,
                "errors": [str(e)]
            }
    
    return results

# Usage
results = integrate_across_managers("production_rag")
for manager, result in results.items():
    print(f"{manager}: {'✓' if result['success'] else '✗'}")
```

## Error Handling

### Integration Result Structure

```python
@dataclass
class IntegrationResult:
    success: bool
    template_name: str
    target_manager: str
    converted_config: Dict[str, Any]
    validation_results: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    timestamp: str
```

### Error Categories

1. **Template Errors**: Template not found, invalid template format
2. **Conversion Errors**: Failed to convert template to target format
3. **Validation Errors**: Schema validation failures
4. **Integration Errors**: Failed to integrate with configuration manager
5. **Environment Errors**: Environment variable processing failures

### Error Handling Patterns

```python
from quick_start.config.integration_factory import IntegrationFactory

def robust_integration(template_name, target_manager):
    """Robust integration with comprehensive error handling."""
    
    factory = IntegrationFactory()
    
    try:
        # Validate before integration
        validation = factory.get_adapter_info(target_manager)
        print(f"Using adapter: {validation['class']}")
        
        # Perform integration
        result = factory.integrate_template(template_name, target_manager)
        
        if result.success:
            # Log warnings if any
            for warning in result.warnings:
                print(f"Warning: {warning}")
            
            return result.converted_config
        else:
            # Handle integration errors
            print(f"Integration failed for template '{template_name}':")
            for error in result.errors:
                print(f"  - {error}")
            
            # Check for specific error types
            if "TemplateNotFoundError" in str(result.errors):
                print("Suggestion: Check template name and availability")
            elif "ValidationError" in str(result.errors):
                print("Suggestion: Review template configuration format")
            
            return None
    
    except ImportError as e:
        print(f"Integration system not available: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during integration: {e}")
        return None

# Usage
config = robust_integration("my_template", "iris_rag")
if config:
    print("Integration successful!")
else:
    print("Integration failed, using fallback configuration")
```

## Testing

### Integration Test Examples

```python
import pytest
from quick_start.config.integration_factory import IntegrationFactory

class TestQuickStartIntegration:
    
    def test_basic_iris_rag_integration(self):
        """Test basic integration with iris_rag manager."""
        factory = IntegrationFactory()
        
        result = factory.integrate_template("basic_rag", "iris_rag")
        
        assert result.success
        assert result.target_manager == "iris_rag"
        assert "database" in result.converted_config
        assert "embeddings" in result.converted_config
    
    def test_rag_templates_integration_with_validation(self):
        """Test rag_templates integration with validation."""
        factory = IntegrationFactory()
        
        result = factory.integrate_template(
            template_name="advanced_rag",
            target_manager="rag_templates",
            options={"validate_schema": True}
        )
        
        assert result.success
        assert result.validation_results is not None
        assert len(result.errors) == 0
    
    def test_environment_variable_integration(self):
        """Test environment variable integration."""
        factory = IntegrationFactory()
        
        env_vars = {
            "RAG_DATABASE__IRIS__HOST": "test-host",
            "RAG_DATABASE__IRIS__PORT": "9999"
        }
        
        result = factory.integrate_template(
            template_name="basic_rag",
            target_manager="iris_rag",
            environment_variables=env_vars
        )
        
        assert result.success
        config = result.converted_config
        assert config["database"]["iris"]["host"] == "test-host"
        assert config["database"]["iris"]["port"] == 9999
    
    def test_integration_error_handling(self):
        """Test error handling for invalid templates."""
        factory = IntegrationFactory()
        
        result = factory.integrate_template("nonexistent_template", "iris_rag")
        
        assert not result.success
        assert len(result.errors) > 0
        assert "TemplateNotFoundError" in result.metadata.get("error_details", {}).get("error_type", "")

    def test_configuration_manager_integration(self):
        """Test integration through configuration managers."""
        from iris_rag.config.manager import ConfigurationManager
        
        manager = ConfigurationManager()
        
        # Test template loading
        config = manager.load_quick_start_template("basic_rag")
        assert config is not None
        
        # Test template listing
        templates = manager.list_quick_start_templates()
        assert templates["integration_factory_available"]
        
        # Test validation
        validation = manager.validate_quick_start_integration("basic_rag")
        assert validation["valid"]
```

### Running Integration Tests

```bash
# Run all integration tests
pytest tests/quick_start/test_config_integration.py -v

# Run specific test
pytest tests/quick_start/test_config_integration.py::TestQuickStartConfigurationIntegration::test_iris_rag_config_manager_integration -v

# Run with coverage
pytest tests/quick_start/test_config_integration.py --cov=quick_start.config --cov-report=html
```

## Migration Guide

### Migrating from Manual Configuration

#### Before (Manual Configuration)

```python
# Old approach - manual configuration
from iris_rag.config.manager import ConfigurationManager

config_dict = {
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
        "model": "all-MiniLM-L6-v2",
        "dimension": 384
    },
    "llm": {
        "provider": "openai",
        "model": "gpt-3.5-turbo"
    }
}

manager = ConfigurationManager()
# Manual configuration loading...
```

#### After (Quick Start Integration)

```python
# New approach - Quick Start template
from iris_rag.config.manager import ConfigurationManager

manager = ConfigurationManager()

# Load template with environment overrides
config = manager.load_quick_start_template(
    template_name="basic_rag",
    environment_variables={
        "RAG_DATABASE__IRIS__USERNAME": "demo",
        "RAG_DATABASE__IRIS__PASSWORD": "demo"
    }
)

# Configuration is automatically loaded and validated
```

### Migration Steps

1. **Identify Current Configuration**: Document your current configuration structure
2. **Choose Appropriate Template**: Select a Quick Start template that matches your needs
3. **Map Environment Variables**: Convert hardcoded values to environment variables
4. **Test Integration**: Use validation methods to test integration
5. **Update Code**: Replace manual configuration with template loading
6. **Verify Functionality**: Ensure all functionality works with new configuration

### Migration Checklist

- [ ] Current configuration documented
- [ ] Quick Start template selected
- [ ] Environment variables mapped
- [ ] Integration tested in development
- [ ] Code updated to use templates
- [ ] Tests updated for new configuration approach
- [ ] Documentation updated
- [ ] Production deployment planned

## Troubleshooting

### Common Issues

#### 1. Template Not Found

**Error**: `TemplateNotFoundError: Template 'my_template' not found`

**Solutions**:
- Verify template name spelling
- Check template availability: `manager.list_quick_start_templates()`
- Ensure Quick Start system is properly installed

#### 2. Integration Factory Not Available

**Error**: `ImportError: Quick Start integration system not available`

**Solutions**:
- Verify `quick_start` package is installed
- Check Python path includes Quick Start modules
- Ensure all dependencies are installed

#### 3. Schema Validation Failures

**Error**: `ValidationError: Missing required section: vector_index`

**Solutions**:
- Review template configuration structure
- Check if template matches target manager requirements
- Use flexible validation: `options={"validate_schema": False}`

#### 4. Environment Variable Type Errors

**Error**: `TypeError: Invalid type for database:iris:port: expected int, got str`

**Solutions**:
- Ensure environment variables are properly typed
- Use proper casting in environment variable names
- Check original configuration for expected types

#### 5. Configuration Merge Conflicts

**Error**: Configuration values being overridden unexpectedly

**Solutions**:
- Review configuration precedence order
- Check environment variable naming
- Use specific configuration keys to avoid conflicts

### Debugging Tools

#### Enable Debug Logging

```python
import logging

# Enable debug logging for integration system
logging.getLogger("quick_start.config").setLevel(logging.DEBUG)
logging.getLogger("iris_rag.config").setLevel(logging.DEBUG)
logging.getLogger("rag_templates.core").setLevel(logging.DEBUG)

# Create console handler
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# Add handler to loggers
logging.getLogger("quick_start.config").addHandler(handler)
```

#### Integration Diagnostics

```python
from quick_start.config.integration_factory import IntegrationFactory

def diagnose_integration(template_name, target_manager):
    """Comprehensive integration diagnostics."""
    
    factory = IntegrationFactory()
    
    print(f"Diagnosing integration: {template_name} -> {target_manager}")
    print("=" * 50)
    
    # Check adapter availability
    try:
        adapters = factory.list_available_adapters()
        print(f"Available adapters: {adapters}")
    except Exception as e:
        print(f"Failed to list adapters: {e}")
        return
    
    # Check adapter info
    try:
        adapter_info = factory.get_adapter_info(target_manager)
        print(f"Target adapter info: {adapter_info}")
    except Exception as e:
        print(f"Failed to get adapter info: {e}")
    
    # Validate integration request
    try:
        from quick_start.config.integration_factory import IntegrationRequest
        request = IntegrationRequest(template_name, target_manager)
        issues = factory.validate_integration_request(request)
        
        if issues:
            print(f"Validation issues: {issues}")
        else:
            print("Validation passed")
    except Exception as e:
        print(f"Validation failed: {e}")
    
    # Attempt integration
    try:
        result = factory.integrate_template(template_name, target_manager)
        
        if result.success:
            print("Integration successful!")
            print(f"Metadata: {result.metadata}")
        else:
            print(f"Integration failed: {result.errors}")
            print(f"Warnings: {result.warnings}")
    except Exception as e:
        print(f"Integration exception: {e}")

# Usage
diagnose_integration("basic_rag", "iris_rag")
```

### Performance Considerations

#### Large Configuration Templates

For large templates or high-frequency integrations:

```python
# Cache integration factory
class CachedIntegrationFactory:
    _instance = None
    _factory = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._factory = IntegrationFactory()
        return cls._instance
    
    def integrate_template(self, *args, **kwargs):
        return self._factory.integrate_template(*args, **kwargs)

# Usage
factory = CachedIntegrationFactory()
result = factory.integrate_template("basic_rag", "iris_rag")
```

#### Batch Integration

For multiple template integrations:

```python
def batch_integrate(template_configs, target_manager):
    """Integrate multiple templates efficiently."""
    
    factory = IntegrationFactory()
    results = []
    
    for template_name, options in template_configs.items():
        try:
            result = factory.integrate_template(
                template_name=template_name,
                target_manager=target_manager,
                **options
            )
            results.append((template_name, result))
        except Exception as e:
            print(f"Failed to integrate {template_name}: {e}")
    
    return results

# Usage
configs = {
    "basic_rag": {"options": {"validate_schema": True}},
    "advanced_rag": {"options": {"ensure_compatibility": True}},
    "production_rag": {"environment_variables": {"RAG_ENV": "prod"}}
}

results = batch_integrate(configs, "iris_rag")
```

---

## Summary

The Quick Start Configuration Templates Integration System provides a powerful and flexible way to integrate template-based configurations with existing configuration managers. Key benefits include:

- **Simplified Configuration**: Zero-config startup with sensible defaults
- **Flexible Integration**: Support for multiple configuration managers
- **Robust Validation**: Comprehensive schema and compatibility validation
- **Environment Integration**: Seamless environment variable support
- **Error Handling**: Structured error reporting and recovery
- **Testing Support**: Comprehensive test coverage and validation tools

For additional support or questions, refer to the test suite in `tests/quick_start/test_config_integration.py` or the implementation in `quick_start/config/`.