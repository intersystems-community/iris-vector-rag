# Configuration Templates Specification

## 1. Overview

The Configuration Templates system provides environment-specific configuration management with inheritance, validation, and automatic environment variable injection. It supports the progression from quick start to production deployments while maintaining consistency and security.

## 2. Template Hierarchy Architecture

### 2.1 Inheritance Structure

```
base_config.yaml (Foundation)
├── quick_start.yaml (Quick Start Base)
│   ├── quick_start_minimal.yaml (10 docs)
│   ├── quick_start_standard.yaml (50 docs)
│   └── quick_start_extended.yaml (100 docs)
├── development.yaml (Development Environment)
│   ├── development_local.yaml
│   └── development_docker.yaml
└── production.yaml (Production Environment)
    ├── production_single.yaml
    └── production_cluster.yaml
```

### 2.2 Template Resolution Flow

```
User Request (profile=quick_start_standard)
         │
         ▼
Template Resolver
         │
         ├─► Load base_config.yaml
         ├─► Load quick_start.yaml (inherits from base)
         └─► Load quick_start_standard.yaml (inherits from quick_start)
         │
         ▼
Environment Variable Injection
         │
         ▼
Validation Engine
         │
         ▼
Runtime Configuration Object
```

## 3. Configuration Schema

### 3.1 Base Configuration Template

```yaml
# quick_start/config/templates/base_config.yaml
metadata:
  version: "1.0.0"
  schema_version: "2024.1"
  description: "Base configuration for RAG Templates"
  
# Database Configuration
database:
  iris:
    driver: "intersystems_iris.dbapi._DBAPI"
    host: "${IRIS_HOST:-localhost}"
    port: "${IRIS_PORT:-1972}"
    namespace: "${IRIS_NAMESPACE:-USER}"
    username: "${IRIS_USERNAME:-_SYSTEM}"
    password: "${IRIS_PASSWORD:-SYS}"
    connection_pool:
      min_connections: 2
      max_connections: 10
      connection_timeout: 30
      idle_timeout: 300

# Storage Configuration
storage:
  iris:
    table_name: "${IRIS_TABLE_NAME:-RAG.SourceDocuments}"
    vector_dimension: "${VECTOR_DIMENSION:-384}"
    index_type: "HNSW"
  
  chunking:
    enabled: true
    strategy: "fixed_size"
    chunk_size: 512
    overlap: 50
    preserve_sentences: true
    min_chunk_size: 100

# Vector Index Configuration
vector_index:
  type: "HNSW"
  M: 16
  efConstruction: 200
  Distance: "COSINE"

# Embeddings Configuration
embeddings:
  default_provider: "sentence_transformers"
  sentence_transformers:
    model_name: "${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
    device: "${EMBEDDING_DEVICE:-cpu}"
    cache_folder: "${EMBEDDING_CACHE_FOLDER:-./models/embeddings}"

# LLM Configuration
llm:
  default_provider: "${LLM_PROVIDER:-openai}"
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "${OPENAI_MODEL:-gpt-3.5-turbo}"
    temperature: 0.1
    max_tokens: 1000
  azure_openai:
    api_key: "${AZURE_OPENAI_API_KEY}"
    endpoint: "${AZURE_OPENAI_ENDPOINT}"
    api_version: "${AZURE_OPENAI_API_VERSION:-2023-12-01-preview}"
    deployment_name: "${AZURE_OPENAI_DEPLOYMENT_NAME}"

# Logging Configuration
logging:
  level: "${LOG_LEVEL:-INFO}"
  format: "structured"
  handlers:
    - type: "console"
      level: "INFO"
    - type: "file"
      level: "DEBUG"
      filename: "${LOG_FILE:-logs/rag_templates.log}"
      max_size: "10MB"
      backup_count: 5

# Security Configuration
security:
  enable_auth: false
  cors:
    enabled: true
    origins: ["http://localhost:3000", "http://localhost:8080"]
  rate_limiting:
    enabled: false
    requests_per_minute: 60

# Performance Configuration
performance:
  batch_size: 32
  max_workers: 4
  timeout: 30
  cache:
    enabled: true
    ttl: 3600
    max_size: 1000
```

### 3.2 Quick Start Configuration Template

```yaml
# quick_start/config/templates/quick_start.yaml
extends: "base_config.yaml"

metadata:
  profile: "quick_start"
  description: "Quick start configuration optimized for community edition"

# Override database settings for community edition
database:
  iris:
    connection_pool:
      min_connections: 1
      max_connections: 5
      connection_timeout: 15

# Quick start specific storage settings
storage:
  iris:
    table_name: "QuickStart.Documents"
    vector_dimension: 384
  
  chunking:
    chunk_size: 256  # Smaller chunks for quick start
    overlap: 25

# Sample data configuration
sample_data:
  enabled: true
  auto_download: true
  source_type: "pmc_api"
  storage_path: "data/quick_start_samples"
  cache_enabled: true
  cleanup_on_success: false

# MCP Server configuration
mcp_server:
  enabled: true
  name: "rag-quick-start"
  description: "RAG Templates Quick Start Server"
  port: "${MCP_SERVER_PORT:-3000}"
  auto_start: true
  demo_mode: true
  
  tools:
    enabled:
      - "rag_basic"
      - "rag_hyde"
      - "rag_crag"
      - "rag_graphrag"
      - "rag_colbert"
      - "rag_noderag"
      - "rag_hybrid_ifind"
      - "rag_sqlrag"
      - "rag_health_check"

# Performance optimized for quick start
performance:
  batch_size: 16  # Smaller batches
  max_workers: 2  # Fewer workers
  timeout: 15
  cache:
    enabled: true
    ttl: 1800  # Shorter TTL
    max_size: 500

# Monitoring configuration
monitoring:
  enabled: true
  metrics:
    enabled: true
    port: "${METRICS_PORT:-9090}"
  health_checks:
    enabled: true
    interval: 30
  
# Documentation server
docs_server:
  enabled: true
  port: "${DOCS_PORT:-8080}"
  auto_generate: true
```

### 3.3 Quick Start Profile Variants

```yaml
# quick_start/config/templates/quick_start_minimal.yaml
extends: "quick_start.yaml"

metadata:
  profile: "quick_start_minimal"
  description: "Minimal quick start with 10 documents"

sample_data:
  document_count: 10
  categories: ["medical"]
  parallel_downloads: 2
  batch_size: 5

performance:
  batch_size: 8
  max_workers: 1

mcp_server:
  tools:
    enabled:
      - "rag_basic"
      - "rag_hyde"
      - "rag_health_check"
```

```yaml
# quick_start/config/templates/quick_start_standard.yaml
extends: "quick_start.yaml"

metadata:
  profile: "quick_start_standard"
  description: "Standard quick start with 50 documents"

sample_data:
  document_count: 50
  categories: ["medical", "research"]
  parallel_downloads: 4
  batch_size: 10

performance:
  batch_size: 16
  max_workers: 2
```

```yaml
# quick_start/config/templates/quick_start_extended.yaml
extends: "quick_start.yaml"

metadata:
  profile: "quick_start_extended"
  description: "Extended quick start with 100 documents"

sample_data:
  document_count: 100
  categories: ["medical", "research", "clinical"]
  parallel_downloads: 6
  batch_size: 15

performance:
  batch_size: 24
  max_workers: 3

database:
  iris:
    connection_pool:
      max_connections: 8
```

## 4. Configuration Engine Architecture

### 4.1 Template Engine Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ConfigurationContext:
    """Context for configuration resolution."""
    profile: str
    environment: str
    overrides: Dict[str, Any]
    template_path: Path
    environment_variables: Dict[str, str]

class IConfigurationTemplate(ABC):
    """Interface for configuration template operations."""
    
    @abstractmethod
    def resolve_template(
        self, 
        context: ConfigurationContext
    ) -> Dict[str, Any]:
        """Resolve configuration template with context."""
        pass
    
    @abstractmethod
    def validate_configuration(
        self, 
        config: Dict[str, Any]
    ) -> List[str]:
        """Validate configuration against schema."""
        pass
    
    @abstractmethod
    def generate_environment_file(
        self, 
        config: Dict[str, Any], 
        output_path: Path
    ) -> None:
        """Generate .env file from configuration."""
        pass
    
    @abstractmethod
    def get_available_profiles(self) -> List[str]:
        """Get list of available configuration profiles."""
        pass
```

### 4.2 Template Engine Implementation

```python
# quick_start/config/template_engine.py
import yaml
import os
from typing import Dict, Any, List
from pathlib import Path
import jsonschema
from jinja2 import Environment, FileSystemLoader

class ConfigurationTemplateEngine(IConfigurationTemplate):
    """Main implementation of configuration template engine."""
    
    def __init__(self, template_directory: Path):
        self.template_directory = template_directory
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_directory),
            undefined=StrictUndefined
        )
        self.schema_validator = ConfigurationSchemaValidator()
        self.inheritance_resolver = InheritanceResolver()
        self.variable_injector = EnvironmentVariableInjector()
    
    def resolve_template(
        self, 
        context: ConfigurationContext
    ) -> Dict[str, Any]:
        """Resolve configuration template with full inheritance chain."""
        
        # Build inheritance chain
        inheritance_chain = self._build_inheritance_chain(context.profile)
        
        # Load and merge configurations
        merged_config = {}
        for template_name in inheritance_chain:
            template_config = self._load_template(template_name)
            merged_config = self._deep_merge(merged_config, template_config)
        
        # Apply context overrides
        if context.overrides:
            merged_config = self._deep_merge(merged_config, context.overrides)
        
        # Inject environment variables
        resolved_config = self.variable_injector.inject_variables(
            merged_config, 
            context.environment_variables
        )
        
        # Validate final configuration
        validation_errors = self.validate_configuration(resolved_config)
        if validation_errors:
            raise ConfigurationError(f"Validation failed: {validation_errors}")
        
        return resolved_config
    
    def _build_inheritance_chain(self, profile: str) -> List[str]:
        """Build inheritance chain for a profile."""
        chain = []
        current_profile = profile
        
        while current_profile:
            template_path = self.template_directory / f"{current_profile}.yaml"
            if not template_path.exists():
                raise ConfigurationError(f"Template not found: {current_profile}")
            
            chain.insert(0, current_profile)
            
            # Load template to check for 'extends' directive
            with open(template_path, 'r') as f:
                template_data = yaml.safe_load(f)
            
            current_profile = template_data.get('extends')
        
        return chain
    
    def _load_template(self, template_name: str) -> Dict[str, Any]:
        """Load a single template file."""
        template_path = self.template_directory / f"{template_name}.yaml"
        
        with open(template_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
```

### 4.3 Environment Variable Injection

```python
# quick_start/config/variable_injector.py
import re
import os
from typing import Dict, Any, Union

class EnvironmentVariableInjector:
    """Handles environment variable injection into configuration."""
    
    # Pattern for environment variable substitution: ${VAR_NAME:-default_value}
    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    def inject_variables(
        self, 
        config: Dict[str, Any], 
        env_vars: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Inject environment variables into configuration."""
        if env_vars is None:
            env_vars = dict(os.environ)
        
        return self._process_value(config, env_vars)
    
    def _process_value(self, value: Any, env_vars: Dict[str, str]) -> Any:
        """Process a configuration value for environment variable substitution."""
        if isinstance(value, str):
            return self._substitute_env_vars(value, env_vars)
        elif isinstance(value, dict):
            return {k: self._process_value(v, env_vars) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._process_value(item, env_vars) for item in value]
        else:
            return value
    
    def _substitute_env_vars(self, text: str, env_vars: Dict[str, str]) -> Union[str, int, float, bool]:
        """Substitute environment variables in text."""
        def replace_var(match):
            var_expr = match.group(1)
            
            # Handle default values: VAR_NAME:-default_value
            if ':-' in var_expr:
                var_name, default_value = var_expr.split(':-', 1)
                value = env_vars.get(var_name, default_value)
            else:
                var_name = var_expr
                value = env_vars.get(var_name, '')
                if not value:
                    raise ConfigurationError(f"Required environment variable not set: {var_name}")
            
            return value
        
        result = self.ENV_VAR_PATTERN.sub(replace_var, text)
        
        # Try to convert to appropriate type
        return self._convert_type(result)
    
    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        """Convert string value to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
```

### 4.4 Configuration Validation

```python
# quick_start/config/validator.py
import jsonschema
from typing import Dict, Any, List
from pathlib import Path

class ConfigurationSchemaValidator:
    """Validates configuration against JSON schema."""
    
    def __init__(self, schema_directory: Path = None):
        self.schema_directory = schema_directory or Path(__file__).parent / "schemas"
        self.schemas = self._load_schemas()
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema."""
        errors = []
        
        try:
            # Validate against main schema
            jsonschema.validate(config, self.schemas['main'])
            
            # Validate specific sections
            for section, schema in self.schemas.items():
                if section != 'main' and section in config:
                    try:
                        jsonschema.validate(config[section], schema)
                    except jsonschema.ValidationError as e:
                        errors.append(f"Section '{section}': {e.message}")
        
        except jsonschema.ValidationError as e:
            errors.append(f"Configuration validation error: {e.message}")
        
        return errors
    
    def _load_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load all JSON schemas."""
        schemas = {}
        
        for schema_file in self.schema_directory.glob("*.json"):
            schema_name = schema_file.stem
            with open(schema_file, 'r') as f:
                schemas[schema_name] = json.load(f)
        
        return schemas
```

## 5. Configuration Profiles

### 5.1 Profile Management

```python
# quick_start/config/profile_manager.py
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path

@dataclass
class ConfigurationProfile:
    """Represents a configuration profile."""
    name: str
    description: str
    template_file: str
    requirements: Dict[str, Any]
    features: List[str]
    resource_estimates: Dict[str, Any]

class ProfileManager:
    """Manages configuration profiles."""
    
    def __init__(self, template_directory: Path):
        self.template_directory = template_directory
        self.profiles = self._discover_profiles()
    
    def get_profile(self, name: str) -> Optional[ConfigurationProfile]:
        """Get profile by name."""
        return self.profiles.get(name)
    
    def list_profiles(self) -> List[ConfigurationProfile]:
        """List all available profiles."""
        return list(self.profiles.values())
    
    def get_profiles_by_category(self, category: str) -> List[ConfigurationProfile]:
        """Get profiles by category (quick_start, development, production)."""
        return [
            profile for profile in self.profiles.values()
            if profile.name.startswith(category)
        ]
    
    def _discover_profiles(self) -> Dict[str, ConfigurationProfile]:
        """Discover available configuration profiles."""
        profiles = {}
        
        for template_file in self.template_directory.glob("*.yaml"):
            if template_file.name.startswith("base_"):
                continue  # Skip base templates
            
            profile_name = template_file.stem
            profile_data = self._load_profile_metadata(template_file)
            
            profiles[profile_name] = ConfigurationProfile(
                name=profile_name,
                description=profile_data.get('metadata', {}).get('description', ''),
                template_file=str(template_file),
                requirements=profile_data.get('requirements', {}),
                features=profile_data.get('features', []),
                resource_estimates=profile_data.get('resource_estimates', {})
            )
        
        return profiles
```

## 6. Environment File Generation

### 6.1 Environment File Generator

```python
# quick_start/config/env_generator.py
from typing import Dict, Any
from pathlib import Path
import re

class EnvironmentFileGenerator:
    """Generates .env files from configuration."""
    
    def generate(
        self, 
        config: Dict[str, Any], 
        output_path: Path,
        include_comments: bool = True
    ) -> None:
        """Generate .env file from configuration."""
        
        env_vars = self._extract_env_vars(config)
        
        with open(output_path, 'w') as f:
            if include_comments:
                f.write("# Generated environment file for RAG Templates\n")
                f.write("# This file contains environment variables extracted from configuration\n\n")
            
            # Group variables by section
            sections = self._group_by_section(env_vars)
            
            for section, variables in sections.items():
                if include_comments:
                    f.write(f"# {section.upper()} Configuration\n")
                
                for var_name, var_info in variables.items():
                    if include_comments and var_info.get('description'):
                        f.write(f"# {var_info['description']}\n")
                    
                    value = var_info['value']
                    if isinstance(value, str) and ' ' in value:
                        value = f'"{value}"'
                    
                    f.write(f"{var_name}={value}\n")
                
                f.write("\n")
    
    def _extract_env_vars(self, config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract environment variables from configuration."""
        env_vars = {}
        
        # Common environment variables
        env_vars.update({
            'IRIS_HOST': {
                'value': config.get('database', {}).get('iris', {}).get('host', 'localhost'),
                'section': 'database',
                'description': 'IRIS database host'
            },
            'IRIS_PORT': {
                'value': config.get('database', {}).get('iris', {}).get('port', 1972),
                'section': 'database',
                'description': 'IRIS database port'
            },
            'IRIS_USERNAME': {
                'value': config.get('database', {}).get('iris', {}).get('username', '_SYSTEM'),
                'section': 'database',
                'description': 'IRIS database username'
            },
            'IRIS_PASSWORD': {
                'value': config.get('database', {}).get('iris', {}).get('password', 'SYS'),
                'section': 'database',
                'description': 'IRIS database password'
            },
            'LOG_LEVEL': {
                'value': config.get('logging', {}).get('level', 'INFO'),
                'section': 'logging',
                'description': 'Logging level'
            }
        })
        
        # LLM configuration
        llm_config = config.get('llm', {})
        if 'openai' in llm_config:
            env_vars['OPENAI_API_KEY'] = {
                'value': '',
                'section': 'llm',
                'description': 'OpenAI API key (required for LLM functionality)'
            }
        
        # MCP Server configuration
        mcp_config = config.get('mcp_server', {})
        if mcp_config.get('enabled'):
            env_vars['MCP_SERVER_PORT'] = {
                'value': mcp_config.get('port', 3000),
                'section': 'mcp',
                'description': 'MCP server port'
            }
        
        return env_vars
    
    def _group_by_section(self, env_vars: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Group environment variables by section."""
        sections = {}
        
        for var_name, var_info in env_vars.items():
            section = var_info.get('section', 'general')
            if section not in sections:
                sections[section] = {}
            sections[section][var_name] = var_info
        
        return sections
```

## 7. Configuration CLI

### 7.1 Configuration Commands

```python
# quick_start/cli/config_commands.py
import click
from pathlib import Path
from typing import Optional

@click.group()
def config():
    """Configuration management commands."""
    pass

@config.command()
@click.option('--profile', '-p', required=True, help='Configuration profile name')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', type=click.Choice(['yaml', 'json', 'env']), default='yaml')
def generate(profile: str, output: Optional[str], format: str):
    """Generate configuration from profile."""
    from quick_start.config.template_engine import ConfigurationTemplateEngine
    from quick_start.config.env_generator import EnvironmentFileGenerator
    
    template_engine = ConfigurationTemplateEngine(Path("quick_start/config/templates"))
    
    context = ConfigurationContext(
        profile=profile,
        environment='quick_start',
        overrides={},
        template_path=Path("quick_start/config/templates"),
        environment_variables=dict(os.environ)
    )
    
    config = template_engine.resolve_template(context)
    
    if output:
        output_path = Path(output)
    else:
        output_path = Path(f"config_{profile}.{format}")
    
    if format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
    elif format == 'env':
        env_generator = EnvironmentFileGenerator()
        env_generator.generate(config, output_path)
    
    click.echo(f"Configuration generated: {output_path}")

@config.command()
def list_profiles():
    """List available configuration profiles."""
    from quick_start.config.profile_manager import ProfileManager
    
    profile_manager = ProfileManager(Path("quick_start/config/templates"))
    profiles = profile_manager.list_profiles()
    
    click.echo("Available Configuration Profiles:")
    click.echo("=" * 40)
    
    for profile in profiles:
        click.echo(f"Name: {profile.name}")
        click.echo(f"Description: {profile.description}")
        click.echo(f"Features: {', '.join(profile.features)}")
        click.echo("-" * 40)

@config.command()
@click.option('--profile', '-p', required=True, help='Configuration profile to validate')
def validate(profile: str):
    """Validate configuration profile."""
    # Implementation for validation
    pass
```

This configuration templates specification provides a comprehensive, extensible system for managing environment-specific configurations while maintaining security and ease of use.