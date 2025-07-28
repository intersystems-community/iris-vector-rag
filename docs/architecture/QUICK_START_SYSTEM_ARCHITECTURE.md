# Quick Start System Architecture

## 1. Executive Summary

This document defines the comprehensive architecture for the RAG Templates Quick Start system, designed to provide a seamless onboarding experience for new users while maintaining the ability to scale to enterprise deployments.

### 1.1 Design Principles

- **Zero-Configuration Start**: Users can experience all 8 RAG techniques with a single command
- **Progressive Complexity**: Clear path from quick start to production deployment
- **Community Edition Compatible**: Works within IRIS Community Edition 10GB limits
- **Modular Architecture**: Each component can be used independently or together
- **Enterprise Scalability**: Quick start components can scale to full enterprise deployment

### 1.2 Core Requirements

- **Sample Data Pipeline**: Automated download and setup for 10-100 PMC documents
- **MCP Server Quick Setup**: One-command setup for MCP server with sample data
- **User Onboarding**: Step-by-step guide for experiencing all 8 RAG techniques
- **Public Repository Ready**: No enterprise dependencies, community-friendly

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Quick Start System                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Sample Data    │  │  Quick Setup    │  │  Configuration  │  │
│  │  Manager        │  │  Orchestrator   │  │  Templates      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│           │                     │                     │         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Documentation │  │  Testing        │  │  MCP Server     │  │
│  │  Generator      │  │  Framework      │  │  Quick Start    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                 Existing Infrastructure                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  8 RAG          │  │  IRIS Database  │  │  MCP Server     │  │
│  │  Techniques     │  │  Integration    │  │  Infrastructure │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  PMC Downloader │  │  Configuration  │  │  Testing        │  │
│  │  System         │  │  Management     │  │  Framework      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Component Architecture

### 3.1 Sample Data Manager

**Purpose**: Automated management of sample PMC documents for quick start scenarios.

**Service Boundaries**:
```
┌─────────────────────────────────────────────────────────────────┐
│                    Sample Data Manager                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Data Source    │  │  Download       │  │  Validation     │  │
│  │  Registry       │  │  Orchestrator   │  │  Engine         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Storage        │  │  Ingestion      │  │  Health         │  │
│  │  Manager        │  │  Pipeline       │  │  Monitor        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Interfaces**:
- `ISampleDataSource`: Abstract interface for data sources
- `IDataDownloader`: Download orchestration interface
- `IDataValidator`: Validation and integrity checking
- `IStorageManager`: Local storage management
- `IIngestionPipeline`: Database ingestion interface

**Configuration**:
```yaml
sample_data:
  sources:
    - name: "pmc_quick_start"
      type: "pmc_subset"
      document_count: 50
      categories: ["medical", "research"]
    - name: "pmc_extended"
      type: "pmc_subset" 
      document_count: 100
      categories: ["medical", "research", "clinical"]
  storage:
    local_path: "data/quick_start_samples"
    cache_enabled: true
    cleanup_policy: "retain_on_success"
  ingestion:
    batch_size: 10
    parallel_workers: 2
    iris_edition: "community"
```

### 3.2 Quick Setup Orchestrator

**Purpose**: Coordinates the entire quick start setup process with dependency management.

**Service Boundaries**:
```
┌─────────────────────────────────────────────────────────────────┐
│                 Quick Setup Orchestrator                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Environment    │  │  Dependency     │  │  Service        │  │
│  │  Detector       │  │  Resolver       │  │  Orchestrator   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Progress       │  │  Error          │  │  Rollback       │  │
│  │  Tracker        │  │  Handler        │  │  Manager        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Setup Phases**:
1. **Environment Detection**: Check system requirements, Docker availability
2. **Dependency Resolution**: Ensure IRIS database, Python environment
3. **Data Preparation**: Download and ingest sample documents
4. **Service Initialization**: Start MCP server, configure endpoints
5. **Validation**: End-to-end testing of all components
6. **User Guidance**: Generate personalized setup completion guide

**Key Interfaces**:
- `IEnvironmentDetector`: System capability detection
- `IDependencyResolver`: Dependency management
- `IServiceOrchestrator`: Service lifecycle management
- `IProgressTracker`: Setup progress monitoring
- `IRollbackManager`: Failure recovery

### 3.3 Configuration Templates

**Purpose**: Environment-specific configuration management with inheritance.

**Service Boundaries**:
```
┌─────────────────────────────────────────────────────────────────┐
│                  Configuration Templates                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Template       │  │  Environment    │  │  Validation     │  │
│  │  Engine         │  │  Resolver       │  │  Engine         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Override       │  │  Secret         │  │  Migration      │  │
│  │  Manager        │  │  Manager        │  │  Handler        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Template Hierarchy**:
```
base_config.yaml
├── quick_start.yaml (inherits from base)
│   ├── quick_start_minimal.yaml
│   └── quick_start_extended.yaml
├── development.yaml (inherits from base)
└── production.yaml (inherits from base)
```

**Key Features**:
- **Environment Variables**: All sensitive data via environment variables
- **Template Inheritance**: Hierarchical configuration with overrides
- **Validation**: Schema validation for all configuration files
- **Migration**: Automatic configuration migration between versions

### 3.4 MCP Server Quick Start

**Purpose**: Streamlined MCP server setup optimized for quick start scenarios.

**Service Boundaries**:
```
┌─────────────────────────────────────────────────────────────────┐
│                 MCP Server Quick Start                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Server         │  │  Tool           │  │  Health         │  │
│  │  Factory        │  │  Registry       │  │  Monitor        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Demo           │  │  Performance    │  │  Documentation  │  │
│  │  Generator      │  │  Monitor        │  │  Generator      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Quick Start Features**:
- **Zero-Config Server**: Automatic server creation with sensible defaults
- **Demo Tool Set**: Pre-configured tools for demonstrating all 8 RAG techniques
- **Interactive Examples**: Built-in examples for each RAG technique
- **Performance Dashboard**: Real-time monitoring of technique performance
- **Auto-Documentation**: Generated API documentation and usage examples

### 3.5 Documentation Generator

**Purpose**: Automated generation of user-specific documentation and tutorials.

**Service Boundaries**:
```
┌─────────────────────────────────────────────────────────────────┐
│                  Documentation Generator                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Template       │  │  Content        │  │  Interactive    │  │
│  │  Engine         │  │  Generator      │  │  Tutorial       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Code           │  │  API            │  │  Deployment     │  │
│  │  Examples       │  │  Reference      │  │  Guide          │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Generated Documentation**:
- **Quick Start Guide**: Step-by-step setup instructions
- **Technique Tutorials**: Interactive tutorials for each RAG technique
- **API Reference**: Auto-generated API documentation
- **Code Examples**: Working code examples in multiple languages
- **Deployment Guides**: Environment-specific deployment instructions

### 3.6 Testing Framework

**Purpose**: Comprehensive validation of quick start system functionality.

**Service Boundaries**:
```
┌─────────────────────────────────────────────────────────────────┐
│                    Testing Framework                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Setup          │  │  Integration    │  │  Performance    │  │
│  │  Validator      │  │  Tester         │  │  Validator      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Smoke          │  │  Regression     │  │  User           │  │
│  │  Tests          │  │  Suite          │  │  Acceptance     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Test Categories**:
- **Setup Validation**: Verify all components are correctly installed
- **Integration Tests**: End-to-end testing of all RAG techniques
- **Performance Tests**: Baseline performance validation
- **Smoke Tests**: Quick validation of core functionality
- **User Acceptance**: Simulated user journey testing

## 4. Data Flow Architecture

### 4.1 Quick Start Flow

```
User Command: `make quick-start`
         │
         ▼
┌─────────────────┐
│ Environment     │ ──► Check Docker, Python, UV
│ Detection       │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Sample Data     │ ──► Download 50 PMC documents
│ Download        │     Validate integrity
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ IRIS Database   │ ──► Start IRIS Community Edition
│ Setup           │     Initialize schema
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │ ──► Process and load documents
│                 │     Create vector embeddings
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ MCP Server      │ ──► Start MCP server
│ Startup         │     Register all 8 techniques
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Validation      │ ──► Test all techniques
│ & Demo          │     Generate demo queries
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ User Guide      │ ──► Generate personalized guide
│ Generation      │     Show next steps
└─────────────────┘
```

### 4.2 Configuration Flow

```
Base Configuration (base_config.yaml)
         │
         ▼
Environment Detection
         │
         ├─► Quick Start ──► quick_start.yaml
         ├─► Development ──► development.yaml
         └─► Production ──► production.yaml
         │
         ▼
Template Resolution
         │
         ▼
Environment Variable Injection
         │
         ▼
Validation & Schema Check
         │
         ▼
Runtime Configuration
```

## 5. File Structure and Organization

```
quick_start/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── orchestrator.py          # Main setup orchestration
│   ├── environment_detector.py  # System capability detection
│   └── progress_tracker.py      # Setup progress monitoring
├── data/
│   ├── __init__.py
│   ├── sample_manager.py        # Sample data management
│   ├── downloader.py           # PMC document downloader
│   ├── validator.py            # Data validation
│   └── ingestion.py            # Database ingestion
├── config/
│   ├── __init__.py
│   ├── template_engine.py      # Configuration templating
│   ├── environment_resolver.py # Environment-specific config
│   └── templates/
│       ├── base_config.yaml
│       ├── quick_start.yaml
│       ├── quick_start_minimal.yaml
│       └── quick_start_extended.yaml
├── mcp/
│   ├── __init__.py
│   ├── quick_server.py         # Quick start MCP server
│   ├── demo_tools.py           # Demo tool implementations
│   └── health_monitor.py       # Server health monitoring
├── docs/
│   ├── __init__.py
│   ├── generator.py            # Documentation generation
│   ├── tutorial_builder.py     # Interactive tutorial builder
│   └── templates/
│       ├── quick_start_guide.md
│       ├── technique_tutorial.md
│       └── api_reference.md
├── testing/
│   ├── __init__.py
│   ├── setup_validator.py      # Setup validation tests
│   ├── integration_tester.py   # Integration test suite
│   ├── smoke_tests.py          # Quick validation tests
│   └── performance_validator.py # Performance baseline tests
└── cli/
    ├── __init__.py
    ├── commands.py             # CLI command implementations
    └── interactive.py          # Interactive setup wizard
```

## 6. Interface Specifications

### 6.1 Sample Data Manager Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class SampleDataConfig:
    source_type: str
    document_count: int
    categories: List[str]
    storage_path: str
    cache_enabled: bool

class ISampleDataManager(ABC):
    @abstractmethod
    async def download_samples(self, config: SampleDataConfig) -> Dict[str, Any]:
        """Download sample documents according to configuration."""
        pass
    
    @abstractmethod
    async def validate_samples(self, storage_path: str) -> bool:
        """Validate downloaded sample documents."""
        pass
    
    @abstractmethod
    async def ingest_samples(self, storage_path: str) -> Dict[str, Any]:
        """Ingest samples into IRIS database."""
        pass
    
    @abstractmethod
    async def cleanup_samples(self, storage_path: str) -> None:
        """Clean up temporary sample files."""
        pass
```

### 6.2 Quick Setup Orchestrator Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
from enum import Enum

class SetupPhase(Enum):
    ENVIRONMENT_CHECK = "environment_check"
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    DATA_PREPARATION = "data_preparation"
    SERVICE_INITIALIZATION = "service_initialization"
    VALIDATION = "validation"
    COMPLETION = "completion"

class IQuickSetupOrchestrator(ABC):
    @abstractmethod
    async def setup(
        self, 
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[SetupPhase, float], None]] = None
    ) -> Dict[str, Any]:
        """Execute complete quick start setup."""
        pass
    
    @abstractmethod
    async def validate_environment(self) -> Dict[str, bool]:
        """Validate system environment for quick start."""
        pass
    
    @abstractmethod
    async def rollback(self, phase: SetupPhase) -> None:
        """Rollback setup to previous state."""
        pass
```

### 6.3 Configuration Template Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pathlib import Path

class IConfigurationTemplate(ABC):
    @abstractmethod
    def resolve_template(
        self, 
        template_name: str, 
        environment: str,
        overrides: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Resolve configuration template with environment-specific values."""
        pass
    
    @abstractmethod
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
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
```

## 7. Deployment Architecture

### 7.1 Docker Compose Quick Start

```yaml
# docker-compose.quick-start.yml
version: '3.8'

services:
  iris_quick_start:
    image: containers.intersystems.com/intersystems/iris:latest
    container_name: iris_quick_start
    environment:
      - IRIS_DOCKER_IMAGE=community
      - IRISNAMESPACE=USER
      - ISC_DEFAULT_PASSWORD=SYS
    ports:
      - "1972:1972"
      - "52773:52773"
    volumes:
      - iris_quick_start_data:/usr/irissys/mgr
    healthcheck:
      test: ["CMD", "/usr/irissys/bin/iris", "session", "iris", "-U%SYS", "##class(%SYSTEM.SQL).Execute(\"SELECT 1\")"]
      interval: 10s
      timeout: 5s
      retries: 3

  rag_quick_start:
    build:
      context: .
      dockerfile: quick_start/Dockerfile
    container_name: rag_quick_start
    depends_on:
      iris_quick_start:
        condition: service_healthy
    environment:
      - IRIS_HOST=iris_quick_start
      - IRIS_PORT=1972
      - QUICK_START_MODE=true
    ports:
      - "3000:3000"  # MCP Server
      - "8080:8080"  # Documentation Server
    volumes:
      - ./data/quick_start_samples:/app/data/samples

volumes:
  iris_quick_start_data:
```

### 7.2 Makefile Integration

```makefile
# Quick Start Commands
.PHONY: quick-start quick-start-minimal quick-start-extended quick-start-clean

quick-start: ## Complete quick start setup (50 documents)
	@echo "🚀 Starting RAG Templates Quick Start..."
	uv run python -m quick_start.cli.commands setup --profile=standard

quick-start-minimal: ## Minimal quick start (10 documents)
	@echo "🚀 Starting RAG Templates Minimal Quick Start..."
	uv run python -m quick_start.cli.commands setup --profile=minimal

quick-start-extended: ## Extended quick start (100 documents)
	@echo "🚀 Starting RAG Templates Extended Quick Start..."
	uv run python -m quick_start.cli.commands setup --profile=extended

quick-start-clean: ## Clean up quick start environment
	@echo "🧹 Cleaning up Quick Start environment..."
	uv run python -m quick_start.cli.commands cleanup

quick-start-validate: ## Validate quick start setup
	@echo "✅ Validating Quick Start setup..."
	uv run python -m quick_start.testing.setup_validator
```

## 8. Security and Compliance

### 8.1 Security Boundaries

- **No Hardcoded Secrets**: All sensitive data via environment variables
- **Minimal Permissions**: Containers run with minimal required permissions
- **Network Isolation**: Services communicate through defined network boundaries
- **Data Encryption**: All data at rest and in transit encrypted
- **Audit Logging**: All setup actions logged for audit purposes

### 8.2 Community Edition Compliance

- **Data Limits**: Respect IRIS Community Edition 10GB limit
- **Resource Constraints**: Optimize for limited resource environments
- **License Compliance**: Ensure all components compatible with community licensing
- **Open Source**: All quick start components use open source dependencies

## 9. Performance and Scalability

### 9.1 Performance Targets

- **Setup Time**: Complete setup in under 5 minutes
- **Memory Usage**: Peak memory usage under 4GB
- **Disk Usage**: Total disk usage under 8GB (within community limits)
- **Response Time**: RAG queries respond within 2 seconds

### 9.2 Scalability Path

- **Horizontal Scaling**: Clear path to scale from quick start to enterprise
- **Configuration Migration**: Automated migration from quick start to production config
- **Data Migration**: Tools to migrate from sample data to production datasets
- **Service Decomposition**: Ability to decompose monolithic quick start into microservices

## 10. Monitoring and Observability

### 10.1 Health Monitoring

- **Component Health**: Real-time health status of all components
- **Performance Metrics**: Response times, throughput, error rates
- **Resource Utilization**: CPU, memory, disk usage monitoring
- **User Journey Tracking**: Track user progress through quick start process

### 10.2 Logging and Debugging

- **Structured Logging**: JSON-formatted logs for all components
- **Debug Mode**: Verbose logging for troubleshooting
- **Error Correlation**: Correlation IDs for tracking errors across components
- **Performance Profiling**: Built-in profiling for performance optimization

This architecture provides a comprehensive foundation for the Quick Start system while maintaining clean separation of concerns and extensibility for future enhancements.