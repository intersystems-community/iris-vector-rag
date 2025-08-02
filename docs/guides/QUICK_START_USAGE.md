# Quick Start Usage Guide

This guide provides comprehensive documentation for using the Quick Start system to set up and configure the RAG Templates project.

## Overview

The Quick Start system provides a one-command setup experience for the RAG Templates project, supporting multiple profiles and configurations to suit different use cases and system requirements.

## Quick Start Commands

### Interactive Setup

For first-time users or when you want to choose your configuration interactively:

```bash
make quick-start
```

This command launches an interactive CLI wizard that will:
- Guide you through profile selection
- Configure environment variables
- Set up the database and dependencies
- Load sample data
- Validate the installation

### Profile-Based Setup

For automated setup with predefined configurations:

#### Minimal Profile (Recommended for Development)
```bash
make quick-start-minimal
```
- **Documents**: 50 PMC documents
- **Memory**: 2GB RAM minimum
- **Setup Time**: ~5 minutes
- **Use Case**: Development, testing, quick demos

#### Standard Profile (Recommended for Most Users)
```bash
make quick-start-standard
```
- **Documents**: 500 PMC documents
- **Memory**: 4GB RAM minimum
- **Setup Time**: ~15 minutes
- **Use Case**: Evaluation, small-scale production

#### Extended Profile (For Comprehensive Testing)
```bash
make quick-start-extended
```
- **Documents**: 5000 PMC documents
- **Memory**: 8GB RAM minimum
- **Setup Time**: ~30 minutes
- **Use Case**: Performance testing, large-scale evaluation

#### Custom Profile
```bash
make quick-start-custom PROFILE=my-custom-profile
```
- Use your own custom profile configuration
- Profile must be defined in `quick_start/config/templates/`

### Management Commands

#### Check System Status
```bash
make quick-start-status
```
Provides comprehensive system health check including:
- Database connectivity
- Docker services status
- Python environment validation
- Pipeline functionality
- Data availability

#### Clean Environment
```bash
make quick-start-clean
```
Safely cleans up the Quick Start environment:
- Removes temporary files
- Resets configuration to defaults
- Preserves important data and settings

## System Requirements

### Minimum Requirements
- **Operating System**: macOS, Linux, or Windows with WSL2
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM (8GB recommended)
- **Disk Space**: 10GB free space
- **Docker**: Docker Desktop or Docker Engine + Docker Compose

### Required Software
- **uv**: Python package manager (auto-installed if missing)
- **Docker**: Container runtime
- **Git**: Version control (for development)

## Setup Process

### 1. Pre-Setup Validation
The system automatically checks:
- Python version compatibility
- Required system dependencies
- Docker availability and status
- Available system resources

### 2. Environment Configuration
- Creates or updates `.env` file with required variables
- Configures database connection parameters
- Sets up Python path and environment variables

### 3. Dependency Installation
- Installs Python packages using uv
- Starts Docker services (IRIS database)
- Validates package imports and functionality

### 4. Database Setup
- Initializes IRIS database schema
- Creates required tables and indexes
- Configures database connections

### 5. Data Loading
- Downloads and processes PMC documents
- Generates embeddings for vector search
- Populates database with sample data

### 6. Validation
- Tests database connectivity
- Validates pipeline functionality
- Confirms system readiness

## Profile Configuration

### Profile Structure
Profiles are defined in YAML format with the following structure:

```yaml
name: "minimal"
description: "Minimal setup for development"
requirements:
  memory_gb: 2
  disk_gb: 5
  documents: 50
environment:
  IRIS_HOST: "localhost"
  IRIS_PORT: "1972"
  LOG_LEVEL: "INFO"
data:
  source: "pmc_sample"
  limit: 50
  embeddings: true
pipelines:
  - "basic"
  - "hyde"
```

### Creating Custom Profiles
1. Create a new YAML file in `quick_start/config/templates/`
2. Define your configuration parameters
3. Use with `make quick-start-custom PROFILE=your-profile`

## Troubleshooting

### Common Issues

#### Docker Not Running
```bash
# Check Docker status
docker info

# Start Docker services
docker-compose up -d

# Verify IRIS container
docker ps | grep iris
```

#### Python Environment Issues
```bash
# Reinstall dependencies
make install

# Check Python environment
uv run python -c "import iris_rag; print('OK')"

# Validate environment
make quick-start-status
```

#### Database Connection Problems
```bash
# Check database connectivity
make test-dbapi

# Restart database
docker-compose restart iris

# Verify environment variables
cat .env | grep IRIS
```

#### Memory or Resource Issues
```bash
# Check system resources
make quick-start-status

# Use minimal profile
make quick-start-minimal

# Clean up and retry
make quick-start-clean && make quick-start-minimal
```

### Getting Help

#### System Status
```bash
make quick-start-status
```
Provides detailed diagnostics and recommendations.

#### Validation
```bash
# Validate specific components
python -m quick_start.scripts.validate_setup --component database
python -m quick_start.scripts.validate_setup --component python
python -m quick_start.scripts.validate_setup --component docker
```

#### Environment Check
```bash
# Check environment setup
python -m quick_start.scripts.setup_environment --check

# Validate environment
python -m quick_start.scripts.setup_environment --validate
```

## Advanced Usage

### Environment Variables

Key environment variables that can be customized:

```bash
# Database Configuration
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USERNAME=_SYSTEM
IRIS_PASSWORD=SYS

# Quick Start Configuration
QUICK_START_MODE=true
LOG_LEVEL=INFO

# Python Configuration
PYTHONPATH=/path/to/project
PYTHONDONTWRITEBYTECODE=1
```

### Integration with Existing Workflows

#### CI/CD Integration
```bash
# Non-interactive setup for CI
make quick-start-minimal

# Validate setup
make quick-start-status

# Run tests
make test-1000
```

#### Development Workflow
```bash
# Quick development setup
make quick-start-minimal

# Test specific pipeline
make test-pipeline PIPELINE=basic

# Run comprehensive tests
make test-1000
```

### Performance Optimization

#### For Development
- Use `minimal` profile for fastest setup
- Limit document count for quick iterations
- Use local caching when available

#### For Production
- Use `standard` or `extended` profiles
- Ensure adequate system resources
- Monitor system performance during setup

## Next Steps

After successful Quick Start setup:

### 1. Validate Installation
```bash
make quick-start-status
make test-pipeline PIPELINE=basic
```

### 2. Explore RAG Pipelines
```bash
# Test different pipeline types
make test-pipeline PIPELINE=hyde
make test-pipeline PIPELINE=colbert
make test-pipeline PIPELINE=graphrag
```

### 3. Run Comprehensive Tests
```bash
# Test with 1000 documents
make test-1000

# Run RAGAS evaluation
make eval-all-ragas-1000
```

### 4. Explore Documentation
- [API Reference](../API_REFERENCE.md)
- [Pipeline Documentation](../reference/)
- [Architecture Overview](../architecture/)

## Support

### Documentation
- [System Architecture](../architecture/SYSTEM_ARCHITECTURE.md)
- [Configuration Guide](../CONFIGURATION.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share experiences
- Documentation: Contribute to guides and examples

### Development
- [Contributing Guide](../../CONTRIBUTING.md)
- [Development Setup](DEVELOPMENT_SETUP.md)
- [Testing Guide](TESTING_GUIDE.md)