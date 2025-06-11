# IPM Installation Guide

This guide covers installing the `intersystems-iris-rag` package using InterSystems Package Manager (IPM/ZPM).

## Overview

The `intersystems-iris-rag` package can be installed in two ways:

1. **Python Package Installation**: `pip install intersystems-iris-rag` (for Python developers)
2. **IPM Installation**: `zpm "install intersystems-iris-rag"` (for ObjectScript developers)

Both methods result in a fully functional RAG system with automated setup.

## Prerequisites

### System Requirements
- InterSystems IRIS 2025.1 or higher
- Python 3.11 or higher
- 2GB+ available memory
- Internet connection for package downloads

### IRIS Configuration
- Vector Search must be available (automatically enabled during installation)
- Embedded Python must be configured
- Sufficient database space for vector storage

## IPM Installation

### Quick Installation

```objectscript
// Install the package
zpm "install intersystems-iris-rag"

// Verify installation
do ##class(RAG.IPMInstaller).DisplayInfo()

// Run system test
do ##class(RAG.PythonBridge).RunSystemTest()
```

### Custom Installation Parameters

```objectscript
// Install with custom parameters
zpm "install intersystems-iris-rag -DNamespace=RAGNS -DPythonPath=/usr/bin/python3.11"

// Install without Python package (if already installed)
zpm "install intersystems-iris-rag -DInstallPythonPackage=false"

// Install with sample data
zpm "install intersystems-iris-rag -DCreateSampleData=true"
```

### Available Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `PYTHON_PATH` | Path to Python executable | `python3` | Any valid Python path |
| `INSTALL_PYTHON_PACKAGE` | Install Python package via pip | `true` | `true`, `false` |
| `ENABLE_VECTOR_SEARCH` | Enable IRIS Vector Search | `true` | `true`, `false` |
| `CREATE_SAMPLE_DATA` | Create sample data for testing | `false` | `true`, `false` |
| `NAMESPACE` | Target namespace for installation | `USER` | Any valid namespace |
| `PYTHON_ENVIRONMENT` | Python virtual environment path | `""` | Path to venv |
| `SKIP_DEPENDENCIES` | Skip dependency validation | `false` | `true`, `false` |

## Installation Process

The IPM installation follows these phases:

### 1. Setup Phase
- Validates IRIS version compatibility
- Checks Python environment
- Enables Vector Search if requested
- Validates system requirements

### 2. Configure Phase
- Installs Python package via pip
- Creates database schema
- Configures vector search tables
- Sets up HNSW indexes

### 3. Activate Phase
- Tests Python integration
- Creates sample data (if requested)
- Runs integration tests
- Validates complete installation

### 4. Test Phase
- Executes comprehensive functionality tests
- Validates all components are working
- Reports any issues found

## Post-Installation Verification

### Basic Verification

```objectscript
// Check installation status
do ##class(RAG.IPMInstaller).DisplayInfo()

// Test Python integration
do ##class(RAG.PythonBridge).TestPythonIntegration()

// Validate environment
do ##class(RAG.PythonBridge).ValidatePythonEnvironment()
```

### Comprehensive Testing

```objectscript
// Run full system test
do ##class(RAG.PythonBridge).RunSystemTest()

// Demonstrate RAG functionality
do ##class(RAG.PythonBridge).DemoRAGFunctionality()
```

### Database Verification

```sql
-- Check tables were created
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = 'RAG'

-- Check vector search is enabled
SELECT %SYSTEM_SQL.GetVectorSearchEnabled()

-- Verify sample data (if created)
SELECT COUNT(*) FROM RAG.SourceDocuments
```

## Configuration

### Generate Configuration Template

```objectscript
// Generate default configuration
do ##class(RAG.PythonBridge).GenerateConfigTemplate("config.yaml")
```

### Sample Configuration

```yaml
database:
  iris:
    host: localhost
    port: 1972
    namespace: USER
    username: demo
    password: demo
    driver: intersystems_iris.dbapi

embeddings:
  primary_backend: sentence_transformers
  sentence_transformers:
    model_name: all-MiniLM-L6-v2
  dimension: 384

pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5

llm:
  provider: openai
  model_name: gpt-3.5-turbo
  api_key: ${OPENAI_API_KEY}
  temperature: 0.0
  max_tokens: 1000
```

## Usage Examples

### Basic RAG Pipeline

```objectscript
// Create and test basic pipeline
do ##class(RAG.PythonBridge).CreateBasicPipeline("config.yaml")

// Load documents
do ##class(RAG.PythonBridge).LoadDocuments("/path/to/documents", "config.yaml")

// Execute query
write ##class(RAG.PythonBridge).ExecuteRAGQuery("What is machine learning?", "config.yaml")
```

### Python Integration

```python
# After IPM installation, use from Python
from iris_rag import create_pipeline

# Create pipeline
pipeline = create_pipeline("basic", config_path="config.yaml")

# Load documents
pipeline.load_documents("./documents")

# Execute query
result = pipeline.execute("What is machine learning?")
print(result["answer"])
```

## Troubleshooting

### Common Issues

#### Python Package Installation Fails
```objectscript
// Check Python environment
do ##class(RAG.PythonBridge).ValidatePythonEnvironment()

// Manual installation
do ##class(RAG.IPMInstaller).InstallPythonPackage()
```

#### Vector Search Not Available
```objectscript
// Enable vector search manually
do ##class(RAG.IPMInstaller).EnableVectorSearch()
```

#### Database Schema Issues
```objectscript
// Recreate schema
do ##class(RAG.IPMInstaller).CreateDatabaseSchema()

// Configure vector tables
do ##class(RAG.IPMInstaller).ConfigureVectorTables()
```

### Diagnostic Commands

```objectscript
// Get detailed installation info
do ##class(RAG.PythonBridge).GetInstallationInfo()

// Check package version
write ##class(RAG.IPMInstaller).GetPackageVersion()

// Validate installation
write ##class(RAG.IPMInstaller).GetInstallationStatus()
```

### Log Analysis

Check IRIS messages log for installation details:
```objectscript
// View recent messages
do ##class(%SYS.System).WriteToConsoleLog("Checking RAG installation...")
```

## Upgrading

### Upgrade via IPM

```objectscript
// Upgrade to latest version
zpm "upgrade intersystems-iris-rag"

// Force reinstall
zpm "uninstall intersystems-iris-rag"
zpm "install intersystems-iris-rag"
```

### Manual Upgrade

```objectscript
// Upgrade Python package
do ##class(RAG.IPMInstaller).UpgradePythonPackage()

// Update database schema
do ##class(RAG.IPMInstaller).UpdateDatabaseSchema()
```

## Uninstallation

### Complete Removal

```objectscript
// Uninstall via IPM (preserves data)
zpm "uninstall intersystems-iris-rag"

// Manual cleanup (removes all data)
do ##class(RAG.IPMInstaller).RemoveDatabaseObjects()
```

### Backup Before Removal

```objectscript
// Create backup
do ##class(RAG.IPMInstaller).BackupBeforeUninstall()
```

## Support

### Getting Help

1. **Documentation**: Check the [main documentation](README.md)
2. **Issues**: Report issues on GitHub
3. **Community**: Join InterSystems Developer Community

### Diagnostic Information

When reporting issues, include:

```objectscript
// Generate diagnostic report
do ##class(RAG.PythonBridge).GetInstallationInfo()
do ##class(RAG.IPMInstaller).DisplayInfo()
```

## Advanced Configuration

### Custom Python Environment

```objectscript
// Install with custom Python environment
zpm "install intersystems-iris-rag -DPythonEnvironment=/path/to/venv"
```

### Enterprise Deployment

```objectscript
// Install in production namespace
zpm "install intersystems-iris-rag -DNamespace=PRODUCTION -DCreateSampleData=false"
```

### Development Setup

```objectscript
// Install with all development features
zpm "install intersystems-iris-rag -DCreateSampleData=true -DSkipDependencies=false"
```

---

For more information, see the [main README](README.md) and [API documentation](API_REFERENCE.md).