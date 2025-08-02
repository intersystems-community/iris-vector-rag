# IPM Installation Guide

## Installing via InterSystems Package Manager (IPM/ZPM)

This guide covers installing the RAG Templates package using InterSystems Package Manager (IPM) directly in your IRIS instance.

## Prerequisites

- InterSystems IRIS 2025.1 or later
- IPM/ZPM installed in your IRIS instance
- Internet access for downloading dependencies
- Python 3.11+ available on the system

## Installation Methods

### Method 1: From Package Manager Registry

```objectscript
// Install from IPM registry
zpm "install intersystems-iris-rag"
```

### Method 2: From GitHub Repository

```objectscript
// Install directly from GitHub
zpm "install https://github.com/intersystems-community/iris-rag-templates"
```

### Method 3: From Local Module

1. Clone the repository:
```bash
git clone https://github.com/intersystems-community/iris-rag-templates.git
cd iris-rag-templates
```

2. Install via IPM:
```objectscript
// In IRIS Terminal
zpm "load /path/to/iris-rag-templates/"
```

## Installation Parameters

The package supports several configuration parameters:

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `PYTHON_PATH` | Path to Python executable | `python3` | Any valid Python path |
| `INSTALL_PYTHON_PACKAGE` | Install Python dependencies | `1` | `0` (skip), `1` (install) |
| `ENABLE_VECTOR_SEARCH` | Enable IRIS Vector Search | `1` | `0` (disable), `1` (enable) |
| `NAMESPACE` | Target installation namespace | `USER` | Any valid namespace |

### Custom Installation with Parameters

```objectscript
// Install with custom parameters
zpm "install intersystems-iris-rag -DParameters=""PYTHON_PATH=/usr/local/bin/python3,NAMESPACE=MYRAG"""
```

## Post-Installation Configuration

### 1. Verify Installation

```objectscript
// Check installation status
Do ##class(RAG.IPMInstaller).Test()
```

### 2. Configure Python Environment

If automatic Python installation was skipped, manually configure:

```bash
# Navigate to installation directory
cd /path/to/iris-installation/

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Initialize Vector Search

```objectscript
// Enable vector search capabilities
Do ##class(RAG.VectorMigration).EnableVectorSearch()
```

### 4. Test RAG Functionality

```objectscript
// Test basic RAG functionality
Set bridge = ##class(RAG.PythonBridge).%New()
Set result = bridge.Query("What is machine learning?", "basic")
Write result.answer
```

## Python Integration

### Environment Setup

The package automatically configures Python integration, but you may need to verify:

```python
# Test Python package import
import iris_rag
from rag_templates import RAG

# Initialize RAG system
rag = RAG()
result = rag.query("test query")
print(result)
```

### Configuration Files

After installation, the following configuration files are available:

- `config/config.yaml` - Main configuration
- `config/pipelines.yaml` - Pipeline configurations
- `requirements.txt` - Python dependencies
- `pyproject.toml` - Package metadata

## Verification Steps

### 1. Database Schema Verification

```objectscript
// Check if RAG tables were created
SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = 'RAG'
```

### 2. Python Package Verification

```python
# Verify all RAG pipelines are available
from iris_rag.validation.factory import get_available_pipelines
pipelines = get_available_pipelines()
print(f"Available pipelines: {pipelines}")
```

### 3. ObjectScript Integration Verification

```objectscript
// Test ObjectScript-Python bridge
Set demo = ##class(RAGDemo.TestBed).%New()
Do demo.RunBasicTests()
```

## Troubleshooting

### Common Issues

**1. Python Import Errors**
```bash
# Ensure Python path is correct
which python3
pip list | grep sentence-transformers
```

**2. Vector Search Not Enabled**
```objectscript
// Enable vector search manually
Do ##class(RAG.VectorMigration).EnableVectorSearch()
```

**3. Missing Dependencies**
```bash
# Reinstall Python dependencies
pip install -r requirements.txt --force-reinstall
```

**4. Namespace Issues**
```objectscript
// Switch to correct namespace
zn "USER"
// Or your target namespace
zn "MYRAG"
```

### Diagnostic Commands

```objectscript
// Run comprehensive diagnostics
Do ##class(RAG.IPMInstaller).ValidateInstallation()

// Check system status
Do ##class(RAG.IPMInstaller).GetSystemStatus()

// Test individual components
Do ##class(RAG.IPMInstaller).TestPythonIntegration()
Do ##class(RAG.IPMInstaller).TestVectorSearch()
Do ##class(RAG.IPMInstaller).TestRAGPipelines()
```

## Uninstallation

To remove the package:

```objectscript
// Uninstall package
zpm "uninstall intersystems-iris-rag"
```

This will:
- Remove ObjectScript classes
- Clean up database schema (optional)
- Remove package metadata

Note: Python dependencies and configuration files may need manual cleanup.

## Advanced Configuration

### Custom Schema Installation

```objectscript
// Install to custom schema
Do ##class(RAG.IPMInstaller).SetParameter("CUSTOM_SCHEMA", "MyCompany")
Do ##class(RAG.IPMInstaller).Configure()
```

### Production Deployment

For production environments:

1. **Set Production Parameters**:
```objectscript
Do ##class(RAG.IPMInstaller).SetParameter("ENVIRONMENT", "PRODUCTION")
Do ##class(RAG.IPMInstaller).SetParameter("LOG_LEVEL", "WARNING")
```

2. **Configure Security**:
```objectscript
// Set up secure database connections
Do ##class(RAG.IPMInstaller).ConfigureProductionSecurity()
```

3. **Enable Monitoring**:
```objectscript
// Enable production monitoring
Do ##class(RAG.IPMInstaller).EnableMonitoring()
```

## Support and Documentation

- **Main Documentation**: [RAG Templates Documentation](../README.md)
- **Configuration Guide**: [Configuration Documentation](CONFIGURATION.md)
- **Troubleshooting**: [Deployment Guide](guides/DEPLOYMENT_GUIDE.md)
- **API Reference**: [Developer Guide](DEVELOPER_GUIDE.md)

## Version Compatibility

| RAG Templates Version | IRIS Version | Python Version | Notes |
|----------------------|--------------|----------------|-------|
| 0.2.0+ | 2025.1+ | 3.11+ | Full feature support |
| 0.1.x | 2024.1+ | 3.9+ | Limited vector search |

For older IRIS versions, consider manual installation following the [Deployment Guide](guides/DEPLOYMENT_GUIDE.md).