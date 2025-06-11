# Python Naming Conventions for RAG Templates

## Overview

This document outlines the Python naming conventions used in the `iris_rag` package, based on analysis of the current codebase and standard Python practices. While we attempted to search InterSystems internal Confluence documentation for specific naming guidelines, the MCP Confluence tool was not accessible during this analysis.

## Current Package Structure Analysis

### Package Naming
- **Main Package**: [`iris_rag`](iris_rag/__init__.py:1) - Uses snake_case with descriptive naming
- **PyPI Package Name**: [`intersystems-iris-rag`](pyproject.toml:2) - Uses kebab-case for distribution

### Module Organization
The package follows a clear hierarchical structure:

```
iris_rag/
├── core/           # Core functionality
├── config/         # Configuration management
├── pipelines/      # Pipeline implementations
├── storage/        # Storage adapters
├── embeddings/     # Embedding management
├── adapters/       # External adapters
├── services/       # Business services
└── utils/          # Utility functions
```

## Naming Convention Standards

### 1. Package and Module Names
- **Format**: `snake_case`
- **Examples**:
  - [`iris_rag`](iris_rag/__init__.py:1)
  - [`config.manager`](iris_rag/config/manager.py:1)
  - [`core.base`](iris_rag/core/base.py:1)

### 2. Class Names
- **Format**: `PascalCase`
- **Examples**:
  - [`RAGPipeline`](rag_templates/core/base.py:3) - Abstract base class
  - [`ConnectionManager`](rag_templates/core/connection.py:1) - Service class
  - [`ConfigurationManager`](rag_templates/config/manager.py:10) - Manager class
  - [`Document`](rag_templates/core/models.py:10) - Data model

### 3. Function and Method Names
- **Format**: `snake_case`
- **Examples**:
  - [`create_pipeline()`](rag_templates/__init__.py:10) - Factory function
  - [`load_documents()`](rag_templates/core/base.py:32) - Abstract method
  - [`default_id_factory()`](rag_templates/core/models.py:5) - Utility function

### 4. Variable Names
- **Format**: `snake_case`
- **Examples**:
  - [`query_text`](rag_templates/core/base.py:13) - Parameter names
  - [`page_content`](rag_templates/core/models.py:21) - Attribute names
  - [`config_path`](rag_templates/config/manager.py:22) - Local variables

### 5. Constants
- **Format**: `UPPER_SNAKE_CASE`
- **Examples**:
  - [`ENV_PREFIX`](rag_templates/config/manager.py:19) - Class constants
  - [`DELIMITER`](rag_templates/config/manager.py:20) - Configuration constants

### 6. Exception Classes
- **Format**: `PascalCase` ending with `Error`
- **Examples**:
  - [`ConfigValidationError`](rag_templates/config/manager.py:6) - Custom exception

## Parameter Naming Standards

Based on the [`.clinerules`](.clinerules:1) file, the project enforces consistent parameter naming:

### Database Connections
- **Use**: `iris_connector` 
- **Avoid**: `connection`

### Function Parameters
- **Use**: `embedding_func`
- **Avoid**: `embed_func`
- **Use**: `llm_func` (consistent across all pipelines)

### Standard Return Format
All pipelines must return a dictionary with standardized keys:
- `"query"`: The original query
- `"answer"`: The generated answer  
- `"retrieved_documents"`: The documents/nodes used

## File and Directory Naming

### Python Files
- **Format**: `snake_case.py`
- **Examples**: [`base.py`](rag_templates/core/base.py:1), [`manager.py`](rag_templates/config/manager.py:1)

### Test Files
- **Format**: `test_*.py`
- **Examples**: [`test_base.py`](tests/test_core/test_base.py:1), [`test_connection.py`](tests/test_core/test_connection.py:1)

### Documentation Files
- **Format**: `UPPER_SNAKE_CASE.md`
- **Examples**: [`README.md`](README.md:1), [`API_REFERENCE.md`](docs/API_REFERENCE.md:1)

## Import Conventions

### Package Imports
```python
# Absolute imports from package root
from iris_rag.core.base import RAGPipeline
from iris_rag.config.manager import ConfigurationManager

# Relative imports within package
from .base import RAGPipeline
from .connection import ConnectionManager
```

### __all__ Declarations
Each module should define [`__all__`](rag_templates/__init__.py:43) to control public API:
```python
__all__ = ["create_pipeline", "RAGPipeline", "ConnectionManager"]
```

## Validation Against Current Implementation

### ✅ Compliant Patterns
- Package structure follows Python standards
- Class names use PascalCase consistently
- Function names use snake_case
- Constants use UPPER_SNAKE_CASE
- Parameter naming follows project rules

### ⚠️ Areas for Review
- Ensure all new modules follow the established patterns
- Verify parameter naming consistency across new RAG techniques
- Maintain standard return format across all pipelines

## Confluence Search Results

**Note**: During this analysis, we attempted to search InterSystems internal Confluence documentation using the MCP Atlassian tool for specific Python naming conventions. However, the tool was not accessible despite the MCP server being active. 

**Confluence URLs**: Not available due to tool access issues
**InterSystems Guidelines**: Could not be retrieved

## Recommendations

1. **Continue Current Patterns**: The existing naming conventions align with Python PEP 8 standards
2. **Document Deviations**: Any InterSystems-specific requirements should be documented when available
3. **Enforce Consistency**: Use linting tools to maintain naming consistency
4. **Update Documentation**: When Confluence access is restored, update this document with official guidelines

## Related Documentation

- [API Reference](docs/API_REFERENCE.md:1) - Complete API documentation
- [Project Rules](.clinerules:1) - Development standards and rules
- [Package Configuration](pyproject.toml:1) - Build and dependency configuration

---

*Last Updated: 2025-06-07*  
*Status: Based on codebase analysis - Confluence search pending*