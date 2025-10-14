# Example Enhancement Guide

Best practices and guidelines for creating, maintaining, and enhancing examples and demonstrations in the RAG-Templates framework.

## Table of Contents

1. [Overview](#overview)
2. [Example Categories](#example-categories)
3. [Development Standards](#development-standards)
4. [Testing Integration](#testing-integration)
5. [Documentation Requirements](#documentation-requirements)
6. [Performance Guidelines](#performance-guidelines)
7. [CI/CD Integration](#cicd-integration)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting](#troubleshooting)

## Overview

Examples serve as the primary interface between the RAG-Templates framework and its users. Well-crafted examples demonstrate capabilities, provide learning pathways, and validate framework functionality across different use cases.

### Design Principles

1. **Clarity**: Examples should be immediately understandable
2. **Completeness**: Cover the full workflow from setup to results
3. **Robustness**: Handle errors gracefully and provide meaningful feedback
4. **Performance**: Execute efficiently and respect resource constraints
5. **Testability**: Support both mock and real execution modes

## Example Categories

### Basic RAG Examples (`scripts/basic/`)

**Purpose**: Demonstrate fundamental RAG operations
**Target Audience**: New users, integration testing
**Complexity**: Low to Medium

#### Guidelines:
- Focus on single RAG pipeline functionality
- Use clear, descriptive variable names
- Include comprehensive error handling
- Provide sample queries and expected outputs
- Keep execution time under 3 minutes

#### Template Structure:
```python
#!/usr/bin/env python3
"""
Basic RAG Pipeline Example

Demonstrates standard vector similarity search and LLM generation
using the iris_rag framework with sample medical documents.

Usage:
    python try_basic_rag_pipeline.py

Requirements:
    - iris_rag package
    - LLM API key (or mock mode)
    - IRIS database (or mock storage)
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_rag import create_pipeline
from common.utils import get_llm_func

def main():
    """Main example execution function."""
    print("🔍 Basic RAG Pipeline Example")
    print("=" * 50)

    try:
        # Configuration
        use_mock = os.getenv('USE_MOCK_LLM', 'true').lower() == 'true'

        # Create pipeline
        pipeline = create_pipeline(
            pipeline_type="basic",
            validate_requirements=True,
            auto_setup=True
        )

        # Configure LLM
        if not use_mock:
            pipeline.llm_func = get_llm_func("openai", "gpt-4o-mini")

        # Test queries
        test_queries = [
            "What is diabetes?",
            "How does insulin work?",
            "What are the symptoms of COVID-19?"
        ]

        # Execute queries
        results = []
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: {query}")

            result = pipeline.query(query, generate_answer=True)

            # Validate result
            if result.get('answer'):
                print(f"✅ Answer: {result['answer'][:100]}...")
                print(f"📚 Sources: {len(result.get('contexts', []))} documents")
                results.append(result)
            else:
                print("❌ No answer generated")

        # Summary
        print(f"\n📊 Summary:")
        print(f"   Queries processed: {len(results)}")
        print(f"   Average answer length: {sum(len(r['answer']) for r in results) // len(results)}")
        print("✅ Basic RAG pipeline demonstration completed")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### Advanced RAG Examples (`scripts/crag/`, `scripts/reranking/`)

**Purpose**: Showcase sophisticated RAG techniques
**Target Audience**: Advanced users, researchers
**Complexity**: Medium to High

#### Guidelines:
- Demonstrate multiple retrieval strategies
- Include performance comparisons
- Show error correction and quality assessment
- Provide detailed metrics and analysis
- Support configurable parameters

#### Advanced Features:
- Relevance evaluation and correction
- Multi-step reasoning
- Graph-based retrieval
- Hybrid search strategies
- Quality scoring and ranking

### Demonstration Scripts (`demo_*.py`)

**Purpose**: Interactive showcases and visualizations
**Target Audience**: Presentations, educational content
**Complexity**: Variable

#### Guidelines:
- Emphasize visual and interactive elements
- Generate exportable artifacts (HTML, JSON, images)
- Include step-by-step explanations
- Support both headless and interactive modes
- Provide comprehensive output validation

## Development Standards

### Code Quality

#### Import Management
```python
# Standard library imports first
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

# Third-party imports
import numpy as np
import pandas as pd

# Project imports
from iris_rag import create_pipeline
from common.utils import get_llm_func
```

#### Error Handling
```python
def robust_pipeline_creation(pipeline_type: str) -> Pipeline:
    """Create pipeline with comprehensive error handling."""
    try:
        pipeline = create_pipeline(
            pipeline_type=pipeline_type,
            validate_requirements=True,
            auto_setup=True
        )
        return pipeline
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Try: pip install -r requirements.txt")
        sys.exit(1)
    except ConnectionError as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Try: docker-compose up -d")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        sys.exit(1)
```

#### Configuration Management
```python
class ExampleConfig:
    """Centralized configuration for examples."""

    def __init__(self):
        self.use_mock = os.getenv('USE_MOCK_LLM', 'true').lower() == 'true'
        self.timeout = int(os.getenv('EXAMPLE_TIMEOUT', '300'))
        self.max_docs = int(os.getenv('MAX_DOCUMENTS', '10'))
        self.llm_model = os.getenv('LLM_MODEL', 'gpt-4o-mini')

    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not self.use_mock and not os.getenv('OPENAI_API_KEY'):
            print("❌ OPENAI_API_KEY required for real mode")
            return False
        return True
```

### Performance Optimization

#### Resource Management
```python
import psutil
from contextlib import contextmanager

@contextmanager
def resource_monitor(name: str):
    """Monitor resource usage during execution."""
    process = psutil.Process()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()

    print(f"🔄 Starting {name}...")

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        print(f"✅ {name} completed:")
        print(f"   Duration: {end_time - start_time:.2f}s")
        print(f"   Memory: {end_memory:.1f}MB (delta: {end_memory - start_memory:+.1f}MB)")

# Usage
with resource_monitor("Pipeline Creation"):
    pipeline = create_pipeline("basic")
```

#### Efficient Data Handling
```python
def load_sample_data(limit: int = 100) -> List[Document]:
    """Load sample data with memory efficiency."""
    documents = []

    # Use generators for large datasets
    for doc_data in document_generator(limit=limit):
        if len(documents) >= limit:
            break
        documents.append(Document(**doc_data))

    return documents
```

## Testing Integration

### Mock Mode Support

Every example must support mock mode for testing:

```python
def get_test_pipeline(pipeline_type: str, use_mock: bool = True):
    """Get pipeline configured for testing."""
    pipeline = create_pipeline(pipeline_type)

    if use_mock:
        # Use mock LLM provider
        from scripts.testing.mock_providers import MockLLMProvider
        mock_llm = MockLLMProvider(mode="realistic")
        pipeline.llm_func = mock_llm.generate_response

    return pipeline
```

### Test Configuration

Add your example to `scripts/testing/config.yaml`:

```yaml
examples:
  "your_category/your_example.py":
    timeout: 240
    expected_outputs: ["answer", "sources", "metadata"]
    test_queries:
      - "Sample query 1"
      - "Sample query 2"
    performance_bounds:
      max_execution_time: 180
      max_memory_mb: 512
    features: ["your_feature_tags"]
    validation_rules:
      answer_min_length: 50
      sources_min_count: 1
```

### Output Validation

Structure outputs for automatic validation:

```python
def generate_example_output(results: List[Dict]) -> Dict:
    """Generate standardized output for validation."""
    return {
        "status": "success",
        "timestamp": time.time(),
        "results": results,
        "metadata": {
            "total_queries": len(results),
            "average_response_time": calculate_avg_time(results),
            "pipeline_type": "basic"
        },
        "summary": {
            "successful_queries": len([r for r in results if r.get('answer')]),
            "total_sources": sum(len(r.get('contexts', [])) for r in results),
            "avg_answer_length": calculate_avg_length(results)
        }
    }
```

## Documentation Requirements

### Docstring Standards

```python
def example_function(query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Process a query using the configured RAG pipeline.

    This function demonstrates the standard RAG workflow including
    document retrieval, relevance scoring, and answer generation.

    Args:
        query: The user's question or search query
        top_k: Maximum number of documents to retrieve (default: 5)

    Returns:
        Dictionary containing:
        - answer: Generated response string
        - contexts: List of retrieved document chunks
        - sources: List of source document metadata
        - metadata: Pipeline execution metadata

    Raises:
        ValueError: If query is empty or invalid
        ConnectionError: If database connection fails

    Example:
        >>> result = example_function("What is diabetes?")
        >>> print(result['answer'])
        "Diabetes is a chronic condition..."
    """
```

### README Requirements

Each example directory should include a README.md:

```markdown
# Category Examples

Brief description of the examples in this category.

## Examples

### `try_example_name.py`
**Purpose**: Brief description
**Complexity**: Low/Medium/High
**Runtime**: ~X minutes
**Dependencies**: List key requirements

**Usage**:
```bash
python try_example_name.py
```

**Expected Output**: Description of what users should see

## Configuration

Environment variables and configuration options.

## Troubleshooting

Common issues and solutions.
```

## Performance Guidelines

### Execution Time Targets

- **Basic examples**: < 3 minutes (mock), < 10 minutes (real)
- **Advanced examples**: < 5 minutes (mock), < 15 minutes (real)
- **Demonstration scripts**: < 10 minutes (mock), < 30 minutes (real)

### Memory Usage

- **Basic examples**: < 512MB peak memory
- **Advanced examples**: < 1GB peak memory
- **Demonstration scripts**: < 2GB peak memory

### Optimization Techniques

#### Lazy Loading
```python
def load_documents_lazy(source_dir: Path):
    """Load documents on-demand to reduce memory usage."""
    for file_path in source_dir.glob("*.txt"):
        yield Document.from_file(file_path)
```

#### Batch Processing
```python
def process_queries_batch(queries: List[str], batch_size: int = 5):
    """Process queries in batches to manage resources."""
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        yield [pipeline.query(q) for q in batch]
```

#### Resource Cleanup
```python
def cleanup_resources(pipeline):
    """Clean up resources after example completion."""
    if hasattr(pipeline, 'vector_store'):
        pipeline.vector_store.close()
    if hasattr(pipeline, 'connection'):
        pipeline.connection.close()
```

## CI/CD Integration

### Environment Detection

```python
def detect_environment() -> str:
    """Detect execution environment for appropriate configuration."""
    if os.getenv('GITHUB_ACTIONS'):
        return 'github_ci'
    elif os.getenv('GITLAB_CI'):
        return 'gitlab_ci'
    elif os.getenv('EXAMPLE_TEST_MODE'):
        return 'testing'
    else:
        return 'development'
```

### CI-Friendly Output

```python
def print_ci_friendly(message: str, level: str = "info"):
    """Print messages in CI-friendly format."""
    timestamp = time.strftime("%H:%M:%S")

    if os.getenv('GITHUB_ACTIONS'):
        if level == "error":
            print(f"::error::[{timestamp}] {message}")
        elif level == "warning":
            print(f"::warning::[{timestamp}] {message}")
        else:
            print(f"[{timestamp}] {message}")
    else:
        # Local development format
        level_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(level, "ℹ️")
        print(f"{level_emoji} [{timestamp}] {message}")
```

## Common Patterns

### Standard Example Structure

```python
#!/usr/bin/env python3
"""
Example Title

Brief description of what this example demonstrates.
"""

import os
import sys
import time
from pathlib import Path

# Project imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_rag import create_pipeline
from common.utils import get_llm_func

class ExampleRunner:
    """Encapsulate example logic for better organization."""

    def __init__(self, config: dict):
        self.config = config
        self.pipeline = None
        self.results = []

    def setup(self):
        """Initialize pipeline and resources."""
        pass

    def run_queries(self, queries: List[str]):
        """Execute test queries."""
        pass

    def generate_report(self):
        """Generate summary report."""
        pass

    def cleanup(self):
        """Clean up resources."""
        pass

def main():
    """Main execution function."""
    config = load_configuration()
    runner = ExampleRunner(config)

    try:
        runner.setup()
        runner.run_queries(config['test_queries'])
        runner.generate_report()
    except Exception as e:
        print(f"❌ Example failed: {e}")
        sys.exit(1)
    finally:
        runner.cleanup()

if __name__ == "__main__":
    main()
```

### Configuration Loading

```python
def load_configuration() -> dict:
    """Load example configuration from multiple sources."""
    config = {
        'use_mock': os.getenv('USE_MOCK_LLM', 'true').lower() == 'true',
        'timeout': int(os.getenv('TIMEOUT', '300')),
        'verbose': os.getenv('VERBOSE', 'false').lower() == 'true',
    }

    # Load from config file if available
    config_file = Path(__file__).parent / 'config.yaml'
    if config_file.exists():
        import yaml
        with open(config_file) as f:
            file_config = yaml.safe_load(f)
        config.update(file_config)

    return config
```

### Progress Reporting

```python
from tqdm import tqdm
import sys

def process_with_progress(items: List, description: str):
    """Process items with progress bar."""
    if os.getenv('CI') or not sys.stdout.isatty():
        # CI environment - simple text progress
        total = len(items)
        for i, item in enumerate(items, 1):
            print(f"[{i}/{total}] Processing {description}...")
            yield item
    else:
        # Interactive environment - progress bar
        for item in tqdm(items, desc=description):
            yield item
```

## Troubleshooting

### Common Issues

#### Import Errors
```python
# Solution: Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
```

#### Database Connection Issues
```python
def check_database_connection():
    """Verify database connectivity before starting."""
    try:
        from iris_rag.core.connection import ConnectionManager
        config = ConfigurationManager()
        conn_mgr = ConnectionManager(config)
        conn = conn_mgr.get_connection()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        print("💡 Try: docker-compose up -d")
        return False
```

#### API Key Issues
```python
def validate_api_keys():
    """Check required API keys are available."""
    required_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys and not os.getenv('USE_MOCK_LLM', '').lower() == 'true':
        print(f"❌ Missing API keys: {', '.join(missing_keys)}")
        print("💡 Set USE_MOCK_LLM=true for testing without APIs")
        return False
    return True
```

### Debug Mode

```python
def enable_debug_mode():
    """Enable comprehensive debugging output."""
    import logging

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Enable detailed HTTP logging
    import urllib3
    urllib3.disable_warnings()

    # Enable framework debug mode
    os.environ['IRIS_RAG_DEBUG'] = 'true'
```

### Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

def profile_example(func):
    """Decorator to profile example execution."""
    def wrapper(*args, **kwargs):
        if os.getenv('PROFILE_EXAMPLE'):
            pr = cProfile.Profile()
            pr.enable()
            result = func(*args, **kwargs)
            pr.disable()

            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats()

            with open('example_profile.txt', 'w') as f:
                f.write(s.getvalue())

            return result
        else:
            return func(*args, **kwargs)
    return wrapper
```

## Best Practices Summary

1. **Always support mock mode** for testing and CI
2. **Handle errors gracefully** with helpful error messages
3. **Provide progress feedback** for long-running operations
4. **Include comprehensive documentation** and examples
5. **Test across different environments** and configurations
6. **Monitor resource usage** and optimize performance
7. **Structure output** for automatic validation
8. **Follow naming conventions** and coding standards
9. **Use configuration management** for flexibility
10. **Clean up resources** properly after execution

By following these guidelines, examples will be robust, maintainable, and provide excellent user experiences while supporting comprehensive testing and validation.