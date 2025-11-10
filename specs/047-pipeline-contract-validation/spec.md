# Feature Spec: Pipeline Contract Validation

**Feature ID**: 047
**Branch**: 047-pipeline-contract-validation
**Status**: Draft
**Created**: 2025-10-18
**Author**: System (prompted by integration test analysis)

---

## Overview

Implement automated validation of RAG pipeline implementations to ensure they adhere to the framework's API contract. This prevents runtime errors from API inconsistencies (like `query_text` vs `query`) and ensures all pipelines provide a consistent developer experience.

## Problem Statement

**Current Issues:**
1. **No API Enforcement**: Pipelines can use any method signatures (e.g., `query_text` vs `query`)
2. **Late Error Detection**: API inconsistencies discovered at runtime, not startup
3. **Inconsistent Responses**: No validation that pipelines return required fields
4. **Poor Developer Experience**: Developers must read source code to understand expected API

**Recent Example:**
- GraphRAG uses `query(query_text: str)`
- Other pipelines use `query(query: str)`
- MCP integration broke because technique handler expected `query` parameter
- Only caught during integration testing, not at registration time

## Goals

### Primary Goals
1. **Validate Pipeline API**: Check that pipelines implement required methods with correct signatures
2. **Validate Response Format**: Ensure pipelines return standardized response structure
3. **Early Detection**: Catch violations at pipeline registration, not runtime
4. **Clear Error Messages**: Provide actionable guidance when validation fails

### Secondary Goals
1. **Document Pipeline Contract**: Define official RAG pipeline API in code
2. **Support Evolution**: Allow contract to evolve while maintaining backward compatibility
3. **Enable Strict Mode**: Optional strict validation for production deployments

## Pipeline API Contract (Specification)

### Required Methods

All RAG pipelines MUST implement:

```python
class RAGPipeline(ABC):
    """Base class for all RAG pipelines."""

    @abstractmethod
    def query(
        self,
        query: str,              # REQUIRED: Standard parameter name
        top_k: int = 5,          # OPTIONAL: Number of documents to retrieve
        generate_answer: bool = True,  # OPTIONAL: Whether to generate LLM answer
        **kwargs                 # OPTIONAL: Pipeline-specific parameters
    ) -> Dict[str, Any]:
        """
        Execute RAG query with standardized API.

        Args:
            query: The query string (NOT query_text!)
            top_k: Number of documents to retrieve
            generate_answer: Whether to generate answer
            **kwargs: Additional pipeline-specific parameters

        Returns:
            Standardized response dictionary (see Response Contract below)
        """
        pass

    @abstractmethod
    def load_documents(
        self,
        documents: List[Document] = None,
        documents_path: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load documents into pipeline's knowledge base.

        Returns:
            Status dictionary with documents_loaded, embeddings_generated, etc.
        """
        pass
```

### Response Contract

All `query()` methods MUST return a dictionary with these fields:

```python
{
    # REQUIRED FIELDS
    "answer": str,                          # LLM-generated answer or None
    "retrieved_documents": List[Document],  # Retrieved documents (LangChain Document objects)
    "contexts": List[str],                  # String contexts for RAGAS evaluation
    "execution_time": float,                # Execution time in seconds
    "sources": List[Dict[str, Any]],        # Source references
    "metadata": {
        # REQUIRED METADATA FIELDS
        "num_retrieved": int,               # Number of documents retrieved
        "pipeline_type": str,               # Pipeline identifier (e.g., "basic_rag")
        "generated_answer": bool,           # Whether LLM answer was generated
        "processing_time": float,           # Processing time in seconds
        "retrieval_method": str,            # Retrieval method used (e.g., "vector", "hybrid")
        "context_count": int,               # Number of contexts provided

        # OPTIONAL METADATA FIELDS (pipeline-specific)
        "correction_applied": bool,         # CRAG: Whether correction was applied
        "retrieval_status": str,            # CRAG: Retrieval quality assessment
        "iris_graph_core_enabled": bool,    # GraphRAG: Whether graph core is available
        # ... additional pipeline-specific fields
    }
}
```

### Backward Compatibility Requirements

To support existing code:

1. **Deprecated Parameters**: Pipelines MAY accept deprecated parameter names (e.g., `query_text`) but MUST also accept the standard name (`query`)
2. **Parameter Aliases**: Use this pattern:
   ```python
   def query(self, query: str = None, query_text: str = None, **kwargs):
       # Support both standard and deprecated names
       if query is None and query_text is None:
           raise ValueError("'query' parameter is required")
       query = query if query is not None else query_text
       # ... rest of implementation
   ```

## Design

### PipelineValidator Class

```python
# iris_rag/core/validators.py (new file)

from typing import Dict, Any, List, Tuple, Type
from inspect import signature, Parameter
import logging

logger = logging.getLogger(__name__)


class PipelineContractViolation:
    """Represents a single contract violation."""

    def __init__(self, severity: str, message: str, suggestion: str = None):
        self.severity = severity  # 'error', 'warning', 'info'
        self.message = message
        self.suggestion = suggestion

    def __str__(self):
        msg = f"[{self.severity.upper()}] {self.message}"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


class PipelineValidator:
    """Validates RAG pipeline implementations against framework contract."""

    # Contract definition
    REQUIRED_METHODS = ['query', 'load_documents']

    QUERY_METHOD_CONTRACT = {
        'required_params': ['query'],
        'deprecated_params': ['query_text'],  # Allowed but discouraged
        'standard_optional_params': ['top_k', 'generate_answer', 'custom_prompt', 'include_sources'],
        'required_response_fields': ['answer', 'retrieved_documents', 'contexts', 'execution_time', 'metadata', 'sources'],
        'required_metadata_fields': ['num_retrieved', 'pipeline_type', 'generated_answer', 'processing_time', 'retrieval_method', 'context_count']
    }

    @classmethod
    def validate_pipeline_class(cls, pipeline_class: Type, pipeline_name: str) -> Tuple[bool, List[PipelineContractViolation]]:
        """
        Validate a pipeline class against the contract.

        Args:
            pipeline_class: The pipeline class to validate
            pipeline_name: Human-readable name for error messages

        Returns:
            (is_valid, violations)
        """
        violations = []

        # Check required methods exist
        for method_name in cls.REQUIRED_METHODS:
            if not hasattr(pipeline_class, method_name):
                violations.append(PipelineContractViolation(
                    'error',
                    f"Pipeline '{pipeline_name}' missing required method: {method_name}",
                    f"Add method: def {method_name}(self, ...)"
                ))

        # Validate query() method signature if it exists
        if hasattr(pipeline_class, 'query'):
            violations.extend(cls._validate_query_signature(pipeline_class.query, pipeline_name))

        # Determine if pipeline is valid (no errors, warnings OK)
        has_errors = any(v.severity == 'error' for v in violations)
        return (not has_errors, violations)

    @classmethod
    def _validate_query_signature(cls, query_method, pipeline_name: str) -> List[PipelineContractViolation]:
        """Validate query() method signature."""
        violations = []

        try:
            sig = signature(query_method)
            params = {name: param for name, param in sig.parameters.items() if name != 'self'}
            param_names = list(params.keys())

            # Check for required 'query' parameter
            has_query = 'query' in param_names
            has_query_text = 'query_text' in param_names

            if not has_query and not has_query_text:
                violations.append(PipelineContractViolation(
                    'error',
                    f"Pipeline '{pipeline_name}' query() method missing required 'query' parameter",
                    "Add parameter: query: str"
                ))
            elif not has_query and has_query_text:
                violations.append(PipelineContractViolation(
                    'warning',
                    f"Pipeline '{pipeline_name}' uses deprecated 'query_text' parameter instead of 'query'",
                    "Support both: def query(self, query: str = None, query_text: str = None, ...)"
                ))

            # Check that query parameter is first positional (after self)
            if has_query:
                query_param = params['query']
                # It's OK if query has a default (for backward compat), but warn if it's not early in signature
                param_position = param_names.index('query')
                if param_position > 2:  # Allow some flexibility
                    violations.append(PipelineContractViolation(
                        'info',
                        f"Pipeline '{pipeline_name}' has 'query' parameter at position {param_position}. Consider moving it earlier for clarity.",
                        "Standard: def query(self, query: str, top_k: int = 5, ...)"
                    ))

            # Check for **kwargs to accept pipeline-specific parameters
            has_kwargs = any(param.kind == Parameter.VAR_KEYWORD for param in params.values())
            if not has_kwargs:
                violations.append(PipelineContractViolation(
                    'warning',
                    f"Pipeline '{pipeline_name}' query() method lacks **kwargs parameter",
                    "Add **kwargs to accept optional parameters: def query(self, query: str, ..., **kwargs)"
                ))

        except Exception as e:
            violations.append(PipelineContractViolation(
                'error',
                f"Failed to inspect query() method signature for '{pipeline_name}': {e}"
            ))

        return violations

    @classmethod
    def validate_response(cls, response: Dict[str, Any], pipeline_name: str) -> Tuple[bool, List[PipelineContractViolation]]:
        """
        Validate a query response against the contract.

        This can be called at runtime to verify response format.
        """
        violations = []

        # Check required top-level fields
        for field in cls.QUERY_METHOD_CONTRACT['required_response_fields']:
            if field not in response:
                violations.append(PipelineContractViolation(
                    'error',
                    f"Pipeline '{pipeline_name}' response missing required field: '{field}'",
                    f"Add '{field}' to the response dictionary"
                ))

        # Check metadata structure
        if 'metadata' in response:
            metadata = response['metadata']
            if not isinstance(metadata, dict):
                violations.append(PipelineContractViolation(
                    'error',
                    f"Pipeline '{pipeline_name}' metadata must be a dict, got {type(metadata)}"
                ))
            else:
                # Check required metadata fields
                for field in cls.QUERY_METHOD_CONTRACT['required_metadata_fields']:
                    if field not in metadata:
                        violations.append(PipelineContractViolation(
                            'error',
                            f"Pipeline '{pipeline_name}' metadata missing required field: '{field}'",
                            f"Add metadata['{field}'] = ... to the response"
                        ))

        has_errors = any(v.severity == 'error' for v in violations)
        return (not has_errors, violations)
```

### Integration into TechniqueHandlerRegistry

```python
# iris_rag/mcp/technique_handlers.py (modified)

from iris_rag.core.validators import PipelineValidator

class TechniqueHandlerRegistry:
    """Registry of all technique handlers with contract validation."""

    def __init__(self, strict_mode: bool = False):
        self._handlers: Dict[str, TechniqueHandler] = {}
        self._strict_mode = strict_mode  # If True, refuse to register invalid pipelines
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register handlers for all 6 RAG pipelines with validation."""
        from iris_rag.pipelines.basic import BasicRAGPipeline
        from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
        from iris_rag.pipelines.crag import CRAGPipeline
        from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        # Register each pipeline with validation
        self._register_with_validation('basic', BasicRAGPipeline)
        self._register_with_validation('basic_rerank', BasicRAGRerankingPipeline)
        self._register_with_validation('crag', CRAGPipeline)
        self._register_with_validation('graphrag', HybridGraphRAGPipeline)

        # ... register other pipelines

    def _register_with_validation(self, technique: str, pipeline_class):
        """Register pipeline with contract validation."""
        # Validate pipeline class
        is_valid, violations = PipelineValidator.validate_pipeline_class(pipeline_class, technique)

        # Log violations
        if violations:
            logger.warning(f"Pipeline '{technique}' contract validation results:")
            for violation in violations:
                if violation.severity == 'error':
                    logger.error(str(violation))
                elif violation.severity == 'warning':
                    logger.warning(str(violation))
                else:
                    logger.info(str(violation))

        # In strict mode, refuse to register invalid pipelines
        if not is_valid and self._strict_mode:
            raise ValueError(
                f"Cannot register pipeline '{technique}' in strict mode. "
                f"Fix the following errors:\n" +
                "\n".join(str(v) for v in violations if v.severity == 'error')
            )

        # Register the handler
        self._handlers[technique] = TechniqueHandler(technique, pipeline_class)

        if is_valid:
            logger.info(f"✅ Pipeline '{technique}' registered successfully (passed validation)")
        else:
            logger.warning(f"⚠️  Pipeline '{technique}' registered with violations (non-strict mode)")
```

## Implementation Plan

### Phase 1: Core Validation Framework
1. Create `iris_rag/core/validators.py` with `PipelineValidator` class
2. Define complete API contract specification
3. Implement signature validation and response validation methods
4. Add comprehensive docstrings and examples

### Phase 2: Integration
1. Modify `TechniqueHandlerRegistry` to use validation
2. Add `strict_mode` configuration option
3. Update existing pipelines to fix violations (GraphRAG `query_text` → `query` support)
4. Log validation results at startup

### Phase 3: Testing
1. Create `tests/contract/test_pipeline_validator.py`
2. Test validation of compliant pipelines (should pass)
3. Test validation of non-compliant mock pipelines (should detect violations)
4. Test strict mode enforcement
5. Test response validation

### Phase 4: Documentation
1. Update `CLAUDE.md` with official Pipeline API Contract section
2. Document how to make pipelines contract-compliant
3. Add troubleshooting guide for common violations
4. Create migration guide for existing custom pipelines

## Testing Strategy

### Unit Tests (tests/contract/test_pipeline_validator.py)

```python
def test_validates_compliant_pipeline():
    """Validator accepts pipeline that follows contract."""
    class CompliantPipeline(RAGPipeline):
        def query(self, query: str, top_k: int = 5, **kwargs):
            return {
                'answer': 'test',
                'retrieved_documents': [],
                'contexts': [],
                'execution_time': 0.1,
                'sources': [],
                'metadata': {
                    'num_retrieved': 0,
                    'pipeline_type': 'test',
                    'generated_answer': True,
                    'processing_time': 0.1,
                    'retrieval_method': 'vector',
                    'context_count': 0
                }
            }

        def load_documents(self, documents=None, **kwargs):
            return {'documents_loaded': 0}

    is_valid, violations = PipelineValidator.validate_pipeline_class(
        CompliantPipeline, 'compliant'
    )

    assert is_valid
    assert len([v for v in violations if v.severity == 'error']) == 0


def test_detects_missing_query_parameter():
    """Validator detects missing 'query' parameter."""
    class NonCompliantPipeline(RAGPipeline):
        def query(self, search_text: str, **kwargs):  # Wrong parameter name!
            return {}

        def load_documents(self, documents=None, **kwargs):
            return {}

    is_valid, violations = PipelineValidator.validate_pipeline_class(
        NonCompliantPipeline, 'non_compliant'
    )

    assert not is_valid
    assert any('query' in str(v) for v in violations)


def test_warns_about_query_text_deprecation():
    """Validator warns about deprecated query_text parameter."""
    class DeprecatedPipeline(RAGPipeline):
        def query(self, query_text: str, **kwargs):  # Deprecated!
            return {}

        def load_documents(self, documents=None, **kwargs):
            return {}

    is_valid, violations = PipelineValidator.validate_pipeline_class(
        DeprecatedPipeline, 'deprecated'
    )

    # Should have warnings but still be valid
    warnings = [v for v in violations if v.severity == 'warning']
    assert len(warnings) > 0
    assert any('query_text' in str(v) and 'deprecated' in str(v).lower() for v in warnings)
```

### Integration Tests

```python
def test_registry_validates_pipelines_on_startup():
    """Registry validates all pipelines during initialization."""
    # Should log validation results but not crash
    registry = TechniqueHandlerRegistry(strict_mode=False)

    # All default pipelines should be registered
    assert 'basic' in registry.list_techniques()
    assert 'graphrag' in registry.list_techniques()


def test_strict_mode_refuses_invalid_pipelines():
    """Strict mode prevents registration of non-compliant pipelines."""
    class BadPipeline:
        def query(self):  # Missing 'query' parameter!
            return {}

    registry = TechniqueHandlerRegistry(strict_mode=True)

    with pytest.raises(ValueError, match="strict mode"):
        registry._register_with_validation('bad', BadPipeline)
```

## Configuration

Add to `iris_rag/config/default_config.yaml`:

```yaml
validation:
  # Enable pipeline contract validation
  enabled: true

  # Strict mode: refuse to register non-compliant pipelines
  strict_mode: false

  # Validate responses at runtime (performance impact)
  validate_responses: false

  # Log level for validation messages
  log_level: "WARNING"  # ERROR, WARNING, INFO, DEBUG
```

## Success Metrics

1. **All 6 default pipelines pass validation** without errors
2. **GraphRAG updated** to support both `query` and `query_text` parameters
3. **Validation catches violations** in test suite (proven by tests)
4. **Zero runtime failures** from API inconsistencies after deployment
5. **Documentation complete** with examples of compliant pipelines

## Future Enhancements

1. **Automated Contract Tests**: Generate contract tests from validator spec
2. **Plugin Validation**: Validate third-party pipelines before loading
3. **Version Compatibility**: Track contract version, warn about breaking changes
4. **Performance Contracts**: Validate p95 latency, token usage, etc.
5. **OpenAPI Generation**: Auto-generate OpenAPI spec from contract

## References

- Current issue: GraphRAG uses `query_text` vs standard `query` parameter
- Related: Feature 043 (Complete MCP Tools Implementation)
- Inspired by: FastAPI's automatic validation, Pydantic models

---

**Next Steps:**
1. Review and approve this spec
2. Create feature branch: `047-pipeline-contract-validation`
3. Implement Phase 1 (Core Validation Framework)
4. Update GraphRAG to be fully compliant
5. Add comprehensive tests
