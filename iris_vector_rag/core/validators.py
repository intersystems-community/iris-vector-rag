"""
Pipeline Contract Validation Framework

This module provides validation for RAG pipeline implementations to ensure
they adhere to the framework contract defined in iris_rag.core.base.RAGPipeline.

The validator checks:
1. Required method signatures (query, load_documents)
2. Response format compliance (standardized fields)
3. Metadata completeness (all required metadata fields)
4. Backward compatibility (deprecated parameter support)

Usage:
    from iris_vector_rag.core.validators import PipelineValidator

    validator = PipelineValidator()

    # Validate pipeline class
    violations = validator.validate_pipeline_class(MyPipeline)
    if violations:
        for violation in violations:
            print(f"{violation.severity}: {violation.message}")

    # Validate response
    violations = validator.validate_response(response_dict, pipeline_name="basic")
    if violations:
        print("Response validation failed!")
"""

import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, get_type_hints

from iris_vector_rag.core.base import RAGPipeline

logger = logging.getLogger(__name__)


class ViolationSeverity(Enum):
    """Severity levels for contract violations."""

    ERROR = "error"  # Critical violation - pipeline cannot be used
    WARNING = "warning"  # Non-critical violation - pipeline may work but violates best practices
    INFO = "info"  # Informational - deprecated patterns or suggestions


@dataclass
class PipelineContractViolation:
    """Represents a violation of the pipeline contract."""

    severity: ViolationSeverity
    category: str  # e.g., "method_signature", "response_format", "metadata"
    message: str
    location: str  # e.g., "MyPipeline.query", "response.metadata"
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        result = f"[{self.severity.value.upper()}] {self.category} - {self.location}: {self.message}"
        if self.suggestion:
            result += f"\n  Suggestion: {self.suggestion}"
        return result


class PipelineValidator:
    """
    Validates RAG pipeline implementations against framework contract.

    The validator ensures pipelines conform to the standardized API defined
    in RAGPipeline base class, enabling:
    - Consistent user experience across all pipelines
    - LangChain & RAGAS compatibility
    - Automated testing and validation
    - Clear error messages for contract violations

    Contract Definition:

    1. Required Methods:
       - query(query: str, top_k: int = 20, **kwargs) -> Dict[str, Any]
       - load_documents(documents_path: str = "", documents: List[Document] = None, **kwargs) -> None

    2. Query Response Format:
       - answer: str (LLM-generated answer)
       - retrieved_documents: List[Document] (retrieved context)
       - contexts: List[str] (text content for RAGAS)
       - sources: List[str] (source references)
       - execution_time: float (total time in seconds)
       - metadata: Dict[str, Any] (pipeline-specific metadata)

    3. Required Metadata Fields:
       - num_retrieved: int (number of documents retrieved)
       - pipeline_type: str (pipeline identifier)
       - generated_answer: bool (whether LLM answer was generated)
       - processing_time: float (same as execution_time)
       - retrieval_method: str (how retrieval was performed)
       - context_count: int (number of contexts returned)

    4. Backward Compatibility:
       - Support deprecated query_text parameter (aliased to query)
    """

    # Required methods all pipelines must implement
    REQUIRED_METHODS = ['query', 'load_documents']

    # Query method contract
    QUERY_METHOD_CONTRACT = {
        'required_params': ['query'],
        'optional_params': ['top_k', 'kwargs'],
        'deprecated_params': ['query_text'],  # Should map to 'query'
        'required_response_fields': [
            'answer',
            'retrieved_documents',
            'contexts',
            'sources',
            'execution_time',
            'metadata'
        ],
        'required_metadata_fields': [
            'num_retrieved',
            'pipeline_type',
            'generated_answer',
            'processing_time',
            'retrieval_method',
            'context_count'
        ]
    }

    # Load documents method contract
    LOAD_DOCUMENTS_CONTRACT = {
        'required_params': [],  # All params are optional
        'optional_params': ['documents_path', 'documents', 'kwargs']
    }

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.

        Args:
            strict_mode: If True, treats warnings as errors
        """
        self.strict_mode = strict_mode

    def validate_pipeline_class(
        self,
        pipeline_class: Type[RAGPipeline]
    ) -> List[PipelineContractViolation]:
        """
        Validate a pipeline class implementation.

        Checks:
        1. Inherits from RAGPipeline
        2. Implements all required methods
        3. Method signatures match contract
        4. Supports deprecated parameters (with warnings)

        Args:
            pipeline_class: The pipeline class to validate

        Returns:
            List of contract violations (empty if valid)
        """
        violations = []

        # Check inheritance
        if not issubclass(pipeline_class, RAGPipeline):
            violations.append(PipelineContractViolation(
                severity=ViolationSeverity.ERROR,
                category="inheritance",
                message=f"{pipeline_class.__name__} does not inherit from RAGPipeline",
                location=f"{pipeline_class.__name__}",
                suggestion="Ensure your pipeline class inherits from iris_vector_rag.core.base.RAGPipeline"
            ))
            return violations  # Cannot continue validation

        # Check required methods exist
        for method_name in self.REQUIRED_METHODS:
            if not hasattr(pipeline_class, method_name):
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.ERROR,
                    category="missing_method",
                    message=f"Required method '{method_name}' not implemented",
                    location=f"{pipeline_class.__name__}.{method_name}",
                    suggestion=f"Implement the {method_name} method as defined in RAGPipeline"
                ))

        # Validate query method signature
        if hasattr(pipeline_class, 'query'):
            violations.extend(
                self._validate_query_signature(pipeline_class)
            )

        # Validate load_documents method signature
        if hasattr(pipeline_class, 'load_documents'):
            violations.extend(
                self._validate_load_documents_signature(pipeline_class)
            )

        return violations

    def validate_response(
        self,
        response: Dict[str, Any],
        pipeline_name: str = "unknown"
    ) -> List[PipelineContractViolation]:
        """
        Validate a pipeline query response.

        Checks:
        1. All required fields are present
        2. Field types are correct
        3. Metadata is complete
        4. Values are reasonable (e.g., non-negative times)

        Args:
            response: The query response dictionary
            pipeline_name: Name of the pipeline for error messages

        Returns:
            List of contract violations (empty if valid)
        """
        violations = []

        # Check required response fields
        for field in self.QUERY_METHOD_CONTRACT['required_response_fields']:
            if field not in response:
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.ERROR,
                    category="missing_field",
                    message=f"Required field '{field}' missing from response",
                    location=f"{pipeline_name}.query() response",
                    suggestion=f"Ensure your query method returns a dict with '{field}' key"
                ))

        # Validate field types
        violations.extend(self._validate_response_types(response, pipeline_name))

        # Validate metadata if present
        if 'metadata' in response:
            violations.extend(
                self._validate_metadata(response['metadata'], pipeline_name)
            )

        # Validate execution_time is reasonable
        if 'execution_time' in response:
            exec_time = response['execution_time']
            if not isinstance(exec_time, (int, float)):
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.ERROR,
                    category="invalid_type",
                    message=f"execution_time must be numeric, got {type(exec_time)}",
                    location=f"{pipeline_name}.query() response",
                    suggestion="Use time.time() to measure execution time"
                ))
            elif exec_time < 0:
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.ERROR,
                    category="invalid_value",
                    message=f"execution_time cannot be negative: {exec_time}",
                    location=f"{pipeline_name}.query() response",
                    suggestion="Check your time measurement logic"
                ))

        return violations

    def _validate_query_signature(
        self,
        pipeline_class: Type[RAGPipeline]
    ) -> List[PipelineContractViolation]:
        """Validate query method signature matches contract."""
        violations = []
        method = getattr(pipeline_class, 'query')
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        # Remove 'self' from params
        if 'self' in params:
            params.remove('self')

        # Check required params exist
        for required_param in self.QUERY_METHOD_CONTRACT['required_params']:
            if required_param not in params:
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.ERROR,
                    category="method_signature",
                    message=f"Required parameter '{required_param}' missing from query method",
                    location=f"{pipeline_class.__name__}.query",
                    suggestion=f"Add '{required_param}' parameter to query method signature"
                ))

        # Check for **kwargs support (allows flexibility)
        has_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )

        if not has_kwargs:
            violations.append(PipelineContractViolation(
                severity=ViolationSeverity.WARNING,
                category="method_signature",
                message="query method should accept **kwargs for flexibility",
                location=f"{pipeline_class.__name__}.query",
                suggestion="Add **kwargs parameter to query method signature"
            ))

        # Check if deprecated query_text is handled
        # This is checked via **kwargs or explicit parameter
        if 'query_text' in params:
            violations.append(PipelineContractViolation(
                severity=ViolationSeverity.INFO,
                category="deprecated_param",
                message="query_text parameter is deprecated, use 'query' instead",
                location=f"{pipeline_class.__name__}.query",
                suggestion="Remove query_text parameter, handle via **kwargs if needed"
            ))

        return violations

    def _validate_load_documents_signature(
        self,
        pipeline_class: Type[RAGPipeline]
    ) -> List[PipelineContractViolation]:
        """Validate load_documents method signature matches contract."""
        violations = []
        method = getattr(pipeline_class, 'load_documents')
        sig = inspect.signature(method)

        # Check for **kwargs support
        has_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in sig.parameters.values()
        )

        if not has_kwargs:
            violations.append(PipelineContractViolation(
                severity=ViolationSeverity.WARNING,
                category="method_signature",
                message="load_documents method should accept **kwargs for flexibility",
                location=f"{pipeline_class.__name__}.load_documents",
                suggestion="Add **kwargs parameter to load_documents signature"
            ))

        return violations

    def _validate_response_types(
        self,
        response: Dict[str, Any],
        pipeline_name: str
    ) -> List[PipelineContractViolation]:
        """Validate response field types."""
        violations = []

        # answer should be string
        if 'answer' in response and not isinstance(response['answer'], str):
            violations.append(PipelineContractViolation(
                severity=ViolationSeverity.ERROR,
                category="invalid_type",
                message=f"'answer' must be string, got {type(response['answer'])}",
                location=f"{pipeline_name}.query() response",
                suggestion="Ensure LLM response is converted to string"
            ))

        # retrieved_documents should be list
        if 'retrieved_documents' in response:
            if not isinstance(response['retrieved_documents'], list):
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.ERROR,
                    category="invalid_type",
                    message=f"'retrieved_documents' must be list, got {type(response['retrieved_documents'])}",
                    location=f"{pipeline_name}.query() response",
                    suggestion="Return list of Document objects"
                ))

        # contexts should be list of strings
        if 'contexts' in response:
            if not isinstance(response['contexts'], list):
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.ERROR,
                    category="invalid_type",
                    message=f"'contexts' must be list, got {type(response['contexts'])}",
                    location=f"{pipeline_name}.query() response",
                    suggestion="Return list of context strings"
                ))
            elif response['contexts'] and not all(isinstance(c, str) for c in response['contexts']):
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.ERROR,
                    category="invalid_type",
                    message="'contexts' must contain only strings",
                    location=f"{pipeline_name}.query() response",
                    suggestion="Extract text content from Document objects"
                ))

        # sources should be list of strings
        if 'sources' in response:
            if not isinstance(response['sources'], list):
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.ERROR,
                    category="invalid_type",
                    message=f"'sources' must be list, got {type(response['sources'])}",
                    location=f"{pipeline_name}.query() response",
                    suggestion="Return list of source identifiers"
                ))

        # metadata should be dict
        if 'metadata' in response and not isinstance(response['metadata'], dict):
            violations.append(PipelineContractViolation(
                severity=ViolationSeverity.ERROR,
                category="invalid_type",
                message=f"'metadata' must be dict, got {type(response['metadata'])}",
                location=f"{pipeline_name}.query() response",
                suggestion="Return metadata as dictionary"
            ))

        return violations

    def _validate_metadata(
        self,
        metadata: Dict[str, Any],
        pipeline_name: str
    ) -> List[PipelineContractViolation]:
        """Validate metadata completeness."""
        violations = []

        # Check required metadata fields
        for field in self.QUERY_METHOD_CONTRACT['required_metadata_fields']:
            if field not in metadata:
                violations.append(PipelineContractViolation(
                    severity=ViolationSeverity.WARNING,
                    category="missing_metadata",
                    message=f"Required metadata field '{field}' missing",
                    location=f"{pipeline_name}.query() metadata",
                    suggestion=f"Add '{field}' to response metadata"
                ))

        # Validate specific metadata field types
        if 'num_retrieved' in metadata and not isinstance(metadata['num_retrieved'], int):
            violations.append(PipelineContractViolation(
                severity=ViolationSeverity.WARNING,
                category="invalid_type",
                message=f"num_retrieved should be int, got {type(metadata['num_retrieved'])}",
                location=f"{pipeline_name}.query() metadata",
                suggestion="Use len(retrieved_documents) for num_retrieved"
            ))

        if 'generated_answer' in metadata and not isinstance(metadata['generated_answer'], bool):
            violations.append(PipelineContractViolation(
                severity=ViolationSeverity.WARNING,
                category="invalid_type",
                message=f"generated_answer should be bool, got {type(metadata['generated_answer'])}",
                location=f"{pipeline_name}.query() metadata",
                suggestion="Set generated_answer to True/False"
            ))

        return violations

    def get_contract_summary(self) -> str:
        """
        Get a human-readable summary of the pipeline contract.

        Returns:
            Formatted string describing the contract requirements
        """
        return """
Pipeline Contract Requirements:

1. Required Methods:
   - query(query: str, top_k: int = 20, **kwargs) -> Dict[str, Any]
   - load_documents(documents_path: str = "", documents: List[Document] = None, **kwargs) -> None

2. Query Response Format:
   Required fields:
   - answer: str                          # LLM-generated answer
   - retrieved_documents: List[Document]  # Retrieved context
   - contexts: List[str]                  # Text content (RAGAS compatible)
   - sources: List[str]                   # Source references
   - execution_time: float                # Total time in seconds
   - metadata: Dict[str, Any]             # Pipeline-specific metadata

3. Required Metadata Fields:
   - num_retrieved: int                   # Number of documents retrieved
   - pipeline_type: str                   # Pipeline identifier
   - generated_answer: bool               # Whether LLM answer was generated
   - processing_time: float               # Same as execution_time
   - retrieval_method: str                # How retrieval was performed
   - context_count: int                   # Number of contexts returned

4. Backward Compatibility:
   - Support deprecated 'query_text' parameter (map to 'query')
   - Handle via **kwargs in query method

5. Best Practices:
   - Use **kwargs for extensibility
   - Include descriptive error messages
   - Log important operations
   - Validate input parameters
   - Return consistent metadata across queries
"""
