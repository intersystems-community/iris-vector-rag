# Specification: Examples and Demos Testing Framework

**Specification ID**: 005-examples-testing-framework
**Version**: 1.0.0
**Status**: Draft
**Author**: Claude Code
**Date**: 2024-09-30

## 🎯 Overview

This specification defines a comprehensive testing framework for examples and demos in the rag-templates project, ensuring they remain functional, educational, and maintainable across different environments and configurations.

## 📋 Requirements

### Functional Requirements

**FR-001**: Example Validation
- All example scripts MUST execute successfully in clean environments
- Examples MUST produce expected outputs with sample data
- Examples MUST handle missing dependencies gracefully
- Examples MUST provide clear error messages for setup issues

**FR-002**: Demo Functionality Testing
- Demo scripts MUST demonstrate core features without external dependencies
- Interactive demos MUST work in both mock and real LLM modes
- Visualization demos MUST generate valid output files
- Demos MUST include self-validation mechanisms

**FR-003**: Documentation Consistency
- All examples MUST include clear setup instructions
- Examples MUST specify required environment variables
- Examples MUST document expected outputs
- Examples MUST include troubleshooting guidance

**FR-004**: Environment Compatibility
- Examples MUST work in development environments
- Examples MUST work in CI/CD environments
- Examples MUST support both real and mocked dependencies
- Examples MUST handle missing optional dependencies

### Non-Functional Requirements

**NFR-001**: Performance
- Example execution time MUST be < 5 minutes for basic demos
- Examples MUST not require > 2GB memory
- Examples MUST handle timeout scenarios gracefully
- Examples MUST provide progress indicators for long operations

**NFR-002**: Maintainability
- Examples MUST follow consistent code structure
- Examples MUST use shared utilities for common operations
- Examples MUST be automatically testable via CI/CD
- Examples MUST have version compatibility checks

**NFR-003**: Educational Value
- Examples MUST demonstrate best practices
- Examples MUST include explanatory comments
- Examples MUST show realistic use cases
- Examples MUST provide clear learning progression

## 🏗️ Architecture

### Current Example Structure
```
scripts/
├── basic/                          # Core pipeline examples
│   ├── try_basic_rag_pipeline.py   # BasicRAG demonstration
│   └── try_hybrid_graphrag_pipeline.py # HybridGraphRAG demonstration
├── crag/                           # Advanced pipeline examples
│   └── try_crag_pipeline.py        # CRAG demonstration
├── reranking/                      # Specialized examples
│   └── try_basic_rerank.py         # Reranking demonstration
├── demo_graph_visualization.py    # Interactive visualization demo
├── demo_ontology_support.py       # Ontology system demo
└── utilities/                      # Shared demo utilities
```

### Proposed Testing Framework
```
scripts/
├── testing/                       # NEW: Testing infrastructure
│   ├── __init__.py
│   ├── example_runner.py          # Test execution framework
│   ├── validation_suite.py        # Output validation
│   ├── mock_providers.py          # Mock LLM/API providers
│   └── fixtures/                  # Test data and expectations
├── examples/                      # REORGANIZED: Structured examples
│   ├── basic/                     # Basic pipeline examples
│   ├── advanced/                  # Advanced pipeline examples
│   ├── demos/                     # Interactive demonstrations
│   └── tutorials/                 # Step-by-step learning
└── validation/                    # NEW: Validation outputs
    ├── reports/                   # Test execution reports
    ├── outputs/                   # Expected vs actual outputs
    └── benchmarks/                # Performance baselines
```

## 🔧 Technical Design

### Component 1: Example Test Runner

**File**: `scripts/testing/example_runner.py`

```python
class ExampleTestRunner:
    """Framework for executing and validating example scripts."""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.mock_providers = MockProviders()
        self.validators = ValidationSuite()

    def run_example(self, script_path: str, mode: str = "mock") -> TestResult:
        """Execute an example script and validate results."""
        # 1. Setup test environment
        # 2. Execute script with appropriate mocks
        # 3. Capture outputs and metrics
        # 4. Validate against expectations
        # 5. Generate test report

    def run_all_examples(self, filter_pattern: str = None) -> List[TestResult]:
        """Execute all examples matching pattern."""

    def generate_report(self, results: List[TestResult]) -> str:
        """Generate comprehensive test report."""
```

**Features**:
- Isolated execution environments
- Mock LLM provider integration
- Output capture and validation
- Performance metrics collection
- Failure diagnosis and reporting

### Component 2: Validation Suite

**File**: `scripts/testing/validation_suite.py`

```python
class ValidationSuite:
    """Validation framework for example outputs."""

    def validate_basic_rag(self, output: Dict) -> ValidationResult:
        """Validate BasicRAG example output."""
        # Check required fields: answer, sources, metadata
        # Validate answer quality metrics
        # Ensure source attribution

    def validate_graph_visualization(self, output_files: List[str]) -> ValidationResult:
        """Validate graph visualization outputs."""
        # Check generated HTML files
        # Validate JSON graph data
        # Ensure interactive elements

    def validate_performance_metrics(self, metrics: Dict) -> ValidationResult:
        """Validate performance characteristics."""
        # Check execution time bounds
        # Validate memory usage
        # Ensure reasonable response times
```

**Validation Categories**:
- **Output Structure**: Required fields, data types, formats
- **Content Quality**: Answer relevance, source attribution, completeness
- **Performance**: Execution time, memory usage, resource utilization
- **Error Handling**: Graceful failures, meaningful error messages

### Component 3: Mock Provider System

**File**: `scripts/testing/mock_providers.py`

```python
class MockLLMProvider:
    """Mock LLM provider for predictable testing."""

    def __init__(self, response_mode: str = "realistic"):
        self.response_mode = response_mode
        self.call_count = 0
        self.call_history = []

    def generate_response(self, prompt: str) -> str:
        """Generate deterministic responses for testing."""
        # Realistic: Context-aware responses
        # Simple: Template-based responses
        # Error: Simulate API failures

    def get_metrics(self) -> Dict:
        """Return usage metrics for validation."""

class MockDataProvider:
    """Mock data provider for consistent test datasets."""

    def get_sample_documents(self, count: int = 10) -> List[Document]:
        """Return sample documents for testing."""

    def get_test_queries(self, category: str = "general") -> List[str]:
        """Return categorized test queries."""
```

**Mock Modes**:
- **Realistic**: Context-aware responses simulating real LLM behavior
- **Deterministic**: Fixed responses for consistent testing
- **Error**: Simulate various failure scenarios
- **Performance**: Controlled latency and token usage

### Component 4: Example Configuration

**File**: `scripts/testing/config.yaml`

```yaml
examples:
  basic:
    try_basic_rag_pipeline:
      timeout: 300
      expected_outputs:
        - answer
        - sources
        - metadata
      performance_bounds:
        max_execution_time: 180
        max_memory_mb: 1024
      test_queries:
        - "What is diabetes?"
        - "How does insulin work?"

  demos:
    demo_graph_visualization:
      timeout: 600
      expected_files:
        - "graph_visualization.html"
        - "graph_data.json"
      performance_bounds:
        max_execution_time: 300

validation:
  output_formats:
    answer: str
    sources: List[Dict]
    metadata: Dict

  quality_metrics:
    min_answer_length: 50
    max_answer_length: 2000
    required_source_count: 1
```

## 📊 Implementation Plan

### Phase 1: Core Testing Infrastructure (3-4 hours)

**Tasks**:
1. Create testing directory structure
2. Implement ExampleTestRunner framework
3. Build MockLLMProvider system
4. Create basic validation framework
5. Establish configuration system

**Deliverables**:
- Functional test runner
- Mock provider integration
- Basic validation suite
- Configuration-driven testing

**Acceptance Criteria**:
- Can execute examples in isolation
- Mock providers work reliably
- Basic validation catches issues
- Configuration drives test behavior

### Phase 2: Example Validation (4-5 hours)

**Tasks**:
1. Implement validation for each example type
2. Create expected output fixtures
3. Build performance benchmarking
4. Add error scenario testing
5. Develop reporting system

**Deliverables**:
- Complete validation suite
- Performance benchmarks
- Error scenario coverage
- Comprehensive reporting

**Acceptance Criteria**:
- All current examples validate successfully
- Performance bounds are established
- Error scenarios are covered
- Reports provide actionable insights

### Phase 3: CI/CD Integration (2-3 hours)

**Tasks**:
1. Create automated test execution
2. Integrate with existing CI pipeline
3. Add example regression testing
4. Implement failure notifications
5. Create performance tracking

**Deliverables**:
- CI/CD integration
- Automated example testing
- Regression detection
- Performance monitoring

**Acceptance Criteria**:
- Examples test automatically on PR
- Regressions are caught immediately
- Performance degradation is detected
- Teams are notified of failures

### Phase 4: Documentation and Enhancement (2-3 hours)

**Tasks**:
1. Update example documentation
2. Add testing guidelines
3. Create contributor guide
4. Implement example templates
5. Add advanced validation rules

**Deliverables**:
- Enhanced documentation
- Contributor guidelines
- Example templates
- Advanced validation

**Acceptance Criteria**:
- Examples are well-documented
- Contributors know how to add examples
- Templates ensure consistency
- Validation catches edge cases

## 🧪 Testing Strategy

### Unit Tests
```python
# Test framework components
def test_mock_llm_provider():
    provider = MockLLMProvider(response_mode="deterministic")
    response = provider.generate_response("Test query")
    assert response == expected_deterministic_response

def test_validation_suite():
    validator = ValidationSuite()
    result = validator.validate_basic_rag(sample_output)
    assert result.is_valid
    assert result.score > 0.8
```

### Integration Tests
```python
# Test complete example execution
def test_basic_rag_example():
    runner = ExampleTestRunner()
    result = runner.run_example("basic/try_basic_rag_pipeline.py", mode="mock")
    assert result.success
    assert result.execution_time < 300
    assert result.validation_score > 0.7
```

### Performance Tests
```python
# Test performance characteristics
def test_example_performance():
    runner = ExampleTestRunner()
    metrics = runner.benchmark_example("basic/try_basic_rag_pipeline.py")
    assert metrics.execution_time < 180
    assert metrics.memory_usage < 1024  # MB
    assert metrics.api_calls < 10
```

### Error Scenario Tests
```python
# Test error handling
def test_missing_dependency_handling():
    runner = ExampleTestRunner()
    result = runner.run_example_with_missing_deps("try_hybrid_graphrag_pipeline.py")
    assert result.graceful_failure
    assert "install rag-templates[hybrid-graphrag]" in result.error_message
```

## 📈 Success Metrics

### Functional Metrics
- ✅ 100% of examples execute successfully in test environment
- ✅ All examples provide meaningful outputs with mock data
- ✅ Examples handle missing dependencies gracefully
- ✅ Error messages provide clear resolution guidance

### Quality Metrics
- ✅ Examples demonstrate realistic use cases
- ✅ Code follows consistent patterns and best practices
- ✅ Documentation is comprehensive and accurate
- ✅ Learning progression is clear and logical

### Performance Metrics
- ✅ Example execution time < 5 minutes for demos
- ✅ Memory usage < 2GB for all examples
- ✅ 95% uptime for automated example testing
- ✅ < 24 hour turnaround for example issue resolution

### Maintainability Metrics
- ✅ Examples updated automatically with framework changes
- ✅ New examples follow established templates
- ✅ Validation rules catch regressions effectively
- ✅ Example test coverage > 90%

## 🔄 Example Categories and Testing Approach

### Basic Examples (`scripts/basic/`)
**Purpose**: Demonstrate core pipeline functionality
**Testing Focus**:
- Pipeline initialization and configuration
- Query execution with sample data
- Output format and quality validation
- Error handling for common issues

**Validation Criteria**:
- Pipeline creates successfully
- Queries return structured responses
- Sources are properly attributed
- Performance within expected bounds

### Advanced Examples (`scripts/crag/`, `scripts/reranking/`)
**Purpose**: Show specialized pipeline capabilities
**Testing Focus**:
- Advanced feature demonstration
- Complex configuration handling
- Integration with external systems
- Performance optimization techniques

**Validation Criteria**:
- Advanced features work as documented
- Configuration options are respected
- Integration points are stable
- Performance improvements are measurable

### Interactive Demos (`demo_*.py`)
**Purpose**: Provide hands-on exploration of capabilities
**Testing Focus**:
- Interactive component functionality
- Visualization output generation
- User experience validation
- Cross-platform compatibility

**Validation Criteria**:
- Interactive elements respond correctly
- Visualizations are generated successfully
- User workflows complete end-to-end
- Output files are valid and viewable

### Tutorial Examples
**Purpose**: Provide step-by-step learning experience
**Testing Focus**:
- Educational progression and clarity
- Code explanation accuracy
- Exercise completion verification
- Prerequisite validation

**Validation Criteria**:
- Steps build logically upon each other
- Code examples execute as shown
- Exercises validate understanding
- Prerequisites are clearly stated

## 📋 Acceptance Criteria

### Primary Criteria
1. **Functional Validation**: All examples execute successfully in test environments
2. **Quality Assurance**: Examples demonstrate realistic, valuable use cases
3. **Maintainability**: Testing framework catches regressions automatically
4. **Documentation**: Examples are well-documented with clear setup instructions

### Secondary Criteria
1. **Performance**: Examples complete within reasonable time bounds
2. **Accessibility**: Examples work across different environments and setups
3. **Educational Value**: Examples provide clear learning progression
4. **Community**: Framework supports community-contributed examples

## 🎯 Definition of Done

- [ ] Core testing infrastructure implemented and functional
- [ ] All existing examples validate successfully
- [ ] Mock provider system supports realistic testing
- [ ] Validation suite catches common issues
- [ ] CI/CD integration provides automated testing
- [ ] Documentation updated with testing guidelines
- [ ] Performance baselines established
- [ ] Example templates created for contributors
- [ ] Error scenarios comprehensively covered
- [ ] Reporting system provides actionable insights

This specification provides a comprehensive framework for ensuring examples and demos remain functional, educational, and maintainable as the rag-templates project evolves.