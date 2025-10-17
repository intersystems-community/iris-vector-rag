# Examples Testing Framework Implementation Plan

## 🎯 Overview

This plan provides detailed implementation steps for the Examples Testing Framework specification, focusing on creating robust, automated testing for all examples and demos in the rag-templates project.

## 📋 Priority Implementation Order

### Phase 1: Core Infrastructure (IMMEDIATE - 3-4 hours)

#### 1.1 Create Testing Directory Structure
```bash
mkdir -p scripts/testing/{fixtures,reports}
mkdir -p scripts/examples/{basic,advanced,demos,tutorials}
mkdir -p scripts/validation/{reports,outputs,benchmarks}
```

#### 1.2 Mock Provider System
**Priority**: HIGH
**File**: `scripts/testing/mock_providers.py`

```python
class MockLLMProvider:
    """Deterministic LLM provider for testing."""

    RESPONSE_TEMPLATES = {
        "diabetes": "Diabetes is a chronic condition affecting blood sugar levels...",
        "insulin": "Insulin is a hormone that regulates blood glucose...",
        "default": "Based on the provided context, this is a comprehensive answer..."
    }

    def __init__(self, mode: str = "realistic"):
        self.mode = mode
        self.call_count = 0
        self.response_time = 0.5  # Simulated response time

    def generate_response(self, prompt: str) -> str:
        """Generate predictable responses for testing."""
        self.call_count += 1

        if self.mode == "error":
            raise Exception("Simulated API error")

        # Extract key terms from prompt for context-aware responses
        for key, template in self.RESPONSE_TEMPLATES.items():
            if key.lower() in prompt.lower():
                return f"{template} (Response #{self.call_count})"

        return f"{self.RESPONSE_TEMPLATES['default']} (Response #{self.call_count})"
```

#### 1.3 Basic Test Runner
**Priority**: HIGH
**File**: `scripts/testing/example_runner.py`

```python
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

class ExampleTestResult:
    """Results from executing an example."""

    def __init__(self):
        self.success: bool = False
        self.execution_time: float = 0.0
        self.exit_code: int = -1
        self.stdout: str = ""
        self.stderr: str = ""
        self.error_message: str = ""
        self.validation_results: Dict = {}

class ExampleTestRunner:
    """Framework for executing and validating examples."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.scripts_dir = self.project_root / "scripts"

    def run_example(self, script_path: str, timeout: int = 300) -> ExampleTestResult:
        """Execute an example script with timeout and capture results."""
        result = ExampleTestResult()
        full_path = self.scripts_dir / script_path

        if not full_path.exists():
            result.error_message = f"Script not found: {full_path}"
            return result

        try:
            start_time = time.time()

            # Execute script in isolated environment
            process = subprocess.run(
                [sys.executable, str(full_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=self._get_test_env()
            )

            result.execution_time = time.time() - start_time
            result.exit_code = process.returncode
            result.stdout = process.stdout
            result.stderr = process.stderr
            result.success = process.returncode == 0

            if not result.success:
                result.error_message = f"Script failed with exit code {process.returncode}"

        except subprocess.TimeoutExpired:
            result.error_message = f"Script timed out after {timeout} seconds"
        except Exception as e:
            result.error_message = f"Execution error: {str(e)}"

        return result

    def _get_test_env(self) -> Dict[str, str]:
        """Get environment variables for test execution."""
        import os
        env = os.environ.copy()

        # Set test-specific environment variables
        env['EXAMPLE_TEST_MODE'] = 'true'
        env['USE_MOCK_LLM'] = 'true'

        return env
```

#### 1.4 Example Configuration System
**Priority**: HIGH
**File**: `scripts/testing/config.yaml`

```yaml
# Example Testing Configuration

global:
  default_timeout: 300
  max_memory_mb: 1024
  mock_llm: true

examples:
  "basic/try_basic_rag_pipeline.py":
    timeout: 180
    expected_outputs: ["answer", "sources"]
    test_queries:
      - "What is diabetes?"
      - "How does insulin work?"
    performance_bounds:
      max_execution_time: 120
      max_memory_mb: 512

  "basic/try_hybrid_graphrag_pipeline.py":
    timeout: 300
    optional_dependencies: ["iris-vector-graph"]
    graceful_fallback: true
    expected_outputs: ["answer", "sources", "metadata"]

  "crag/try_crag_pipeline.py":
    timeout: 240
    expected_outputs: ["answer", "sources", "relevance_score"]

  "demo_graph_visualization.py":
    timeout: 600
    expected_files:
      - "graph_visualization.html"
      - "graph_data.json"
    interactive: true

validation:
  answer:
    min_length: 20
    max_length: 2000
    required_keywords: []

  sources:
    min_count: 1
    max_count: 10
    required_fields: ["content", "metadata"]
```

### Phase 2: Validation Suite (HIGH PRIORITY - 2-3 hours)

#### 2.1 Output Validation Framework
**File**: `scripts/testing/validation_suite.py`

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import re

@dataclass
class ValidationResult:
    """Result of validating example output."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    details: Dict[str, Any]

class ValidationSuite:
    """Comprehensive validation for example outputs."""

    def __init__(self, config: Dict = None):
        self.config = config or {}

    def validate_example_output(self, script_name: str, output: str) -> ValidationResult:
        """Validate output based on example type."""

        # Parse JSON output if possible
        try:
            parsed_output = json.loads(output)
        except json.JSONDecodeError:
            # Try to extract JSON from text output
            parsed_output = self._extract_json_from_text(output)

        if not parsed_output:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["No valid JSON output found"],
                details={"raw_output": output[:500]}
            )

        # Get validation rules for this example
        example_config = self.config.get("examples", {}).get(script_name, {})

        return self._validate_parsed_output(parsed_output, example_config)

    def _validate_parsed_output(self, output: Dict, config: Dict) -> ValidationResult:
        """Validate parsed output against configuration rules."""
        issues = []
        score = 1.0
        details = {}

        # Check required outputs
        expected_outputs = config.get("expected_outputs", [])
        for expected in expected_outputs:
            if expected not in output:
                issues.append(f"Missing required output: {expected}")
                score -= 0.3

        # Validate answer quality if present
        if "answer" in output:
            answer_result = self._validate_answer(output["answer"])
            if not answer_result.is_valid:
                issues.extend(answer_result.issues)
                score -= 0.2
            details["answer_validation"] = answer_result.details

        # Validate sources if present
        if "sources" in output:
            sources_result = self._validate_sources(output["sources"])
            if not sources_result.is_valid:
                issues.extend(sources_result.issues)
                score -= 0.2
            details["sources_validation"] = sources_result.details

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            details=details
        )

    def _validate_answer(self, answer: str) -> ValidationResult:
        """Validate answer quality metrics."""
        issues = []
        details = {}

        # Length validation
        answer_config = self.config.get("validation", {}).get("answer", {})
        min_length = answer_config.get("min_length", 10)
        max_length = answer_config.get("max_length", 5000)

        if len(answer) < min_length:
            issues.append(f"Answer too short: {len(answer)} < {min_length}")
        elif len(answer) > max_length:
            issues.append(f"Answer too long: {len(answer)} > {max_length}")

        details["length"] = len(answer)
        details["word_count"] = len(answer.split())

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=1.0 if len(issues) == 0 else 0.5,
            issues=issues,
            details=details
        )

    def _validate_sources(self, sources: List[Dict]) -> ValidationResult:
        """Validate source attribution."""
        issues = []
        details = {}

        sources_config = self.config.get("validation", {}).get("sources", {})
        min_count = sources_config.get("min_count", 1)
        max_count = sources_config.get("max_count", 20)

        if len(sources) < min_count:
            issues.append(f"Too few sources: {len(sources)} < {min_count}")
        elif len(sources) > max_count:
            issues.append(f"Too many sources: {len(sources)} > {max_count}")

        # Validate source structure
        required_fields = sources_config.get("required_fields", ["content"])
        for i, source in enumerate(sources):
            for field in required_fields:
                if field not in source:
                    issues.append(f"Source {i} missing field: {field}")

        details["count"] = len(sources)
        details["has_content"] = all("content" in s for s in sources)

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=1.0 if len(issues) == 0 else 0.7,
            issues=issues,
            details=details
        )

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Try to extract JSON from text output."""
        # Look for JSON patterns in text
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None
```

### Phase 3: Integration and Automation (MEDIUM PRIORITY - 2-3 hours)

#### 3.1 Main Test Execution Script
**File**: `scripts/testing/run_example_tests.py`

```python
#!/usr/bin/env python3
"""
Main script for executing example tests.
"""

import argparse
import yaml
from pathlib import Path
from typing import List

from example_runner import ExampleTestRunner, ExampleTestResult
from validation_suite import ValidationSuite
from mock_providers import MockLLMProvider

def main():
    parser = argparse.ArgumentParser(description="Run example tests")
    parser.add_argument("--pattern", help="Pattern to match example names")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize components
    runner = ExampleTestRunner()
    validator = ValidationSuite(config)

    # Get examples to test
    examples = get_examples_to_test(config, args.pattern)

    print(f"Running {len(examples)} example tests...")

    results = []
    for example in examples:
        print(f"Testing: {example}")
        result = runner.run_example(example, timeout=args.timeout)

        if result.success and result.stdout:
            validation = validator.validate_example_output(example, result.stdout)
            result.validation_results = validation

        results.append((example, result))

        if args.verbose:
            print_result_summary(example, result)

    # Generate final report
    generate_test_report(results)

def get_examples_to_test(config: dict, pattern: str = None) -> List[str]:
    """Get list of examples to test."""
    examples = list(config.get("examples", {}).keys())

    if pattern:
        examples = [e for e in examples if pattern in e]

    return examples

def print_result_summary(example: str, result: ExampleTestResult):
    """Print summary of test result."""
    status = "✅ PASS" if result.success else "❌ FAIL"
    print(f"  {status} {example} ({result.execution_time:.1f}s)")

    if not result.success:
        print(f"    Error: {result.error_message}")

    if hasattr(result, 'validation_results') and result.validation_results:
        val = result.validation_results
        print(f"    Validation: {val.score:.2f} score, {len(val.issues)} issues")

def generate_test_report(results: List):
    """Generate comprehensive test report."""
    total = len(results)
    passed = sum(1 for _, r in results if r.success)

    print(f"\n{'='*50}")
    print(f"Test Results: {passed}/{total} examples passed")
    print(f"{'='*50}")

    for example, result in results:
        status = "✅" if result.success else "❌"
        print(f"{status} {example}")

        if not result.success:
            print(f"   Error: {result.error_message}")

if __name__ == "__main__":
    main()
```

#### 3.2 CI/CD Integration
**File**: `scripts/ci/test-examples.sh`

```bash
#!/bin/bash
# CI script for testing examples

set -e

echo "🧪 Running Example Tests..."

# Setup test environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export EXAMPLE_TEST_MODE=true
export USE_MOCK_LLM=true

# Run example tests
python scripts/testing/run_example_tests.py \
    --mode mock \
    --timeout 300 \
    --verbose

echo "✅ Example tests completed"
```

### Phase 4: Enhanced Features (LOWER PRIORITY - 2-3 hours)

#### 4.1 Performance Benchmarking
**File**: `scripts/testing/benchmark_runner.py`

```python
import psutil
import time
from typing import Dict, Any

class PerformanceBenchmark:
    """Performance monitoring for example execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = None
        self.start_time = None

    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_memory = self.process.memory_info().rss
        self.start_time = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        current_memory = self.process.memory_info().rss
        current_time = time.time()

        return {
            "execution_time": current_time - self.start_time if self.start_time else 0,
            "memory_usage_mb": (current_memory - self.start_memory) / 1024 / 1024 if self.start_memory else 0,
            "peak_memory_mb": current_memory / 1024 / 1024,
            "cpu_percent": self.process.cpu_percent()
        }
```

#### 4.2 Example Templates
**File**: `scripts/testing/templates/basic_example_template.py`

```python
"""
Template for creating new basic RAG examples.

This template provides the standard structure for basic RAG pipeline examples
with proper error handling, logging, and test integration.
"""

import logging
import sys
import os
from pathlib import Path

# Standard path setup for examples
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_rag import create_pipeline
from common.utils import get_llm_func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main example execution."""
    logger.info("Starting {EXAMPLE_NAME} example")

    try:
        # Check if running in test mode
        test_mode = os.getenv("EXAMPLE_TEST_MODE", "false").lower() == "true"
        use_mock_llm = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

        # Setup LLM function
        if use_mock_llm or test_mode:
            llm_func = mock_llm_function
        else:
            llm_func = get_llm_func("openai", "gpt-4o-mini")

        # Create pipeline
        pipeline = create_pipeline("{PIPELINE_TYPE}")

        # Example-specific code here
        # ...

        logger.info("Example completed successfully")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        sys.exit(1)

def mock_llm_function(prompt: str) -> str:
    """Mock LLM function for testing."""
    return "This is a mock response for testing purposes."

if __name__ == "__main__":
    main()
```

## 🔄 Migration Strategy

### Existing Examples Enhancement
1. **Add test mode detection** to all existing examples
2. **Implement mock LLM integration** for predictable testing
3. **Standardize output formats** for easier validation
4. **Add progress indicators** for long-running examples
5. **Enhance error messages** with resolution guidance

### New Example Requirements
1. **Use example templates** for consistency
2. **Include test validation** in all new examples
3. **Document expected outputs** in example headers
4. **Provide mock data options** for offline testing
5. **Follow performance guidelines** for resource usage

## 📊 Success Metrics and Validation

### Implementation Success Criteria
- [ ] All existing examples execute successfully in test mode
- [ ] Mock providers generate realistic outputs
- [ ] Validation suite catches common issues
- [ ] CI/CD integration provides automated testing
- [ ] Performance benchmarks establish baselines

### Quality Assurance Checklist
- [ ] Examples follow consistent patterns
- [ ] Error handling is comprehensive
- [ ] Documentation is accurate and helpful
- [ ] Test coverage includes edge cases
- [ ] Performance is within acceptable bounds

This implementation plan provides a structured approach to creating a robust testing framework for examples and demos, ensuring they remain functional and educational as the project evolves.