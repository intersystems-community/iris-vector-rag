"""
Validation Suite for rag-templates examples.

This module provides comprehensive validation of example outputs,
including structure validation, content quality assessment, and
performance benchmarking.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating example output."""

    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    details: Dict[str, Any]
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "issues": self.issues,
            "details": self.details,
            "category": self.category,
        }


class ValidationSuite:
    """
    Comprehensive validation framework for example outputs.

    Validates structure, content quality, and performance characteristics
    of RAG pipeline examples based on configurable rules.
    """

    def __init__(self, config: Dict = None, clean_iris_mode: bool = False):
        """
        Initialize validation suite.

        Args:
            config: Configuration dictionary with validation rules
            clean_iris_mode: Enable lenient validation for clean IRIS testing
        """
        self.clean_iris_mode = clean_iris_mode
        self.config = config or self._get_default_config()

        # Adjust config for clean IRIS mode
        if self.clean_iris_mode:
            self._adjust_config_for_clean_iris()

    def validate_example_output(
        self, script_name: str, output: str, performance_metrics: Dict = None
    ) -> ValidationResult:
        """
        Validate example output based on script type and expected format.

        Args:
            script_name: Name of the example script
            output: Raw output from script execution
            performance_metrics: Optional performance data

        Returns:
            ValidationResult with comprehensive assessment
        """
        # Determine example category
        category = self._categorize_example(script_name)

        # Parse output
        parsed_output = self._parse_output(output)
        if not parsed_output:
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["Could not parse example output"],
                details={
                    "raw_output": output[:500] + "..." if len(output) > 500 else output
                },
                category=category,
            )

        # Get validation rules for this example
        example_config = self.config.get("examples", {}).get(script_name, {})

        # Perform validation based on category
        if category == "basic_rag":
            return self._validate_basic_rag_output(
                parsed_output, example_config, performance_metrics
            )
        elif category == "graph_visualization":
            return self._validate_visualization_output(
                parsed_output, example_config, performance_metrics
            )
        elif category == "advanced_rag":
            return self._validate_advanced_rag_output(
                parsed_output, example_config, performance_metrics
            )
        else:
            return self._validate_general_output(
                parsed_output, example_config, performance_metrics
            )

    def _categorize_example(self, script_name: str) -> str:
        """Categorize example script by name and functionality."""
        script_lower = script_name.lower()

        if "basic" in script_lower and "rag" in script_lower:
            return "basic_rag"
        elif "crag" in script_lower:
            return "advanced_rag"
        elif "rerank" in script_lower:
            return "advanced_rag"
        elif "hybrid" in script_lower:
            return "advanced_rag"
        elif "visualization" in script_lower or "graph" in script_lower:
            return "graph_visualization"
        elif "demo" in script_lower:
            return "demo"
        else:
            return "general"

    def _parse_output(self, output: str) -> Optional[Dict]:
        """Parse example output to extract structured data."""
        # Try to find JSON in output
        json_output = self._extract_json_from_text(output)
        if json_output:
            return json_output

        # Try to parse as key-value output
        structured_output = self._extract_structured_data(output)
        if structured_output:
            return structured_output

        # Return raw output for text-based validation
        return {"raw_output": output, "type": "text"}

    def _validate_basic_rag_output(
        self, output: Dict, config: Dict, metrics: Dict = None
    ) -> ValidationResult:
        """Validate basic RAG pipeline output."""
        issues = []
        score = 1.0
        details = {}

        # Check required fields
        required_fields = config.get("expected_outputs", ["answer", "sources"])

        # In clean IRIS mode, only require answer field
        if self.clean_iris_mode:
            required_fields = [f for f in required_fields if f == "answer"]

        for field in required_fields:
            if field not in output:
                issues.append(f"Missing required field: {field}")
                # Reduced penalty for clean IRIS mode
                penalty = 0.2 if self.clean_iris_mode else 0.3
                score -= penalty

        # Validate answer quality
        if "answer" in output:
            answer_result = self._validate_answer_quality(output["answer"])
            if not answer_result.is_valid:
                issues.extend(answer_result.issues)
                score -= 0.2
            details["answer_validation"] = answer_result.details

        # Validate sources
        if "sources" in output:
            sources_result = self._validate_sources(output["sources"])
            if not sources_result.is_valid:
                issues.extend(sources_result.issues)
                # Reduced penalty for clean IRIS mode
                penalty = 0.1 if self.clean_iris_mode else 0.2
                score -= penalty
            details["sources_validation"] = sources_result.details
        elif not self.clean_iris_mode:
            # Only penalize missing sources if not in clean IRIS mode
            issues.append("No sources found in output")
            score -= 0.2

        # Validate performance if provided
        if metrics:
            perf_result = self._validate_performance(
                metrics, config.get("performance_bounds", {})
            )
            if not perf_result.is_valid:
                issues.extend(perf_result.issues)
                score -= 0.1
            details["performance_validation"] = perf_result.details

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            details=details,
            category="basic_rag",
        )

    def _validate_advanced_rag_output(
        self, output: Dict, config: Dict, metrics: Dict = None
    ) -> ValidationResult:
        """Validate advanced RAG pipeline output (CRAG, reranking, etc.)."""
        # Start with basic validation
        basic_result = self._validate_basic_rag_output(output, config, metrics)

        # Add advanced-specific validations
        additional_issues = []
        additional_details = {}

        # Check for relevance scores (CRAG)
        if "relevance_score" in config.get("expected_outputs", []):
            if "relevance_score" not in output:
                additional_issues.append("Missing relevance score for CRAG pipeline")
            elif not isinstance(output["relevance_score"], (int, float)):
                additional_issues.append("Relevance score must be numeric")
            elif not 0.0 <= output["relevance_score"] <= 1.0:
                additional_issues.append("Relevance score must be between 0.0 and 1.0")

        # Check for reranking metadata
        if "rerank" in config.get("features", []):
            if "rerank_scores" not in output.get("metadata", {}):
                additional_issues.append("Missing reranking scores in metadata")

        # Combine results
        all_issues = basic_result.issues + additional_issues
        all_details = {**basic_result.details, **additional_details}

        return ValidationResult(
            is_valid=len(all_issues) == 0,
            score=basic_result.score - (len(additional_issues) * 0.1),
            issues=all_issues,
            details=all_details,
            category="advanced_rag",
        )

    def _validate_visualization_output(
        self, output: Dict, config: Dict, metrics: Dict = None
    ) -> ValidationResult:
        """Validate graph visualization output."""
        issues = []
        score = 1.0
        details = {}

        # Check for expected files
        expected_files = config.get("expected_files", [])
        generated_files = output.get("generated_files", [])

        for expected_file in expected_files:
            if expected_file not in generated_files:
                issues.append(f"Missing expected file: {expected_file}")
                score -= 0.3

        # Validate graph data structure
        if "graph_data" in output:
            graph_result = self._validate_graph_data(output["graph_data"])
            if not graph_result.is_valid:
                issues.extend(graph_result.issues)
                score -= 0.2
            details["graph_validation"] = graph_result.details

        # Check visualization metrics
        if "visualization_stats" in output:
            stats = output["visualization_stats"]
            if stats.get("node_count", 0) < 1:
                issues.append("Graph must contain at least one node")
                score -= 0.2
            if stats.get("edge_count", 0) < 1:
                issues.append("Graph should contain edges for meaningful visualization")
                score -= 0.1

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            details=details,
            category="graph_visualization",
        )

    def _validate_answer_quality(self, answer: str) -> ValidationResult:
        """Validate answer quality metrics."""
        issues = []
        details = {}

        # Basic validation
        answer_config = self.config.get("validation", {}).get("answer", {})
        min_length = answer_config.get("min_length", 20)
        max_length = answer_config.get("max_length", 2000)

        if len(answer) < min_length:
            issues.append(f"Answer too short: {len(answer)} < {min_length} characters")
        elif len(answer) > max_length:
            issues.append(f"Answer too long: {len(answer)} > {max_length} characters")

        # Content quality checks
        word_count = len(answer.split())
        sentence_count = len([s for s in answer.split(".") if s.strip()])

        if word_count < 10:
            issues.append("Answer contains too few words to be meaningful")

        # Check for placeholder or template responses
        placeholder_patterns = [
            r"this is a (?:dummy|mock|test) (?:answer|response)",
            r"based on the (?:provided )?context",
            r"i don't have enough information",
            r"please provide more details",
        ]

        for pattern in placeholder_patterns:
            if re.search(pattern, answer.lower()):
                issues.append("Answer appears to be a placeholder or template response")
                break

        details.update(
            {
                "length": len(answer),
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": (
                    word_count / sentence_count if sentence_count > 0 else 0
                ),
            }
        )

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=1.0 if len(issues) == 0 else 0.7,
            issues=issues,
            details=details,
        )

    def _validate_sources(
        self, sources: Union[List[Dict], List[str]]
    ) -> ValidationResult:
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
        if sources and isinstance(sources[0], dict):
            required_fields = sources_config.get("required_fields", ["content"])
            for i, source in enumerate(sources):
                for field in required_fields:
                    if field not in source:
                        issues.append(f"Source {i} missing field: {field}")

                # Check content quality
                if "content" in source and len(source["content"]) < 10:
                    issues.append(f"Source {i} content too short")

        details.update(
            {
                "count": len(sources),
                "type": (
                    "structured"
                    if sources and isinstance(sources[0], dict)
                    else "simple"
                ),
                "avg_content_length": self._calculate_avg_source_length(sources),
            }
        )

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=1.0 if len(issues) == 0 else 0.8,
            issues=issues,
            details=details,
        )

    def _validate_performance(self, metrics: Dict, bounds: Dict) -> ValidationResult:
        """Validate performance metrics against bounds."""
        issues = []
        details = metrics.copy()

        # Check execution time
        max_time = bounds.get("max_execution_time", 300)
        if metrics.get("execution_time", 0) > max_time:
            issues.append(
                f"Execution time exceeded: {metrics['execution_time']:.1f}s > {max_time}s"
            )

        # Check memory usage
        max_memory = bounds.get("max_memory_mb", 1024)
        if metrics.get("peak_memory_mb", 0) > max_memory:
            issues.append(
                f"Memory usage exceeded: {metrics['peak_memory_mb']:.1f}MB > {max_memory}MB"
            )

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=1.0 if len(issues) == 0 else 0.8,
            issues=issues,
            details=details,
        )

    def _validate_graph_data(self, graph_data: Dict) -> ValidationResult:
        """Validate graph data structure."""
        issues = []
        details = {}

        required_keys = ["nodes", "edges"]
        for key in required_keys:
            if key not in graph_data:
                issues.append(f"Graph data missing required key: {key}")

        if "nodes" in graph_data:
            nodes = graph_data["nodes"]
            if not isinstance(nodes, list):
                issues.append("Graph nodes must be a list")
            elif len(nodes) == 0:
                issues.append("Graph must contain at least one node")
            else:
                # Validate node structure
                for i, node in enumerate(nodes[:5]):  # Check first 5 nodes
                    if not isinstance(node, dict):
                        issues.append(f"Node {i} must be a dictionary")
                        continue
                    if "id" not in node:
                        issues.append(f"Node {i} missing required 'id' field")

            details["node_count"] = len(nodes) if isinstance(nodes, list) else 0

        if "edges" in graph_data:
            edges = graph_data["edges"]
            if not isinstance(edges, list):
                issues.append("Graph edges must be a list")
            else:
                details["edge_count"] = len(edges)

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=1.0 if len(issues) == 0 else 0.7,
            issues=issues,
            details=details,
        )

    def _validate_general_output(
        self, output: Dict, config: Dict, metrics: Dict = None
    ) -> ValidationResult:
        """General validation for unspecified example types."""
        issues = []
        details = {"output_keys": list(output.keys())}

        # Basic structure check
        if len(output) == 0:
            issues.append("Output is empty")

        # Check for error indicators
        if "error" in output or "exception" in output:
            issues.append("Output contains error indicators")

        return ValidationResult(
            is_valid=len(issues) == 0,
            score=0.8 if len(issues) == 0 else 0.5,
            issues=issues,
            details=details,
            category="general",
        )

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Extract JSON data from text output."""
        # Look for JSON blocks
        json_patterns = [
            r"```json\s*(\{.*?\})\s*```",  # JSON code blocks
            r'(\{[^{}]*"[^"]*"[^{}]*\})',  # Simple JSON objects
            r"(\{.*\})",  # Any text that looks like JSON
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return None

    def _extract_structured_data(self, text: str) -> Optional[Dict]:
        """Extract structured data from text output using patterns."""
        result = {}

        # Look for answer pattern
        answer_match = re.search(
            r"(?:Answer|Response):\s*(.+?)(?:\n\n|\nSources?:|$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            result["answer"] = answer_match.group(1).strip()

        # Look for sources pattern
        sources_match = re.search(
            r"Sources?:\s*(.+?)(?:\n\n|$)", text, re.DOTALL | re.IGNORECASE
        )
        if sources_match:
            sources_text = sources_match.group(1).strip()
            # Split by lines or bullet points
            sources_lines = [
                line.strip("- •") for line in sources_text.split("\n") if line.strip()
            ]
            result["sources"] = sources_lines

        # Look for metadata or performance info
        time_match = re.search(
            r"(?:Time|Duration|Execution time):\s*([0-9.]+)", text, re.IGNORECASE
        )
        if time_match:
            result["metadata"] = {"execution_time": float(time_match.group(1))}

        return result if result else None

    def _calculate_avg_source_length(self, sources: List) -> float:
        """Calculate average length of source content."""
        if not sources:
            return 0.0

        total_length = 0
        count = 0

        for source in sources:
            if isinstance(source, dict) and "content" in source:
                total_length += len(source["content"])
                count += 1
            elif isinstance(source, str):
                total_length += len(source)
                count += 1

        return total_length / count if count > 0 else 0.0

    def _get_default_config(self) -> Dict:
        """Get default validation configuration."""
        return {
            "validation": {
                "answer": {"min_length": 20, "max_length": 2000},
                "sources": {
                    "min_count": 1,
                    "max_count": 20,
                    "required_fields": ["content"],
                },
            },
            "examples": {
                "basic/try_basic_rag_pipeline.py": {
                    "expected_outputs": ["answer", "sources"],
                    "performance_bounds": {
                        "max_execution_time": 180,
                        "max_memory_mb": 512,
                    },
                },
                "crag/try_crag_pipeline.py": {
                    "expected_outputs": ["answer", "sources", "relevance_score"],
                    "performance_bounds": {
                        "max_execution_time": 240,
                        "max_memory_mb": 768,
                    },
                },
            },
        }

    def _adjust_config_for_clean_iris(self):
        """Adjust validation configuration for clean IRIS testing scenarios."""
        # Relax source requirements - clean IRIS may not have data loaded
        if "validation" in self.config and "sources" in self.config["validation"]:
            self.config["validation"]["sources"]["min_count"] = 0  # Allow zero sources

        # Reduce scoring penalties for missing sources in clean scenarios
        # This affects the validation scoring logic to be more lenient
