"""
Contract tests for graph inspector diagnostic script.

These tests validate the graph_inspector_contract.md specification.
Tests MUST FAIL initially (FileNotFoundError) before script implementation.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestGraphInspectorContract:
    """Contract tests for scripts/inspect_knowledge_graph.py"""

    SCRIPT_PATH = Path("scripts/inspect_knowledge_graph.py")
    # Use the same Python interpreter running the tests
    PYTHON_EXE = sys.executable

    def test_graph_inspector_exists(self):
        """CA-SETUP: Verify script file exists at expected location."""
        assert self.SCRIPT_PATH.exists(), (
            f"Graph inspector script not found at {self.SCRIPT_PATH}"
        )
        assert self.SCRIPT_PATH.is_file()

    def test_graph_inspector_executable(self):
        """CA-SETUP: Verify script can be executed."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Script should run (exit code 0, 1, 2, or 3 are all valid)
        assert result.returncode in [0, 1, 2, 3], (
            f"Unexpected exit code {result.returncode}. "
            f"Stderr: {result.stderr}"
        )

    def test_graph_inspector_output_format(self):
        """CA-9: Verify output is valid JSON with required fields."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Parse JSON output
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            pytest.fail(f"Output is not valid JSON: {e}\nOutput: {result.stdout}")

        # Verify required top-level fields
        assert isinstance(output, dict)
        assert "check_name" in output
        assert "timestamp" in output
        assert "tables_exist" in output
        assert "counts" in output
        assert "sample_entities" in output
        assert "document_links" in output
        assert "data_quality" in output
        assert "diagnosis" in output

        # Verify nested structure
        assert isinstance(output["tables_exist"], dict)
        assert "entities" in output["tables_exist"]
        assert "relationships" in output["tables_exist"]
        assert "communities" in output["tables_exist"]

        assert isinstance(output["counts"], dict)
        assert "entities" in output["counts"]
        assert "relationships" in output["counts"]
        assert "communities" in output["counts"]

        assert isinstance(output["diagnosis"], dict)
        assert "severity" in output["diagnosis"]
        assert "message" in output["diagnosis"]
        assert "suggestions" in output["diagnosis"]

    def test_graph_inspector_empty_graph_detection(self):
        """CA-3: Verify exit code 1 when all counts == 0 (empty graph)."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)

        # If all tables exist but all counts are 0, exit code MUST be 1
        if all(output["tables_exist"].values()):
            if all(count == 0 for count in output["counts"].values()):
                assert result.returncode == 1, (
                    "Empty graph (all counts == 0) should return exit code 1"
                )
                assert output["diagnosis"]["severity"] == "error"
                assert "empty" in output["diagnosis"]["message"].lower()

    def test_graph_inspector_tables_missing_detection(self):
        """CA-4: Verify exit code 2 when any table does not exist."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)

        # If any table is missing, exit code MUST be 2
        if not all(output["tables_exist"].values()):
            assert result.returncode == 2, (
                "Missing tables should return exit code 2"
            )
            assert output["diagnosis"]["severity"] == "critical"
            assert "schema" in output["diagnosis"]["message"].lower()

    def test_graph_inspector_sample_limit(self):
        """CA-5: Verify sample_entities contains at most 5 items."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        sample_entities = output["sample_entities"]

        assert isinstance(sample_entities, list)
        assert len(sample_entities) <= 5, (
            f"sample_entities should contain at most 5 items, got {len(sample_entities)}"
        )

    def test_graph_inspector_document_link_consistency(self):
        """CA-6: Verify linked + orphaned == total_entities."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        doc_links = output["document_links"]
        counts = output["counts"]

        assert doc_links["linked"] + doc_links["orphaned"] == doc_links["total_entities"], (
            "linked + orphaned must equal total_entities"
        )
        assert doc_links["total_entities"] == counts["entities"], (
            "total_entities must equal entity count"
        )

    def test_graph_inspector_completeness_score_bounds(self):
        """CA-7: Verify completeness score is between 0.0 and 1.0."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        completeness_score = output["data_quality"]["completeness_score"]

        assert 0.0 <= completeness_score <= 1.0, (
            f"Completeness score must be 0.0-1.0, got {completeness_score}"
        )

    def test_graph_inspector_suggestions_on_error(self):
        """CA-8: Verify suggestions provided when exit code != 0."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)

        # If exit code is 1 (empty graph), must have at least 2 suggestions
        if result.returncode == 1:
            suggestions = output["diagnosis"]["suggestions"]
            assert len(suggestions) >= 2, (
                "Empty graph should provide at least 2 actionable suggestions"
            )
            assert any("extract" in s.lower() for s in suggestions), (
                "Suggestions should mention entity extraction"
            )

    def test_graph_inspector_connection_error_handling(self):
        """CA-10: Verify exit code 3 on database connection error."""
        # This test validates error handling structure
        # Actual connection error requires stopping IRIS
        # For now, verify the script handles exit code 3 scenario
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # If exit code is 3, verify error diagnosis
        if result.returncode == 3:
            output = json.loads(result.stdout)
            assert output["diagnosis"]["severity"] == "critical"
            suggestions = output["diagnosis"]["suggestions"]
            assert any("connection" in s.lower() for s in suggestions), (
                "Connection error should provide connection troubleshooting"
            )
