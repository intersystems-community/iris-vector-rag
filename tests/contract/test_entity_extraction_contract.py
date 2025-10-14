"""
Contract tests for entity extraction verifier diagnostic script.

These tests validate the entity_extraction_verification_contract.md specification.
Tests MUST FAIL initially (FileNotFoundError) before script implementation.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


class TestEntityExtractionContract:
    """Contract tests for scripts/verify_entity_extraction.py"""

    SCRIPT_PATH = Path("scripts/verify_entity_extraction.py")
    # Use the same Python interpreter running the tests
    PYTHON_EXE = sys.executable

    def test_entity_extraction_verifier_exists(self):
        """CA-SETUP: Verify script file exists at expected path."""
        assert self.SCRIPT_PATH.exists(), (
            f"Entity extraction verifier script not found at {self.SCRIPT_PATH}"
        )
        assert self.SCRIPT_PATH.is_file()

    def test_entity_extraction_verifier_executable(self):
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

    def test_entity_extraction_verifier_output_format(self):
        """CA-10: Verify output is valid JSON with required fields."""
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
        assert "service_status" in output
        assert "llm_status" in output
        assert "ontology_status" in output
        assert "extraction_config" in output
        assert "ingestion_hooks" in output
        assert "test_extraction" in output
        assert "diagnosis" in output

        # Verify nested structure
        assert isinstance(output["service_status"], dict)
        assert "available" in output["service_status"]
        assert "import_error" in output["service_status"]

        assert isinstance(output["llm_status"], dict)
        assert "configured" in output["llm_status"]
        assert "provider" in output["llm_status"]
        assert "model" in output["llm_status"]

        assert isinstance(output["diagnosis"], dict)
        assert "severity" in output["diagnosis"]
        assert "message" in output["diagnosis"]

    def test_service_availability_detection(self):
        """CA-1: Verify service import status is checked."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        service_status = output["service_status"]

        # If service unavailable, exit code MUST be 2
        if not service_status["available"]:
            assert result.returncode == 2
            assert service_status["import_error"] is not None
            assert output["diagnosis"]["severity"] == "critical"

    def test_llm_configuration_check(self):
        """CA-2: Verify LLM configuration is validated."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        llm_status = output["llm_status"]

        # Verify LLM config fields present
        assert "configured" in llm_status
        assert "provider" in llm_status
        assert "model" in llm_status
        assert "api_key_set" in llm_status

        # If provider is not stub, api_key_set should be checked
        if llm_status["provider"] != "stub":
            assert isinstance(llm_status["api_key_set"], bool)

    def test_ontology_status_check(self):
        """CA-3: Verify ontology plugin status is checked."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        ontology_status = output["ontology_status"]

        # Verify ontology status fields
        assert "enabled" in ontology_status
        assert "domain" in ontology_status
        assert "concept_count" in ontology_status
        assert "plugin_loaded" in ontology_status

        # If ontology enabled, plugin should be loaded and have concepts
        if ontology_status["enabled"]:
            assert ontology_status["plugin_loaded"] is True
            assert ontology_status["concept_count"] > 0
            assert ontology_status["domain"] is not None

    def test_extraction_config_validity(self):
        """CA-4: Verify extraction method is valid."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        extraction_config = output["extraction_config"]

        valid_methods = ["ontology_hybrid", "rule_based", "llm_only", "llm_basic", "pattern_based"]
        assert extraction_config["method"] in valid_methods, (
            f"Extraction method '{extraction_config['method']}' not in {valid_methods}"
        )

    def test_confidence_threshold_bounds(self):
        """CA-5: Verify confidence threshold is between 0.0 and 1.0."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        threshold = output["extraction_config"]["confidence_threshold"]

        assert 0.0 <= threshold <= 1.0, (
            f"Confidence threshold must be 0.0-1.0, got {threshold}"
        )

    def test_test_extraction_execution(self):
        """CA-6: Verify test extraction runs and returns valid results."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        test_extraction = output["test_extraction"]

        # Verify test extraction fields
        assert "success" in test_extraction
        assert "sample_text" in test_extraction
        assert "entities_found" in test_extraction
        assert "sample_entities" in test_extraction
        assert "error" in test_extraction

        # If success, must have valid entity count
        if test_extraction["success"]:
            assert test_extraction["entities_found"] >= 0
            assert test_extraction["error"] is None
        else:
            assert test_extraction["error"] is not None

    def test_sample_entity_format(self):
        """CA-7: Verify each sample entity has required fields."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        sample_entities = output["test_extraction"]["sample_entities"]

        for entity in sample_entities:
            assert "name" in entity
            assert "type" in entity
            assert "confidence" in entity
            assert 0.0 <= entity["confidence"] <= 1.0, (
                f"Entity confidence must be 0.0-1.0, got {entity['confidence']}"
            )

    def test_ingestion_hook_detection(self):
        """CA-8: Verify ingestion hook invocation tracking."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)
        ingestion_hooks = output["ingestion_hooks"]

        # Verify hook tracking fields
        assert "extraction_called" in ingestion_hooks
        assert "hook_location" in ingestion_hooks
        assert "invocation_count" in ingestion_hooks

        assert isinstance(ingestion_hooks["extraction_called"], bool)
        assert isinstance(ingestion_hooks["invocation_count"], int)
        assert ingestion_hooks["invocation_count"] >= 0

    def test_not_invoked_diagnosis(self):
        """CA-9: Verify exit code 1 when extraction not invoked."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)

        # If extraction not called, exit code should be 1
        if not output["ingestion_hooks"]["extraction_called"]:
            assert result.returncode == 1, (
                "Extraction not invoked should return exit code 1"
            )
            assert output["diagnosis"]["severity"] in ["warning", "error"]
            assert "not invoked" in output["diagnosis"]["message"].lower()

            # Must provide suggestions
            suggestions = output["diagnosis"]["suggestions"]
            assert len(suggestions) >= 2
            assert any("load_documents" in s.lower() for s in suggestions)

    def test_import_error_handling(self):
        """CA-SETUP: Verify exit code 2 on import failure."""
        result = subprocess.run(
            [self.PYTHON_EXE, str(self.SCRIPT_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = json.loads(result.stdout)

        # If service unavailable, exit code must be 2
        if not output["service_status"]["available"]:
            assert result.returncode == 2
            assert output["diagnosis"]["severity"] == "critical"
            assert output["service_status"]["import_error"] is not None
