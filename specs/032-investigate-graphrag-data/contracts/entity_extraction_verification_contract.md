# Contract: Entity Extraction Verifier

**Feature**: 032-investigate-graphrag-data | **Date**: 2025-10-06
**Script**: `scripts/verify_entity_extraction.py`
**Purpose**: Verify entity extraction service configuration and invocation status

## Command Interface

### Execution
```bash
python scripts/verify_entity_extraction.py
```

### Exit Codes
- **0**: Enabled and functional - Extraction service works correctly
- **1**: Disabled or not invoked - Service exists but not called during load_data
- **2**: Service error - Import failed, LLM unavailable, or module missing
- **3**: Configuration error - Invalid settings or missing required config

### Output Format

**Medium**: JSON to stdout

**Schema**:
```json
{
  "check_name": "entity_extraction_verification",
  "timestamp": "2025-10-06T12:34:56.789Z",
  "execution_time_ms": 156.7,

  "service_status": {
    "available": boolean,
    "import_error": string | null,
    "version": string,
    "module_path": string
  },

  "llm_status": {
    "configured": boolean,
    "provider": string,
    "model": string,
    "api_key_set": boolean,
    "api_key_valid": boolean | null
  },

  "ontology_status": {
    "enabled": boolean,
    "domain": string | null,
    "concept_count": integer,
    "plugin_loaded": boolean,
    "plugin_type": string | null
  },

  "extraction_config": {
    "method": string,
    "confidence_threshold": float,
    "enabled_types": [string],
    "max_entities": integer
  },

  "ingestion_hooks": {
    "extraction_called": boolean,
    "hook_location": string,
    "invocation_count": integer,
    "last_invocation": string | null
  },

  "test_extraction": {
    "success": boolean,
    "sample_text": string,
    "entities_found": integer,
    "sample_entities": [
      {
        "name": string,
        "type": string,
        "confidence": float
      }
    ],
    "error": string | null
  },

  "diagnosis": {
    "severity": "info" | "warning" | "error" | "critical",
    "message": string,
    "root_cause": string | null,
    "suggestions": [string],
    "next_steps": [string]
  }
}
```

## Contract Assertions

### CA-1: Service Availability Check
```python
# GIVEN: Entity extraction module import attempted
# WHEN: Checking service status
# THEN: If import fails, service_status.available MUST be False
#       AND import_error MUST contain error message

if import_failed:
    assert service_status["available"] == False
    assert service_status["import_error"] is not None
    assert exit_code == 2
```

### CA-2: LLM Configuration Validation
```python
# GIVEN: Service available
# WHEN: Checking LLM configuration
# THEN: If provider is not "stub", api_key_set MUST be checked

if llm_status["provider"] != "stub":
    assert "api_key_set" in llm_status
    if llm_status["api_key_set"]:
        # Optionally validate key (may require API call)
        assert "api_key_valid" in llm_status
```

### CA-3: Ontology Status Consistency
```python
# GIVEN: Ontology enabled
# WHEN: Checking ontology status
# THEN: If enabled, plugin MUST be loaded and concept_count > 0

if ontology_status["enabled"]:
    assert ontology_status["plugin_loaded"] == True
    assert ontology_status["concept_count"] > 0
    assert ontology_status["domain"] is not None
```

### CA-4: Extraction Method Validity
```python
# GIVEN: Extraction config loaded
# WHEN: Checking extraction method
# THEN: Method MUST be one of: ontology_hybrid, rule_based, llm_only, pattern_based

valid_methods = ["ontology_hybrid", "rule_based", "llm_only", "pattern_based"]
assert extraction_config["method"] in valid_methods
```

### CA-5: Confidence Threshold Bounds
```python
# GIVEN: Extraction config loaded
# WHEN: Checking confidence threshold
# THEN: Threshold MUST be between 0.0 and 1.0

assert 0.0 <= extraction_config["confidence_threshold"] <= 1.0
```

### CA-6: Test Extraction Execution
```python
# GIVEN: Service available and configured
# WHEN: Running test extraction on sample text
# THEN: If success True, MUST have entities_found >= 0
#       If success False, MUST have error message

if test_extraction["success"]:
    assert test_extraction["entities_found"] >= 0
    assert test_extraction["error"] is None
else:
    assert test_extraction["error"] is not None
    assert exit_code in [2, 3]
```

### CA-7: Sample Entity Format
```python
# GIVEN: Test extraction found entities
# WHEN: Returning sample entities
# THEN: Each entity MUST have name, type, confidence fields

for entity in test_extraction["sample_entities"]:
    assert "name" in entity
    assert "type" in entity
    assert "confidence" in entity
    assert 0.0 <= entity["confidence"] <= 1.0
```

### CA-8: Ingestion Hook Detection
```python
# GIVEN: Service functional
# WHEN: Checking if extraction called during load_data
# THEN: If extraction_called False, exit code SHOULD be 1 (not invoked)

if not ingestion_hooks["extraction_called"]:
    assert exit_code == 1
    assert diagnosis["severity"] in ["warning", "error"]
    assert "not invoked" in diagnosis["message"].lower()
```

### CA-9: Suggestions for Not Invoked
```python
# GIVEN: Exit code == 1 (not invoked)
# WHEN: Generating diagnosis
# THEN: MUST provide suggestions for enabling extraction

if exit_code == 1:
    assert len(diagnosis["suggestions"]) >= 2
    assert any("load_documents" in s.lower() for s in diagnosis["suggestions"])
```

### CA-10: JSON Output Validity
```python
# GIVEN: Script execution completes
# WHEN: Parsing stdout
# THEN: Output MUST be valid JSON

import json
output = json.loads(stdout)
assert isinstance(output, dict)
assert "check_name" in output
assert "diagnosis" in output
```

## Example Scenarios

### Scenario 1: Service Configured But Not Invoked (Expected Initial State)

**Input**: Framework configured, LLM available, but extraction not called during load_data

**Expected Output**:
```json
{
  "check_name": "entity_extraction_verification",
  "timestamp": "2025-10-06T12:00:00.000Z",
  "execution_time_ms": 123.4,
  "service_status": {
    "available": true,
    "import_error": null,
    "version": "1.0.0",
    "module_path": "iris_rag/services/entity_extraction.py"
  },
  "llm_status": {
    "configured": true,
    "provider": "openai",
    "model": "gpt-4",
    "api_key_set": true,
    "api_key_valid": true
  },
  "ontology_status": {
    "enabled": true,
    "domain": "biomedical",
    "concept_count": 1250,
    "plugin_loaded": true,
    "plugin_type": "GeneralOntologyPlugin"
  },
  "extraction_config": {
    "method": "ontology_hybrid",
    "confidence_threshold": 0.7,
    "enabled_types": ["ENTITY", "CONCEPT", "PROCESS"],
    "max_entities": 100
  },
  "ingestion_hooks": {
    "extraction_called": false,
    "hook_location": "GraphRAGPipeline.load_documents",
    "invocation_count": 0,
    "last_invocation": null
  },
  "test_extraction": {
    "success": true,
    "sample_text": "COVID-19 is caused by SARS-CoV-2 virus and affects the respiratory system.",
    "entities_found": 3,
    "sample_entities": [
      {"name": "COVID-19", "type": "DISEASE", "confidence": 0.95},
      {"name": "SARS-CoV-2", "type": "VIRUS", "confidence": 0.92},
      {"name": "respiratory system", "type": "ANATOMY", "confidence": 0.88}
    ],
    "error": null
  },
  "diagnosis": {
    "severity": "warning",
    "message": "Entity extraction service is functional but not invoked during document loading",
    "root_cause": "GraphRAGPipeline.load_documents does not call entity extraction service",
    "suggestions": [
      "Add entity extraction invocation to GraphRAGPipeline.load_documents method",
      "Create separate make target for GraphRAG-specific data loading with extraction",
      "Verify entity_extraction.enabled=true in configuration"
    ],
    "next_steps": [
      "Review GraphRAGPipeline source code for extraction hooks",
      "Compare with working entity extraction examples",
      "Test extraction manually with sample documents"
    ]
  }
}
```

**Exit Code**: 1

### Scenario 2: LLM Not Configured (Service Error)

**Input**: Entity extraction service available but LLM API key not set

**Expected Output**:
```json
{
  "check_name": "entity_extraction_verification",
  "timestamp": "2025-10-06T12:00:00.000Z",
  "execution_time_ms": 89.2,
  "service_status": {
    "available": true,
    "import_error": null,
    "version": "1.0.0",
    "module_path": "iris_rag/services/entity_extraction.py"
  },
  "llm_status": {
    "configured": false,
    "provider": "openai",
    "model": "gpt-4",
    "api_key_set": false,
    "api_key_valid": null
  },
  "ontology_status": {
    "enabled": true,
    "domain": "biomedical",
    "concept_count": 1250,
    "plugin_loaded": true,
    "plugin_type": "GeneralOntologyPlugin"
  },
  "extraction_config": {
    "method": "ontology_hybrid",
    "confidence_threshold": 0.7,
    "enabled_types": ["ENTITY", "CONCEPT", "PROCESS"],
    "max_entities": 100
  },
  "ingestion_hooks": {
    "extraction_called": false,
    "hook_location": "GraphRAGPipeline.load_documents",
    "invocation_count": 0,
    "last_invocation": null
  },
  "test_extraction": {
    "success": true,
    "sample_text": "COVID-19 is caused by SARS-CoV-2 virus.",
    "entities_found": 2,
    "sample_entities": [
      {"name": "COVID-19", "type": "CONCEPT", "confidence": 0.85},
      {"name": "SARS-CoV-2", "type": "CONCEPT", "confidence": 0.82}
    ],
    "error": null
  },
  "diagnosis": {
    "severity": "warning",
    "message": "LLM not configured but extraction still functional using ontology-based method",
    "root_cause": "OPENAI_API_KEY environment variable not set",
    "suggestions": [
      "Set OPENAI_API_KEY in .env file for LLM-enhanced extraction",
      "Ontology-based extraction still works but may have lower accuracy",
      "Consider using rule-based method if LLM is unavailable"
    ],
    "next_steps": [
      "Configure LLM API key in .env file",
      "Test extraction quality with ontology-only method",
      "Review extraction method configuration"
    ]
  }
}
```

**Exit Code**: 0 (service works with ontology-only method)

### Scenario 3: Import Error (Critical Failure)

**Input**: Entity extraction module cannot be imported

**Expected Output**:
```json
{
  "check_name": "entity_extraction_verification",
  "timestamp": "2025-10-06T12:00:00.000Z",
  "execution_time_ms": 12.3,
  "service_status": {
    "available": false,
    "import_error": "ModuleNotFoundError: No module named 'iris_rag.services.entity_extraction'",
    "version": null,
    "module_path": null
  },
  "llm_status": {
    "configured": false,
    "provider": "unknown",
    "model": "unknown",
    "api_key_set": false,
    "api_key_valid": null
  },
  "ontology_status": {
    "enabled": false,
    "domain": null,
    "concept_count": 0,
    "plugin_loaded": false,
    "plugin_type": null
  },
  "extraction_config": {
    "method": "unknown",
    "confidence_threshold": 0.0,
    "enabled_types": [],
    "max_entities": 0
  },
  "ingestion_hooks": {
    "extraction_called": false,
    "hook_location": "unknown",
    "invocation_count": 0,
    "last_invocation": null
  },
  "test_extraction": {
    "success": false,
    "sample_text": "",
    "entities_found": 0,
    "sample_entities": [],
    "error": "Cannot test extraction - module import failed"
  },
  "diagnosis": {
    "severity": "critical",
    "message": "Entity extraction service unavailable - module import failed",
    "root_cause": "iris_rag.services.entity_extraction module missing or corrupted",
    "suggestions": [
      "Verify iris_rag package installation: pip install -e .",
      "Check for Python environment issues",
      "Reinstall dependencies: uv sync"
    ],
    "next_steps": [
      "Check package installation",
      "Review import errors in logs",
      "Verify Python environment is activated"
    ]
  }
}
```

**Exit Code**: 2

### Scenario 4: Fully Functional and Invoked (Success Case)

**Input**: Extraction service configured, invoked, and working

**Expected Output**:
```json
{
  "check_name": "entity_extraction_verification",
  "timestamp": "2025-10-06T12:00:00.000Z",
  "execution_time_ms": 234.5,
  "service_status": {
    "available": true,
    "import_error": null,
    "version": "1.0.0",
    "module_path": "iris_rag/services/entity_extraction.py"
  },
  "llm_status": {
    "configured": true,
    "provider": "openai",
    "model": "gpt-4",
    "api_key_set": true,
    "api_key_valid": true
  },
  "ontology_status": {
    "enabled": true,
    "domain": "biomedical",
    "concept_count": 1250,
    "plugin_loaded": true,
    "plugin_type": "GeneralOntologyPlugin"
  },
  "extraction_config": {
    "method": "ontology_hybrid",
    "confidence_threshold": 0.7,
    "enabled_types": ["ENTITY", "CONCEPT", "PROCESS"],
    "max_entities": 100
  },
  "ingestion_hooks": {
    "extraction_called": true,
    "hook_location": "GraphRAGPipeline.load_documents",
    "invocation_count": 10,
    "last_invocation": "2025-10-06T11:45:23.456Z"
  },
  "test_extraction": {
    "success": true,
    "sample_text": "COVID-19 is caused by SARS-CoV-2 virus and affects the respiratory system.",
    "entities_found": 3,
    "sample_entities": [
      {"name": "COVID-19", "type": "DISEASE", "confidence": 0.95},
      {"name": "SARS-CoV-2", "type": "VIRUS", "confidence": 0.92},
      {"name": "respiratory system", "type": "ANATOMY", "confidence": 0.88}
    ],
    "error": null
  },
  "diagnosis": {
    "severity": "info",
    "message": "Entity extraction service is fully functional and actively invoked",
    "root_cause": null,
    "suggestions": [],
    "next_steps": []
  }
}
```

**Exit Code**: 0

## Contract Tests

**Test File**: `tests/contract/test_entity_extraction_contract.py`

### Test Cases

1. **test_entity_extraction_verifier_exists**: Verify script file exists
2. **test_entity_extraction_verifier_executable**: Verify script can be executed
3. **test_entity_extraction_verifier_output_format**: Verify valid JSON output
4. **test_service_availability_detection**: Verify service import status checked
5. **test_llm_configuration_check**: Verify LLM config validation
6. **test_ontology_status_check**: Verify ontology plugin status
7. **test_extraction_config_validity**: Verify extraction method is valid
8. **test_confidence_threshold_bounds**: Verify threshold 0.0-1.0
9. **test_test_extraction_execution**: Verify test extraction runs
10. **test_ingestion_hook_detection**: Verify hook invocation tracking
11. **test_not_invoked_diagnosis**: Verify exit code 1 when not invoked
12. **test_import_error_handling**: Verify exit code 2 on import failure

---

**Contract Specification Complete** - Ready for TDD implementation
