# Contract: Graph Inspector

**Feature**: 032-investigate-graphrag-data | **Date**: 2025-10-06
**Script**: `scripts/inspect_knowledge_graph.py`
**Purpose**: Inspect IRIS knowledge graph state for GraphRAG pipeline

## Command Interface

### Execution
```bash
python scripts/inspect_knowledge_graph.py
```

### Exit Codes
- **0**: Success - Knowledge graph populated with data
- **1**: Empty graph - Tables exist but contain no data
- **2**: Tables missing - Schema not initialized for GraphRAG
- **3**: Database connection error

### Output Format

**Medium**: JSON to stdout (one object per line for streaming)

**Schema**:
```json
{
  "check_name": "knowledge_graph_inspection",
  "timestamp": "2025-10-06T12:34:56.789Z",
  "execution_time_ms": 245.3,

  "tables_exist": {
    "entities": boolean,
    "relationships": boolean,
    "communities": boolean
  },

  "counts": {
    "entities": integer,
    "relationships": integer,
    "communities": integer
  },

  "sample_entities": [
    {
      "id": string,
      "name": string,
      "type": string,
      "document_id": string | null
    }
  ],

  "document_links": {
    "total_entities": integer,
    "linked": integer,
    "orphaned": integer
  },

  "data_quality": {
    "entities_with_embeddings": integer,
    "completeness_score": float  // 0.0 - 1.0
  },

  "diagnosis": {
    "severity": "info" | "warning" | "error" | "critical",
    "message": string,
    "suggestions": [string],
    "next_steps": [string]
  }
}
```

## Contract Assertions

### CA-1: Table Existence Consistency
```python
# GIVEN: Database connection is successful
# WHEN: Checking table existence
# THEN: Either all three tables exist OR none exist
#       (Partial schema is a critical error)

if any(tables_exist.values()):
    assert all(tables_exist.values()), "Partial schema detected - critical error"
```

### CA-2: Count Validity
```python
# GIVEN: tables_exist all True
# WHEN: Counting rows in tables
# THEN: All count values must be >= 0

assert counts["entities"] >= 0
assert counts["relationships"] >= 0
assert counts["communities"] >= 0
```

### CA-3: Empty Graph Detection
```python
# GIVEN: tables_exist all True
# WHEN: All counts == 0
# THEN: Exit code MUST be 1 (empty graph)

if all(count == 0 for count in counts.values()):
    assert exit_code == 1
    assert diagnosis["severity"] == "error"
    assert diagnosis["message"].lower().contains("empty")
```

### CA-4: Table Missing Detection
```python
# GIVEN: Database connection successful
# WHEN: Any table does not exist
# THEN: Exit code MUST be 2 (tables missing)

if not all(tables_exist.values()):
    assert exit_code == 2
    assert diagnosis["severity"] == "critical"
    assert "schema" in diagnosis["message"].lower()
```

### CA-5: Sample Entity Limit
```python
# GIVEN: entities count > 0
# WHEN: Returning sample entities
# THEN: Return at most 5 samples for brevity

assert len(sample_entities) <= 5
```

### CA-6: Document Link Consistency
```python
# GIVEN: Entity count > 0
# WHEN: Calculating document links
# THEN: linked + orphaned MUST equal total_entities

assert document_links["linked"] + document_links["orphaned"] == document_links["total_entities"]
assert document_links["total_entities"] == counts["entities"]
```

### CA-7: Completeness Score Bounds
```python
# GIVEN: Data quality metrics calculated
# WHEN: Computing completeness score
# THEN: Score must be between 0.0 and 1.0

assert 0.0 <= data_quality["completeness_score"] <= 1.0
```

### CA-8: Suggestions for Empty Graph
```python
# GIVEN: Exit code == 1 (empty graph)
# WHEN: Generating diagnosis
# THEN: Must provide at least 2 actionable suggestions

if exit_code == 1:
    assert len(diagnosis["suggestions"]) >= 2
    assert any("extract" in s.lower() for s in diagnosis["suggestions"])
```

### CA-9: JSON Output Validity
```python
# GIVEN: Script execution completes
# WHEN: Parsing stdout
# THEN: Output must be valid JSON

import json
output = json.loads(stdout)
assert isinstance(output, dict)
assert "check_name" in output
assert "diagnosis" in output
```

### CA-10: Database Connection Error Handling
```python
# GIVEN: Database connection fails
# WHEN: Script executes
# THEN: Exit code MUST be 3, provide connection troubleshooting

if connection_error:
    assert exit_code == 3
    assert diagnosis["severity"] == "critical"
    assert any("connection" in s.lower() for s in diagnosis["suggestions"])
```

## Example Scenarios

### Scenario 1: Empty Knowledge Graph (Expected Initial State)

**Input**: Database with vector data loaded, but no entity extraction run

**Expected Output**:
```json
{
  "check_name": "knowledge_graph_inspection",
  "timestamp": "2025-10-06T12:00:00.000Z",
  "execution_time_ms": 123.4,
  "tables_exist": {
    "entities": true,
    "relationships": true,
    "communities": true
  },
  "counts": {
    "entities": 0,
    "relationships": 0,
    "communities": 0
  },
  "sample_entities": [],
  "document_links": {
    "total_entities": 0,
    "linked": 0,
    "orphaned": 0
  },
  "data_quality": {
    "entities_with_embeddings": 0,
    "completeness_score": 0.0
  },
  "diagnosis": {
    "severity": "error",
    "message": "Knowledge graph is empty - no entities, relationships, or communities found",
    "suggestions": [
      "Run entity extraction on loaded documents using GraphRAG pipeline",
      "Check entity_extraction.enabled configuration in config file",
      "Verify ontology is loaded for entity type detection"
    ],
    "next_steps": [
      "Run verify_entity_extraction.py to check extraction service status",
      "Review GraphRAG load_documents workflow for extraction invocation"
    ]
  }
}
```

**Exit Code**: 1

### Scenario 2: Tables Missing (Schema Not Initialized)

**Input**: Fresh database without GraphRAG schema initialization

**Expected Output**:
```json
{
  "check_name": "knowledge_graph_inspection",
  "timestamp": "2025-10-06T12:00:00.000Z",
  "execution_time_ms": 45.2,
  "tables_exist": {
    "entities": false,
    "relationships": false,
    "communities": false
  },
  "counts": {
    "entities": 0,
    "relationships": 0,
    "communities": 0
  },
  "sample_entities": [],
  "document_links": {
    "total_entities": 0,
    "linked": 0,
    "orphaned": 0
  },
  "data_quality": {
    "entities_with_embeddings": 0,
    "completeness_score": 0.0
  },
  "diagnosis": {
    "severity": "critical",
    "message": "Knowledge graph schema not initialized - RAG.Entities, RAG.Relationships, RAG.Communities tables missing",
    "suggestions": [
      "Run schema initialization for GraphRAG pipeline",
      "Ensure SchemaManager.create_schema_manager('graphrag', ...) is called",
      "Check database permissions for table creation"
    ],
    "next_steps": [
      "Review schema_manager.py implementation",
      "Verify GraphRAG pipeline initialization workflow"
    ]
  }
}
```

**Exit Code**: 2

### Scenario 3: Populated Knowledge Graph (Success Case)

**Input**: Database with 10 documents loaded and entity extraction completed

**Expected Output**:
```json
{
  "check_name": "knowledge_graph_inspection",
  "timestamp": "2025-10-06T12:00:00.000Z",
  "execution_time_ms": 234.1,
  "tables_exist": {
    "entities": true,
    "relationships": true,
    "communities": true
  },
  "counts": {
    "entities": 127,
    "relationships": 348,
    "communities": 23
  },
  "sample_entities": [
    {"id": "e001", "name": "COVID-19", "type": "DISEASE", "document_id": "PMC123"},
    {"id": "e002", "name": "SARS-CoV-2", "type": "VIRUS", "document_id": "PMC123"},
    {"id": "e003", "name": "Spike Protein", "type": "PROTEIN", "document_id": "PMC124"},
    {"id": "e004", "name": "Antibody Response", "type": "PROCESS", "document_id": "PMC125"},
    {"id": "e005", "name": "Vaccine Efficacy", "type": "CONCEPT", "document_id": "PMC126"}
  ],
  "document_links": {
    "total_entities": 127,
    "linked": 127,
    "orphaned": 0
  },
  "data_quality": {
    "entities_with_embeddings": 127,
    "completeness_score": 1.0
  },
  "diagnosis": {
    "severity": "info",
    "message": "Knowledge graph is healthy - 127 entities, 348 relationships, 23 communities",
    "suggestions": [],
    "next_steps": []
  }
}
```

**Exit Code**: 0

## Contract Tests

**Test File**: `tests/contract/test_graph_inspector_contract.py`

### Test Cases

1. **test_graph_inspector_exists**: Verify script file exists at expected path
2. **test_graph_inspector_executable**: Verify script can be executed
3. **test_graph_inspector_output_format**: Verify output is valid JSON with required fields
4. **test_graph_inspector_empty_graph_detection**: Verify exit code 1 when graph is empty
5. **test_graph_inspector_tables_missing_detection**: Verify exit code 2 when tables missing
6. **test_graph_inspector_sample_limit**: Verify sample_entities contains at most 5 items
7. **test_graph_inspector_document_link_consistency**: Verify linked + orphaned == total
8. **test_graph_inspector_completeness_score_bounds**: Verify score is 0.0-1.0
9. **test_graph_inspector_suggestions_on_error**: Verify suggestions provided when exit != 0
10. **test_graph_inspector_connection_error_handling**: Verify exit code 3 on connection failure

---

**Contract Specification Complete** - Ready for TDD implementation
