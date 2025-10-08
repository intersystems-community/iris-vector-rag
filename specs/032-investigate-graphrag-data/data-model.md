# Data Model: GraphRAG Investigation

**Feature**: 032-investigate-graphrag-data | **Date**: 2025-10-06
**Phase**: 1 (Design) | **Input**: Research findings + spec requirements

## Core Entities

### KnowledgeGraphState

Represents the current state of the knowledge graph in IRIS database.

```python
@dataclass
class KnowledgeGraphState:
    """State snapshot of GraphRAG knowledge graph tables."""

    # Table existence
    entities_table_exists: bool
    relationships_table_exists: bool
    communities_table_exists: bool

    # Row counts
    entity_count: int  # Total entities in RAG.Entities
    relationship_count: int  # Total relationships in RAG.Relationships
    community_count: int  # Total communities in RAG.Communities

    # Data quality metrics
    document_entity_links: int  # Entities with document_id set
    orphaned_entities: int  # Entities without document links
    entities_with_embeddings: int  # Entities with vector embeddings

    # Configuration state
    extraction_enabled: bool  # Whether entity extraction is configured
    schema_version: str  # Knowledge graph schema version

    # Timestamps
    last_entity_created: Optional[datetime]
    last_relationship_created: Optional[datetime]
```

**Validation Rules**:
- If any table exists, all three should exist (consistency check)
- If `entity_count > 0`, should have `document_entity_links > 0`
- `orphaned_entities` should be `<= entity_count`
- `entities_with_embeddings` should be `<= entity_count`

### EntityExtractionStatus

Represents the configuration and operational status of entity extraction service.

```python
@dataclass
class EntityExtractionStatus:
    """Status of entity extraction service configuration."""

    # Service availability
    service_available: bool  # Can import entity_extraction module
    import_error: Optional[str]  # Error message if import failed

    # LLM configuration
    llm_configured: bool  # LLM available for extraction
    llm_provider: str  # LLM provider name (OpenAI, Anthropic, stub)
    llm_model: str  # Model name

    # Ontology configuration
    ontology_enabled: bool  # Ontology plugin loaded
    ontology_domain: Optional[str]  # Detected ontology domain
    ontology_concept_count: int  # Number of ontology concepts loaded

    # Extraction settings
    extraction_method: str  # Method (ontology_hybrid, rule_based, llm_only)
    confidence_threshold: float  # Minimum confidence for entities
    enabled_entity_types: List[str]  # Entity types to extract
    max_entities_per_doc: int  # Maximum entities per document

    # Runtime state
    extraction_triggered: bool  # Whether extraction called during load
    documents_processed: int  # Documents that went through extraction
    entities_created: int  # Entities successfully created
    errors: List[str]  # Extraction errors if any
```

**Validation Rules**:
- If `service_available == False`, all other fields should indicate unavailability
- If `llm_configured == False`, extraction may still work with rule-based method
- If `ontology_enabled == True`, should have `ontology_concept_count > 0`
- `entities_created <= documents_processed * max_entities_per_doc`

### PipelineDataComparison

Represents data availability across different pipeline types for comparison.

```python
@dataclass
class PipelineDataComparison:
    """Comparison of data availability across pipelines."""

    pipeline_name: str  # Pipeline type (basic, crag, graphrag, etc.)

    # Vector store data
    vector_table_exists: bool
    vector_table_rows: int
    vector_dimension: int

    # Metadata table data
    metadata_table_exists: bool
    metadata_table_rows: int

    # Knowledge graph data (GraphRAG only)
    knowledge_graph_exists: bool  # All three KG tables exist
    knowledge_graph_rows: int  # Total entities + relationships

    # Retrieval performance
    retrieval_success_rate: float  # % of queries that return results
    average_documents_returned: float  # Avg docs per query

    # Data completeness score
    data_completeness: float  # 0.0-1.0 score of required data availability
```

**Validation Rules**:
- All pipelines should have `vector_table_exists == True` after `make load-data`
- GraphRAG should have `knowledge_graph_exists == True` for proper operation
- `retrieval_success_rate` should be `> 0` if `data_completeness > 0.5`
- Basic pipeline: `data_completeness` only depends on vector table
- GraphRAG: `data_completeness` depends on vector + knowledge graph

### DiagnosticResult

Unified result format for all diagnostic operations.

```python
@dataclass
class DiagnosticResult:
    """Result of a diagnostic check operation."""

    check_name: str  # Name of diagnostic check
    success: bool  # Whether check passed
    severity: str  # "info", "warning", "error", "critical"

    message: str  # Human-readable result message
    details: Dict[str, Any]  # Structured diagnostic data

    # Actionable recommendations
    suggestions: List[str]  # Suggested fixes
    next_steps: List[str]  # What to investigate next

    # Metadata
    timestamp: datetime
    execution_time_ms: float
```

**Validation Rules**:
- If `success == False`, should have at least one suggestion
- `severity` must be one of: "info", "warning", "error", "critical"
- `details` should contain all data needed to reproduce check

## API Contracts

### Graph Inspector Contract

**Script**: `scripts/inspect_knowledge_graph.py`

**Input**: None (reads from IRIS database via config)

**Output Format** (JSON to stdout):
```json
{
  "check_name": "knowledge_graph_inspection",
  "timestamp": "2025-10-06T12:34:56",
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
  "sample_entities": [
    {
      "id": "entity_001",
      "name": "Example Entity",
      "type": "CONCEPT",
      "document_id": "doc_123"
    }
  ],
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
    "message": "Knowledge graph is empty",
    "suggestions": [
      "Run entity extraction on loaded documents",
      "Check entity_extraction configuration in config"
    ]
  }
}
```

**Exit Codes**:
- `0`: Success (knowledge graph populated with data)
- `1`: Empty graph (tables exist but no data)
- `2`: Tables missing (schema not initialized)
- `3`: Database connection error

**State Transitions**:
- `Tables Missing (2)` → Create schema → `Empty Graph (1)` → Extract entities → `Success (0)`

### Entity Extraction Verifier Contract

**Script**: `scripts/verify_entity_extraction.py`

**Input**: None (reads from IRIS and framework config)

**Output Format** (JSON to stdout):
```json
{
  "check_name": "entity_extraction_verification",
  "timestamp": "2025-10-06T12:34:56",
  "service_status": {
    "available": true,
    "import_error": null,
    "version": "1.0.0"
  },
  "llm_status": {
    "configured": true,
    "provider": "openai",
    "model": "gpt-4",
    "api_key_set": true
  },
  "ontology_status": {
    "enabled": true,
    "domain": "biomedical",
    "concept_count": 1250,
    "plugin_loaded": true
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
    "invocation_count": 0
  },
  "test_extraction": {
    "success": true,
    "sample_text": "COVID-19 is caused by SARS-CoV-2 virus.",
    "entities_found": 2,
    "sample_entities": [
      {"name": "COVID-19", "type": "DISEASE"},
      {"name": "SARS-CoV-2", "type": "VIRUS"}
    ],
    "error": null
  },
  "diagnosis": {
    "severity": "warning",
    "message": "Entity extraction configured but not invoked during load_data",
    "suggestions": [
      "Add entity extraction call to load_documents workflow",
      "Create separate make target for GraphRAG data loading"
    ]
  }
}
```

**Exit Codes**:
- `0`: Enabled and functional (extraction works)
- `1`: Disabled or not invoked (service exists but not called)
- `2`: Service error (import failed, LLM unavailable)
- `3`: Configuration error (invalid settings)

### Pipeline Data Comparison Contract

**Script**: `scripts/compare_pipeline_data.py`

**Input**: None (inspects all pipeline data tables)

**Output Format** (JSON to stdout):
```json
{
  "check_name": "pipeline_data_comparison",
  "timestamp": "2025-10-06T12:34:56",
  "pipelines": {
    "basic": {
      "vector_table_rows": 142,
      "metadata_table_rows": 142,
      "knowledge_graph_rows": 0,
      "data_completeness": 1.0,
      "retrieval_success_rate": 0.85
    },
    "crag": {
      "vector_table_rows": 142,
      "metadata_table_rows": 142,
      "knowledge_graph_rows": 0,
      "data_completeness": 1.0,
      "retrieval_success_rate": 0.80
    },
    "graphrag": {
      "vector_table_rows": 142,
      "metadata_table_rows": 142,
      "knowledge_graph_rows": 0,
      "data_completeness": 0.5,
      "retrieval_success_rate": 0.0
    }
  },
  "diagnosis": {
    "severity": "error",
    "message": "GraphRAG missing knowledge graph data while other pipelines have sufficient vector data",
    "root_cause": "Entity extraction not executed during load_data",
    "suggestions": [
      "Add entity extraction to GraphRAG load_documents method",
      "Create make load-data-graphrag target for GraphRAG-specific loading"
    ]
  }
}
```

**Exit Codes**:
- `0`: All pipelines have required data
- `1`: Some pipelines missing required data
- `2`: Database connection error

## Data Relationships

```
KnowledgeGraphState
    ├─ Validates against EntityExtractionStatus
    │  (If extraction disabled, explains empty graph)
    │
    └─ Compares with PipelineDataComparison
       (GraphRAG should have KG data, others don't need it)

EntityExtractionStatus
    ├─ Explains KnowledgeGraphState
    │  (Why is graph empty? Extraction not configured/invoked)
    │
    └─ Guides fix recommendations
       (Enable extraction, configure LLM, load ontology)

PipelineDataComparison
    └─ Provides context for GraphRAG issue
       (Other pipelines work = infrastructure OK, GraphRAG-specific issue)

DiagnosticResult
    └─ Unified wrapper for all checks
       (Standard format for success/failure/suggestions)
```

## Diagnostic Workflow

```
1. Run inspect_knowledge_graph.py
   ↓
   Exit Code 1 (Empty Graph) or 2 (Tables Missing)
   ↓
2. Run verify_entity_extraction.py
   ↓
   Exit Code 1 (Not Invoked) - ROOT CAUSE FOUND
   ↓
3. Run compare_pipeline_data.py
   ↓
   Confirms: Basic/CRAG have data, GraphRAG missing KG data
   ↓
4. Generate Investigation Report
   ↓
   Root Cause: Entity extraction not invoked during load_data
   Fix: Add extraction to GraphRAG pipeline or create separate make target
```

---

**Data Model Complete** - Ready for contract implementation
