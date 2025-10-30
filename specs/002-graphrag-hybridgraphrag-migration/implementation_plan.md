# Implementation Plan: GraphRAG → HybridGraphRAG Migration

## Phase 1: Technical Fixes (IMMEDIATE)

### 1.1 Schema Management - Missing Tables
**Issue**: `KG_NODEEMBEDDINGS_OPTIMIZED`, `RDF_EDGES`, `RDF_LABELS`, `RDF_PROPS` tables missing

**Implementation**:
```python
# Create HybridGraphRAGSchemaManager
class HybridGraphRAGSchemaManager(SchemaManager):
    def ensure_iris_graph_core_tables(self):
        """Ensure all iris_graph_core tables exist."""
        required_tables = [
            "KG_NODEEMBEDDINGS_OPTIMIZED",
            "RDF_EDGES",
            "RDF_LABELS",
            "RDF_PROPS"
        ]
        # Implementation details...
```

### 1.2 EmbeddingManager Integration
**Issue**: `'EmbeddingManager' object has no attribute 'get_embeddings'`

**Implementation**:
```python
# Fix EmbeddingManager.get_embeddings() method
def get_embeddings(self, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text list."""
    return self.embedding_model.encode(texts)
```

### 1.3 Hybrid Fusion Engine
**Issue**: Hybrid fusion returning 0 contexts

**Investigation**:
- Debug RRF (Reciprocal Rank Fusion) logic
- Check vector/text/graph result aggregation
- Validate query routing logic

## Phase 2: Validation Framework

### 2.1 Head-to-Head Comparison Test
```python
def test_graphrag_vs_hybridgraphrag_comparison():
    """Execute comprehensive comparison between pipelines."""

    test_scenarios = [
        ("simple_factual", simple_queries),
        ("multi_hop_reasoning", multihop_queries),
        ("complex_entity_relations", complex_queries)
    ]

    results = {}
    for scenario_name, queries in test_scenarios:
        results[scenario_name] = compare_pipelines(
            graphrag_pipeline,
            hybridgraphrag_pipeline,
            queries
        )

    return generate_comparison_report(results)
```

### 2.2 Performance Validation
- Response time benchmarking
- Memory usage profiling
- Concurrent query testing
- RAGAS score comparison

## Phase 3: Migration Strategy

### 3.1 Feature Flag Implementation
```python
# Add to pipeline factory
def create_pipeline(pipeline_type: str, **kwargs):
    if pipeline_type == "graphrag":
        use_hybrid = kwargs.get('use_hybrid_graphrag', True)  # Default to hybrid
        if use_hybrid:
            return create_pipeline("hybrid_graphrag", **kwargs)
    return _create_legacy_pipeline(pipeline_type, **kwargs)
```

### 3.2 Gradual Rollout
- Phase 1: Opt-in hybrid for new deployments
- Phase 2: Default hybrid with GraphRAG fallback
- Phase 3: Deprecate GraphRAG after validation period

---

## IMMEDIATE EXECUTION PLAN

Let's start implementing Phase 1 right now...