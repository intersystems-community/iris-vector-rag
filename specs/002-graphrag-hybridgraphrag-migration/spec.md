# GraphRAG → HybridGraphRAG Migration Specification

**Specification ID**: 002-graphrag-hybridgraphrag-migration
**Version**: 1.0
**Status**: Draft
**Created**: 2024-09-29

## 1. Executive Summary

This specification defines the complete migration path from GraphRAG to HybridGraphRAG pipeline, including technical fixes, validation testing, and rollout strategy.

## 2. Goals and Objectives

### Primary Goals
- **Replace GraphRAG with HybridGraphRAG** as the default graph-based RAG solution
- **Maintain or improve performance** on all existing GraphRAG use cases
- **Add multi-modal retrieval capabilities** (vector + text + graph fusion)
- **Ensure backward compatibility** during transition period

### Success Criteria
- HybridGraphRAG matches or exceeds GraphRAG performance on graph reasoning tasks
- All missing table/schema issues resolved
- Comprehensive validation test suite passes
- Zero regression in existing functionality

## 3. Current State Analysis

### GraphRAG Current Performance
- **RAGAS Score**: 85.6% overall
- **Success Rate**: 100% on multi-hop queries
- **Retrieval Method**: `knowledge_graph_traversal`
- **Status**: ✅ Working with knowledge graph

### HybridGraphRAG Current Issues
```
❌ Missing Tables: KG_NODEEMBEDDINGS_OPTIMIZED, RDF_EDGES
❌ EmbeddingManager: 'get_embeddings' method missing
❌ Hybrid Fusion: Returning 0 contexts
❌ iris_graph_core: Incomplete schema setup
```

## 4. Technical Requirements

### 4.1 Schema Requirements
- **Create missing iris_graph_core tables**:
  - `KG_NODEEMBEDDINGS_OPTIMIZED`
  - `RDF_EDGES`
  - `RDF_LABELS`
  - `RDF_PROPS`
- **Ensure schema compatibility** with both Community and Licensed IRIS editions
- **Implement auto-migration** from GraphRAG schema to HybridGraphRAG schema

### 4.2 Integration Requirements
- **Fix EmbeddingManager integration** - implement `get_embeddings()` method
- **Ensure hybrid fusion works** - return meaningful contexts (>0)
- **Maintain GraphRAG compatibility** - `knowledge_graph_traversal` method intact
- **Add method selection logic** - intelligent routing between graph/vector/text/hybrid

### 4.3 Performance Requirements
- **Match GraphRAG performance** on existing benchmarks
- **Improve on complex queries** with hybrid fusion
- **Maintain sub-3s response times** for single-hop queries
- **Support concurrent queries** without schema conflicts

## 5. Implementation Plan

### Phase 1: Technical Fixes (Critical Path)
1. **Schema Management**
   - Create `HybridGraphRAGSchemaManager`
   - Implement missing table creation
   - Add schema validation and auto-repair

2. **EmbeddingManager Integration**
   - Fix `get_embeddings()` method
   - Ensure vector search integration
   - Test embedding retrieval pipeline

3. **Hybrid Fusion Engine**
   - Debug 0-context issue
   - Implement proper result aggregation
   - Add fallback mechanisms

### Phase 2: Validation Framework (Quality Assurance)
1. **Comparison Test Suite**
   - Head-to-head GraphRAG vs HybridGraphRAG
   - Same queries, same evaluation metrics
   - Performance and quality comparison

2. **Regression Testing**
   - Ensure no degradation in graph reasoning
   - Validate multi-hop query performance
   - Test knowledge graph traversal accuracy

3. **Stress Testing**
   - Concurrent query handling
   - Large document corpus scaling
   - Memory and performance profiling

### Phase 3: Migration Strategy (Risk Management)
1. **Feature Flag Implementation**
   - `--use-hybrid-graphrag` flag
   - `--use-legacy-graphrag` fallback
   - Configuration-driven pipeline selection

2. **Gradual Rollout**
   - Default to HybridGraphRAG for new deployments
   - Maintain GraphRAG for existing deployments
   - Monitoring and rollback capabilities

3. **Deprecation Timeline**
   - 2 release cycles with both pipelines
   - GraphRAG deprecation warnings
   - Final removal after validation period

## 6. Validation Testing Specification

### 6.1 Functional Validation
```python
def test_graphrag_vs_hybrid_comparison():
    """Direct head-to-head comparison test."""
    test_queries = [
        # Simple factual queries
        "What are the symptoms of diabetes?",
        "How is COVID-19 transmitted?",

        # Multi-hop reasoning queries
        "What drugs treat diseases that cause the same symptoms as diabetes?",
        "How are COVID transmission methods related to respiratory treatments?",

        # Complex entity relationship queries
        "What medications for cancer have side effects treated by heart disease drugs?",
        "Which vaccines work against diseases with similar transmission patterns?"
    ]

    for query in test_queries:
        graphrag_result = test_graphrag_pipeline(query)
        hybrid_result = test_hybrid_pipeline(query)

        assert hybrid_result.quality >= graphrag_result.quality
        assert hybrid_result.contexts > 0  # No empty results
        assert hybrid_result.answer_length > 50  # Meaningful answers
```

### 6.2 Performance Validation
- **Response Time**: HybridGraphRAG ≤ GraphRAG + 20% latency tolerance
- **Throughput**: Support same concurrent query load
- **Memory Usage**: No significant regression in memory consumption
- **Accuracy**: RAGAS scores within 5% margin

### 6.3 Integration Validation
- **Schema Compatibility**: Works with existing IRIS instances
- **API Compatibility**: Drop-in replacement for GraphRAG
- **Configuration**: Honors existing GraphRAG configurations
- **Error Handling**: Graceful degradation and error recovery

## 7. Risk Assessment

### High Risk
- **Schema Migration Failures**: Could break existing deployments
- **Performance Regression**: HybridGraphRAG slower than GraphRAG
- **Data Loss**: Migration process corrupts existing knowledge graphs

### Medium Risk
- **Compatibility Issues**: iris_graph_core dependencies
- **Feature Gaps**: Missing GraphRAG functionality in Hybrid
- **Resource Usage**: Higher memory/CPU requirements

### Mitigation Strategies
- **Comprehensive Testing**: Full validation before migration
- **Rollback Plan**: Ability to revert to GraphRAG quickly
- **Staged Deployment**: Gradual rollout with monitoring
- **Data Backup**: Schema and data backup before migration

## 8. Acceptance Criteria

### ✅ Technical Acceptance
- [ ] All missing tables created and functional
- [ ] EmbeddingManager integration working
- [ ] Hybrid fusion returns >0 contexts for test queries
- [ ] No SQL errors in HybridGraphRAG execution

### ✅ Performance Acceptance
- [ ] HybridGraphRAG ≥ 85% RAGAS score (matches current GraphRAG)
- [ ] Response times within 20% of GraphRAG baseline
- [ ] 100% success rate on validation query suite
- [ ] Memory usage within acceptable limits

### ✅ Quality Acceptance
- [ ] Head-to-head comparison shows HybridGraphRAG ≥ GraphRAG
- [ ] Knowledge graph traversal quality maintained
- [ ] Multi-hop reasoning capability preserved
- [ ] No regression in existing functionality

### ✅ Integration Acceptance
- [ ] Drop-in replacement for GraphRAG in existing code
- [ ] Configuration compatibility maintained
- [ ] Error handling and fallback mechanisms working
- [ ] Documentation updated with migration guide

## 9. Implementation Timeline

```
Week 1: Technical Fixes
- Day 1-2: Schema management and missing tables
- Day 3-4: EmbeddingManager integration
- Day 5: Hybrid fusion debugging

Week 2: Validation Framework
- Day 1-2: Comparison test suite development
- Day 3-4: Performance and regression testing
- Day 5: Stress testing and optimization

Week 3: Migration Implementation
- Day 1-2: Feature flags and configuration
- Day 3-4: Documentation and migration guides
- Day 5: Final validation and sign-off
```

## 10. Next Steps

1. **Immediate Actions**:
   - Create missing schema tables for iris_graph_core
   - Fix EmbeddingManager `get_embeddings()` method
   - Debug hybrid fusion 0-context issue

2. **Validation Setup**:
   - Implement head-to-head comparison framework
   - Create comprehensive test query suite
   - Set up performance monitoring

3. **Migration Execution**:
   - Implement feature flags for pipeline selection
   - Create migration documentation
   - Execute staged rollout plan

---

**Approval Required**: Technical Lead, Product Owner
**Implementation Owner**: Development Team
**Timeline**: 3 weeks
**Priority**: High