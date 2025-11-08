# IRIS DSPy Adapter - Implementation Checklist

**Feature**: 048-dspy-iris-adapter
**Goal**: Create IRIS database adapter for retrieve-dspy library

---

## ✅ Completed

- [x] Analyze Weaviate adapter interface
- [x] Understand ObjectFromDB model
- [x] Design IRIS adapter architecture
- [x] Write comprehensive feature spec

---

## 📋 Ready to Implement

### Phase 1: Development Environment Setup
- [ ] Fork retrieve-dspy repository
- [ ] Clone locally: `git clone https://github.com/YOUR_USERNAME/retrieve-dspy.git`
- [ ] Create feature branch: `git checkout -b feature/iris-adapter`
- [ ] Install dependencies: `pip install -e .`
- [ ] Verify tests run: `pytest tests/`

### Phase 2: Core Adapter Implementation
**File**: `retrieve_dspy/database/iris_database.py`

- [ ] Create file structure and imports
- [ ] Implement `_get_iris_connection_from_env()`
- [ ] Implement `_get_query_embedding()` using iris_rag
- [ ] Implement `_parse_vector_string()` helper
- [ ] Implement `_vector_search()` with VECTOR_COSINE
- [ ] Implement main `iris_search_tool()` function
- [ ] Implement `async_iris_search_tool()` wrapper
- [ ] Add docstrings and type hints

### Phase 3: Advanced Search Methods
- [ ] Implement `_hybrid_search()` combining vector + text
- [ ] Implement `_text_search()` using SQL LIKE
- [ ] Implement `_rrf_fusion_search()` for multi-signal retrieval
- [ ] Add tag filtering support across all methods
- [ ] Add error handling and logging

### Phase 4: Testing
**File**: `tests/database/test_iris_database.py`

- [ ] Test basic vector search
- [ ] Test async search
- [ ] Test tag filtering
- [ ] Test hybrid search
- [ ] Test RRF fusion
- [ ] Test error handling (connection failures, etc.)
- [ ] Test integration with DSPy retriever

### Phase 5: Examples
**Files**:
- `examples/iris_basic_rag.py`
- `examples/iris_multi_query.py`
- `examples/iris_clustering.py`

- [ ] Example: Basic RAG with IRIS backend
- [ ] Example: Multi-query fusion
- [ ] Example: Document clustering
- [ ] Example: Reranking with IRIS retrieval
- [ ] Add example data setup scripts

### Phase 6: Documentation
**Files**:
- `retrieve-dspy/README.md`
- `retrieve-dspy/docs/databases/iris.md` (new)

- [ ] Add IRIS section to main README
- [ ] Create dedicated IRIS documentation
- [ ] Document environment variables
- [ ] Add configuration examples
- [ ] Document IRIS-specific features (graph, SQL filters)

### Phase 7: Integration with rag-templates
**Files in rag-templates repo**:
- `examples/dspy/iris_dspy_basic.py`
- `examples/dspy/iris_dspy_advanced.py`
- `docs/DSPY_INTEGRATION.md`

- [ ] Create examples directory for DSPy
- [ ] Add basic DSPy + IRIS example
- [ ] Add advanced example with clustering
- [ ] Update CLAUDE.md with DSPy integration guide
- [ ] Add DSPy to recommended integrations

### Phase 8: Upstream Contribution
- [ ] Run all retrieve-dspy tests locally
- [ ] Ensure code style matches repository (black, ruff)
- [ ] Create pull request with detailed description
- [ ] Add IRIS to supported databases in README
- [ ] Respond to code review feedback

---

## 🎯 Quick Start Commands

```bash
# 1. Set up development environment
cd /path/to/workspace
git clone https://github.com/YOUR_USERNAME/retrieve-dspy.git
cd retrieve-dspy
git checkout -b feature/iris-adapter
pip install -e ".[dev]"

# 2. Create adapter file
mkdir -p retrieve_dspy/database
touch retrieve_dspy/database/iris_database.py

# 3. Run existing tests to verify setup
pytest tests/ -v

# 4. Implement core adapter (see spec.md)
# Edit retrieve_dspy/database/iris_database.py

# 5. Create tests
mkdir -p tests/database
touch tests/database/test_iris_database.py

# 6. Run IRIS-specific tests
pytest tests/database/test_iris_database.py -v

# 7. Create examples
mkdir -p examples/iris
touch examples/iris/basic_rag.py
touch examples/iris/multi_query.py

# 8. Test examples
python examples/iris/basic_rag.py
```

---

## 📦 Dependencies

### Required in retrieve-dspy
```toml
# pyproject.toml additions
[project.optional-dependencies]
iris = [
    "iris-vector-graph>=2.0.0",  # If using graph features
]
```

### Required Environment Variables
```bash
# IRIS Connection
export IRIS_HOST="localhost"
export IRIS_PORT="1972"
export IRIS_NAMESPACE="USER"
export IRIS_USERNAME="_SYSTEM"
export IRIS_PASSWORD="SYS"

# Optional: Use existing iris_rag config
export IRIS_RAG_CONFIG_PATH="/path/to/iris_rag/config"
```

---

## 🧪 Testing Checklist

### Unit Tests
- [ ] Test iris_search_tool with mock connection
- [ ] Test _vector_search SQL generation
- [ ] Test _parse_vector_string with various formats
- [ ] Test ObjectFromDB conversion
- [ ] Test error handling for missing connection
- [ ] Test error handling for invalid table names

### Integration Tests (requires running IRIS)
- [ ] Test against real IRIS database
- [ ] Test with 1K document corpus
- [ ] Test with 10K document corpus
- [ ] Benchmark latency (should be < 100ms p95)
- [ ] Test concurrent queries
- [ ] Test with different embedding dimensions

### DSPy Integration Tests
- [ ] Test dspy.Retrieve with IRIS backend
- [ ] Test MultiQueryRetriever
- [ ] Test ClusterDocuments with IRIS vectors
- [ ] Test RetrieveRerank pipeline
- [ ] Test with dspy.ChainOfThought

---

## 🎨 Code Style Guidelines

### Follow retrieve-dspy Patterns
```python
# Use type hints
def iris_search_tool(
    query: str,
    collection_name: str,
    ...
) -> list[ObjectFromDB]:

# Use docstrings in Google style
"""Search IRIS database using vector similarity.

Args:
    query: Search query text
    collection_name: IRIS table name

Returns:
    List of search results as ObjectFromDB objects
"""

# Use logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"Searching IRIS: {collection_name}")
```

### Error Handling
```python
try:
    connection = _get_iris_connection_from_env()
except Exception as e:
    logger.error(f"Failed to connect to IRIS: {e}")
    raise ConnectionError(
        f"Cannot connect to IRIS. Check environment variables: {e}"
    )
```

---

## 🚀 Performance Targets

- Vector search latency: < 100ms p95 for 10K docs
- Hybrid search latency: < 200ms p95 for 10K docs
- RRF fusion latency: < 300ms p95 for 10K docs
- Embedding generation: < 50ms for single query
- Connection pool: Reuse connections (don't create per query)

---

## 📊 Success Criteria

### MVP (Minimum Viable Product)
- [ ] iris_search_tool() works with vector search
- [ ] Returns correctly formatted ObjectFromDB objects
- [ ] Has basic tests
- [ ] Has one working example
- [ ] Documentation in README

### Full Feature Parity
- [ ] All search methods implemented (vector, hybrid, text, RRF)
- [ ] Async support working
- [ ] Tag filtering working
- [ ] Comprehensive test coverage (>80%)
- [ ] Multiple examples
- [ ] Full documentation

### Upstream Contribution
- [ ] Pull request submitted
- [ ] Tests passing in CI
- [ ] Code review completed
- [ ] Merged to main branch
- [ ] IRIS listed in supported databases

---

## 💡 IRIS-Specific Features to Highlight

When documenting, emphasize these IRIS advantages:

1. **Enterprise Grade**: IRIS is a battle-tested enterprise database
2. **Native SQL**: Combine vector search with complex SQL queries
3. **HNSW Optimization**: 50x faster than basic vector search
4. **Graph Integration**: Enhance retrieval with knowledge graph traversal
5. **Hybrid Search**: Native RRF fusion of multiple search signals
6. **Production Ready**: Connection pooling, transactions, ACID guarantees

---

## 🤝 Collaboration Opportunities

### With retrieve-dspy Maintainers
- Discuss API design choices
- Get feedback on implementation approach
- Coordinate on testing requirements
- Plan documentation structure

### With IRIS Community
- Share DSPy integration in IRIS forums
- Create blog post or tutorial
- Present at community events
- Gather feedback from early adopters

---

## 📚 Additional Resources

- retrieve-dspy repo: https://github.com/weaviate/retrieve-dspy
- DSPy documentation: https://dspy-docs.vercel.app/
- IRIS Vector Search docs: (internal docs)
- Example Weaviate adapter: retrieve_dspy/database/weaviate_database.py

---

**Current Status**: Spec complete, ready to implement Phase 1
**Next Action**: Fork retrieve-dspy repository and set up development environment
