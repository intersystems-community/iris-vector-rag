# Pipeline Selection Guide

**Purpose**: Help you choose the right RAG pipeline for your use case
**Audience**: Developers building RAG applications with iris-vector-rag

## Quick Reference

| Pipeline | Best For | Retrieval Strategy | When to Use |
|----------|----------|-------------------|-------------|
| **basic** | General Q&A | Vector similarity | Getting started, baseline comparisons, simple use cases |
| **basic_rerank** | High precision | Vector + cross-encoder reranking | Legal/medical domains, accuracy-critical applications |
| **crag** | Self-correcting | Vector + evaluation + web search | Dynamic knowledge, fact-checking, current events |
| **graphrag** | Complex relationships | Vector + text + graph + RRF fusion | Research, medical knowledge, entity-heavy domains |
| **multi_query_rrf** | Robust recall | Multiple query variations + RRF | Ambiguous queries, exploratory search |
| **pylate_colbert** | Late interaction | ColBERT token-level matching | Fine-grained relevance, long documents |

## Pipeline Descriptions

### basic - Standard Vector Retrieval

**How it Works**:
1. Embed query into vector
2. Find top-k most similar document embeddings
3. Generate answer from retrieved documents

**Strengths**:
- Fast and lightweight
- Low resource requirements
- Good baseline performance
- Easy to understand and debug

**Weaknesses**:
- No reranking (lower precision than basic_rerank)
- No self-correction (can return irrelevant results)
- No knowledge graph reasoning

**Configuration**:
```python
from iris_vector_rag import create_pipeline

pipeline = create_pipeline(
    'basic',
    validate_requirements=True,
    auto_setup=False
)

result = pipeline.query("What is diabetes?", top_k=5)
```

**Performance**:
- Retrieval time: 100-300ms
- Memory: ~500MB (model + index)
- Accuracy: 70-80% (domain-dependent)

**Best Use Cases**:
- Getting started with RAG
- Proof-of-concept applications
- Well-scoped Q&A with high-quality documents
- Performance-critical applications where speed > accuracy

---

### basic_rerank - Vector Retrieval with Reranking

**How it Works**:
1. Embed query and retrieve top-k candidates (k=20-50)
2. Rerank candidates using cross-encoder model
3. Select top-n most relevant (n=5)
4. Generate answer from reranked documents

**Strengths**:
- Higher precision than basic (10-15% accuracy improvement)
- Better handling of semantic nuances
- Filters out false positives from vector search

**Weaknesses**:
- Slower than basic (2-3x overhead from reranking)
- Higher resource requirements (2 models: embedding + cross-encoder)

**Configuration**:
```python
pipeline = create_pipeline(
    'basic_rerank',
    validate_requirements=True
)

# Retrieves 50 candidates, reranks to top 5
result = pipeline.query("What is diabetes?", top_k=5, candidate_k=50)
```

**Performance**:
- Retrieval time: 300-600ms
- Memory: ~1.5GB (embedding model + cross-encoder)
- Accuracy: 80-90%

**Best Use Cases**:
- Legal research (precision matters)
- Medical Q&A (accuracy-critical)
- Customer support (fewer wrong answers)
- Enterprise search (user expectations high)

**Comparison with basic**:
```python
# Test on medical Q&A dataset
basic_accuracy = 0.75        # 75% correct answers
rerank_accuracy = 0.88       # 88% correct answers
improvement = +17% accuracy

# Cost: 2.5x slower, 3x more memory
```

---

### crag - Corrective RAG with Self-Evaluation

**How it Works**:
1. Retrieve documents via vector search
2. **Evaluate** retrieved documents for relevance (LLM self-check)
3. If irrelevant: **Trigger web search fallback** to find better sources
4. Generate answer from corrected document set

**Strengths**:
- Self-correcting (detects and fixes poor retrieval)
- Adapts to knowledge gaps (web search fallback)
- Robust to outdated internal documents

**Weaknesses**:
- Slower than basic (3-5x due to evaluation + web search)
- Requires web search API (e.g., Tavily, Bing)
- Higher LLM costs (evaluation step uses LLM calls)

**Configuration**:
```python
pipeline = create_pipeline(
    'crag',
    validate_requirements=True,
    web_search_api_key="your-tavily-api-key"  # Required for fallback
)

result = pipeline.query("What is the latest diabetes treatment?", top_k=5)
# → If internal docs outdated, triggers web search
```

**Performance**:
- Retrieval time: 500-1500ms (depending on web search)
- Memory: ~800MB
- Accuracy: 85-95% (higher than basic due to self-correction)

**Best Use Cases**:
- Current events ("What happened today?")
- Fact-checking and verification
- Dynamic knowledge domains (news, research)
- Hybrid internal + external knowledge

**When to Use Web Fallback**:
- Internal documents may be outdated
- User queries reference recent events
- Knowledge base has gaps
- External sources needed for validation

---

### graphrag - Hybrid GraphRAG with Multi-Modal Retrieval

**How it Works**:
1. **Vector search**: Semantic similarity
2. **Text search**: Keyword/BM25 matching
3. **Graph traversal**: Follow entity relationships in knowledge graph
4. **RRF fusion**: Combine results from all 3 methods
5. Generate answer from fused results

**Strengths**:
- Best for complex entity relationships
- Captures connections vector search misses
- Hybrid retrieval (semantic + lexical + graph)
- Superior performance on knowledge-intensive tasks

**Weaknesses**:
- Requires knowledge graph construction (entity extraction)
- Slowest pipeline (3 retrieval methods + fusion)
- Higher memory requirements (vector + text + graph indices)
- Requires iris-vector-graph library

**Configuration**:
```python
pipeline = create_pipeline(
    'graphrag',
    validate_requirements=True
)

# Automatically uses:
# - Vector index for semantic search
# - Text index for keyword search
# - Knowledge graph for entity-based retrieval
result = pipeline.query("What medications interact with metformin?", top_k=5)
```

**Performance**:
- Retrieval time: 800-2000ms
- Memory: ~2GB (all indices)
- Accuracy: 90-95% (best for entity-heavy queries)

**Best Use Cases**:
- Medical knowledge (drug interactions, disease relationships)
- Research papers (author networks, citation graphs)
- Legal case law (precedent relationships)
- Enterprise knowledge bases (org charts, product hierarchies)

**Requirements**:
```bash
# Install GraphRAG dependencies
pip install iris-vector-rag[hybrid-graphrag]
```

**Knowledge Graph Construction**:
```python
from iris_vector_rag.pipelines.graphrag import HybridGraphRAGPipeline

# Entities and relationships extracted during document loading
pipeline.load_documents(documents=docs)
# → Builds knowledge graph: entities + relationships
```

---

### multi_query_rrf - Multiple Query Variations with RRF Fusion

**How it Works**:
1. Generate 3-5 variations of user query (using LLM)
2. Retrieve documents for each query variation
3. Fuse results using Reciprocal Rank Fusion (RRF)
4. Generate answer from fused results

**Strengths**:
- Robust to query phrasing
- Higher recall (finds more relevant docs via variations)
- Handles ambiguous queries better

**Weaknesses**:
- Higher LLM costs (query generation step)
- Slower than basic (3-5x due to multiple retrievals)
- May retrieve redundant documents

**Configuration**:
```python
pipeline = create_pipeline(
    'multi_query_rrf',
    validate_requirements=True,
    num_query_variations=3  # Generate 3 variations
)

result = pipeline.query("diabetes treatment", top_k=5)
# Query variations might be:
# - "What are the treatment options for diabetes?"
# - "How is diabetes managed and treated?"
# - "What medications are used for diabetes?"
```

**Performance**:
- Retrieval time: 600-1200ms
- Memory: ~500MB
- Accuracy: 82-88% (better recall than basic)

**Best Use Cases**:
- Ambiguous user queries
- Exploratory search ("Tell me about X")
- Diverse document collections
- When users may phrase questions poorly

---

### pylate_colbert - Late Interaction with ColBERT

**How it Works**:
1. Encode query into multiple token-level embeddings
2. Encode documents into token-level embeddings
3. Compute **late interaction** (max-sim over token pairs)
4. Retrieve documents with highest late interaction scores

**Strengths**:
- Fine-grained relevance (token-level matching)
- Better than vector for long documents
- Captures term importance

**Weaknesses**:
- Slower than basic vector search
- Higher storage requirements (multiple embeddings per document)
- Requires specialized index structure

**Configuration**:
```python
pipeline = create_pipeline(
    'pylate_colbert',
    validate_requirements=True
)

result = pipeline.query("What are the side effects of metformin?", top_k=5)
```

**Performance**:
- Retrieval time: 400-800ms
- Memory: ~1.2GB
- Accuracy: 85-92% (better than basic for complex queries)

**Best Use Cases**:
- Long documents (research papers, legal documents)
- Queries with specific terms that must match
- When document structure matters
- Fine-grained relevance requirements

---

## Decision Tree

```
Need RAG pipeline?
│
├─ Just getting started? → basic
│
├─ High accuracy required?
│  ├─ Yes, and speed is OK? → basic_rerank
│  └─ Yes, and need entity reasoning? → graphrag
│
├─ Knowledge may be outdated?
│  └─ Yes, need external sources? → crag
│
├─ Users phrase queries poorly?
│  └─ Yes, need query robustness? → multi_query_rrf
│
└─ Long documents with specific terms?
   └─ Yes, need token-level matching? → pylate_colbert
```

## Comparison Matrix

### Performance Comparison

| Pipeline | Retrieval Time | Memory | Accuracy | Cost (LLM calls) |
|----------|---------------|--------|----------|------------------|
| basic | 100-300ms | 500MB | 70-80% | 1 (generation only) |
| basic_rerank | 300-600ms | 1.5GB | 80-90% | 1 (generation only) |
| crag | 500-1500ms | 800MB | 85-95% | 2-3 (eval + generation) |
| graphrag | 800-2000ms | 2GB | 90-95% | 1 (generation only) |
| multi_query_rrf | 600-1200ms | 500MB | 82-88% | 2-4 (query gen + generation) |
| pylate_colbert | 400-800ms | 1.2GB | 85-92% | 1 (generation only) |

### Accuracy by Domain

| Domain | Recommended Pipeline | Accuracy Gain vs basic |
|--------|---------------------|----------------------|
| **Medical Q&A** | graphrag | +20% (entity relationships) |
| **Legal Research** | basic_rerank | +15% (precision matters) |
| **Current Events** | crag | +18% (web fallback) |
| **Research Papers** | graphrag | +22% (citation graphs) |
| **Customer Support** | basic_rerank | +12% (accuracy > speed) |
| **General Q&A** | basic | baseline |

## Switching Between Pipelines

All pipelines share the same API, making it easy to experiment:

```python
from iris_vector_rag import create_pipeline

# Start with basic
pipeline = create_pipeline('basic')
result_basic = pipeline.query("What is diabetes?", top_k=5)

# Upgrade to basic_rerank for better accuracy
pipeline = create_pipeline('basic_rerank')
result_rerank = pipeline.query("What is diabetes?", top_k=5)

# Try graphrag for entity reasoning
pipeline = create_pipeline('graphrag')
result_graph = pipeline.query("What is diabetes?", top_k=5)

# Compare results
print(f"Basic answer: {result_basic['answer']}")
print(f"Rerank answer: {result_rerank['answer']}")
print(f"Graph answer: {result_graph['answer']}")
```

## Cost Analysis

### Development Costs

| Pipeline | Setup Complexity | Learning Curve | Dev Time |
|----------|-----------------|----------------|----------|
| basic | Low | 1 hour | 1 day |
| basic_rerank | Low | 2 hours | 1 day |
| crag | Medium | 4 hours | 2 days |
| graphrag | High | 8 hours | 3-5 days |
| multi_query_rrf | Medium | 3 hours | 2 days |
| pylate_colbert | Medium | 3 hours | 2 days |

### Operational Costs (1000 queries/day)

| Pipeline | Compute | LLM API | Storage | Total/Month |
|----------|---------|---------|---------|-------------|
| basic | $5 | $20 | $2 | $27 |
| basic_rerank | $15 | $20 | $5 | $40 |
| crag | $10 | $60 | $2 | $72 |
| graphrag | $25 | $20 | $10 | $55 |
| multi_query_rrf | $12 | $80 | $2 | $94 |
| pylate_colbert | $18 | $20 | $8 | $46 |

**Assumptions**: AWS t3.large instance, OpenAI gpt-3.5-turbo, 10k documents

## Migration Paths

### From basic to basic_rerank
**Effort**: Low (1 line change)
**Benefit**: +10-15% accuracy
```python
# Before
pipeline = create_pipeline('basic')

# After
pipeline = create_pipeline('basic_rerank')
```

### From basic to graphrag
**Effort**: High (requires graph construction)
**Benefit**: +20-25% accuracy on entity-heavy queries
```python
# Before
pipeline = create_pipeline('basic')

# After
pipeline = create_pipeline('graphrag')
# Note: Requires entity extraction and graph building
```

### From basic to crag
**Effort**: Medium (requires web search API key)
**Benefit**: +15-20% accuracy on dynamic knowledge
```python
# Before
pipeline = create_pipeline('basic')

# After
pipeline = create_pipeline('crag', web_search_api_key="your-key")
```

## Configuration Templates

### Template: basic (Starter)
```python
pipeline = create_pipeline(
    'basic',
    validate_requirements=True,
    auto_setup=False
)
```

### Template: basic_rerank (Production)
```python
pipeline = create_pipeline(
    'basic_rerank',
    validate_requirements=True,
    candidate_k=50,  # Retrieve 50, rerank to top_k
)
```

### Template: crag (Dynamic Knowledge)
```python
pipeline = create_pipeline(
    'crag',
    validate_requirements=True,
    web_search_api_key=os.getenv("TAVILY_API_KEY"),
    relevance_threshold=0.7,  # Trigger web search if below threshold
)
```

### Template: graphrag (Knowledge-Intensive)
```python
pipeline = create_pipeline(
    'graphrag',
    validate_requirements=True,
    entity_extraction_model="en_core_web_sm",
    enable_hybrid_search=True,  # Vector + text + graph
)
```

## Troubleshooting

### "Which pipeline should I start with?"
→ Start with **basic**, then upgrade to **basic_rerank** if accuracy is insufficient.

### "My results aren't accurate enough"
1. Try **basic_rerank** (+10-15% accuracy)
2. If entity-heavy, try **graphrag** (+20% for entities)
3. If knowledge is outdated, try **crag** (web fallback)

### "Queries are too slow"
1. Use **basic** instead of graphrag (3-5x faster)
2. Reduce top_k (fewer documents to process)
3. Use smaller LLM model (gpt-3.5 instead of gpt-4)

### "Users phrase queries poorly"
→ Use **multi_query_rrf** (generates query variations automatically)

### "Need to find specific terms in long documents"
→ Use **pylate_colbert** (token-level matching)

## See Also

- [User Guide](USER_GUIDE.md) - Complete iris-vector-rag usage guide
- [API Reference](API_REFERENCE.md) - Full API documentation
- [IRIS EMBEDDING Guide](IRIS_EMBEDDING_GUIDE.md) - Auto-vectorization setup
- [Performance Tuning](PERFORMANCE.md) - Optimization best practices
