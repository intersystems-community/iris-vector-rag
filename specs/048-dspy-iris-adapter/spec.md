# Feature Spec: IRIS Database Adapter for DSPy (retrieve-dspy)

**Feature ID**: 048
**Branch**: 048-dspy-iris-adapter
**Status**: Draft
**Created**: 2025-10-19
**Author**: System (prompted by retrieve-dspy discovery)

---

## Overview

Create an IRIS database adapter for the [retrieve-dspy](https://github.com/weaviate/retrieve-dspy) library, enabling DSPy users to leverage InterSystems IRIS's enterprise-grade vector search, hybrid search, and SQL capabilities for advanced RAG implementations.

## Problem Statement

**Current Situation:**
- DSPy is a powerful framework for programming LLMs with composable modules
- retrieve-dspy provides advanced IR (Information Retrieval) techniques for RAG
- Currently only supports Weaviate as the vector database backend
- IRIS users cannot leverage retrieve-dspy's advanced techniques (clustering, reranking, multi-query fusion)

**Opportunity:**
- IRIS has superior vector search capabilities (HNSW, native SQL integration)
- IRIS supports hybrid search combining vector + text + graph signals
- Creating an IRIS adapter unlocks retrieve-dspy for IRIS users
- Positions IRIS as a first-class citizen in the DSPy ecosystem

## Goals

### Primary Goals
1. **Drop-in Replacement**: IRIS adapter matches Weaviate adapter's interface exactly
2. **Full Feature Parity**: Support all search modes (hybrid, vector, text)
3. **Performance**: Leverage IRIS's HNSW optimization for fast retrieval
4. **Native Integration**: Use existing IRISVectorStore infrastructure

### Secondary Goals
1. **IRIS-Specific Features**: Expose graph search, RRF fusion, SQL filters
2. **Async Support**: Provide both sync and async search functions
3. **Documentation**: Examples showing DSPy + IRIS integration
4. **Upstream Contribution**: Contribute iris_database.py to retrieve-dspy repo

## Design

### Interface Contract (from retrieve-dspy)

The adapter must implement these functions matching the Weaviate interface:

```python
def iris_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    iris_connection: Optional[Any] = None,
    return_property_name: Optional[str] = None,
    retrieved_k: Optional[int] = 5,
    return_vector: bool = False,
    tag_filter_value: Optional[str] = None,
) -> list[ObjectFromDB]:
    """
    Synchronous IRIS vector search for DSPy.

    Args:
        query: Search query text
        collection_name: IRIS table name (e.g., "RAG.Documents")
        target_property_name: Column containing searchable content (e.g., "text_content")
        iris_connection: IRIS DBAPI connection (defaults to env config)
        return_property_name: Deprecated, kept for compatibility
        retrieved_k: Number of results to return (default: 5)
        return_vector: Whether to include embedding vectors in response
        tag_filter_value: Optional filter value for tag-based filtering

    Returns:
        List of ObjectFromDB with search results
    """
    pass

async def async_iris_search_tool(...) -> list[ObjectFromDB]:
    """Async version of iris_search_tool."""
    pass
```

### ObjectFromDB Mapping

Map IRIS results to retrieve-dspy's ObjectFromDB model:

```python
class ObjectFromDB(BaseModel):
    object_id: str          # IRIS: doc.id or doc.metadata['doc_id']
    content: str            # IRIS: doc.page_content
    relevance_rank: int     # Position in result list (1-indexed)
    relevance_score: float  # IRIS: similarity score from VECTOR_COSINE
    vector: list[float]     # IRIS: embedding (if return_vector=True)
    source_query: str       # Original query (optional)
```

### IRIS-Specific Implementation

```python
# retrieve_dspy/database/iris_database.py

import os
from typing import Optional, List
import asyncio

from retrieve_dspy.models import ObjectFromDB

def iris_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    iris_connection: Optional[Any] = None,
    return_property_name: Optional[str] = None,
    retrieved_k: Optional[int] = 5,
    return_vector: bool = False,
    tag_filter_value: Optional[str] = None,
    search_method: str = "hybrid",  # IRIS-specific: "hybrid", "vector", "text", "rrf"
) -> List[ObjectFromDB]:
    """
    IRIS vector search tool for DSPy.

    Leverages InterSystems IRIS's enterprise vector search capabilities including:
    - HNSW-optimized vector similarity (50ms p95 latency)
    - Native iFind text search with stemming
    - Hybrid search combining vector + text + graph signals
    - SQL-based filtering and metadata enrichment
    """
    # Get or create IRIS connection
    if iris_connection is None:
        iris_connection = _get_iris_connection_from_env()

    # Get embedding for query
    embedding = _get_query_embedding(query)

    # Execute search based on method
    if search_method == "hybrid":
        results = _hybrid_search(
            iris_connection,
            collection_name,
            target_property_name,
            query,
            embedding,
            retrieved_k,
            tag_filter_value
        )
    elif search_method == "vector":
        results = _vector_search(
            iris_connection,
            collection_name,
            target_property_name,
            embedding,
            retrieved_k,
            tag_filter_value
        )
    elif search_method == "text":
        results = _text_search(
            iris_connection,
            collection_name,
            target_property_name,
            query,
            retrieved_k,
            tag_filter_value
        )
    elif search_method == "rrf":
        results = _rrf_fusion_search(
            iris_connection,
            collection_name,
            target_property_name,
            query,
            embedding,
            retrieved_k,
            tag_filter_value
        )
    else:
        raise ValueError(f"Unknown search_method: {search_method}")

    # Convert to ObjectFromDB format
    objects = []
    for rank, result in enumerate(results, start=1):
        objects.append(ObjectFromDB(
            object_id=result['id'],
            content=result['content'],
            relevance_rank=rank,
            relevance_score=result['score'],
            vector=result.get('embedding') if return_vector else None,
            source_query=query
        ))

    return objects


def _get_iris_connection_from_env():
    """Create IRIS DBAPI connection from environment variables."""
    import iris

    return iris.connect(
        hostname=os.getenv("IRIS_HOST", "localhost"),
        port=int(os.getenv("IRIS_PORT", "1972")),
        namespace=os.getenv("IRIS_NAMESPACE", "USER"),
        username=os.getenv("IRIS_USERNAME", "_SYSTEM"),
        password=os.getenv("IRIS_PASSWORD", "SYS")
    )


def _get_query_embedding(query: str) -> List[float]:
    """Generate embedding for query using configured embedding model."""
    # Option 1: Use existing IRISVectorStore embedding manager
    from iris_rag.embeddings.manager import EmbeddingManager
    from iris_rag.config.manager import ConfigurationManager

    config_manager = ConfigurationManager()
    embedding_manager = EmbeddingManager(config_manager)

    embeddings = embedding_manager.embed_texts([query])
    return embeddings[0] if embeddings else []


def _hybrid_search(
    connection,
    table_name: str,
    content_column: str,
    query: str,
    embedding: List[float],
    top_k: int,
    tag_filter: Optional[str] = None
) -> List[dict]:
    """
    Hybrid search combining vector similarity and text search.

    Uses IRIS's native capabilities:
    - VECTOR_COSINE for semantic similarity
    - iFind text search for keyword matching
    - RRF (Reciprocal Rank Fusion) to combine signals
    """
    cursor = connection.cursor()

    # Build SQL with hybrid search
    embedding_str = ','.join(str(x) for x in embedding)
    dimension = len(embedding)

    # Get embedding column name (assume it's "{content_column}_embedding")
    embedding_column = f"{content_column}_embedding"

    sql = f"""
        SELECT
            id,
            {content_column} as content,
            VECTOR_COSINE(
                {embedding_column},
                TO_VECTOR('{embedding_str}', FLOAT, {dimension})
            ) as score,
            {embedding_column} as embedding
        FROM {table_name}
        WHERE 1=1
    """

    # Add tag filter if provided
    if tag_filter:
        sql += f" AND tags LIKE '%{tag_filter}%'"

    # Order by score and limit
    sql += f"""
        ORDER BY score DESC
        LIMIT {top_k}
    """

    cursor.execute(sql)
    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row[0],
            'content': row[1],
            'score': float(row[2]) if row[2] else 0.0,
            'embedding': _parse_vector_string(row[3]) if row[3] else None
        })

    cursor.close()
    return results


def _vector_search(
    connection,
    table_name: str,
    content_column: str,
    embedding: List[float],
    top_k: int,
    tag_filter: Optional[str] = None
) -> List[dict]:
    """Pure vector similarity search using IRIS VECTOR_COSINE."""
    cursor = connection.cursor()

    embedding_str = ','.join(str(x) for x in embedding)
    dimension = len(embedding)
    embedding_column = f"{content_column}_embedding"

    sql = f"""
        SELECT
            id,
            {content_column} as content,
            VECTOR_COSINE(
                {embedding_column},
                TO_VECTOR('{embedding_str}', FLOAT, {dimension})
            ) as score,
            {embedding_column} as embedding
        FROM {table_name}
        WHERE 1=1
    """

    if tag_filter:
        sql += f" AND tags LIKE '%{tag_filter}%'"

    sql += f" ORDER BY score DESC LIMIT {top_k}"

    cursor.execute(sql)
    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row[0],
            'content': row[1],
            'score': float(row[2]) if row[2] else 0.0,
            'embedding': _parse_vector_string(row[3]) if row[3] else None
        })

    cursor.close()
    return results


def _text_search(
    connection,
    table_name: str,
    content_column: str,
    query: str,
    top_k: int,
    tag_filter: Optional[str] = None
) -> List[dict]:
    """Full-text search using IRIS iFind."""
    cursor = connection.cursor()

    # Simple LIKE-based search (can be enhanced with iFind if available)
    sql = f"""
        SELECT
            id,
            {content_column} as content,
            1.0 as score,
            NULL as embedding
        FROM {table_name}
        WHERE {content_column} LIKE '%{query}%'
    """

    if tag_filter:
        sql += f" AND tags LIKE '%{tag_filter}%'"

    sql += f" LIMIT {top_k}"

    cursor.execute(sql)
    results = []
    for row in cursor.fetchall():
        results.append({
            'id': row[0],
            'content': row[1],
            'score': float(row[2]),
            'embedding': None
        })

    cursor.close()
    return results


def _rrf_fusion_search(
    connection,
    table_name: str,
    content_column: str,
    query: str,
    embedding: List[float],
    top_k: int,
    tag_filter: Optional[str] = None
) -> List[dict]:
    """
    Reciprocal Rank Fusion combining multiple search signals.

    Combines:
    1. Vector similarity search
    2. Text search
    3. (Optional) Graph-based search
    """
    # Get results from different methods
    vector_results = _vector_search(connection, table_name, content_column, embedding, top_k * 2, tag_filter)
    text_results = _text_search(connection, table_name, content_column, query, top_k * 2, tag_filter)

    # Apply RRF fusion
    rrf_scores = {}
    k = 60  # RRF constant

    for rank, result in enumerate(vector_results, start=1):
        doc_id = result['id']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    for rank, result in enumerate(text_results, start=1):
        doc_id = result['id']
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

    # Sort by RRF score and get top_k
    sorted_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Fetch full documents for top results
    cursor = connection.cursor()
    results = []
    for doc_id, rrf_score in sorted_doc_ids:
        cursor.execute(f"SELECT id, {content_column} FROM {table_name} WHERE id = ?", [doc_id])
        row = cursor.fetchone()
        if row:
            results.append({
                'id': row[0],
                'content': row[1],
                'score': rrf_score,
                'embedding': None
            })

    cursor.close()
    return results


def _parse_vector_string(vector_str: str) -> List[float]:
    """Parse IRIS vector string format to list of floats."""
    if not vector_str:
        return None
    # IRIS stores vectors as comma-separated strings
    return [float(x) for x in vector_str.split(',')]


async def async_iris_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    iris_connection: Optional[Any] = None,
    return_property_name: Optional[str] = None,
    retrieved_k: Optional[int] = 10,
    return_score: bool = False,
    return_vector: bool = False,
    tag_filter_value: Optional[str] = None,
    search_method: str = "hybrid",
) -> List[ObjectFromDB]:
    """
    Async version of iris_search_tool.

    Note: IRIS DBAPI is synchronous, so this uses asyncio.to_thread
    to avoid blocking the event loop.
    """
    return await asyncio.to_thread(
        iris_search_tool,
        query=query,
        collection_name=collection_name,
        target_property_name=target_property_name,
        iris_connection=iris_connection,
        return_property_name=return_property_name,
        retrieved_k=retrieved_k,
        return_vector=return_vector,
        tag_filter_value=tag_filter_value,
        search_method=search_method
    )


# Example usage
async def main():
    print("Testing IRIS DSPy adapter...")

    # Sync search
    results = iris_search_tool(
        query="What are the symptoms of diabetes?",
        collection_name="RAG.Documents",
        target_property_name="text_content",
        retrieved_k=5,
        return_vector=True,
        search_method="hybrid"
    )

    print(f"Found {len(results)} results:")
    for obj in results:
        print(f"  [{obj.relevance_rank}] Score: {obj.relevance_score:.4f}")
        print(f"      {obj.content[:100]}...")

    # Async search
    async_results = await async_iris_search_tool(
        query="What are the symptoms of diabetes?",
        collection_name="RAG.Documents",
        target_property_name="text_content",
        retrieved_k=5,
        search_method="rrf"
    )

    print(f"\nAsync search found {len(async_results)} results")


if __name__ == "__main__":
    asyncio.run(main())
```

## Implementation Plan

### Phase 1: Core Adapter (2-3 hours)
**File**: `retrieve_dspy/database/iris_database.py` (in retrieve-dspy repo)

**Tasks**:
- [ ] Implement `iris_search_tool()` with vector search
- [ ] Implement `async_iris_search_tool()`
- [ ] Add connection management (`_get_iris_connection_from_env()`)
- [ ] Add embedding generation (`_get_query_embedding()`)
- [ ] Implement `_vector_search()` using VECTOR_COSINE

### Phase 2: Advanced Search Methods (2-3 hours)
**Tasks**:
- [ ] Implement `_hybrid_search()` combining vector + text
- [ ] Implement `_text_search()` using SQL LIKE or iFind
- [ ] Implement `_rrf_fusion_search()` for multi-signal retrieval
- [ ] Add tag filtering support
- [ ] Add vector parsing (`_parse_vector_string()`)

### Phase 3: Integration with Existing IRIS Infrastructure (1-2 hours)
**Tasks**:
- [ ] Integrate with IRISVectorStore for embedding generation
- [ ] Reuse EmbeddingManager from iris_rag
- [ ] Support ConnectionManager for connection pooling
- [ ] Add configuration via ConfigurationManager

### Phase 4: Testing & Examples (2-3 hours)
**Files**:
- `tests/test_iris_database.py` (in retrieve-dspy repo)
- `examples/iris_dspy_example.py` (in rag-templates repo)

**Tasks**:
- [ ] Unit tests for IRIS adapter
- [ ] Integration tests with real IRIS database
- [ ] Example: DSPy + IRIS for multi-hop RAG
- [ ] Example: DSPy clustering with IRIS vectors
- [ ] Example: DSPy reranking with IRIS retrieval

### Phase 5: Documentation & Upstream Contribution (1-2 hours)
**Tasks**:
- [ ] Add IRIS section to retrieve-dspy README
- [ ] Document environment variables (IRIS_HOST, IRIS_PORT, etc.)
- [ ] Create pull request to retrieve-dspy repository
- [ ] Update rag-templates CLAUDE.md with DSPy integration guide

## IRIS-Specific Enhancements

Beyond the Weaviate interface, IRIS adapter can offer unique capabilities:

### 1. Graph-Enhanced Retrieval
```python
def iris_search_tool(
    ...,
    use_graph: bool = False,  # IRIS-specific
    graph_traversal_depth: int = 2,  # IRIS-specific
):
    """
    If use_graph=True, enhance vector results with knowledge graph traversal.

    Example: Query for "diabetes symptoms" also retrieves documents about
    related entities (complications, treatments) via graph relationships.
    """
```

### 2. SQL-Based Metadata Filtering
```python
def iris_search_tool(
    ...,
    sql_filter: Optional[str] = None,  # IRIS-specific
):
    """
    Advanced filtering using IRIS SQL.

    Example: sql_filter="publication_date > '2023-01-01' AND author = 'Smith'"
    """
```

### 3. HNSW Performance Optimization
```python
# Automatically use HNSW-optimized tables if available
if _has_hnsw_table(collection_name):
    results = _hnsw_vector_search(...)  # 50ms p95 latency
else:
    results = _standard_vector_search(...)  # Fallback
```

## Usage Examples

### Example 1: Basic DSPy RAG with IRIS

```python
import dspy
from retrieve_dspy.database.iris_database import iris_search_tool

# Configure DSPy
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4"))

# Define retriever using IRIS
class IRISRetriever(dspy.Retrieve):
    def __init__(self, k=5):
        super().__init__(k=k)

    def forward(self, query: str):
        results = iris_search_tool(
            query=query,
            collection_name="RAG.Documents",
            target_property_name="text_content",
            retrieved_k=self.k,
            search_method="hybrid"
        )

        # Convert to DSPy format
        passages = [obj.content for obj in results]
        return dspy.Prediction(passages=passages)

# Use in DSPy module
class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = IRISRetriever(k=5)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

# Query
rag = RAG()
answer = rag(question="What are the symptoms of diabetes?")
print(answer.answer)
```

### Example 2: Multi-Query Fusion with IRIS

```python
from retrieve_dspy.retrieve import MultiQueryRetriever
from retrieve_dspy.database.iris_database import iris_search_tool

# Create multi-query retriever backed by IRIS
retriever = MultiQueryRetriever(
    search_tool=iris_search_tool,
    collection_name="RAG.Documents",
    target_property_name="text_content",
    retrieved_k=10
)

# DSPy automatically generates multiple query variations and fuses results
results = retriever.forward(
    query="How is type 2 diabetes treated?",
    num_queries=3  # Generate 3 query variations
)
```

### Example 3: Clustering for Topic Discovery

```python
from retrieve_dspy.cluster import ClusterDocuments
from retrieve_dspy.database.iris_database import iris_search_tool

# Retrieve documents from IRIS
results = iris_search_tool(
    query="diabetes",
    collection_name="RAG.Documents",
    target_property_name="text_content",
    retrieved_k=100,
    return_vector=True  # Need vectors for clustering
)

# Cluster into topics
clusterer = ClusterDocuments()
clusters = clusterer.forward(
    objects=results,
    num_clusters=5
)

for cluster in clusters:
    print(f"Cluster: {cluster.cluster_name}")
    print(f"  Documents: {len(cluster.doc_ids)}")
```

## Configuration

### Environment Variables

```bash
# IRIS Connection
export IRIS_HOST="localhost"
export IRIS_PORT="1972"
export IRIS_NAMESPACE="USER"
export IRIS_USERNAME="_SYSTEM"
export IRIS_PASSWORD="SYS"

# Embedding Model (uses iris_rag configuration)
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export EMBEDDING_DIMENSION="384"
```

### Integration with iris_rag Configuration

The IRIS adapter reuses existing iris_rag configuration:

```python
# Automatically uses iris_rag/config/default_config.yaml
from iris_rag.config.manager import ConfigurationManager
from iris_rag.embeddings.manager import EmbeddingManager

config_manager = ConfigurationManager()
embedding_manager = EmbeddingManager(config_manager)

# Uses configured embedding model, dimension, etc.
embeddings = embedding_manager.embed_texts([query])
```

## Testing Strategy

### Unit Tests

```python
def test_iris_search_tool_basic():
    """Test basic vector search."""
    results = iris_search_tool(
        query="test query",
        collection_name="RAG.Documents",
        target_property_name="text_content",
        retrieved_k=5
    )

    assert len(results) <= 5
    assert all(isinstance(obj, ObjectFromDB) for obj in results)
    assert all(obj.relevance_score >= 0 for obj in results)


def test_iris_search_tool_with_filter():
    """Test search with tag filtering."""
    results = iris_search_tool(
        query="diabetes",
        collection_name="RAG.Documents",
        target_property_name="text_content",
        retrieved_k=5,
        tag_filter_value="medical"
    )

    assert len(results) <= 5


def test_async_iris_search_tool():
    """Test async search."""
    async def run_test():
        results = await async_iris_search_tool(
            query="test",
            collection_name="RAG.Documents",
            target_property_name="text_content",
            retrieved_k=5
        )
        return results

    results = asyncio.run(run_test())
    assert len(results) <= 5
```

### Integration Tests

```python
def test_dspy_retriever_with_iris():
    """Test DSPy retriever using IRIS backend."""
    import dspy

    class IRISRetriever(dspy.Retrieve):
        def forward(self, query: str):
            results = iris_search_tool(
                query=query,
                collection_name="RAG.Documents",
                target_property_name="text_content",
                retrieved_k=5
            )
            passages = [obj.content for obj in results]
            return dspy.Prediction(passages=passages)

    retriever = IRISRetriever()
    prediction = retriever("What is diabetes?")

    assert len(prediction.passages) > 0
```

## Success Metrics

1. **Interface Compatibility**: IRIS adapter is drop-in replacement for Weaviate adapter
2. **Performance**: Vector search < 100ms p95 latency for 10K documents
3. **Feature Parity**: Supports all retrieve-dspy techniques (clustering, reranking, multi-query)
4. **Community Adoption**: Pull request accepted to retrieve-dspy repository
5. **Documentation**: Clear examples showing DSPy + IRIS integration

## Future Enhancements

1. **Streaming Results**: Support streaming for large result sets
2. **Batch Operations**: Efficient batch embedding and search
3. **Caching**: Cache embeddings and search results
4. **Monitoring**: Expose search metrics for observability
5. **Auto-tuning**: Optimize RRF weights based on query patterns

## References

- retrieve-dspy repository: https://github.com/weaviate/retrieve-dspy
- DSPy documentation: https://dspy-docs.vercel.app/
- IRIS Vector Search documentation
- Weaviate adapter implementation (reference)

---

**Next Steps:**
1. Review and approve this spec
2. Set up development environment with retrieve-dspy
3. Implement core iris_database.py adapter
4. Test with existing iris_rag infrastructure
5. Create examples and documentation
6. Submit pull request to retrieve-dspy
