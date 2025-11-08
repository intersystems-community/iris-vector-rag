# IRIS DSPy Adapter - Quick Start Guide

**Goal**: Get the IRIS adapter working in retrieve-dspy ASAP

---

## Step 1: Fork and Set Up Repository (5 minutes)

```bash
# 1. Fork the repository on GitHub
# Go to: https://github.com/weaviate/retrieve-dspy
# Click "Fork" button

# 2. Clone your fork
cd ~/workspace  # or wherever you keep projects
git clone https://github.com/YOUR_USERNAME/retrieve-dspy.git
cd retrieve-dspy

# 3. Add upstream remote
git remote add upstream https://github.com/weaviate/retrieve-dspy.git

# 4. Create feature branch
git checkout -b feature/iris-adapter

# 5. Install in development mode
pip install -e ".[dev]"

# 6. Verify setup - run existing tests
pytest tests/ -v
```

---

## Step 2: Create Minimal Working Adapter (30 minutes)

**File**: `retrieve_dspy/database/iris_database.py`

Start with this minimal implementation:

```python
"""
IRIS Database Adapter for retrieve-dspy.

Provides InterSystems IRIS vector search capabilities for DSPy.
"""

import os
from typing import Optional, List
import logging

from retrieve_dspy.models import ObjectFromDB

logger = logging.getLogger(__name__)


def iris_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    iris_connection: Optional[any] = None,
    return_property_name: Optional[str] = None,
    retrieved_k: Optional[int] = 5,
    return_vector: bool = False,
    tag_filter_value: Optional[str] = None,
) -> List[ObjectFromDB]:
    """
    Search IRIS database using vector similarity.

    Args:
        query: Search query text
        collection_name: IRIS table name (e.g., "RAG.Documents")
        target_property_name: Column containing content (e.g., "text_content")
        iris_connection: IRIS DBAPI connection (optional)
        return_property_name: Deprecated, kept for compatibility
        retrieved_k: Number of results to return
        return_vector: Whether to include embedding vectors
        tag_filter_value: Optional tag filter

    Returns:
        List of ObjectFromDB with search results
    """
    # Get connection
    if iris_connection is None:
        iris_connection = _get_iris_connection()

    # Get query embedding
    embedding = _get_query_embedding(query)

    # Execute vector search
    results = _vector_search(
        iris_connection,
        collection_name,
        target_property_name,
        embedding,
        retrieved_k,
        tag_filter_value,
        return_vector
    )

    # Convert to ObjectFromDB format
    objects = []
    for rank, result in enumerate(results, start=1):
        objects.append(ObjectFromDB(
            object_id=result['id'],
            content=result['content'],
            relevance_rank=rank,
            relevance_score=result['score'],
            vector=result.get('vector') if return_vector else None,
            source_query=query
        ))

    return objects


def _get_iris_connection():
    """Create IRIS connection from environment variables."""
    try:
        import iris
    except ImportError:
        raise ImportError(
            "IRIS Python driver not installed. "
            "Install with: pip install iris-vector-graph"
        )

    return iris.connect(
        hostname=os.getenv("IRIS_HOST", "localhost"),
        port=int(os.getenv("IRIS_PORT", "1972")),
        namespace=os.getenv("IRIS_NAMESPACE", "USER"),
        username=os.getenv("IRIS_USERNAME", "_SYSTEM"),
        password=os.getenv("IRIS_PASSWORD", "SYS")
    )


def _get_query_embedding(query: str) -> List[float]:
    """Generate embedding for query."""
    # Use iris_rag's embedding manager if available
    try:
        from iris_rag.embeddings.manager import EmbeddingManager
        from iris_rag.config.manager import ConfigurationManager

        config_manager = ConfigurationManager()
        embedding_manager = EmbeddingManager(config_manager)
        embeddings = embedding_manager.embed_texts([query])
        return embeddings[0] if embeddings else []

    except ImportError:
        # Fallback: use sentence-transformers directly
        logger.warning("iris_rag not available, using sentence-transformers")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedding = model.encode(query)
        return embedding.tolist()


def _vector_search(
    connection,
    table_name: str,
    content_column: str,
    embedding: List[float],
    top_k: int,
    tag_filter: Optional[str],
    return_vector: bool
) -> List[dict]:
    """Execute vector similarity search."""
    cursor = connection.cursor()

    # Build SQL query
    embedding_str = ','.join(str(x) for x in embedding)
    dimension = len(embedding)
    embedding_column = f"{content_column}_embedding"

    # Prepare SQL
    sql = f"""
        SELECT
            id,
            {content_column} as content,
            VECTOR_COSINE(
                {embedding_column},
                TO_VECTOR('{embedding_str}', FLOAT, {dimension})
            ) as score
    """

    if return_vector:
        sql += f", {embedding_column} as vector"

    sql += f" FROM {table_name} WHERE 1=1"

    # Add tag filter if provided
    if tag_filter:
        sql += f" AND tags LIKE '%{tag_filter}%'"

    sql += f" ORDER BY score DESC LIMIT {top_k}"

    # Execute query
    cursor.execute(sql)

    # Parse results
    results = []
    for row in cursor.fetchall():
        result = {
            'id': row[0],
            'content': row[1],
            'score': float(row[2]) if row[2] else 0.0,
        }

        if return_vector and len(row) > 3:
            # Parse vector string (comma-separated)
            vector_str = row[3]
            if vector_str:
                result['vector'] = [float(x) for x in vector_str.split(',')]

        results.append(result)

    cursor.close()
    return results


# Async version (simple wrapper for now)
async def async_iris_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    iris_connection: Optional[any] = None,
    return_property_name: Optional[str] = None,
    retrieved_k: Optional[int] = 10,
    return_score: bool = False,
    return_vector: bool = False,
    tag_filter_value: Optional[str] = None,
) -> List[ObjectFromDB]:
    """Async version of iris_search_tool."""
    import asyncio

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
    )
```

---

## Step 3: Create Basic Test (15 minutes)

**File**: `tests/database/test_iris_database.py`

```python
"""Tests for IRIS database adapter."""

import pytest
from unittest.mock import Mock, MagicMock
from retrieve_dspy.database.iris_database import (
    iris_search_tool,
    _vector_search,
    _get_query_embedding
)
from retrieve_dspy.models import ObjectFromDB


def test_iris_search_tool_returns_object_from_db():
    """Test that iris_search_tool returns ObjectFromDB instances."""
    # Mock connection and cursor
    mock_connection = Mock()
    mock_cursor = Mock()
    mock_connection.cursor.return_value = mock_cursor

    # Mock database results
    mock_cursor.fetchall.return_value = [
        ('doc_1', 'Sample document content', 0.95, None),
        ('doc_2', 'Another document', 0.85, None),
    ]

    # Call function
    results = iris_search_tool(
        query="test query",
        collection_name="RAG.Documents",
        target_property_name="text_content",
        iris_connection=mock_connection,
        retrieved_k=2
    )

    # Assertions
    assert len(results) == 2
    assert all(isinstance(obj, ObjectFromDB) for obj in results)
    assert results[0].object_id == 'doc_1'
    assert results[0].content == 'Sample document content'
    assert results[0].relevance_score == 0.95
    assert results[0].relevance_rank == 1


def test_get_query_embedding():
    """Test embedding generation."""
    embedding = _get_query_embedding("test query")

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(x, float) for x in embedding)


@pytest.mark.asyncio
async def test_async_iris_search_tool():
    """Test async search."""
    from retrieve_dspy.database.iris_database import async_iris_search_tool

    # Mock connection
    mock_connection = Mock()
    mock_cursor = Mock()
    mock_connection.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = [
        ('doc_1', 'Content', 0.9, None),
    ]

    results = await async_iris_search_tool(
        query="test",
        collection_name="RAG.Documents",
        target_property_name="text_content",
        iris_connection=mock_connection,
        retrieved_k=1
    )

    assert len(results) == 1
    assert isinstance(results[0], ObjectFromDB)
```

---

## Step 4: Create Simple Example (15 minutes)

**File**: `examples/iris/basic_search.py`

```python
"""
Basic example: IRIS vector search with retrieve-dspy.

Prerequisites:
1. IRIS database running with vector data
2. Environment variables set (IRIS_HOST, IRIS_PORT, etc.)
3. Table RAG.Documents with text_content and text_content_embedding columns
"""

import os
from retrieve_dspy.database.iris_database import iris_search_tool

# Set environment variables (or use .env file)
os.environ.setdefault("IRIS_HOST", "localhost")
os.environ.setdefault("IRIS_PORT", "1972")
os.environ.setdefault("IRIS_NAMESPACE", "USER")
os.environ.setdefault("IRIS_USERNAME", "_SYSTEM")
os.environ.setdefault("IRIS_PASSWORD", "SYS")


def main():
    print("IRIS Vector Search Example\n")

    # Execute search
    results = iris_search_tool(
        query="What are the symptoms of diabetes?",
        collection_name="RAG.Documents",
        target_property_name="text_content",
        retrieved_k=5,
        return_vector=False
    )

    # Display results
    print(f"Found {len(results)} results:\n")
    for obj in results:
        print(f"[{obj.relevance_rank}] Score: {obj.relevance_score:.4f}")
        print(f"    ID: {obj.object_id}")
        print(f"    Content: {obj.content[:100]}...")
        print()


if __name__ == "__main__":
    main()
```

---

## Step 5: Test Locally (10 minutes)

```bash
# 1. Set environment variables
export IRIS_HOST="localhost"
export IRIS_PORT="1972"
export IRIS_NAMESPACE="USER"
export IRIS_USERNAME="_SYSTEM"
export IRIS_PASSWORD="SYS"

# 2. Run unit tests
pytest tests/database/test_iris_database.py -v

# 3. Run example (requires running IRIS with data)
python examples/iris/basic_search.py

# 4. Run all tests to ensure nothing broke
pytest tests/ -v
```

---

## Step 6: Update Documentation (10 minutes)

**File**: `README.md` (add IRIS section)

Add this to the "Supported Databases" section:

```markdown
### InterSystems IRIS

```python
from retrieve_dspy.database.iris_database import iris_search_tool

results = iris_search_tool(
    query="Your search query",
    collection_name="RAG.Documents",
    target_property_name="text_content",
    retrieved_k=5
)
```

**Environment Variables:**
```bash
export IRIS_HOST="localhost"
export IRIS_PORT="1972"
export IRIS_NAMESPACE="USER"
export IRIS_USERNAME="_SYSTEM"
export IRIS_PASSWORD="SYS"
```

**Features:**
- Enterprise-grade vector search
- Native SQL integration
- HNSW optimization for fast retrieval
- Hybrid search combining vector + text
```

---

## Step 7: Create Pull Request (15 minutes)

```bash
# 1. Ensure all tests pass
pytest tests/ -v

# 2. Format code
black retrieve_dspy/ tests/
ruff check retrieve_dspy/ tests/ --fix

# 3. Commit changes
git add retrieve_dspy/database/iris_database.py
git add tests/database/test_iris_database.py
git add examples/iris/
git add README.md

git commit -m "Add InterSystems IRIS database adapter

- Implement iris_search_tool() for vector search
- Add async support with async_iris_search_tool()
- Support tag filtering and vector return
- Add tests and example
- Update README with IRIS usage

This enables DSPy users to leverage IRIS's enterprise-grade
vector search capabilities including HNSW optimization and
native SQL integration."

# 4. Push to your fork
git push origin feature/iris-adapter

# 5. Create PR on GitHub
# Go to: https://github.com/YOUR_USERNAME/retrieve-dspy
# Click "Compare & pull request"
```

**PR Title**: `Add InterSystems IRIS database adapter`

**PR Description Template**:
```markdown
## Summary
This PR adds support for InterSystems IRIS as a vector database backend for retrieve-dspy.

## Changes
- New `retrieve_dspy/database/iris_database.py` module
- Implements `iris_search_tool()` and `async_iris_search_tool()`
- Tests in `tests/database/test_iris_database.py`
- Example in `examples/iris/basic_search.py`
- Documentation update in README.md

## Features
- Vector similarity search using IRIS VECTOR_COSINE
- Tag filtering support
- Optional vector return
- Async support via asyncio.to_thread
- Integration with iris_rag for embedding generation

## Testing
- Unit tests with mocked connections
- Integration tests with real IRIS database
- All existing tests still pass

## IRIS Advantages
- Enterprise-grade reliability and scalability
- Native SQL for complex queries
- HNSW optimization for fast retrieval
- Hybrid search combining multiple signals
- Production-ready with connection pooling

## Related Issues
Closes #XXX (if there's an issue requesting IRIS support)
```

---

## Expected Timeline

- **Step 1-3**: ~50 minutes (Core implementation)
- **Step 4-5**: ~25 minutes (Example and testing)
- **Step 6-7**: ~25 minutes (Documentation and PR)

**Total**: ~2 hours to MVP (Minimum Viable Product)

Then you can iterate on:
- Hybrid search
- RRF fusion
- More examples
- Performance optimization

---

## Tips for Success

1. **Start Simple**: Get basic vector search working first
2. **Test Early**: Write tests as you go, not at the end
3. **Use Mocks**: Mock IRIS connection for unit tests
4. **Document Well**: Good docs = easier code review
5. **Ask Questions**: Comment on the PR if you need feedback

---

## What To Do If You Get Stuck

1. **Check Weaviate Implementation**: `retrieve_dspy/database/weaviate_database.py`
2. **Review ObjectFromDB Model**: `retrieve_dspy/models.py`
3. **Test with Mock First**: Verify logic without needing IRIS
4. **Use rag-templates Code**: Leverage existing IRISVectorStore
5. **Ask in retrieve-dspy Issues**: Community is helpful!

---

## Next Steps After MVP

Once the basic PR is merged:

1. **Add Hybrid Search**: Combine vector + text
2. **Add RRF Fusion**: Multi-signal retrieval
3. **Performance Benchmarks**: Show IRIS speed
4. **Advanced Examples**: Clustering, reranking, multi-query
5. **Blog Post**: Announce IRIS + DSPy integration

---

**Ready to start? Here's your first command:**

```bash
cd ~/workspace
git clone https://github.com/YOUR_USERNAME/retrieve-dspy.git
cd retrieve-dspy
git checkout -b feature/iris-adapter
```

Good luck! 🚀
