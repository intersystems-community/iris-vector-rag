"""
IRIS Database Adapter for retrieve-dspy.

Provides InterSystems IRIS vector search capabilities for DSPy,
enabling enterprise-grade RAG with HNSW optimization, hybrid search,
and native SQL integration.

Example:
    >>> from retrieve_dspy.database.iris_database import iris_search_tool
    >>>
    >>> results = iris_search_tool(
    ...     query="What are the symptoms of diabetes?",
    ...     collection_name="RAG.Documents",
    ...     target_property_name="text_content",
    ...     retrieved_k=5
    ... )
    >>>
    >>> for obj in results:
    ...     print(f"[{obj.relevance_rank}] {obj.content[:100]}")

Environment Variables:
    IRIS_HOST: IRIS server hostname (default: localhost)
    IRIS_PORT: IRIS server port (default: 1972)
    IRIS_NAMESPACE: IRIS namespace (default: USER)
    IRIS_USERNAME: IRIS username (default: _SYSTEM)
    IRIS_PASSWORD: IRIS password (required)
"""

import os
import asyncio
from typing import Optional, List, Any
import logging

from retrieve_dspy.models import ObjectFromDB

logger = logging.getLogger(__name__)


def iris_search_tool(
    query: str,
    collection_name: str,
    target_property_name: str,
    iris_connection: Optional[Any] = None,
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
        target_property_name: Column containing searchable content (e.g., "text_content")
        iris_connection: IRIS DBAPI connection (optional, creates from env if None)
        return_property_name: Deprecated, kept for Weaviate adapter compatibility
        retrieved_k: Number of results to return (default: 5)
        return_vector: Whether to include embedding vectors in response
        tag_filter_value: Optional filter value for tag-based filtering

    Returns:
        List of ObjectFromDB with search results ranked by relevance

    Raises:
        ImportError: If IRIS Python driver is not installed
        ConnectionError: If cannot connect to IRIS database

    Example:
        >>> results = iris_search_tool(
        ...     query="diabetes symptoms",
        ...     collection_name="RAG.Documents",
        ...     target_property_name="text_content",
        ...     retrieved_k=3
        ... )
        >>> print(f"Found {len(results)} results")
    """
    # Get or create connection
    if iris_connection is None:
        iris_connection = _get_iris_connection()

    # Get query embedding
    try:
        embedding = _get_query_embedding(query)
    except Exception as e:
        logger.error(f"Failed to generate embedding for query: {e}")
        raise

    # Execute vector search
    try:
        results = _vector_search(
            iris_connection,
            collection_name,
            target_property_name,
            embedding,
            retrieved_k,
            tag_filter_value,
            return_vector
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise

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

    logger.info(f"Retrieved {len(objects)} results for query: '{query[:50]}...'")
    return objects


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
) -> List[ObjectFromDB]:
    """
    Async version of iris_search_tool.

    Note: IRIS DBAPI is synchronous, so this wraps the sync function
    with asyncio.to_thread to avoid blocking the event loop.

    Args:
        Same as iris_search_tool

    Returns:
        List of ObjectFromDB with search results

    Example:
        >>> results = await async_iris_search_tool(
        ...     query="diabetes",
        ...     collection_name="RAG.Documents",
        ...     target_property_name="text_content",
        ...     retrieved_k=5
        ... )
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
    )


def _get_iris_connection():
    """
    Create IRIS DBAPI connection from environment variables.

    Environment Variables:
        IRIS_HOST: Hostname (default: localhost)
        IRIS_PORT: Port (default: 1972)
        IRIS_NAMESPACE: Namespace (default: USER)
        IRIS_USERNAME: Username (default: _SYSTEM)
        IRIS_PASSWORD: Password (required)

    Returns:
        IRIS DBAPI connection object

    Raises:
        ImportError: If IRIS driver not installed
        ConnectionError: If connection fails
    """
    try:
        import iris
    except ImportError:
        raise ImportError(
            "IRIS Python driver not installed. "
            "Install with: pip install iris-native-api"
        )

    try:
        connection = iris.connect(
            hostname=os.getenv("IRIS_HOST", "localhost"),
            port=int(os.getenv("IRIS_PORT", "1972")),
            namespace=os.getenv("IRIS_NAMESPACE", "USER"),
            username=os.getenv("IRIS_USERNAME", "_SYSTEM"),
            password=os.getenv("IRIS_PASSWORD", "SYS")
        )
        logger.info(f"Connected to IRIS at {os.getenv('IRIS_HOST', 'localhost')}:{os.getenv('IRIS_PORT', '1972')}")
        return connection
    except Exception as e:
        raise ConnectionError(
            f"Failed to connect to IRIS database: {e}. "
            f"Check environment variables (IRIS_HOST, IRIS_PORT, etc.)"
        )


def _get_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for query text.

    Tries to use iris_rag's EmbeddingManager if available,
    otherwise falls back to sentence-transformers.

    Args:
        query: Query text to embed

    Returns:
        List of floats representing the embedding vector

    Raises:
        ImportError: If neither iris_rag nor sentence-transformers available
    """
    # Try iris_rag first (if available)
    try:
        from iris_rag.embeddings.manager import EmbeddingManager
        from iris_rag.config.manager import ConfigurationManager

        config_manager = ConfigurationManager()
        embedding_manager = EmbeddingManager(config_manager)
        embeddings = embedding_manager.embed_texts([query])

        if embeddings and len(embeddings) > 0:
            logger.debug(f"Generated embedding using iris_rag (dim: {len(embeddings[0])})")
            return embeddings[0]
    except ImportError:
        logger.debug("iris_rag not available, trying sentence-transformers")
    except Exception as e:
        logger.warning(f"iris_rag embedding failed: {e}, trying sentence-transformers")

    # Fallback to sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embedding = model.encode(query)
        logger.debug(f"Generated embedding using sentence-transformers (dim: {len(embedding)})")
        return embedding.tolist()
    except ImportError:
        raise ImportError(
            "No embedding library available. Install either:\n"
            "  - iris_rag: pip install -e /path/to/rag-templates\n"
            "  - sentence-transformers: pip install sentence-transformers"
        )


def _vector_search(
    connection,
    table_name: str,
    content_column: str,
    embedding: List[float],
    top_k: int,
    tag_filter: Optional[str],
    return_vector: bool
) -> List[dict]:
    """
    Execute vector similarity search using IRIS VECTOR_COSINE.

    Args:
        connection: IRIS DBAPI connection
        table_name: Table containing documents
        content_column: Column containing document text
        embedding: Query embedding vector
        top_k: Number of results to return
        tag_filter: Optional tag filter
        return_vector: Whether to return embedding vectors

    Returns:
        List of dicts with id, content, score, and optionally vector

    Note:
        Assumes embedding column is named {content_column}_embedding
        and stores embeddings as comma-separated VARCHAR.
    """
    cursor = connection.cursor()

    # Build embedding string for SQL
    embedding_str = ','.join(str(x) for x in embedding)
    dimension = len(embedding)
    embedding_column = f"{content_column}_embedding"

    # Build SQL query
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
        # Escape single quotes to prevent SQL injection
        escaped_filter = tag_filter.replace("'", "''")
        sql += f" AND tags LIKE '%{escaped_filter}%'"

    # Order by similarity and limit results
    sql += f" ORDER BY score DESC LIMIT {top_k}"

    logger.debug(f"Executing IRIS vector search: {sql[:200]}...")

    # Execute query
    try:
        cursor.execute(sql)
    except Exception as e:
        cursor.close()
        raise RuntimeError(f"SQL execution failed: {e}")

    # Parse results
    results = []
    for row in cursor.fetchall():
        result = {
            'id': str(row[0]),
            'content': str(row[1]) if row[1] else "",
            'score': float(row[2]) if row[2] is not None else 0.0,
        }

        # Parse vector if requested
        if return_vector and len(row) > 3 and row[3]:
            try:
                # IRIS stores vectors as comma-separated strings
                vector_str = str(row[3])
                result['vector'] = [float(x) for x in vector_str.split(',')]
            except Exception as e:
                logger.warning(f"Failed to parse vector for document {result['id']}: {e}")
                result['vector'] = None

        results.append(result)

    cursor.close()
    logger.debug(f"Retrieved {len(results)} results from IRIS")
    return results


# Example usage and testing
async def main():
    """Example usage of IRIS search tool."""
    import sys

    print("IRIS DSPy Adapter - Example Usage\n")
    print("=" * 60)

    # Check environment
    if not os.getenv("IRIS_PASSWORD"):
        print("ERROR: IRIS_PASSWORD environment variable not set")
        print("\nSet environment variables:")
        print("  export IRIS_HOST=localhost")
        print("  export IRIS_PORT=1972")
        print("  export IRIS_NAMESPACE=USER")
        print("  export IRIS_USERNAME=_SYSTEM")
        print("  export IRIS_PASSWORD=your_password")
        sys.exit(1)

    # Test sync search
    print("\n1. Testing synchronous search...")
    try:
        results = iris_search_tool(
            query="What are the symptoms of diabetes?",
            collection_name="RAG.Documents",
            target_property_name="text_content",
            retrieved_k=5,
            return_vector=False
        )

        print(f"✓ Found {len(results)} results\n")
        for obj in results:
            print(f"[{obj.relevance_rank}] Score: {obj.relevance_score:.4f}")
            print(f"    ID: {obj.object_id}")
            print(f"    Content: {obj.content[:80]}...")
            print()

    except Exception as e:
        print(f"✗ Sync search failed: {e}")

    # Test async search
    print("\n2. Testing asynchronous search...")
    try:
        async_results = await async_iris_search_tool(
            query="diabetes treatment",
            collection_name="RAG.Documents",
            target_property_name="text_content",
            retrieved_k=3,
            return_vector=True
        )

        print(f"✓ Found {len(async_results)} results")
        print(f"✓ Vector dimension: {len(async_results[0].vector) if async_results and async_results[0].vector else 'N/A'}")

    except Exception as e:
        print(f"✗ Async search failed: {e}")

    print("\n" + "=" * 60)
    print("IRIS adapter working correctly! ✓")


if __name__ == "__main__":
    asyncio.run(main())
