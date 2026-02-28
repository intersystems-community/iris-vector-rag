# Contract: GraphRAGToolSet

## Overview
`GraphRAGToolSet` is an `iris_llm.ToolSet` subclass in `iris_vector_rag.tools` that exposes GraphRAG pipeline operations as agent-callable tools. It is the reusable tool surface — consumer packages wire a connection and get tools, with no GraphRAG code in the consumer.

## Location
`iris_vector_rag/tools/graphrag.py`

## Availability
Only importable when `iris_llm` is installed. Raises `ImportError` with install instructions otherwise.

## Constructor

```python
class GraphRAGToolSet(ToolSet):
    def __init__(self, executor: SqlExecutor):
        """
        Args:
            executor: A SqlExecutor instance. The toolset owns the pipeline.
        """
```

## Tools

### search_entities
```python
@tool
def search_entities(self, query: str, limit: int = 5) -> str:
    """
    Search the medical knowledge graph for entities matching query terms.
    Returns JSON: {"query": str, "entities_found": int, "entities": [...]}
    """
```

### traverse_relationships
```python
@tool
def traverse_relationships(self, entity_text: str, max_depth: int = 2) -> str:
    """
    Traverse knowledge graph relationships from a seed entity (1–3 hops).
    Returns JSON: {"seed_entity": str, "entities_found": int, "relationships_found": int, "graph": {...}}
    """
```

### hybrid_search
```python
@tool
def hybrid_search(self, query: str, top_k: int = 5) -> str:
    """
    Combined vector + graph search using Reciprocal Rank Fusion.
    Returns JSON: {"query": str, "fused_results": int, "top_documents": [...]}
    """
```

## Contract Rules

1. All tools return JSON strings (never raw Python objects).
2. `max_depth` is clamped to [1, 3] internally — out-of-range values do not raise.
3. Empty results return valid JSON with zero counts, never exceptions.
4. Pipeline exceptions are NOT caught — they propagate to the caller (who wraps in `ToolError` if needed).
5. `GraphRAGToolSet` does NOT perform RBAC checks — that remains the caller's responsibility.

## Usage Pattern (consumer)

```python
from iris_vector_rag.tools import GraphRAGToolSet
from iris_vector_rag import SqlExecutor

executor = MyExecutorAdapter(iris_connection)
toolset = GraphRAGToolSet(executor=executor)

# Register with iris_llm Agent
agent.add_tool_set(toolset)

# Or convert to LangChain tools
from iris_llm.langchain import IrisTool
lc_tools = IrisTool.from_toolset(toolset)
```
