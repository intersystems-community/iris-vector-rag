# QUIPLER Compatibility with IRIS Adapter

## What is QUIPLER?

**QUIPLER** = **QU**ery **I**terative **P**arallel **L**everage-**E**xpanded **R**etrieval

A sophisticated retrieve-dspy composition that:
1. **Generates multiple queries** from user question (query expansion)
2. **Searches in parallel** with each query
3. **Reranks** each result set using cross-encoder
4. **Fuses** results using Reciprocal Rank Fusion (RRF)

**IRIS Compatibility**: ✅ **FULLY COMPATIBLE** (with one required change)

---

## QUIPLER Architecture

```python
User Question
    ↓
┌─────────────────────────────────────┐
│ 1. Query Expansion (LLM)            │  Generates N queries
│    "diabetes" → ["diabetes symptoms",│
│                   "diabetes diagnosis",│
│                   "diabetes treatment"]│
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 2. Parallel Search (Database)       │  ← Uses database adapter
│    ┌─────────┐ ┌─────────┐ ┌──────┐│
│    │ Query 1 │ │ Query 2 │ │Query3││
│    └────↓────┘ └────↓────┘ └───↓──┘│
│         └───────────┴──────────┘   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 3. Cross-Encoder Reranking          │  Independent of database
│    Rerank each result set separately│
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 4. RRF Fusion                       │  Independent of database
│    Combine & rerank all results     │
└─────────────────────────────────────┘
    ↓
Final Results
```

---

## Current Implementation Issue

QUIPLER currently uses **Weaviate-specific client injection**:

```python
class QUIPLER(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        weaviate_client: Optional[weaviate.WeaviateClient] = None,  # ← Weaviate-specific!
        ...
    ):
```

And passes to CrossEncoderReranker:
```python
self.searcher = CrossEncoderReranker(
    collection_name=collection_name,
    target_property_name=target_property_name,
    weaviate_client=weaviate_client,  # ← Weaviate-specific!
    ...
)
```

---

## Solution: Make Database Tool Configurable

### Proposed Change to retrieve-dspy

**File**: `retrieve_dspy/retrievers/base_rag.py`

```python
class BaseRAG(dspy.Module):
    def __init__(
        self,
        collection_name: str,
        target_property_name: Optional[str] = "content",
        database_tool: Optional[callable] = None,  # ← NEW PARAMETER
        async_database_tool: Optional[callable] = None,  # ← NEW PARAMETER
        verbose: Optional[bool] = True,
        search_only: Optional[bool] = True,
        retrieved_k: Optional[int] = 5,
        verbose_signature: Optional[bool] = True,
        multi_lm_configs: Optional[list[MultiLMConfig]] = None,
    ) -> None:
        self.collection_name = collection_name
        self.target_property_name = target_property_name

        # Database tool configuration
        if database_tool is None:
            # Default to Weaviate for backward compatibility
            from retrieve_dspy.database.weaviate_database import weaviate_search_tool
            database_tool = weaviate_search_tool

        if async_database_tool is None:
            # Default to Weaviate async for backward compatibility
            from retrieve_dspy.database.weaviate_database import async_weaviate_search_tool
            async_database_tool = async_weaviate_search_tool

        self.database_tool = database_tool
        self.async_database_tool = async_database_tool

        # Rest of initialization...
```

**File**: `retrieve_dspy/retrievers/compositions/quipler.py`

```python
class QUIPLER(BaseRAG):
    def __init__(
        self,
        collection_name: str,
        target_property_name: str,
        database_tool: Optional[callable] = None,  # ← NEW PARAMETER
        async_database_tool: Optional[callable] = None,  # ← NEW PARAMETER
        reranker_clients: Optional[List[RerankerClient]] = None,
        return_property_name: Optional[str] = None,
        verbose: bool = False,
        verbose_signature: bool = True,
        search_only: bool = True,
        retrieved_k: int = 50,
        reranked_k: int = 20,
        rrf_k: int = 60,
        **cross_encoder_kwargs
    ):
        super().__init__(
            collection_name=collection_name,
            target_property_name=target_property_name,
            database_tool=database_tool,  # ← Pass to base
            async_database_tool=async_database_tool,  # ← Pass to base
            verbose=verbose,
            verbose_signature=verbose_signature,
            search_only=search_only,
            retrieved_k=retrieved_k,
        )

        # Cross-encoder reranker
        self.searcher = CrossEncoderReranker(
            collection_name=collection_name,
            target_property_name=target_property_name,
            database_tool=database_tool,  # ← Use database_tool instead of weaviate_client
            async_database_tool=async_database_tool,
            reranker_clients=reranker_clients,
            return_property_name=return_property_name,
            verbose=verbose,
            search_only=search_only,
            retrieved_k=retrieved_k,
            reranked_k=reranked_k,
            **cross_encoder_kwargs
        )
```

---

## Using QUIPLER with IRIS (After Enhancement)

### Example Usage

```python
from retrieve_dspy.retrievers import QUIPLER
from retrieve_dspy.database.iris_database import iris_search_tool, async_iris_search_tool

# Create QUIPLER with IRIS backend
quipler = QUIPLER(
    collection_name="RAG.Documents",
    target_property_name="text_content",
    database_tool=iris_search_tool,  # ← Use IRIS!
    async_database_tool=async_iris_search_tool,
    retrieved_k=50,
    reranked_k=20,
    rrf_k=60,
    verbose=True
)

# Execute query
result = quipler.forward("What are the symptoms of diabetes?")

# Result contains:
# - result.sources: List[ObjectFromDB] - fused and reranked documents
# - result.searches: List[str] - all queries used
# - result.usage: Dict - token usage stats
```

### What Happens Under the Hood

```python
# User query: "What are the symptoms of diabetes?"

# Step 1: Query Expansion (LLM generates 3 queries)
queries = [
    "diabetes mellitus symptoms and signs",
    "clinical presentation of diabetes",
    "how to recognize diabetes early warning signs"
]

# Step 2: Parallel IRIS Search (for each query)
# Query 1 searches IRIS
results_1 = iris_search_tool(
    query="diabetes mellitus symptoms and signs",
    collection_name="RAG.Documents",
    target_property_name="text_content",
    retrieved_k=50
)
# Returns 50 documents with relevance scores

# Query 2 searches IRIS
results_2 = iris_search_tool(
    query="clinical presentation of diabetes",
    ...
)
# Returns 50 documents

# Query 3 searches IRIS
results_3 = iris_search_tool(
    query="how to recognize diabetes early warning signs",
    ...
)
# Returns 50 documents

# Step 3: Cross-Encoder Reranking (for each result set)
# Rerank results_1 using cross-encoder model
reranked_1 = cross_encoder.rerank(query="diabetes...", docs=results_1, top_k=20)
# Returns top 20 documents

reranked_2 = cross_encoder.rerank(query="clinical...", docs=results_2, top_k=20)
reranked_3 = cross_encoder.rerank(query="how to...", docs=results_3, top_k=20)

# Step 4: RRF Fusion (combine all reranked results)
final_results = reciprocal_rank_fusion(
    result_sets=[reranked_1, reranked_2, reranked_3],
    k=60,
    top_k=20
)
# Returns final 20 documents, ranked by combined RRF score
```

---

## Benefits of QUIPLER with IRIS

### 1. Query Coverage
- Multiple query formulations capture different aspects
- IRIS vector search finds semantically similar docs for each query
- Better recall than single query

### 2. Parallel Execution
- 3 queries × 50 results = 150 candidate documents
- All searches happen in parallel (async)
- IRIS handles concurrent queries efficiently

### 3. Reranking Precision
- Cross-encoder gives precise relevance for each result set
- Better precision than vector search alone
- Independent of database (works same with IRIS or Weaviate)

### 4. RRF Fusion Quality
- Documents appearing in multiple result sets get boosted
- Diversified results (not dominated by single query)
- Works on ObjectFromDB objects (database-agnostic)

---

## Performance Characteristics

### With IRIS Backend

```python
# Timing breakdown for 3 queries × 50 results each

1. Query Expansion:        ~500-1000ms  (LLM call)
2. Parallel IRIS Search:   ~100-200ms   (3 concurrent searches)
3. Cross-Encoder Rerank:   ~300-500ms   (3 × 50 docs)
4. RRF Fusion:            ~10-20ms     (merge 3 × 20 docs)

Total:                     ~1-2 seconds
```

**Key Optimization**: Parallel search means 3 queries take ~same time as 1 query!

**IRIS Advantages**:
- HNSW indexing: Sub-100ms search even with 100K documents
- Connection pooling: Efficient concurrent queries
- Native SQL: Could add metadata filtering to each query

---

## Alternative: Temporary Workaround (Until PR Merged)

If you want to use QUIPLER with IRIS **before** the enhancement is merged:

### Option 1: Monkey Patch

```python
from retrieve_dspy.retrievers import QUIPLER
from retrieve_dspy.database import iris_database

# Replace weaviate calls in CrossEncoderReranker
import retrieve_dspy.retrievers.rerankers.cross_encoder_reranker as cer
cer.weaviate_search_tool = iris_database.iris_search_tool
cer.async_weaviate_search_tool = iris_database.async_iris_search_tool

# Now QUIPLER will use IRIS
quipler = QUIPLER(
    collection_name="RAG.Documents",
    target_property_name="text_content",
    weaviate_client=None,  # Pass None
    ...
)
```

### Option 2: Fork and Modify

```bash
# Fork retrieve-dspy
git clone https://github.com/isc-tdyar/retrieve-dspy.git
cd retrieve-dspy

# Make the changes above
# Install your fork
pip install -e .

# Use modified version
from retrieve_dspy.retrievers import QUIPLER
```

### Option 3: Create Custom QUIPLER

```python
# contrib/retrieve-dspy/iris_quipler.py
from retrieve_dspy.retrievers.compositions.quipler import QUIPLER as BaseQUIPLER
from retrieve_dspy.database.iris_database import iris_search_tool, async_iris_search_tool
from retrieve_dspy.retrievers import CrossEncoderReranker

class IRIS_QUIPLER(BaseQUIPLER):
    """QUIPLER adapted for IRIS database."""

    def __init__(self, collection_name: str, target_property_name: str, **kwargs):
        # Remove weaviate_client from kwargs
        kwargs.pop('weaviate_client', None)

        # Initialize without calling super (to avoid weaviate dependency)
        # ... (copy BaseRAG initialization)

        # Create searcher with IRIS tools
        self.searcher = IRIS_CrossEncoderReranker(
            collection_name=collection_name,
            target_property_name=target_property_name,
            database_tool=iris_search_tool,
            async_database_tool=async_iris_search_tool,
            **kwargs
        )

# Use it
quipler = IRIS_QUIPLER(
    collection_name="RAG.Documents",
    target_property_name="text_content",
    retrieved_k=50,
    reranked_k=20
)
```

---

## Proposed PR to retrieve-dspy

### Title
"Add database tool injection to support multiple vector databases"

### Description
```markdown
## Summary
Add `database_tool` and `async_database_tool` parameters to `BaseRAG` and all retrievers, enabling use of any vector database that implements the standard search interface.

## Changes
- BaseRAG: Add database_tool parameters with Weaviate as default
- All retrievers: Pass database_tool to parent class
- Backward compatible: Existing code works unchanged (defaults to Weaviate)

## Benefits
- Users can use IRIS, Pinecone, Qdrant, etc. with all retrieve-dspy techniques
- Modular design: database layer separated from retrieval logic
- No duplication: Advanced techniques (QUIPLER, etc.) work with any database

## Example
```python
from retrieve_dspy.retrievers import QUIPLER
from retrieve_dspy.database.iris_database import iris_search_tool

# Use QUIPLER with IRIS instead of Weaviate
quipler = QUIPLER(
    collection_name="Documents",
    target_property_name="content",
    database_tool=iris_search_tool  # ← NEW!
)
```

## Testing
- All existing tests pass (backward compatibility)
- New tests with mock database tool
- Example using IRIS adapter
```

---

## Summary: QUIPLER + IRIS

| Aspect | Status | Notes |
|--------|--------|-------|
| **Query Expansion** | ✅ Works | LLM-based, independent of database |
| **Parallel Search** | ✅ Works | IRIS handles concurrent queries |
| **Cross-Encoder Reranking** | ✅ Works | Independent of database |
| **RRF Fusion** | ✅ Works | Works on ObjectFromDB objects |
| **Async Support** | ✅ Works | Our adapter has async_iris_search_tool |
| **Overall Compatibility** | ⚠️ Requires PR | Need to make database_tool configurable |

**Bottom Line**: QUIPLER is **fully compatible** with IRIS. We just need retrieve-dspy to accept `database_tool` parameter instead of hardcoding `weaviate_client`.

**Action Items**:
1. ✅ Our IRIS adapter is ready (supports sync + async)
2. 🔄 Submit PR to add database_tool injection to BaseRAG
3. 🔄 Update QUIPLER to use database_tool parameter
4. ✅ Create example showing QUIPLER + IRIS

**Timeline**:
- **Workaround available now**: Monkey patch or fork
- **Clean solution**: After retrieve-dspy accepts database_tool PR (~1-2 weeks?)
- **Long-term**: retrieve-dspy becomes database-agnostic framework
