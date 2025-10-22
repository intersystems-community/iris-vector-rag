# InterSystems IRIS Section for retrieve-dspy README

Add this section to the "Supported Databases" section of retrieve-dspy/README.md:

---

### InterSystems IRIS

[InterSystems IRIS](https://www.intersystems.com/products/intersystems-iris/) is an enterprise-grade data platform with native vector search capabilities, HNSW optimization, and hybrid search combining vector + text + graph traversal.

#### Installation

```bash
# Install IRIS Python driver
pip install iris-native-api

# Or for advanced features (graph, hybrid search)
pip install iris-vector-graph
```

#### Basic Usage

```python
from retrieve_dspy.database.iris_database import iris_search_tool

results = iris_search_tool(
    query="What are the symptoms of diabetes?",
    collection_name="RAG.Documents",
    target_property_name="text_content",
    retrieved_k=5
)

for obj in results:
    print(f"[{obj.relevance_rank}] {obj.content}")
```

#### Environment Configuration

Set these environment variables to configure IRIS connection:

```bash
export IRIS_HOST="localhost"
export IRIS_PORT="1972"
export IRIS_NAMESPACE="USER"
export IRIS_USERNAME="_SYSTEM"
export IRIS_PASSWORD="SYS"
```

Alternatively, pass a connection object directly:

```python
import iris

connection = iris.connect(
    hostname="localhost",
    port=1972,
    namespace="USER",
    username="_SYSTEM",
    password="SYS"
)

results = iris_search_tool(
    query="search query",
    collection_name="RAG.Documents",
    target_property_name="text_content",
    iris_connection=connection,
    retrieved_k=5
)
```

#### Advanced Features

**Tag Filtering:**
```python
results = iris_search_tool(
    query="diabetes treatment",
    collection_name="RAG.Documents",
    target_property_name="text_content",
    tag_filter_value="medical",  # Filter by tags column
    retrieved_k=5
)
```

**Return Vectors:**
```python
results = iris_search_tool(
    query="search query",
    collection_name="RAG.Documents",
    target_property_name="text_content",
    return_vector=True,  # Include embedding vectors in results
    retrieved_k=5
)

# Access vectors
for obj in results:
    print(f"Vector dimension: {len(obj.vector)}")
```

**Async Search:**
```python
import asyncio
from retrieve_dspy.database.iris_database import async_iris_search_tool

async def search():
    results = await async_iris_search_tool(
        query="search query",
        collection_name="RAG.Documents",
        target_property_name="text_content",
        retrieved_k=5
    )
    return results

results = asyncio.run(search())
```

#### Key Features

- **Enterprise-grade reliability**: Battle-tested in healthcare, finance, and government
- **Native SQL integration**: Combine vector search with complex SQL queries
- **HNSW optimization**: 50x faster vector search with approximate nearest neighbor indexing
- **Hybrid search**: Combine vector + text + graph traversal (via iris-vector-graph)
- **Production-ready**: Connection pooling, transactions, ACID guarantees
- **High performance**: ~50-100ms p95 latency for 10K documents

#### Database Schema

IRIS expects a table with the following structure:

```sql
CREATE TABLE RAG.Documents (
    id VARCHAR(255) PRIMARY KEY,
    text_content VARCHAR(50000),
    text_content_embedding VECTOR(FLOAT, 384),  -- or your embedding dimension
    tags VARCHAR(1000),  -- optional, for tag filtering
    -- additional metadata columns...
)
```

The embedding column name should follow the pattern `{content_column}_embedding`.

#### Example with DSPy

```python
import dspy
from retrieve_dspy.database.iris_database import iris_search_tool

# Configure DSPy
lm = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=lm)

# Create custom retriever using IRIS
class IRISRetriever(dspy.Retrieve):
    def __init__(self, k=5):
        super().__init__(k=k)

    def forward(self, query: str) -> list:
        results = iris_search_tool(
            query=query,
            collection_name="RAG.Documents",
            target_property_name="text_content",
            retrieved_k=self.k
        )
        # Convert ObjectFromDB to dspy.Prediction format
        return [obj.content for obj in results]

# Use in DSPy program
retriever = IRISRetriever(k=5)
documents = retriever("What are the symptoms of diabetes?")
```

#### Troubleshooting

**Connection Error:**
- Verify IRIS is running and accessible
- Check that IRIS_HOST and IRIS_PORT are correct
- Verify credentials (IRIS_USERNAME, IRIS_PASSWORD)

**ImportError: No module named 'iris':**
```bash
pip install iris-native-api
```

**SQLCODE -259 (Vector datatype mismatch):**
- Ensure embedding column is defined as `VECTOR(FLOAT, dimension)`
- Verify dimension matches your embedding model (e.g., 384 for all-MiniLM-L6-v2)

**No results returned:**
- Verify table has data: `SELECT COUNT(*) FROM RAG.Documents`
- Check embedding column is populated
- Try lowering retrieved_k if corpus is small

#### Resources

- [IRIS Documentation](https://docs.intersystems.com/)
- [IRIS Vector Search Guide](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_vector)
- [iris-vector-graph Package](https://github.com/intersystems/iris-vector-graph)
- [Example Implementation](examples/iris/basic_search.py)

---

**Note**: This section should be inserted into the existing "Supported Databases" section of the README, alongside Weaviate, Pinecone, and other database backends.
