# Quickstart: Feature 065 — iris_llm as IVR LLM Substrate

## Prerequisites

- Python 3.12
- `iris_llm` wheel for your platform (from `../ai-hub/wheels/` or internal registry):
  ```bash
  pip install iris_llm-0.1.0-cp39-abi3-macosx_11_0_arm64.whl
  ```
- `OPENAI_API_KEY` set (or configure a gateway `base_url`)

## Install IVR with iris_llm extra

```bash
cd iris-vector-rag-private
pip install -e ".[iris_llm]"
```

> Note: The `[iris_llm]` extra has no PyPI dependencies — it signals intent. Install the wheel separately (step above).

---

## Use iris_llm as the LLM provider

```python
from iris_vector_rag.common.utils import get_llm_func

llm_func = get_llm_func(provider="iris_llm", model_name="gpt-4o-mini")

# Use with any IVR pipeline
from iris_rag import create_pipeline
pipeline = create_pipeline("graphrag", llm_func=llm_func)
result = pipeline.query("What are the symptoms of pneumonia?")
print(result["answer"])
```

## Use with a gateway (IRIS AI Hub / NIM / Ollama)

```python
llm_func = get_llm_func(
    provider="iris_llm",
    model_name="llama-3.1-8b",
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)
```

---

## Use GraphRAGToolSet with an iris_llm Agent

```python
from iris_llm import Provider, Agent
from iris_vector_rag.tools import GraphRAGToolSet
from iris_vector_rag import SqlExecutor

# Provide a SqlExecutor (example: wrapping a direct IRIS connection)
class DirectIrisExecutor:
    def __init__(self, connection):
        self._conn = connection

    def execute(self, sql, params=None):
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql, params or [])
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()

import iris
conn = iris.createConnection("localhost", 1972, "USER", "SuperUser", "SYS")
executor = DirectIrisExecutor(conn)

# Build toolset and agent
toolset = GraphRAGToolSet(executor=executor)
provider = Provider.new_openai(api_key="sk-...")
agent = Agent.with_provider("gpt-4o", provider)
agent.add_tool_set(toolset)

result = agent.run("Find entities related to respiratory infections and their relationships")
print(result)
```

---

## Use IrisLLMDSPyAdapter for DSPy entity extraction

```python
import dspy
from iris_llm import Provider
from iris_llm.langchain import ChatIris
from iris_vector_rag.dspy_modules.iris_llm_lm import IrisLLMDSPyAdapter

provider = Provider.new_openai(api_key="sk-...")
chat = ChatIris(provider=provider, model="gpt-4o-mini")
adapter = IrisLLMDSPyAdapter(chat_iris=chat, model="gpt-4o-mini")

dspy.configure(lm=adapter)

# Now DSPy modules use iris_llm
from iris_vector_rag.dspy_modules.entity_extraction_module import TrakCareEntityExtractionModule
extractor = TrakCareEntityExtractionModule()
result = extractor("Patient presents with fever and cough")
print(result.entities)
```

---

## Run unit tests (no IRIS instance needed)

```bash
cd iris-vector-rag-private
pytest tests/unit/test_sql_executor.py tests/unit/test_graphrag_toolset.py tests/unit/test_iris_llm_substrate.py -v
```

## Run integration tests (requires iris_llm wheel + OPENAI_API_KEY)

```bash
pytest tests/integration/test_iris_llm_external.py -v
```
