# Research Spike: `langchain-intersystems` Integration

**Created**: 2026-03-22
**Owner**: Thomas Dyar
**Target**: Complete week of March 29 (before AIML71 deck due April 13)
**Branch for results**: `feat/langchain-intersystems-adapter`
**PR target**: public `iris-vector-rag` repo

---

## Context

Aohan Dang's `langchain-intersystems` package (`langchain_intersystems.IRISVectorStore`) is
the official InterSystems LangChain integration, currently in Perforce (not yet released to
PyPI). A demo copy of the wheel ships with the READY 2026 hackathon kit:

```
~/ws/ready2026-hackathon/demos/langchain-vectorstore/
  dist/langchain_intersystems-0.0.1-py3-none-any.whl
  demo.py          ← working demo against State of the Union text
  requirements.txt ← pins intersystems-irispython~=5.3
  LANGCHAIN_INTERSYSTEMS_DOCS.md  ← copy of the Perforce README
```

**Why this matters for READY 2026**:
- AIML71 (RAG with Nick Petrocelli, Tue Apr 28): your section covers "RAG things users can try at home" — this package is that story for LangChain users
- Hackathon kit (Mon Apr 27): Gabriel's ChatFHIR demo uses LangChain — should use official ISC package
- `iris-vector-rag` needs an adapter so it adopts the official package when it ships

---

## What We Know (Pre-Spike)

### API Surface (from docs + demo)

```python
from langchain_intersystems import IRISVectorStore, SimilarityMetric, Predicate

vs = IRISVectorStore(
    embeddings,                          # any LangChain Embeddings object
    connect_kwargs={                     # DB-API connection, NOT Native API
        'hostname': 'localhost',
        'port': 1972,
        'namespace': 'USER',
        'username': '_SYSTEM',
        'password': 'SYS'
    },
    collection_name='my_collection',
    replace_collection=True,             # drop + recreate on init
    similarity_metric=SimilarityMetric.COSINE
)

vs.add_documents(docs)
vs.similarity_search_with_score(query, k=5, filter={...})
```

### Metadata Filtering (the headline feature)

Composable SQL-mapped predicate system:

```python
# Equality (shorthand)
filter={'source': 'radiology_notes'}

# Range
filter={'note_date': (Predicate.BETWEEN, date(2024,1,1), date(2024,12,31))}

# Multi-value
filter={'patient_id': (Predicate.IN, ['P001', 'P002', 'P003'])}

# Boolean logic
filter={
    Predicate.AND: [
        {'resource_type': 'DiagnosticReport'},
        {'status': (Predicate.NOT_EQUAL, 'cancelled')}
    ]
}

# Substring (useful for FHIR references)
filter={'subject': (Predicate.STARTS_WITH, 'Patient/')}
```

Full predicate table: `$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$between`, `$in`,
`$isnull`, `$notnull`, `$contains`, `$startswith`, `$like`, `$matches`, `$pattern`.

### Metadata Constraints (important limitations)

- Keys: `str`, alphanumeric + underscore only, max 126 chars, stored lowercase
- Values: `str` (max 1024), `int`, `float`, `bool`, `datetime.date/time/datetime`, `uuid.UUID`, `bytes`, `decimal.Decimal`
- **All values for a key must have the same type across all documents** — mixing `int` and `str` for the same key raises an exception
- `None` values stored as SQL NULL

### Requires

- IRIS 2025.1+
- Python 3.10+
- `intersystems-irispython~=5.3` (DB-API driver)

---

## What `iris-vector-rag` Does Differently

`iris-vector-rag` is NOT a LangChain VectorStore — it's a full RAG pipeline framework:
- Multiple pipeline types: BasicRAG, GraphRAG, CRAG, BasicRAGReranking
- GraphRAG with entity extraction, KG construction, community detection
- DSPy retriever adapter (`contrib/retrieve-dspy/`)
- Evaluation framework
- Chunking service

The overlap is ONLY at the storage/retrieval layer. `IRISVectorStore` replaces the
low-level IRIS vector SQL calls in `iris_vector_rag/core/`. An adapter in `contrib/`
would let LangChain users plug into the full pipeline.

---

## Questions This Spike Must Answer

### Q1 — Schema: How does `IRISVectorStore` store metadata?

**Why it matters**: Determines query performance on large collections (MIMIC has 200k+ notes).

**What to do**:
```python
vs = IRISVectorStore(embeddings, connect_kwargs=..., collection_name='test_spike')
vs.add_documents([doc_with_metadata])
# Then inspect the IRIS schema:
import intersystems_iris.dbapi as dbapi
conn = dbapi.connect(...)
cur = conn.cursor()
cur.execute("SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'test_spike'")
print(cur.fetchall())
```

**Hypothesis**: Each unique metadata key gets its own SQL column (dynamic schema expansion).
**If wrong**: It's a JSON blob column — filtering requires JSON extraction, slower.

**Record**: Column names, types, whether indexes are created automatically on metadata columns.

---

### Q2 — FHIR Metadata Compatibility

**Why it matters**: The hackathon uses FHIR clinical data. Can we use `IRISVectorStore`
for FHIR documents with FHIR-shaped metadata, or is `fhir-017 DocChunk` the right tool?

**What to do**: Load a small FHIR bundle (10-20 DiagnosticReports from MIMIC) as documents
with FHIR-derived metadata:

```python
# Simulate FHIR-derived metadata
docs = [
    Document(
        page_content=note_text,
        metadata={
            'resource_type': 'DiagnosticReport',
            'subject': 'Patient/10406825',
            'status': 'final',
            'category': 'radiology',
            'effective_date': datetime.date(2024, 3, 15),
            'encounter': 'Encounter/98765'
        }
    )
    for note_text in mimic_radiology_notes
]
vs.add_documents(docs)
```

Then test:
```python
# Patient-scoped search
vs.similarity_search('chest infiltrate', filter={'subject': 'Patient/10406825'})

# Date range + resource type
vs.similarity_search(
    'pneumonia',
    filter={
        Predicate.AND: [
            {'resource_type': 'DiagnosticReport'},
            {'effective_date': (Predicate.BETWEEN,
                datetime.date(2024,1,1), datetime.date(2024,12,31))}
        ]
    }
)
```

**Record**: Does it work? Any type coercion issues? How does performance compare to
`fhir-017 DocChunk` which has first-class indexed properties for SubjectKey, DateKey, StatusKey?

**Key question**: Is `IRISVectorStore` viable as a "bring your own FHIR metadata" store,
or does FHIR search require the native `_v_content` parameter approach?

---

### Q3 — Performance on Realistic Data

**Why it matters**: AIML71 makes a claim about "production-ready RAG" — need actual numbers.

**What to do**: Load MIMIC radiology notes (start with 1k, then 10k) and benchmark:

```python
import time

# Ingest benchmark
start = time.time()
vs.add_documents(docs_1k)
print(f"1k docs ingested in {time.time()-start:.1f}s")

# Query benchmark (unfiltered vs filtered)
start = time.time()
for _ in range(100):
    vs.similarity_search('chest X-ray findings', k=5)
print(f"100 unfiltered queries: {time.time()-start:.1f}s")

start = time.time()
for _ in range(100):
    vs.similarity_search('chest X-ray findings', k=5,
        filter={'category': 'radiology'})
print(f"100 filtered queries: {time.time()-start:.1f}s")
```

**Record**: Ingest rate (docs/sec), query latency (ms p50/p95), filtering overhead.

---

### Q4 — `replace_collection=True` Behavior

**Why it matters**: The hackathon demo uses `replace_collection=True` — attendees will
re-run the demo. Need to understand if this drops the table, truncates it, or something else.

**What to do**: Run `add_documents` twice with `replace_collection=True`, inspect row counts.
Also test what happens with `replace_collection=False` on an existing collection.

**Record**: SQL DDL generated, behavior on re-init, whether HNSW index is rebuilt.

---

### Q5 — MMR (Maximal Marginal Relevance) Support

**Why it matters**: `iris-vector-rag` uses MMR for diversity in retrieval. Does
`IRISVectorStore` support `max_marginal_relevance_search()`?

**What to do**: Check if the method exists and works:
```python
results = vs.max_marginal_relevance_search('chest findings', k=5, fetch_k=20)
```

**Record**: Supported or not. If not, is it in the roadmap?

---

### Q6 — Async Support

**Why it matters**: `iris-vector-rag` has async pipeline variants. LangChain's async
VectorStore methods (`asimilarity_search`, `aadd_documents`) matter for production use.

**What to do**: Test:
```python
import asyncio
results = asyncio.run(vs.asimilarity_search('chest findings', k=5))
```

**Record**: Works, raises NotImplementedError, or something else.

---

### Q7 — Embedding Model Flexibility

**Why it matters**: Hackathon attendees may not have OpenAI keys. Need to confirm the
package works with local embeddings (Ollama) and other providers.

**What to do**: Test with `OllamaEmbeddings` (mxbai-embed-large) if Ollama is running
on `dpgenai1`. Also test with `HuggingFaceEmbeddings`.

**Record**: Any provider-specific issues? Vector dimension handling?

---

### Q8 — Comparison to `langchain-iris` (caretdev/Dmitry)

**Why it matters**: Dmitry's community package exists on PyPI today. Need to know
what `langchain-intersystems` adds and whether community.intersystems.com should recommend
migrating.

**What to do**:
```bash
pip show langchain-iris  # check if installed
```
Read Dmitry's implementation, compare:
- Metadata filtering support (Predicate system vs basic dict)
- Connection method (DB-API vs Native API vs SQLAlchemy)
- HNSW index creation (automatic vs manual)
- LangChain test suite compliance

**Record**: Feature delta table. Migration guide if needed.

---

## How to Run the Spike

```bash
# 1. Install the wheel from the hackathon repo
pip install ~/ws/ready2026-hackathon/demos/langchain-vectorstore/dist/langchain_intersystems-0.0.1-py3-none-any.whl

# 2. Set up IRIS connection (dpgenai1 or los-iris)
export IRIS_HOSTNAME=dpgenai1
export IRIS_PORT=<port>
export IRIS_NAMESPACE=USER
export OPENAI_API_KEY=sk-...  # or use Ollama

# 3. Run the baseline demo first
cd ~/ws/ready2026-hackathon/demos/langchain-vectorstore
python demo.py

# 4. Run the MIMIC/FHIR spike
# (create spike_fhir.py in this dir following Q2 above)

# 5. Document all findings in SPIKE_NOTES.md (create alongside this file)
```

---

## What to Write When Done

After running the spike, create `SPIKE_NOTES.md` in this docs/ folder with:

1. Answers to Q1–Q8 above
2. A **"AIML71 talking points"** section — 3-5 concrete claims you can make on stage
   with actual numbers ("ingest 1k docs in Xs, sub-Yms filtered search")
3. A **"hackathon recommendation"** — should Gabriel's ChatFHIR use `IRISVectorStore`
   directly, wrap it, or use a different path?
4. A **"migration path"** section — what `iris-vector-rag` needs to change to adopt
   `langchain-intersystems` as its storage backend

---

## Adapter Branch Plan (post-spike)

Once spike is done, implement in `feat/langchain-intersystems-adapter`:

**File**: `contrib/langchain_intersystems/iris_rag_retriever.py`

```python
"""
iris-vector-rag retriever backed by langchain-intersystems IRISVectorStore.

Wraps IRISVectorStore as a LangChain BaseRetriever so it can be dropped
into any iris-vector-rag pipeline that accepts a retriever.

Usage:
    from iris_vector_rag.contrib.langchain_intersystems import IRISRAGRetriever

    retriever = IRISRAGRetriever(
        embeddings=OpenAIEmbeddings(),
        connect_kwargs={...},
        collection_name='my_docs',
        search_kwargs={'k': 5, 'filter': {'category': 'radiology'}}
    )
    pipeline = BasicRAGPipeline(retriever=retriever, llm=...)
"""
```

PR this to public `iris-vector-rag` once `langchain-intersystems` ships on PyPI.

---

## READY 2026 Narrative (working draft — update after spike)

**For AIML71 (RAG section)**:

> "There are two ways to do vector search on IRIS. For FHIR-native applications,
> Elijah's `_v_content` search parameter gives you semantic search inside the FHIR
> search framework — patient-scoped, date-filtered, no extra infrastructure.
> For Python/LangChain applications, `IRISVectorStore` gives you a drop-in vector
> store with a metadata filtering system that maps directly to IRIS SQL — every
> predicate you'd write in a WHERE clause, expressed as a Python dict."

**For the hackathon**:

> "If you're a LangChain developer, `pip install langchain-intersystems` and you're
> done. Same API you use with Pinecone or Chroma. Better metadata filtering than
> either of them, because it's backed by IRIS SQL."

**Numbers needed from spike**: ingest rate, query latency, filter overhead.
