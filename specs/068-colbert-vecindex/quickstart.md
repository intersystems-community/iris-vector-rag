# Quick Start: ColBERT VecIndex (068)

## Prerequisites

```bash
# iris-vector-graph 1.21.0 must be installed
pip install "iris-vector-graph>=1.21.0"

# Deploy VecIndex.cls to IRIS container
bash scripts/deploy_vecindex.sh iris-langchain-spike
```

## Ingest — Dual-Write Tokens to VecIndex

```python
import iris.dbapi as dbapi
from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor
from iris_vector_rag.pipelines.colbert_iris.schema import ColBERTSchema

conn = dbapi.connect(hostname="localhost", port=13972,
                     namespace="USER", username="_SYSTEM", password="SYS")

schema = ColBERTSchema(conn)
schema.create_tables()

ingestor = ColBERTIngestor(conn, model=my_colbert_model, token_dim=128)

docs = [{"doc_id": f"doc_{i}", "text": f"Document {i}", "metadata": {}} 
        for i in range(5000)]

stats = ingestor.ingest_documents(
    docs,
    use_vecindex=True,   # enables dual-write to ^VecIdx globals
)
print(f"Ingested {stats['docs_ingested']} docs, {stats['vecindex_count']} tokens to VecIndex")
print(f"VecIndex build: {stats['vecindex_build_ms']:.0f}ms")
```

## Search — MaxSim via VecIndex

```python
import numpy as np
from iris_vector_rag.pipelines.colbert_iris.vecindex_phase2 import VecIndexSearcher

searcher = VecIndexSearcher(conn, index_name="colbert_tokens", token_dim=128)

# Encode query — 4 token vectors of dim 128
q_vecs = my_colbert_model.encode(["IRIS vector search performance"], is_query=True)[0]
q_vecs = np.array(q_vecs, dtype=np.float32)
if q_vecs.ndim == 1:
    q_vecs = q_vecs.reshape(1, -1)

# Search — nprobe=2 is balanced; nprobe=4 for higher recall
results, meta = searcher.search(q_vecs, top_k=10, nprobe=2)

for doc_id, score in results:
    print(f"  {doc_id}: {score:.4f}")

print(f"Candidates: {meta['n_candidates']}, Total: {meta['total_ms']:.0f}ms")
```

## Check Index Status

```python
info = searcher.info()
# {"name": "colbert_tokens", "count": 267063, "dim": 128, "metric": "dot", "trees": 4, "leafSize": 50}
print(info)
```

## Benchmark vs Phase 2 SQL HNSW

```bash
IRIS_PORT=13972 python tests/colbert_iris/benchmark_scale.py \
  --tiers 5000 --queries 15 \
  --skip-baseline --skip-plaid --skip-sp \
  --output /tmp/colbert_vecindex_bench.json
```

Output includes `phase2_vecindex` tier with p50/p95/p99, recall@10, speedup vs Phase 2.

## Rebuild Index

```python
# Drop and rebuild (e.g., after corpus changes)
searcher.drop()
stats = ingestor.ingest_documents(docs, use_vecindex=True)
```

## IRIS Quirks Relevant to VecIndex

| Issue | Behavior | Fix |
|---|---|---|
| `$vectorop` SIMD | Available IRIS 2024.1+ on all tiers | No workaround needed |
| Globals persistence | `^VecIdx` survives IRIS restart | No re-ingest after restart |
| No class compile lock | VecIndex uses no DDL | No `-110` errors |
| `vec_insert` overhead | 128 SET ops per token via `^tmpvec` | Use direct gref if available |
| `intersystems_iris` package | Required for `IRISGraphEngine._iris_obj()` | Pre-installed with irispython>=5.1 |
