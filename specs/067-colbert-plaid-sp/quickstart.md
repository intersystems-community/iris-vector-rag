# Quickstart: Deploy and Test RAG.ColBERTSearch

## Prerequisites
- `iris-langchain-spike` container running on port 13972
- Feature 066 corpus loaded (5000 docs, 267K tokens, K=512 PLAID centroids built)

## Step 1: Install numpy in IRIS container
```bash
bash scripts/setup_spike_env.sh
# or manually:
docker exec iris-langchain-spike pip3 install --target /usr/irissys/mgr/python numpy==1.26.4
```

## Step 2: Deploy the SP
```bash
bash scripts/deploy_colbert_sp.sh iris-langchain-spike
# or manually:
docker cp iris_vector_rag/pipelines/colbert_iris/sp/ColBERTSearch.cls iris-langchain-spike:/tmp/
docker exec -i iris-langchain-spike bash -c "echo \"set sc=\$SYSTEM.OBJ.Load('/tmp/ColBERTSearch.cls','ck') write sc halt\" | /usr/irissys/bin/irissession IRIS -U USER"
```

## Step 3: Run tests
```bash
IRIS_PORT=13972 /tmp/spike-venv-312/bin/python -m pytest tests/colbert_iris/test_colbert_sp.py -v -m integration
```

## Step 4: Run benchmark
```bash
IRIS_PORT=13972 /tmp/spike-venv-312/bin/python tests/colbert_iris/benchmark_scale.py \
  --tiers 5000 --queries 15 --skip-baseline \
  --output /tmp/colbert_sp_bench.json
```

## Calling the SP from Python
```python
import json, numpy as np, iris.dbapi as dbapi

conn = dbapi.connect(hostname="localhost", port=13972,
                     namespace="USER", username="_SYSTEM", password="SYS")

q_vecs = np.random.rand(4, 128).astype(np.float32)
q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
q_json = json.dumps(q_vecs.tolist())

cur = conn.cursor()
cur.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", [q_json, 10, 4])
row = cur.fetchone()
result = json.loads(row[0])
print(result["results"])      # [["doc_id", score], ...]
print(result["total_ms"])     # ~190ms at T5K warm
```

## Known IRIS Quirks
See `research.md` for the complete quirks reference table. Key points:
- Class name: `RAG.ColBERTSearch` (no underscores)
- `#tmp` tables NOT supported — SP uses Python dict for accumulation
- `iris.sql.exec(sql, *list)` fails at >50 args — use string interpolation
- VECTOR columns cannot be fetched — use `VECTOR_DOT_PRODUCT` inline
- First call per connection may be slower (~30ms Python module import overhead)
