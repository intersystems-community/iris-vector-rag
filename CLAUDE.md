# iris-vector-rag-private

Python RAG framework built on InterSystems IRIS vector search. Pipelines: BasicRAG, CRAG,
HybridGraphRAG, ColBERT. Public mirror: `isc-tdyar/iris-vector-rag`.

## IRIS — THIS PROJECT ONLY

**Container:** `iris-vector-rag-iris` · **Port:** `51972`

```bash
docker start iris-vector-rag-iris   # start if stopped
```

**VECTOR column gotcha:** `INFORMATION_SCHEMA` and dbapi report `VECTOR(FLOAT, N)` as
`varchar`. This is a driver limitation — NOT a schema defect. Never ALTER TABLE based on
this. To verify: `SELECT VECTOR_COSINE(embedding, embedding) FROM table` — if it returns
a result, the column is correctly typed.

Never skip integration tests because "IRIS is unavailable" — start the container.

## Build and Test

```bash
make setup-env && make install      # uv-based setup
source .venv/bin/activate

./scripts/ci/run-tests.sh           # all tests
./scripts/ci/run-tests.sh -t unit   # unit only
./scripts/ci/run-tests.sh -t integration -v
pytest tests/ --cov=iris_rag        # with coverage

make test-community                 # Community Edition mode (1 connection)
make test-enterprise                # Enterprise Edition mode

black . && isort .                  # format
```

## iris-agentic-dev MCP

No `.iris-agentic-dev.toml` here — this project uses `iris-vector-rag-iris:51972`.
To use MCP tools against it, create `.iris-agentic-dev.toml`:

```toml
host = "localhost"
web_port = 52773   # check: docker port iris-vector-rag-iris 52773
namespace = "USER"
```

Run `/iris-agentic-dev` skill if `IRIS_UNREACHABLE`.

## Key Paths

| What             | Where                 |
| ---------------- | --------------------- |
| Pipeline factory | `iris_rag/pipelines/` |
| Vector store     | `iris_rag/storage/`   |
| Config           | `iris_rag/config/`    |
| Gate scripts     | `.specify/gates/`     |

## Quality Gates

`.specify/gates/policy.json` — one policy, three boundaries (agent / git / CI).

```bash
bash .specify/gates/verify.sh --boundary agent   # prettier, markdownlint, shellcheck, spec
bash .specify/gates/doctor.sh                    # hooks wired? tooling present?
bash .specify/gates/canary.sh                    # prove gates still block
python3 scripts/gates/antipatterns.py            # shipped-bug detectors vs baseline
```

Formatters pinned in `package.json` (`npm ci`). New instances of a bug class in
`antipatterns.py` fail the gate; a fixed instance must be removed from
`antipatterns-baseline.txt`. The `tdd-compliance` pre-commit hook rewrites the working
tree — commit with `SKIP=tdd-compliance`.

## Hard Rules

- `SELECT TOP n` not `FETCH FIRST` (IRIS SQL)
- Never cross containers — `iris-vector-rag-iris` only
- Test-first; E2E gate before next phase
- `IRIS_BACKEND_MODE=community` for single-connection test runs
