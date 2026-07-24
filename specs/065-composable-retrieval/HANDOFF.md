# Handoff: Composable Query-Time Retrieval (Feature 065)

**Branch**: `claude/mongodb-vector-search-devx-ws3v6o` · **Spec slot**: `specs/065-composable-retrieval/`
**Last updated**: 2026-07-22 · **State**: speckit spec→plan→tasks→analyze complete; first-PR slice (US1+US8) implemented, tested, pushed.

## Resume locally

```bash
git fetch origin claude/mongodb-vector-search-devx-ws3v6o
git checkout claude/mongodb-vector-search-devx-ws3v6o
git pull origin claude/mongodb-vector-search-devx-ws3v6o
uv sync                      # creates .venv, installs deps (pulls torch — a few min)
```

⚠️ **Branch rule**: all work stays on `claude/mongodb-vector-search-devx-ws3v6o`. The speckit
scripts key off branch name; they were run with `SPECIFY_FEATURE=065-composable-retrieval`
to point at the spec without cutting a `065-*` branch. Keep doing that:
```bash
SPECIFY_FEATURE=065-composable-retrieval .specify/scripts/bash/check-prerequisites.sh --json
```

## What this feature is

Bring vector-search DevX to parity with MongoDB's composable model (`$rankFusion` /
`$scoreFusion` / `$rerank`): make retrieval **mode, fusion weights, filtering, and reranking
query-time options on every pipeline**, instead of baked into the pipeline *type*. Plus fix
the correctness/ergonomics defects found in the current code. Full context: `spec.md`,
`plan.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`.

## Done so far

- **Full speckit artifact set** in `specs/065-composable-retrieval/` (spec, plan, research,
  data-model, 3 contracts, quickstart, tasks, requirements checklist). `/speckit.analyze` ran
  and its 6 findings (C1/C2/U1/G1/R1/F1) were remediated.
- **First-PR slice shipped** (commits up to `3c4899b`):
  - **US1** — filter bug fix: `BasicRAGPipeline.query()` now forwards `metadata_filter` and
    applies `similarity_threshold` (were silently discarded); invalid filter key now raises
    `VectorStoreConfigurationError` instead of being swallowed. `iris_vector_rag/pipelines/basic.py`.
  - **US8** — README imports `iris_rag`→`iris_vector_rag` + package docstring. `README.md`,
    `iris_vector_rag/__init__.py`.
  - Tests: `tests/contract/test_metadata_filter_applied.py` (hermetic, mocked store),
    `tests/contract/test_readme_imports.py`, `tests/integration/test_filtered_search.py` (skips
    without IRIS).

### Test status
```bash
uv run pytest tests/contract/test_metadata_filter_applied.py tests/contract/test_readme_imports.py -q
# 12 passed
uv run pytest tests/integration/test_filtered_search.py -q
# 2 skipped (needs a live IRIS DB — docker-compose up -d)
```
Known **pre-existing, unrelated** failures (verified on clean baseline, NOT caused by this work):
`tests/unit/test_pipelines_unit.py::*load_documents*` (signature bug) and `tests/unit/api/*`
(missing optional dep `bcrypt`).

## Next up (in order)

Tasks live in `tasks.md` (47 tasks, 11 phases, TDD-first). Recommended path:

1. **Foundational (T004–T008)** — `core/query_options.py` (`QueryOptions` + `normalize_query_params`,
   `query`/`query_text` alias, canonical = `query`), `core/composable_query.py`
   (`ComposableQueryMixin`), golden-response harness for all 6 pipelines. Blocks US2/3/4.
2. **US2 (T014–T020)** — route every pipeline's `query()` through the mixin; consistent defaults.
3. **US3 (T021–T024)** — `rerank=bool|str|callable`, cached cross-encoder in `retrieval/rerank.py`.
4. **US4 (T025–T031)** — `retrieval=vector|text|hybrid|rrf` + weights via `retrieval/engine.py`,
   reusing `HybridRetrievalMethods`/`_hybrid_utils`; text side = iris-vector-graph BM25.
5. **US5/US6/US7 (P3)**, then Polish (T043–T047).

Run with: `/speckit.implement` (scope it per slice), or implement tasks directly.

## Gotchas / decisions already made (don't re-litigate)

- **Registered pipeline → file map** (verified in `iris_vector_rag/__init__.py`):
  `basic`→basic.py, `basic_rerank`→basic_rerank.py, `crag`→crag.py,
  **`graphrag`→hybrid_graphrag.py** (NOT graphrag.py — that + graphrag_merged.py +
  iris_global_graphrag.py are legacy/out of scope), **`pylate_colbert`→colbert_pylate/pylate_pipeline.py**,
  `multi_query_rrf`→multi_query_rrf.py.
- **T001–T003 deferred** (YAGNI): they're plumbing for US2+, unused by the slice. Do them
  when landing Foundational/US2, not before.
- **`.DAT` fixtures**: only `mcp-basic-rag-5docs` (5 docs) exists. `medical-graphrag-20`
  (referenced in CLAUDE.md examples) does **not** exist here. US1 test uses <10 programmatic
  docs (allowed by Principle II). **US4 (T026) needs a new ≥10-doc `.DAT` fixture with a BM25
  index** — `make fixture-create`.
- **Constitution gates** (`.specify/memory/constitution.md`): TDD tests-first (III,
  non-negotiable), new options default-disabled / zero breaking changes (IV, non-negotiable),
  `.DAT` fixtures for ≥10 entities (II), parameterized SQL for filters (VIII), <5ms overhead
  when disabled (VI).
- **`similarity_search` return type** (U3): fix *additively* — add `search_by_text` /
  `search_by_vector`; leave the polymorphic method untouched (Principle IV).
- **Filter SQL note**: `IRISVectorStore` filters via `LIKE` on serialized JSON with
  single-quote escaping + a key whitelist (`MetadataFilterManager`). If a reviewer pushes on
  Principle VIII, consider switching the LIKE clause to bound parameters (follow-up, not in the slice).
- **Integration tests need IRIS**: `docker-compose up -d` then `make setup-db`. No DB in the
  cloud session, so integration/DB tests were authored to skip cleanly there.

## No PR opened yet

The slice is pushed but no PR exists (wasn't requested). Open one when ready.
