# Changelog

## v0.11.4

- Consolidate all IRIS connection logic through `get_iris_connection()` — removes
  duplicate `get_iris_connector_for_embedded()` from `common/utils.py` and the
  3-way `hasattr` fan-out in `hybrid_graphrag.py`.
- Route `colbert_iris/plaid.py` and `vecindex_phase2.py` through
  `get_iris_connection()`; remove direct `intersystems_iris.createConnection`
  bypass.
- Fix top-level `import iris.dbapi` in `common/connection_pool.py` that raised
  `ImportError` at import time — now lazy-loaded inside `_create_connection()`.
- Add embedded-mode support to `get_iris_connection()`: detects
  `embedded-kernel` / `embedded-local` runtime via `iris.runtime`, and
  auto-configures embedded-local when `IRISINSTALLDIR` is set — skips TCP
  probe in both cases.
- Merge Dependabot PRs: `actions/checkout` 6→7, `gitleaks/gitleaks-action`
  2→3.

## v0.5.4

- Fix IRIS connection API usage (replace unsupported `iris.connect()` with
  supported APIs).
- Require `iris-vector-graph` for GraphRAG pipelines; fail fast with clear
  ImportError when missing.
- Auto-initialize GraphRAG schema tables with validation, logging, and
  per-table error reporting.
