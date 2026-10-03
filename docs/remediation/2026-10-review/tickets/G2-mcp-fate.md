# Decide the fate of `mcp/`

- type: grilling (HITL)
- status: closed
- blocked-by: none (R3 closed)
- claimed-by: Tom, driving session (2026-10-02)

Facts: [research/R3-support-status-facts.md](../research/R3-support-status-facts.md).

## Question

Is `iris_vector_rag/mcp/` supported product? The review found an authentication check that
never validates, a CLI that raises on default paths and still names the old package,
hard-coded health and metrics values, and a `python`/`both` mode that only sleeps. Fix,
mark experimental, or delete? If it stays: what auth model, and does it depend on `api/`
(see [Decide the fate of `api/`](G1-api-fate.md))?

Findings that move with the decision: VM-01, VM-03 to VM-06, VM-13, VM-16.

## Resolution

Settled with the maintainer on 2026-10-02: **rebuild the transport thin and keep what works**
(option 2). Recorded as [ADR 0002](../../../adr/0002-mcp-server-is-python-over-stdio.md).

1. **The Node server is retired.** `tools/nodejs/` (8 TypeScript files), `Dockerfile.mcp`,
   `scripts/docker/docker-entrypoint-mcp.sh` and `config/mcp_config.yaml` are deleted. It only
   worked through an HTTP bridge at `localhost:8001` that was never defined. The last commit
   containing it is tagged `archive/mcp-node-v1` and the tag is named in the CHANGELOG.
2. **Six tools ship:** `rag_basic`, `rag_basic_rerank`, `rag_crag`, `rag_graphrag`,
   `rag_pylate_colbert`, and `rag_health_check` with real checks (IRIS reachable, pipelines
   constructible, embedder loadable; no invented values). `rag_metrics` is dropped (there are no
   counters) and `rag_iris_global_graphrag` is dropped (it depends on an external app and its
   load path is broken, PL-05); either can return when sound.
3. **stdio only, no authentication.** The process boundary is the trust boundary.
   `MCP_AUTH_MODE` and the api_key machinery are deleted, not fixed (VM-01). A remote or HTTP
   MCP server with a real auth model is a separate future decision (map fog).
4. **One entry point.** A console script `iris-vector-rag-mcp`, also
   `python -m iris_vector_rag.mcp`, both start the stdio server. No subcommands:
   `cli.py` and the `__main__` modes go. The IRIS connection comes from the Connection
   settings resolver. `docs/MCP_INTEGRATION.md` is rewritten with a working Claude Desktop
   configuration, and "Production-Ready" is replaced by what is tested.
5. **Tests.** Unit-test the server through the SDK's in-memory client (no IRIS; runs in CI).
   Keep the live-IRIS integration tests behind the `mcp` marker, run on demand against the
   project container. Delete the tests of the removed parts (HTTP bridge, CLI, REST
   consistency, `test_bridge_imports_pipeline_manager`).
6. **The `mcp` extra** becomes the official `mcp` SDK only; fastapi, uvicorn, websockets and
   the duplicate dependency-group entry go. The dead `api.services` import (VM-05) and the
   stale `uvicorn` target (VM-04) disappear with the bridge and CLI, which resolves what the
   `api/` ticket left open.

Also kept: strict input schemas (`additionalProperties: false`) so unknown keys are rejected
rather than passed to `pipeline.query(**params)`, and handler errors that do not echo
internals (VM-13).

Findings assigned to feature `mcp-stdio-server`: VM-01, VM-03, VM-04, VM-05, VM-06, VM-13,
VM-17, OS-25. VM-16 is bundled: its `mcp/` items (`__main__`, `validation.py` range
validators, the `technique_handlers.py` logging and import guards) are handled by the rebuild;
its other items stay open. Announced under Unreleased 0.15, Changed and Removed.
