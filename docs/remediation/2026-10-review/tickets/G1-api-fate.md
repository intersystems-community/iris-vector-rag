# Decide the fate of `api/`

- type: grilling (HITL)
- status: closed
- blocked-by: none (R3 closed)
- claimed-by: none

Facts: [research/R3-support-status-facts.md](../research/R3-support-status-facts.md).

## Question

Is `iris_vector_rag/api/` supported product? Fix it, mark it experimental, or delete it?
The review found an app that cannot serve a request (import-time `create_app()`), an
upload endpoint that stores nothing, WebSocket paths with no authorization, and
unauthenticated endpoints leaking error text. Decide which, with the facts from the
support-status ticket. If it stays: what auth model, and is the WebSocket surface kept?

Findings that move with the decision: AP-01 to AP-22, VM-04 (shares `uvicorn` target),
CM-05 (shares config).

## Resolution

Settled with the maintainer on 2026-10-02: **delete `api/`, in two releases, keeping the
design re-hydratable.** Recorded as [ADR 0001](../../../adr/0001-retire-the-rest-api.md).

1. **Deprecate first.** The next release warns on import of `iris_vector_rag.api`, adds a
   README and CHANGELOG notice, marks the `api` extra deprecated, and drops `api` from the
   `all` extra and the CI install. The release after removes the code. Announced under
   Changed and Removed, as with the other policy decisions.
2. **Delete, do not extract.** No new repository. A git tag (`archive/rest-api-v1`) on the
   last commit containing the code and its tests is named in the deprecation message.
3. **Keep the design re-hydratable.** Before removal, write a design pack under
   `docs/archived/rest-api/`. No design document exists outside the code and the two
   `IMPLEMENTATION_*.md` files (there is no spec-kit feature for it), so the pack has to
   hold: the architecture and module map; the route table with request and response models;
   the auth model (API keys, secrets hashing, permissions, expiry); the rate-limit scheme
   (windows, concurrency, Redis layout); the WebSocket protocol; the operation and
   upload-progress model and the migration SQL; the configuration keys in
   `config/api_config.yaml`; the original `README.md` and `IMPLEMENTATION_*.md`; and a
   "what version 2 must do differently" list drawn from findings AP-01 to AP-24.
4. **Patch nothing** in the deprecation release. AP-01 means the app cannot serve a request,
   so AP-08 and AP-09 are not exploitable today.
5. **Sweep at removal:** `iris_vector_rag/api/`, `tests_api/`, `tests/integration/api/`,
   `config/api_config.yaml`, the `api` extra and its lines in `all`, the API docs and README
   claims, the `api` extra in the CI install lines, and `release.yml`'s hand-installed
   `bcrypt` and `email-validator`.
6. **Left to the `mcp/` fate ticket:** the dead `api.services` import in `mcp/bridge.py`
   (VM-05), the stale `uvicorn` target in `mcp/cli.py` (VM-04), and the FastAPI, uvicorn and
   websockets lines in the `mcp` extra. `test_bridge_imports_pipeline_manager` fails at
   removal unless that ticket deals with it first.

Findings AP-01 to AP-24 get the fate `deleted-with:api (feature:retire-rest-api)`, and so
does the `make api-*` part of OS-27. Thirteen of the 35 stale test patch targets in
`docs/DEBT.md` entry 7 disappear with `tests_api/`.
