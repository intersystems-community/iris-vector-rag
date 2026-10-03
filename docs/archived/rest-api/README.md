# Archived design: the REST API (`iris_vector_rag.api`)

Status: archived. Deprecated in 0.15, removed in 0.16
([ADR 0001](../../adr/0001-retire-the-rest-api.md)).

`iris_vector_rag/api/` was a FastAPI service for the RAG pipelines: API-key
authentication, Redis rate limiting, a WebSocket event stream and a document upload
endpoint. It could not serve a request as built, its upload path stored nothing, and no
code or CI job ran it. We did not fix it. We kept the design so a second version does
not start from zero and does not repeat the first version's defects.

## Where the code is

- Git tag **`archive/rest-api-v1`**: the last commit that contains `iris_vector_rag/api/`,
  `tests_api/`, `tests/integration/api/` and `config/api_config.yaml`.
  Recover it with `git show archive/rest-api-v1:iris_vector_rag/api/main.py`, or
  `git worktree add ../rest-api-v1 archive/rest-api-v1`.
- The tag is local until a maintainer pushes it: `git push origin archive/rest-api-v1`.
- Nothing in this pack is executable. The SQL, YAML and READMEs under
  [`original/`](original/) are verbatim copies taken at the tag.

## Contents

| File                                                                 | What it holds                                                           |
| -------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| [architecture.md](architecture.md)                                   | Module map, start-up sequence, middleware stack, services               |
| [routes.md](routes.md)                                               | Route table with request and response models, error model               |
| [auth-and-rate-limits.md](auth-and-rate-limits.md)                   | API keys, secret hashing, permissions, expiry, rate-limit scheme, Redis |
| [websocket-protocol.md](websocket-protocol.md)                       | `/ws` handshake, message types, events, heartbeat                       |
| [data-model.md](data-model.md)                                       | Tables, views, operation and upload-progress model, migration           |
| [configuration.md](configuration.md)                                 | Every key in `config/api_config.yaml`, and the defaults in code         |
| [version-2-must-do-differently.md](version-2-must-do-differently.md) | What a rebuild must change, from findings AP-01 to AP-24                |
| [original/](original/)                                               | Original `README.md`, `IMPLEMENTATION_*.md`, config and SQL             |

## Caveats

- There was never a spec-kit feature for this; the only design text is the code and the two
  `IMPLEMENTATION_*.md` files. Comments in the code cite "Feature 042" and
  `specs/042-full-rest-api/`, which is not in the repository. This pack is reconstructed
  from the code at the tag, so where it and the original READMEs disagree, the code
  (and this pack) is right.
- The original README describes a working product. It was never one: see
  [version-2-must-do-differently.md](version-2-must-do-differently.md).
- The review that found the defects is in
  `docs/remediation/2026-10-review/findings.md` (AP-01 to AP-24).
