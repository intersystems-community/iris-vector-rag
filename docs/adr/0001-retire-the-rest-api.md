---
status: accepted
---

# Retire the REST API, and keep its design re-hydratable

`iris_vector_rag/api/` (a FastAPI service with API-key auth, rate limiting, WebSocket
streaming and document upload) is deprecated in one release and removed in the next. It
cannot start as built, its upload path stores nothing, its operator tooling was deleted in
May 2026, no code or CI job runs it, and nothing has added behaviour to it since October 2025. Fixing it would mean building missing features on a security-sensitive network
surface with no evidence of demand.

The code is not what we keep. We keep the **design**: a git tag on the last commit that
contains the code and its tests, plus an archived design pack under
`docs/archived/rest-api/` that records the architecture, route table, auth and rate-limit
model, WebSocket protocol, data model and configuration keys, and the known defects of the
first version so a rebuild does not repeat them. See the remediation map for the detail.

## Considered options

- **Fix it and keep it supported.** Rejected: the upload feature does not exist, the auth
  model would have to be designed, Redis becomes a hard dependency, and no test exercises it.
- **Mark it experimental.** Rejected: it leaves twenty-four known defects, including a
  leaked error text and default `demo`/`demo` credentials, in the tree, and tends to become
  permanent.
- **Extract it to its own repository.** Rejected for now: it needs an owner, a release
  process and a repo, and nothing suggests anyone wants one. The tag is the starting point
  if someone does.

## Consequences

Anyone using `pip install iris-vector-rag[api]` is affected; the deprecation release is the
notice. `mcp/` carries a dead import of `api.services` and shares the FastAPI extras; what
happens to those is decided with the fate of `mcp/`.
