---
status: accepted
---

# The MCP server is Python over stdio; the Node server is retired

`iris_vector_rag/mcp/` becomes a real MCP server built on the official `mcp` Python SDK over
stdio, exposing six tools (`rag_basic`, `rag_basic_rerank`, `rag_crag`, `rag_graphrag`,
`rag_pylate_colbert`, and a `rag_health_check` that really checks). The TypeScript server in
`tools/nodejs/` is retired. It could only work through an HTTP bridge at `localhost:8001`
that was never defined, so it had nothing to call, and keeping two servers for the same tools
would double the maintenance. The handlers, schemas and validators are kept.

Stdio means the process boundary is the trust boundary, so there is no authentication to
build or get wrong. A remote or HTTP MCP server, with a real auth model, is a separate future
decision.

## Considered options

- **Fix it as built.** Rejected: there was no working server to fix, and the architecture
  (Node server calling a Python bridge) had to be decided first anyway.
- **Mark it experimental.** Rejected: it leaves a package that cannot start, and false
  "Production-Ready" documentation, in the tree.
- **Delete the Python side.** Rejected: MCP is the obvious route for an agent to drive these
  pipelines, and the handlers are working code.

## Consequences

Anyone running the Node server or `Dockerfile.mcp` is affected; both were already unable to
work. The last commit containing the Node server is tagged `archive/mcp-node-v1`.
