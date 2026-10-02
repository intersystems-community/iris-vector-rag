# Decide the fate of `api/`

- type: grilling (HITL)
- status: open
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

(pending)
