# Decide the fate of `api/`

- type: grilling (HITL)
- status: open
- blocked-by: R3 (Gather support-status facts)
- claimed-by: none

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
