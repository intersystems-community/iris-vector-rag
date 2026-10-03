# What version 2 must do differently

Drawn from findings AP-01 to AP-24 in `docs/remediation/2026-10-review/findings.md`. Each
item names the finding that motivates it.

## Build and start

1. **Use an application factory.** `app = create_app()` at import time ran before
   `lifespan` had filled the state, so every router and middleware was built with `None`
   and no request could succeed. Build dependencies in the factory or inject them
   through FastAPI dependencies, and import the module in a test on day one (AP-01).
2. **Have a real entry point.** `uvicorn.run("iris_rag.api.main:app")` named a package
   that does not exist. Ship a console script and test that it starts (AP-16).
3. **Check that models and services agree.** `PipelineManager` passed fields the model
   lacks, and `lifespan` called a pool method that did not exist. Run start-up in CI
   against a real IRIS (AP-23, AP-24).
4. **Do not ship default credentials.** Refuse to start without configured database
   credentials instead of falling back to `demo`/`demo` (AP-09).
5. **Make the configuration file and the code agree**, with one typed settings object
   (see [configuration.md](configuration.md)).

## Authentication and authorisation

6. **Enforce permissions on every transport.** The WebSocket path skipped
   `check_permission` and the REST input limits (AP-05). Put authorisation in one shared
   dependency used by HTTP and WebSocket.
7. **Check ownership on every object.** `watch_upload` streamed any operation's progress
   to any key (AP-04).
8. **Return one error for every credential failure**, do not echo the key id, and always
   run the hash (AP-10).
9. **Compare times in UTC**, with timezone-aware `datetime` (AP-12, AP-21).
10. **Decide how keys are issued** before building anything else; `AuthService` was never
    reachable (AP-21).
11. **Authenticate the WebSocket within a deadline** and drop idle, unauthenticated
    sockets (AP-15).
12. **Make public and protected routes match their documentation** (AP-03), and keep
    error text that carries hosts, ports or exception strings out of unauthenticated
    responses (AP-08).

## Correctness

13. **Do not raise `HTTPException` inside `BaseHTTPMiddleware`.** It bypasses the
    exception handlers, so a 401 or 429 became a 500. Return a response, or use
    dependencies or pure ASGI middleware (AP-02, AP-14).
14. **Put CORS outermost**, and get the order right (the last middleware added is the
    outermost), so 401 and 429 carry CORS headers (AP-17).
15. **Serialise responses with `model_dump(mode="json")`** or return the model, so the
    503 path works exactly when IRIS is down (AP-07).
16. **Do not block the event loop.** Run pipeline calls, IRIS, bcrypt and Redis in
    threads or use async clients (AP-13).
17. **Make the rate limiter atomic** (one Lua script or `MULTI`), key members uniquely,
    fail open or closed by an explicit setting when Redis is down, and release
    concurrency slots in a `finally` with a short TTL (AP-14).
18. **Keep task handles** from `create_task` so they are not collected (AP-15).
19. **Bound every input**: a missing `file.size`, and `limit` with no bounds, became a 500
    and an unchecked SQL parameter (AP-11). Rely on the Pydantic bounds and delete the
    duplicate hand checks (AP-22).

## Features that must be real

20. **Upload must store and index documents** or not exist. Version 1 read nothing,
    slept, and reported success (AP-06). Write the end-to-end test first: upload, then a
    query that finds the content.
21. **An in-memory fallback limiter must exist** if the configuration promises one.

## Code shape

22. One mapper for row to model, and one error constructor instead of a 40-line
    `ErrorResponse(...)` repeated (AP-18).
23. One moving-average update, without identical branches (AP-19).
24. Use the injected collaborator instead of building a throwaway one (AP-20).

## Process

Write a design document and a spec before the code (none existed). Give the service an
owner, a launcher and a CI job that starts it, because what nobody runs stops working.
See also OS-27 in the findings: the documentation promised `make api-*` targets, a CLI
and a Dockerfile that were deleted or never written.
