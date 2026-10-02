# Decide the connection defaults policy

- type: grilling (HITL)
- status: open
- blocked-by: none
- claimed-by: none

## Question

What should an unconfigured connection do? Today: `get_iris_connection()` auto-detects the
first Docker container whose line mentions "iris" and "1972" (any project's), a
password-expiry error can run `UnExpireUserPasswords("*")` on a container chosen by port
substring, and the defaults are `localhost:1972`, `_SYSTEM`, `SYS`. The project owns
`iris-vector-rag-iris:51972`, and the global rule is "always pass the container
explicitly, never auto-discover".

Decide: remove auto-detection entirely, or keep it behind an explicit opt-in; fail fast
with a clear error when nothing is configured, or default to this project's own port; what
to do about the password-fix helper; one environment-variable precedence
(`IRIS_USERNAME` vs `IRIS_USER`) and where it is defined. Also how the change is announced
(it breaks anyone relying on the defaults).

Findings: CM-01, CM-02, CM-03, CM-05, CM-06, CM-09.

## Resolution

(pending)
