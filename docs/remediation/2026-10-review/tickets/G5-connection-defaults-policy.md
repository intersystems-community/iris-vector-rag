# Decide the connection defaults policy

- type: grilling (HITL)
- status: closed
- blocked-by: none
- claimed-by: Tom, driving session (2026-10-02)

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

## Decisions

Settled with the maintainer on 2026-10-02, aligned with iris-devtester's configuration
(`iris_devtester/config/discovery.py`, read-only):

1. **No Docker scanning.** `auto_detect_iris_port()` and its `docker ps` call are removed.
   The only Docker inspection kept is the pinned form: when `IRIS_CONTAINER_NAME` is set,
   inspect exactly that container to find its mapped port. No substring matching.
2. **Defaults follow iris-devtester:** host `localhost`, port `1972`, namespace `USER`,
   user `_SYSTEM`, password `SYS`. One log line names the `host:port` and user actually
   used. This replaces the earlier recommendation to raise when the port is unset.
3. **Password-fix helper removed** (`_hard_fix_iris_passwords`). An expired password
   raises the driver's error with a hint; handling it for test containers belongs to
   iris-devtester.
4. **One resolver, devtester's vocabulary and precedence:** explicit settings, then
   environment, then `.env`, then the pinned container, then defaults. Variables:
   `IRIS_HOST`, `IRIS_PORT`, `IRIS_NAMESPACE`, `IRIS_USERNAME`, `IRIS_PASSWORD`,
   `IRIS_TIMEOUT`, `IRIS_CONTAINER_NAME`. `IRIS_USER` stays as a deprecated alias with a
   warning. Every entry point reads through the resolver, not `os.environ` directly.
   Environment beats file in every getter, logging included.
5. **No new runtime dependency.** The settings object uses devtester's `IRISConfig` field
   names, and a devtester `IRISConfig` can be passed in as-is.
6. Not a decision, just a fix: stop printing the DB URL (CM-09).

## Resolution

Settled with the maintainer on 2026-10-02; the decisions above are final, plus:

1. **Refusal hint.** When the resolved settings came entirely from defaults (no explicit
   value, environment, `.env` or container pin) and the failure is a refused or unreachable
   connection, the error says so: it names the `host:port` and user used and points at
   `IRIS_PORT` and `IRIS_CONTAINER_NAME`. Explicitly configured settings get the driver's
   own error, unchanged.
2. **Announcement.** Listed under the existing `## Unreleased` section of `CHANGELOG.md`
   (the 0.15 work), under "Changed" and "Removed". No major bump. The `IRIS_USER` alias
   stays for one minor release.
3. **Glossary.** `CONTEXT.md` gained **Connection settings** and **Pinned container**.

Behaviour that changes for callers: Docker scanning and the password-fix helper are gone;
the port default is unchanged (1972); `IRIS_USER` warns; `LOG_LEVEL` now beats the YAML
file. Only someone who relied on the scan to find a non-default port notices, and the
refusal hint covers them.

Findings assigned: CM-01, CM-02, CM-03, CM-05, CM-06, CM-09, CM-16 (feature
`connection-settings`). CM-03 is reclassified: the defaults stay (they are iris-devtester's
and the standard IRIS values); what is removed is the guessing. Not decided here and left to
the planning step: CM-04, CM-07, CM-08, CM-10, CM-11.
