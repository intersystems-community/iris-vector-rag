# Decide the fate of enterprise storage and the devtools bridge

- type: grilling (HITL)
- status: closed
- blocked-by: none (R3 closed)
- claimed-by: Tom, driving session (2026-10-02)

Facts: [research/R3-support-status-facts.md](../research/R3-support-status-facts.md).

## Question

Two pieces look unused and are dangerous: `storage/enterprise_storage.py` (`IRISStorage`
can drop a populated table and takes SQL-injectable filter keys; no caller found outside
`storage/__init__.py`) and `testing/iris_devtools_bridge.py` (`DROP NAMESPACE` with no
container guard). Fix, mark experimental, or delete each? If deleted, what is the
deprecation path for anyone importing them?

Findings that move with the decision: ST-01, ST-02, ST-03, ST-20, VM-10.

## Resolution

Settled with the maintainer on 2026-10-02: **delete both, directly, with no deprecation release.**
The maintainer knows of no one importing `IRISStorage` or running the bridge. No ADR and no
archive tag: `IRISStorage` duplicates `IRISVectorStore` and holds no design worth preserving,
and the bridge duplicates `iris-devtester`.

1. **Delete `IRISStorage`** (`storage/enterprise_storage.py`), its test file
   `tests/test_enterprise_storage.py`, and its re-export and `__all__` entry in
   `storage/__init__.py`. CHANGELOG line under Removed. Unlike `api/`, it was never documented,
   and waiting a release would ship a known data-loss path for one more cycle.
2. **Delete `IrisDevToolsBridge`** (`testing/iris_devtools_bridge.py`), its re-export in
   `testing/__init__.py`, the unused `iris_devtools_bridge` session fixture in
   `tests/conftest.py`, and the bridge call in `examples/scripts/create_mcp_test_fixture.py`
   (the call already fails and the script falls back; whether the script survives the `mcp/`
   rebuild is for the plan). Tests that need a container use `iris-devtester` directly.
3. **Housekeeping the plan must do:** remove `docs/DEBT.md` entry 3 (the failing enterprise
   storage tests) and the three enterprise-storage sites in entry 4; update the reference in
   entry 5; mark the open "enterprise_storage.py reads sql/schema.sql" checkbox in
   `specs/070-objectscript-sdk/tasks.md` as dropped.
4. **Not decided here:** the rest of `testing/` (`backend_manager`, `connection_pool`,
   `validators`, `exceptions`), which nine test files use, and the inverted dependency where
   production `config/backend_modes.py` imports `testing.exceptions` (part of VM-16). Recorded
   in the map's fog.

Findings assigned to feature `remove-dead-code`: ST-01, ST-02, ST-03 (including its second
injection site), ST-20, VM-10, VM-18. ST-21 is bundled: the `_convert_clob_to_string`
duplication in `enterprise_storage.py` goes with it; the other two duplications stay open.
