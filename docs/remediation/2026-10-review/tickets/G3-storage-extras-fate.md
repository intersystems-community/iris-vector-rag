# Decide the fate of enterprise storage and the devtools bridge

- type: grilling (HITL)
- status: open
- blocked-by: R3 (Gather support-status facts)
- claimed-by: none

## Question

Two pieces look unused and are dangerous: `storage/enterprise_storage.py` (`IRISStorage`
can drop a populated table and takes SQL-injectable filter keys; no caller found outside
`storage/__init__.py`) and `testing/iris_devtools_bridge.py` (`DROP NAMESPACE` with no
container guard). Fix, mark experimental, or delete each? If deleted, what is the
deprecation path for anyone importing them?

Findings that move with the decision: ST-01, ST-02, ST-03, ST-20, VM-10.

## Resolution

(pending)
