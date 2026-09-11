# Known test-order dependencies

Tests that have failed in one order and passed in every other. The suite pins
`-p no:randomly` in `pytest.ini`, so the default order is the file order and these
do not fail day to day — which is exactly why they need writing down. Each entry
records what was observed, not a theory about it.

Add an entry when a test fails in one ordering and passes in isolation. Delete the
entry when the dependency is removed, not when it stops reproducing.

## `tests/contract/test_mcp_bridge_contract.py::TestMCPBridgeIntegration::test_bridge_imports_pipeline_manager`

**Observed:** failed once during a run with a randomised order, on the import of
`iris_vector_rag.api.services.PipelineManager`. Passed in isolation, passed in the
full contract suite in the pinned order, and passed in every rerun afterwards,
including the CI-equivalent contract command.

**Symptom:** the failure is an `ImportError` the test converts to
`pytest.fail("Cannot import PipelineManager: ...")`. `iris_vector_rag.api.services`
is imported by several other tests, some of which patch or insert entries in
`sys.modules` for API and pipeline modules; a partially initialised or replaced
module left in `sys.modules` by an earlier test makes this import fail for the
next one. Which test leaves it is not established.

**No seed was captured.** The random order that produced the failure was not
recorded, so the exact ordering cannot be replayed. Reproducing it means running
the contract suite with `-p randomly` over several seeds and recording the seed
that fails:

```bash
python -m pytest tests/contract/test_mcp_bridge_contract.py \
    -p randomly -p no:cacheprovider -q
# the seed is printed in the header: "Using --randomly-seed=..."
```

**Status:** open. Not deselected in `tests/contract/ci_known_failures.txt` — it
does not fail in the pinned order, and deselecting it would hide the import
contract it checks rather than the ordering problem underneath it.
