# Known test-order dependencies

Tests that have failed in one order and passed in every other. The default order is
collection order, so these do not fail day to day — which is exactly why they need
writing down. Each entry records what was observed, not a theory about it.

The order is not pinned by configuration. `pytest.ini:34` does contain
`-p no:randomly`, but that whole `addopts` block never reaches pytest: the file's
section header is `[tool:pytest]`, which is only valid in `setup.cfg` — in
`pytest.ini` it must be `[pytest]`. Nothing in it applies, which is easy to confirm
from the fact that `--verbose` and `--cov-fail-under=80` are also in that block and
neither affects a run (`docs/DEBT.md` entry 8). `pytest-randomly` is not installed
either, so there is nothing randomising the order to disable. Both facts mean the
same thing for these entries — a random order is not the default — but neither is a
guarantee: installing `pytest-randomly` would start randomising immediately, because
the opt-out is not in effect.

Add an entry when a test fails in one ordering and passes in isolation. Delete the
entry when the dependency is removed, not when it stops reproducing.
