# Plan: CI Tiers

## Overview

This plan implements a two-tier CI strategy to catch formatting issues, lint violations, and real IRIS integration bugs early in the PR cycle. The feature gates PRs to `main` with both a fast (sub-3min) format/lint check and a selective IRIS smoke test for changes that affect core functionality.

**Key outcome**: All PRs pass black format check + full ruff lint before merge; PRs to `main` also verify basic IRIS ingest+query works.

---

## Key Files

### CI Configuration

- `.github/workflows/ci.yml` — Existing; add `format-check` step, upgrade ruff ruleset
- `.github/workflows/pr-integration.yml` — New; IRIS service container + smoke tests
- `.github/workflows/security.yml` — Existing; remove 4 `if: false` jobs (lines 20, 62, 140, 250)

### Python Config & Tests

- `pyproject.toml` — Add `[tool.ruff]` section with ruleset and ignores
- `pytest.ini` — Add `smoke` marker (line 29), remove DeprecationWarning filter (line 34)
- `tox.ini` — Replace deleted paths (`src/`, `iris_rag/`, `mem0_integration/`) with `iris_vector_rag/`
- `tests/contract/test_smoke.py` — New; minimal IRIS ingest+query test

### Formatting

- 245 files need black reformatting (bulk pass in Phase 2)

---

## Implementation Approach

### Phase 1 — Test markers and contract test first

**Goal**: Define the smoke marker and write the test that CI will gate on.

1. Add `smoke` marker to `pytest.ini` (line 29, after `contract`):

   ```ini
   smoke: Smoke tests (minimal IRIS ingest+query, gates PR-integration)
   ```

2. Remove `ignore::DeprecationWarning` from `pytest.ini` `filterwarnings` (line 34) so library deprecation warnings are visible.

3. Create `tests/contract/test_smoke.py`:

   ```python
   @pytest.mark.smoke
   @pytest.mark.requires_database
   def test_basic_ingest_and_query():
       """Minimal smoke test: ingest 1 doc, query, verify result."""
       # Create pipeline with validated factory
       pipeline = create_validated_pipeline(
           pipeline_type="basic",
           auto_setup=True,
           validate_requirements=True
       )

       # Ingest 1 document
       doc = Document(
           page_content="IRIS vector database stores embeddings.",
           metadata={"source": "smoke_test"}
       )
       pipeline.load_documents([doc])

       # Query
       result = pipeline.query(
           "What does IRIS do?",
           top_k=1,
           generate_answer=False
       )

       # Assert non-empty retrieval
       assert len(result["retrieved_documents"]) > 0, "Smoke test: no docs retrieved"
       assert len(result["contexts"]) > 0, "Smoke test: no contexts"
   ```

   - Use `create_validated_pipeline` to ensure IRIS schema is set up.
   - Mark with both `@pytest.mark.smoke` and `@pytest.mark.requires_database`.
   - Keeps it minimal (~20s).

4. **Smoke test contract**: Verify locally:

   ```bash
   make docker-up
   make setup-db
   pytest tests/contract/test_smoke.py -m smoke -v
   ```

---

### Phase 2 — Bulk format pass

**Goal**: Reformat 245 files so the format gate doesn't immediately block CI.

1. Run black on all Python files:

   ```bash
   black iris_vector_rag/ tests/
   ```

2. Commit as one large, clearly labeled commit:

   ```text
   style: apply black formatting to all Python files

   245 files reformatted for consistency. No logic changes.
   This bulk pass precedes adding the format-check gate to CI.
   ```

3. Push to the feature branch.

---

### Phase 3 — Update ci.yml (PR-fast tier)

**Goal**: Add format and full ruff checks; keep build, test, lint all under 3 minutes.

1. Rename the existing `lint` job to `lint-ruff` and update its ruff call:

   ```yaml
   - name: Check ruff (E, F, W, I, UP, B)
     run: |
       ruff check iris_vector_rag/ tests/ \
         --select=E,F,W,I,UP,B \
         --ignore=E501,W503,F401,F811 \
         --output-format=github
   ```

   Rationale:
   - `E`: PEP 8 errors
   - `F`: PyFlakes (undefined names, unused imports)
   - `W`: PEP 8 warnings
   - `I`: isort import sorting
   - `UP`: pyupgrade
   - `B`: flake8-bugbear
   - Ignore `E501` (line length, handled by black), `W503` (black prefers it), `F401` (unused imports in `__init__.py`), `F811` (redefined names in test fixtures)

2. Add `format-check` step after setup-python, before install (so ruff config is ready):

   ```yaml
   - name: Check black formatting
     run: |
       pip install black
       black --check iris_vector_rag/ tests/
   ```

3. Update install to add `ruff` and `black` as dev deps:

   ```yaml
   - name: Install package
     run: pip install -e ".[dev]"
   ```

4. Reorder steps: checkout → setup-python → format-check → lint → install → test.

---

### Phase 4 — Create pr-integration.yml workflow (PR-integration tier)

**Goal**: New workflow for PRs to `main` that spins up IRIS and runs smoke tests.

Create `.github/workflows/pr-integration.yml`:

```yaml
name: PR Integration Tests

on:
  pull_request:
    branches: [main]
    types: [opened, synchronize, reopened]

env:
  PYTHON_DEFAULT_VERSION: "3.12"
  IRIS_PORT: 1972

jobs:
  smoke-tests:
    name: IRIS Smoke Tests
    runs-on: ubuntu-latest
    timeout-minutes: 10

    services:
      iris:
        image: intersystemsdc/iris-community:latest
        env:
          IRIS_USERNAME: SuperUser
          IRIS_PASSWORD: SYS
        options: >-
          --health-cmd="iris status /usr/irissys"
          --health-interval=10s
          --health-timeout=5s
          --health-retries=30
        ports:
          - 1972:1972

    steps:
      - uses: actions/checkout@v7

      - name: Set up Python ${{ env.PYTHON_DEFAULT_VERSION }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_DEFAULT_VERSION }}

      - name: Install package and test dependencies
        run: |
          pip install -e ".[dev,evaluation]" \
            intersystems-irispython>=5.1.2 \
            iris-vector-graph>=2.1.0

      - name: Wait for IRIS to be healthy
        run: |
          timeout 60 bash -c 'until nc -z localhost 1972; do sleep 1; done'

      - name: Initialize IRIS schema
        env:
          IRIS_HOST: localhost
          IRIS_PORT: 1972
          IRIS_USERNAME: SuperUser
          IRIS_PASSWORD: SYS
          IRIS_NAMESPACE: USER
        run: |
          python -m pytest tests/contract/test_smoke.py::test_basic_ingest_and_query \
            -m smoke \
            -v \
            --tb=short \
            -x

      - name: Run smoke tests
        env:
          IRIS_HOST: localhost
          IRIS_PORT: 1972
          IRIS_USERNAME: SuperUser
          IRIS_PASSWORD: SYS
          IRIS_NAMESPACE: USER
        run: |
          python -m pytest tests/contract/ \
            -m smoke \
            -v \
            --tb=short \
            -x
```

Key points:

- Service container on port 1972 with health check (status command).
- Waits for IRIS with netcat before proceeding.
- Sets `IRIS_*` env vars so connection manager finds it.
- Runs smoke tests with `-x` (exit on first failure) for fast feedback.
- Timeout 10 min total.

---

### Phase 5 — Configuration & cleanup

**Goal**: Fix stale CI config, add ruff rules to `pyproject.toml`, clean up `tox.ini`.

1. **Update `pyproject.toml`**: Add ruff config after `[tool.black]` section:

   ```toml
   [tool.ruff]
   select = ["E", "F", "W", "I", "UP", "B"]
   ignore = ["E501", "W503", "F401", "F811"]
   line-length = 88
   target-version = "py311"

   [tool.ruff.isort]
   known-first-party = ["iris_vector_rag", "adapters", "evaluation_framework"]
   ```

2. **Fix `tox.ini`**: Replace all instances of deleted paths:
   - Replace `src` → `iris_vector_rag`
   - Replace `iris_rag` → `iris_vector_rag`
   - Replace `mem0_integration` → (delete from lint/format targets)
   - In `[testenv]` and `[testenv:coverage]`: Replace cov paths.
   - In `[testenv:lint]`, `[testenv:format]`, `[testenv:type-check]`, `[testenv:security]`: Replace all path references.

   Example changes:

   ```ini
   # OLD
   commands =
       black --check --diff src iris_rag mem0_integration tests

   # NEW
   commands =
       black --check --diff iris_vector_rag tests
   ```

3. **Remove `if: false` jobs from `security.yml`**:
   - Delete `dependency-scan` job (lines 17–57).
   - Delete `code-security-scan` job (lines 59–111).
   - Delete `docker-security-scan` job (lines 137–184).
   - Delete `codeql-analysis` job (lines 247–278).
   - Keep `secret-scan`, `infrastructure-scan`, `compliance-check`, `security-scorecard`, `security-policy-check` (these have no `if: false`).

4. **Verify cleanup**:

   ```bash
   grep -r "if: false" .github/workflows/ # Should return 0 results
   tox --listenvs  # Should not error on deleted paths
   ```

---

## Risks & Constraints

### Black Reformatting Impact

- **245 files** will be reformatted in Phase 2 — a large diff but single-commit.
- Risk: Difficult to review the commit in GitHub. **Mitigation**: Label clearly as style-only, no review needed.
- Risk: Merge conflicts if other branches are active. **Mitigation**: Land Phase 2 commit before other work branches merge to main.

### Ruff Full Ruleset

- Enabling `E, F, W, I, UP, B` may surface violations in existing code not currently caught by `E9,F63,F7,F82`.
- Risk: New ruff hits block CI before the smoke test even runs. **Mitigation**: Phase 2 (black) should resolve most formatting; Phase 3 (ruff config) documents the known ignores (`E501`, `W503`, `F401`, `F811`). If new violations surface, add them to the ignore list with a comment explaining why.

### IRIS Service Container Timing

- PR-integration workflow adds ~5 min to PRs targeting `main`. Not all PRs will need it, but GHA doesn't support conditional job triggers on code diff yet.
- Risk: Slow feedback loop for non-core changes. **Mitigation**: Keep smoke test minimal; document in CONTRIBUTING.md that PR-fast tier is the main gate, PR-integration is an extra catch for database bugs.

### Deprecation Warnings

- Removing `ignore::DeprecationWarning` from `pytest.ini` will expose library warnings (e.g., from dependencies). They won't block the tests but will clutter output.
- Risk: Noisy logs. **Mitigation**: This is intentional; library issues should be visible. Filter by module if needed (e.g., ignore warnings only from `iris_vector_rag`, not dependencies) once we're further in the project.

---

## Dependencies

1. **No new external dependencies** — all tools already in `[tool.dev]`: `black`, `ruff`, `pytest`.
2. **Docker image**: `intersystemsdc/iris-community:latest` (publicly available).
3. **Feature order**:
   - Phase 1 must complete before Phase 2 (test must exist).
   - Phase 2 must complete before Phase 3 (format pass before gate).
   - Phase 3 and 4 are independent (can be in parallel workflows).
   - Phase 5 is cleanup; can happen in parallel with 3–4.

---

## Success Criteria

| Criterion | Check                                                                                   |
| --------- | --------------------------------------------------------------------------------------- |
| SC-001    | Black formatting passes locally: `black --check iris_vector_rag/ tests/` → exit 0       |
| SC-002    | PR-fast completes in <3 min (GHA workflow duration for build+format+lint+test)          |
| SC-003    | Unused import blocks PR (introduce unused import, ruff catches it)                      |
| SC-004    | IRIS ingest+query verified (PR-integration smoke test passes)                           |
| SC-005    | tox config is clean (`tox --listenvs` succeeds; `grep "if: false" .github` → 0 results) |
