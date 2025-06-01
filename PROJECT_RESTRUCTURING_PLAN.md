# Project Status Restructuring Plan

**Goal:** To create a transparent, accurate, and easily maintainable system for tracking the true operational status of all RAG techniques and project components. This plan aims to eliminate the "documentation-reality gap" and provide clear visibility into what is working, what is broken, and what is in progress.

## I. New Documentation Structure & Central Dashboard

1.  **Create a Central Status Dashboard:**
    *   **File:** `PROJECT_STATUS_DASHBOARD.md` (or similar, e.g., `STATUS_INDEX.md`) at the root of the project.
    *   **Purpose:** This will be the **single source of truth** for the at-a-glance status of all RAG techniques.
    *   **Content:**
        *   **Overall Project Health:** A brief summary statement (e.g., "X of Y techniques fully operational. Z techniques with known issues.").
        *   **RAG Technique Status Table:**
            *   **Columns:**
                *   `Technique Name` (e.g., BasicRAG, ColBERT)
                *   `Current Status` (Enum: `WORKING`, `BROKEN`, `UNTESTED`, `IN_PROGRESS`, `DEPRECATED`) - Visually distinct (e.g., using emojis ✅, ❌, ❓, 🚧, 🗑️).
                *   `Last Successful Test Date` (YYYY-MM-DD or "N/A")
                *   `Last Test Result Link` (Link to the specific test report/log file, e.g., a JSON or MD report from `test_results/` or `benchmark_results/`)
                *   `Known Issues Summary` (Brief description of major blockers or bugs, e.g., "Vector length mismatch", "0 documents retrieved")
                *   `Link to Detailed Status/Fix Log` (Link to a dedicated status file for that component, see point 2)
                *   `Primary Implementation File(s)` (Link to the core pipeline code, e.g., `basic_rag/pipeline_final.py`)
                *   `Relevant Documentation` (Link to specific implementation docs, e.g., `docs/COLBERT_IMPLEMENTATION.md`)
        *   **Data Source:** This dashboard will be manually updated initially, with a long-term goal of automation (see Section IV).
        *   **Update Frequency:** Must be updated *immediately* after any significant test run or status change.
    *   **Mermaid Diagram for Status Flow (Optional but Recommended):**
        ```mermaid
        graph TD
            A[Test Execution] --> B{Test Passed?};
            B -- Yes --> C[Update Dashboard: WORKING];
            C --> D[Link to Test Report];
            B -- No --> E[Update Dashboard: BROKEN];
            E --> F[Link to Test Report/Error Log];
            F --> G[Create/Update Issue Ticket];
            H[Development/Fix] --> A;
            G --> H;
        ```

2.  **Standardized Component Status Files:**
    *   **Naming Convention:** `COMPONENT_STATUS_ComponentShortName.md` (e.g., `COMPONENT_STATUS_ColBERT.md`).
    *   **Location:** A new top-level directory, e.g., `project_status_logs/`.
    *   **Purpose:** To provide a detailed, chronological log of status changes, issues, fixes, and relevant test results for *each individual RAG technique or major component*.
    *   **Content Template (for each file):**
        *   `Component Name:`
        *   `Current Overall Status:` (WORKING/BROKEN/UNTESTED/IN_PROGRESS/DEPRECATED)
        *   `Last Checked:` (YYYY-MM-DD HH:MM UTC)
        *   `Status History Log:`
            *   `YYYY-MM-DD HH:MM UTC - Status: [STATUS] - Notes: [Details of test, issue found, fix applied] - Evidence: [Link to test report, commit, PR, issue ticket]`
            *   *(Example Entry)* `2025-06-01 09:00 UTC - Status: WORKING - Notes: ColBERT E2E test passed with 100 docs after query encoder fix. - Evidence: [Link to COLBERT_PIPELINE_FIX_SUMMARY.md], [Link to test_results/colbert_e2e_20250601_0900.json]`
            *   *(Example Entry)* `2025-05-31 12:40 UTC - Status: BROKEN - Notes: BasicRAG failing all queries due to vector length mismatch. - Evidence: [Link to ragas_basic_rag_multi_query_test_20250531_124028.json]`
    *   **Linkage:** The Central Status Dashboard will link to these detailed logs.

3.  **Deprecate/Consolidate Old Status Documents:**
    *   Review all files in `docs/` and the root directory that report status (e.g., `docs/PROJECT_STATUS.md`, `PLAN_STATUS.md`, various "SUMMARY" or "REPORT" files that are outdated).
    *   Extract any still-relevant *historical* information into the new `COMPONENT_STATUS_ComponentShortName.md` files or an archive.
    *   Clearly mark outdated files as `DEPRECATED_*.md` or move them to an `archive/docs_archive/` directory.
    *   Update the main `README.md` and `docs/README.md` to point *exclusively* to `PROJECT_STATUS_DASHBOARD.md` for current status. Remove all conflicting "100% success" claims from these general READMEs.

## II. Project File Reorganization

1.  **Goal:** To clearly separate components based on their development lifecycle stage.
2.  **Proposed Directory Structure Changes:**
    *   `src/` (or `pipelines/` or `rag_techniques/`) - New or renamed top-level directory.
        *   `src/working/`: Contains code for RAG techniques and components that are **verified WORKING** according to the `PROJECT_STATUS_DASHBOARD.md`.
            *   `src/working/basic_rag/`
            *   `src/working/colbert/` (once scaled tests pass)
            *   ...
        *   `src/experimental/` (or `src/in_progress/`): Contains code for RAG techniques that are `IN_PROGRESS`, `BROKEN` (actively being debugged), or `UNTESTED`.
            *   `src/experimental/basic_rag/` (current location until fixed)
            *   `src/experimental/noderag/`
            *   ...
        *   `src/deprecated/`: Contains code for components that are no longer maintained or used.
            *   This can be an alternative or supplement to `archived_pipelines/`. The key is clear labeling.
    *   `tests/`
        *   Align test structure with `src/`. Tests for `src/working/foo` should be clearly identifiable and expected to pass. Tests for `src/experimental/bar` might be expected to fail or be incomplete.
    *   `test_results/` and `benchmark_results/`:
        *   Maintain these for storing raw output.
        *   Implement a clear naming convention for output files that includes date, component, and status (e.g., `BasicRAG_RAGAS_FAIL_20250531_124028.json`, `ColBERT_E2E_PASS_20250601_0900.json`).
    *   `docs/`:
        *   Focus on *technical implementation details*, *design documents*, and *user guides* rather than volatile status.
        *   `docs/archive/` for outdated conceptual documents.
    *   `project_status_logs/`: New directory for the `COMPONENT_STATUS_*.md` files.

3.  **Process for Moving Files:**
    *   Start by creating the new directory structure.
    *   Based on the *actual, verified current status* (derived from recent test results like `ragas_basic_rag_multi_query_test_20250531_124028.json` and `COLBERT_PIPELINE_FIX_SUMMARY.md`), move component code into the appropriate `working/`, `experimental/`, or `deprecated/` subdirectories.
    *   Update import paths in scripts and tests accordingly. This will be a significant refactoring step.

## III. Recommendations for Automated Status Tracking

1.  **Test Result Parsing:**
    *   Develop scripts (Python, using libraries like `json` and `junitparser`) that can parse the output of test runs (e.g., JSON reports from RAGAS, pytest JUnitXML output).
    *   These scripts should extract:
        *   Technique name
        *   Test date
        *   Overall status (Pass/Fail)
        *   Key metrics (e.g., success rate, number of documents retrieved, error messages if any).

2.  **Dashboard Auto-Update (Ambitious, Long-Term):**
    *   The parsing scripts could potentially update a data file (e.g., a central JSON or YAML file).
    *   Another script could then read this data file and regenerate parts of `PROJECT_STATUS_DASHBOARD.md`. This is complex due to Markdown manipulation but achievable with tools or careful string templating.
    *   **Simpler First Step:** The parsing script could generate a *text-based report* or a *snippet of Markdown* that a human then copies into the main dashboard.

3.  **CI/CD Integration:**
    *   Integrate test execution into a CI/CD pipeline (e.g., GitHub Actions, Jenkins).
    *   After each run (e.g., on every commit to `main` or on a nightly schedule), the pipeline would:
        *   Execute all relevant tests.
        *   Run the parsing scripts.
        *   Archive test results (making them linkable).
        *   (If auto-update is implemented) Update the status dashboard or the data file it uses.
        *   (If manual update) Post a summary to a chat channel (e.g., Slack) or create a "pending status update" task, prompting a team member to update the dashboard.

4.  **Pre-commit Hooks:**
    *   Consider pre-commit hooks that remind developers to update status logs if they are working on files within `src/experimental/` or if tests related to a component fail.

## IV. Action Plan & Timeline (High-Level)

1.  **Phase 1: Immediate Stabilization & Transparency (1-2 weeks)**
    *   **Task 1.1:** Create the `PROJECT_STATUS_DASHBOARD.md` file at the root. Manually populate it based on the *actual* findings from May 31st/June 1st (BasicRAG broken, ColBERT working, others likely broken or unverified).
    *   **Task 1.2:** Create the `project_status_logs/` directory. Create initial `COMPONENT_STATUS_ComponentShortName.md` files for each RAG technique, backfilling with information from the recent JSON reports and fix summaries.
    *   **Task 1.3:** Update the main `README.md` and `docs/README.md` to remove false "100% success" claims and point to the new dashboard as the single source of truth for status. Add a prominent disclaimer about past inaccuracies.
    *   **Task 1.4:** Identify and mark clearly outdated status documents in `docs/` for archival.

2.  **Phase 2: Project Structure Refactoring (2-4 weeks, depends on complexity)**
    *   **Task 2.1:** Define and create the new `src/working/`, `src/experimental/`, `src/deprecated/` directory structure.
    *   **Task 2.2:** Carefully move existing RAG pipeline code into the appropriate new directories based on their verified status from Task 1.1.
    *   **Task 2.3:** Update all import statements across the codebase (pipelines, tests, scripts) to reflect the new locations. This will require careful testing.
    *   **Task 2.4:** Reorganize the `tests/` directory to mirror the new `src/` structure if deemed beneficial.

3.  **Phase 3: Automation Foundation (Ongoing, start after Phase 1)**
    *   **Task 3.1:** Develop initial scripts to parse key test result JSON files (e.g., RAGAS outputs, benchmark summaries) to extract status and metrics.
    *   **Task 3.2:** Establish a clear, consistent naming convention and output format for all test and benchmark result files to facilitate parsing.

4.  **Phase 4: CI/CD Integration & Full Automation (Long-term, ongoing)**
    *   **Task 4.1:** Integrate automated test runs into a CI/CD pipeline.
    *   **Task 4.2:** Enhance parsing scripts and explore methods for automatically updating (or generating snippets for) the `PROJECT_STATUS_DASHBOARD.md`.