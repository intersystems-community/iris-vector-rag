# InterSystems IRIS: Postmortem on Vector Search, SQL Syntax, Compilation, Driver, and Stored Procedure Interactions

## Preamble: An "Innocuous" Project Meets an "OMFG" Reality

The goal was straightforward: develop a suite of RAG (Retrieval Augmented Generation) templates in Python, leveraging InterSystems IRIS for its vector search capabilities. This was envisioned as a convenient, portable `docker-compose` project, complete with benchmarks to evaluate different RAG techniques working in IRIS. On paper, a standard, modern development task.

In practice, this "innocuous" project descended into an extraordinarily challenging debugging saga. Early phases (with IRIS 2024.1.2) were dominated by fundamental, systemic issues encountered with InterSystems IRIS's behavior related to automated ObjectScript class compilation, the reliability of SQL procedure projection, and query/catalog caching in a Dockerized environment. Later phases (with IRIS 2025.1), focused on direct vector search implementation, encountered a different set of challenges related to specific SQL vector syntax, driver behaviors, and documentation interpretation. This document details that journey, the failure modes encountered, their root causes (where understood), and the role of documentation.

## Executive Summary

InterSystems IRIS, while a powerful platform, presented significant developer experience hurdles during this RAG template project.
Initial efforts with IRIS 2024.1.2 revealed critical flaws in mechanisms for projecting ObjectScript class methods as SQL stored procedures, query caching, and SQL catalog management, especially in automated, Dockerized environments. This led to "zombie" procedures and unreliable compilation.

Subsequent work with IRIS 2025.1, focusing on direct SQL vector search, highlighted nuances in `TO_VECTOR()` syntax, HNSW index requirements, Python driver limitations for the `VECTOR` type, and undocumented behaviors of vector functions within Stored Procedures.

Key findings indicate:
*   **Initial SP/Compilation (IRIS 2024.1.2):**
    *   SQL catalog entries for projected SPs did not reliably update.
    *   Cached queries/routines persisted, executing obsolete logic.
    *   Automated class compilation via `docker exec` was unreliable, with spurious compiler errors and inaccessible system classes (e.g., `%SYS.SQL.Schema`).
*   **Vector Syntax & Usage (IRIS 2025.1):**
    *   **`TO_VECTOR()` DML Parameter Binding:** Clear documentation for correct syntax (e.g., `TO_VECTOR(?, FLOAT)`) *exists* on the main "Using Vector Search" (`GSQL_vecsearch`) page. Initial challenges were likely due to information discoverability and precise interpretation of the `type` argument (unquoted keyword vs. string/parameter).
    *   **HNSW Index Requirements:** The same `GSQL_vecsearch` page clearly specifies that HNSW indexes require native `VECTOR` or `EMBEDDING` type columns. Assumptions that `TO_VECTOR()` could adapt `VARCHAR` columns for HNSW base indexing were contrary to this documented prerequisite.
    *   **Python Driver `VECTOR` Type Support:** The Python DB-API documentation (`BPYNAT_pyapi`) is silent on the `VECTOR` data type, implying no special native handling. This aligns with observed behavior (vectors returned as strings, requiring `TO_VECTOR()` for inserts), indicating a driver limitation or a documentation gap for this type.
    *   **Vector Functions in SPs/UDFs:** Neither the general SQL Stored Procedure documentation (`GSQL_procedures`) nor the `GSQL_vecsearch` page provide specific guidance on using vector functions within SPs/UDFs, their limitations (e.g., with dynamic parameters), or best practices. This represents a documentation gap.
*   **General:** Error messages were often misleading. Full Docker volume wipes were sometimes the only resort for a clean state during early SP work.

These issues, spanning different IRIS versions and feature sets, created a challenging development experience. This report provides a timeline, analyzes root causes in light of available documentation, and underscores the importance of clear, easily discoverable, and comprehensive documentation for all usage contexts.

**IRIS Version Context:** The initial SP projection, compilation, and caching issues were primarily with `IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2024.1.2 (Build 398U)`. The subsequent vector syntax and driver analysis was conducted with IRIS 2025.1.0.225.1.

## Technical Environment Information (for IRIS 2025.1 phase)

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2025.1.0.225.1 |
| Python Version | 3.12.x |
| Client Libraries | intersystems-iris, pyodbc, sqlalchemy |
| Operating System | macOS (arm64) |

For detailed technical information about specific vector search limitations and syntax findings, see:
*   [`bug_reproductions/bug1_parameter_binding.py`](bug_reproductions/bug1_parameter_binding.py)
*   [`bug_reproductions/bug2_hnsw_varchar.py`](bug_reproductions/bug2_hnsw_varchar.py)
*   [`bug_reproductions/bug3_vector_driver_support.py`](bug_reproductions/bug3_vector_driver_support.py)
*   [`bug_reproductions/bug4_stored_procedures.py`](bug_reproductions/bug4_stored_procedures.py)

## Detailed Timeline of Failures & Attempts (Primarily with IRIS 2024.1.2 SPs)

The core task involved calling an SQL stored procedure, `RAG.VSU_SearchDocsV3` (projected from `RAG.VectorSearchUtils.SearchSourceDocuments`), from Python using `pyodbc`.

**1. Initial ODBC Failure & Query Complexity Hypothesis**
*   **Symptom:** ODBC calls to `RAG.VSU_SearchDocsV3` failed with `[SQLCODE: <-460>:<General error>]`.
*   **Finding:** The `-460` error persisted even with the simplest query, indicating a more fundamental issue than query complexity.

**2. Automated Class Compilation & SQL Projection Challenges (IRIS 2024.1.2)**
*   **Attempt 2.1: Custom SQL SP for File-Based Load (`RAG.CompileClassFromFile`)**
    *   **Result:** `[SQLCODE: <-400>:<Fatal error occurred>] [Error: <<ENDOFFILE>LoadUDL+17^%apiOBJ>]`. Likely IRIS process permissions issue in Docker.
*   **Attempt 2.2: Custom SQL SP for Stream-Based Load (`RAG.LoadClassFromStream`)**
    *   **Result:** `[SQLCODE: <-400>:<Fatal error occurred>] [Location: <SPFunction>]`. Possible crash in `$System.OBJ.LoadStream`.
*   **Attempt 2.3: Direct DBAPI Execution of System Utilities**
    *   **Result:** Various SQLCODE errors (`<-51>`, `<-12>`, `<-359>`, `<-428>`), proving unreliable.
*   **Attempt 2.4: Automated Compilation via `docker exec` (Python `subprocess`)**
    *   **Persistent Compiler Errors:** `ERROR #1038: Private variable not allowed : 'varname'` for valid `New stmt As %SQL.Statement`. `ERROR #5559: The class definition ... could not be parsed correctly...`.
    *   **Eventual Success (ultra-minimal class):** A class with one parameterless method `Quit "Hardcoded String"` eventually compiled.
    *   **Analysis:** The `docker exec` mechanism worked, but the compiler showed extreme sensitivity or bugs with more complex, yet valid, classes in this context.

**3. SQL Projection Visibility & Catalog/Cache Issues (IRIS 2024.1.2)**
*   **Symptom:** Even after reported successful compilation, projected SQL procedures were often NOT found in `INFORMATION_SCHEMA.ROUTINES`.
*   **Attempts to Fix Visibility:**
    *   `Do $SYSTEM.SQL.Purge(0)`: No effect.
    *   `Do ##class(%SYS.SQL.Schema).Refresh("RAG",1)`: Failed with `<CLASS DOES NOT EXIST> *%SYS.SQL.Schema`. This is a critical failure.
*   **ODBC Call Behavior (compiled `MinimalTest.cls`):** Scalar `Quit <value>` methods returned empty strings via ODBC.
*   **Analysis:** Inability to access `%SYS.SQL.Schema` likely prevented catalog refresh. Data marshalling issues for scalar returns via SQL projection/ODBC.

## Analysis of Vector Search Syntax & Documentation (IRIS 2025.1)

The following summarizes findings from specific bug reproduction scripts and review of IRIS 2025.1 documentation:

*   **`TO_VECTOR()` Parameter Binding (Bug #1):**
    *   The "Using Vector Search" documentation (`GSQL_vecsearch`) clearly shows `TO_VECTOR(?, FLOAT)` for DML. The initial project challenge was likely in locating this specific guidance or misinterpreting the `type` argument (which must be an unquoted keyword).
*   **HNSW Index on `VARCHAR` (Bug #2):**
    *   The `GSQL_vecsearch` documentation explicitly states HNSW indexes require a `VECTOR` or `EMBEDDING` typed field. The assumption that `TO_VECTOR()` in `CREATE INDEX` could adapt a `VARCHAR` column was contrary to this.
*   **Python Driver `VECTOR` Type Support (Bug #3):**
    *   The Python DB-API documentation (`BPYNAT_pyapi`) does *not* mention the `VECTOR` data type or its specific handling. This silence implies a lack of native support, consistent with observations that vectors are fetched as strings and require `TO_VECTOR()` for inserts via string parameters. This is a documentation gap for `VECTOR` type specifics in the Python driver context or reflects a driver feature limitation.
*   **Vector Functions in SPs/UDFs (Bug #4):**
    *   Neither the general SQL Stored Procedure documentation (`GSQL_procedures`) nor the `GSQL_vecsearch` page detail the use, limitations, or best practices for vector functions (`TO_VECTOR`, `VECTOR_COSINE`) *within* SPs/UDFs, especially concerning dynamic parameters. This is a documentation gap.

## Root Causes (Consolidated View)

1.  **SQL Projection & Caching (IRIS 2024.1.2):** Systemic issues with catalog updates and cache persistence for SPs in automated Docker environments. Inability to access `%SYS.SQL.Schema` was a critical symptom.
2.  **Compiler Behavior (IRIS 2024.1.2):** Spurious compiler errors (e.g., #1038) for valid ObjectScript syntax when loaded via `docker exec iris session`.
3.  **Documentation Discoverability & Specificity (IRIS 2025.1):**
    *   While core DML `TO_VECTOR` syntax and HNSW prerequisites were documented, finding and precisely applying this information proved challenging initially.
    *   Specific documentation for Python driver handling of `VECTOR` types, and for using vector functions within SPs/UDFs (including limitations), was found to be lacking in the consulted pages.
4.  **Client Driver Limitations (Python):** Lack of native `VECTOR` type support in the `intersystems-iris` Python driver.
5.  **Misleading Error Messages:** Generic or incorrect errors often masked the true nature of problems across versions and components.

## Conclusion: A Call for Enhanced Developer Experience and Comprehensive Documentation

The journey documented highlights significant friction points. Early issues with IRIS 2024.1.2's SP projection and compilation in Docker were severe. Later investigations with IRIS 2025.1 revealed that while some core vector search DML and indexing information was documented, challenges arose from discoverability, interpretation for specific nuances, and clear gaps in documentation for client driver specifics (`VECTOR` type) and advanced SQL contexts (vector functions in SPs/UDFs).

For InterSystems IRIS to be a leading platform for AI/ML applications, addressing both foundational platform stability (compilation, caching, projection) and ensuring comprehensive, easily discoverable documentation for all common use-cases of its advanced features (like vector search across different client languages and procedural SQL) is crucial. This includes explicitly stating limitations or recommended workarounds where native features are still evolving.
