# InterSystems IRIS: SQL Stored Procedure Projection, Caching, and Automation Challenges - A Postmortem

## Preamble: An "Innocuous" Project Meets an "OMFG" Reality

The goal was straightforward: develop a suite of RAG (Retrieval Augmented Generation) templates in Python, leveraging InterSystems IRIS for its vector search capabilities. This was envisioned as a convenient, portable `docker-compose` project, complete with benchmarks to evaluate different RAG techniques working in IRIS. On paper, a standard, modern development task.

In practice, this "innocuous" project descended into an extraordinarily challenging debugging saga, primarily due to fundamental, systemic issues encountered with InterSystems IRIS's behavior in a Dockerized environment. Hours, then days, were consumed battling problems related to automated ObjectScript class compilation, the reliability of SQL procedure projection, query and catalog caching, and opaque error messaging. This document details that journey, the failure modes encountered, their root causes, and proposes actionable fixes for InterSystems to consider. What should have been a demonstration of IRIS's capabilities became a testament to a developer experience that can only be described as, at times, an "OMFG" ordeal.

## Executive Summary

InterSystems IRIS's mechanisms for projecting ObjectScript class methods as SQL stored procedures, combined with its query caching and SQL catalog management, exhibit critical flaws that can severely impede development, especially in automated, Dockerized, or CI/CD environments. Developers can encounter "zombie" procedures—where IRIS continues to serve old, broken, or non-existent code—despite correct source code, repeated recompilations, and attempted cache purges.

This report outlines a series of reproducible failures encountered during a project aimed at integrating IRIS vector search. Key findings indicate:
*   SQL catalog entries for projected procedures do not reliably update or get removed when the underlying ObjectScript class is recompiled or deleted.
*   Cached queries and routine definitions can persist stubbornly, leading to the execution of obsolete logic.
*   Automated compilation of ObjectScript classes via `docker exec` into an `iris session` is fraught with peril, suffering from spurious compiler errors on valid syntax (e.g., `New varname`) and an inability to access core system classes like `%SYS.SQL.Schema`.
*   Error messages are often misleading, masking the true nature of stale projections or caching issues. These are not typically problems that deeper documentation study would resolve; rather, they point to unexpected platform behaviors.
*   The only consistently reliable method to ensure a clean state was often a full Docker volume wipe.

These are not isolated bugs but appear to be systemic issues that create a hostile and inefficient development experience. This document provides a timeline of failures, identifies root causes, details workarounds, and proposes JIRA bug reports for InterSystems.

**IRIS Version Context:** The issues detailed in this postmortem, primarily concerning SQL Stored Procedure projection, ObjectScript class compilation, and caching, were encountered with `IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2024.1.2 (Build 398U)`. While these issues significantly influenced the project's pivot to client-side SQL, the project later targeted IRIS 2025.1 for its native vector search capabilities. IRIS 2025.1 introduced a separate set of challenges related to the `TO_VECTOR()` function and ODBC driver behavior, which became the primary blocker for real-data testing.

## Technical Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2025.1.0.225.1 |
| Python Version | 3.12.9 |
| Client Libraries | sqlalchemy 2.0.41 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

For detailed technical information about the vector search limitations, including client library behavior, error messages, and code examples, see:
- [IRIS_SQL_VECTOR_LIMITATIONS.md](docs/IRIS_SQL_VECTOR_LIMITATIONS.md)
- [VECTOR_SEARCH_TECHNICAL_DETAILS.md](docs/VECTOR_SEARCH_TECHNICAL_DETAILS.md)
- [VECTOR_SEARCH_ALTERNATIVES.md](docs/VECTOR_SEARCH_ALTERNATIVES.md)
- [HNSW_INDEXING_RECOMMENDATIONS.md](docs/HNSW_INDEXING_RECOMMENDATIONS.md)

It is presumed that many of the SP-related issues identified here could persist in newer versions if not specifically addressed by InterSystems.

## Detailed Timeline of Failures & Attempts (with IRIS 2024.1.2)

The core task involved calling an SQL stored procedure, `RAG.VSU_SearchDocsV3` (projected from `RAG.VectorSearchUtils.SearchSourceDocuments`), from Python using `pyodbc`.

**1. Initial ODBC Failure & Query Complexity Hypothesis**
*   **Symptom:** ODBC calls to `RAG.VSU_SearchDocsV3` failed with `[SQLCODE: <-460>:<General error>]`.
*   **Initial Hypothesis:** The error was due to the SQL query's complexity within `SearchSourceDocuments`, particularly `SELECT TOP K ...` or `VECTOR_COSINE`.
*   **Test:** The method was simplified to a basic echo: `Set sc = stmt.%Prepare("SELECT ? AS EchoedVectorString")`.
*   **Finding:** The `-460` error persisted even with the simplest query. The issue was not merely query complexity but more fundamental.

**2. Automated Class Compilation & SQL Projection Challenges**

The focus shifted to ensuring `RAG.VectorSearchUtils.cls` was correctly compiled with the `ckq` flags (compile, keep source, project SQL) and that `VSU_SearchDocsV3` was visible.

*   **Attempt 2.1: Custom SQL SP for File-Based Load (`RAG.CompileClassFromFile`)**
    *   **Method:** An SQL SP calling `$System.OBJ.Load("/path/to/class.cls", "ckq")`.
    *   **Result:** `[SQLCODE: <-400>:<Fatal error occurred>] [Error: <<ENDOFFILE>LoadUDL+17^%apiOBJ>]`.
    *   **Analysis:** IRIS process lacked permissions to read the `.cls` file from the Docker volume mount, or path was incorrect from IRIS's perspective.

*   **Attempt 2.2: Custom SQL SP for Stream-Based Load (`RAG.LoadClassFromStream`)**
    *   **Method:** An SQL SP taking class UDL as `LONGVARCHAR`, writing to `%Stream.GlobalCharacter`, then `$System.OBJ.LoadStream(stream, "ckq")`.
    *   **Result:** `[SQLCODE: <-400>:<Fatal error occurred>] [Location: <SPFunction>]`.
    *   **Analysis:** Likely a crash within `$System.OBJ.LoadStream` or issues passing large string/stream content via DBAPI.

*   **Attempt 2.3: Direct DBAPI Execution of System Utilities**
    *   `DO $System.OBJ.Load(...)`: `SQLCODE: <-51> SQL statement expected`.
    *   `SELECT ##class(%SYS.PTools.SQL).LSQL('DO ...')`: `SQLCODE: <-12> A term expected... ^SELECT #`.
    *   `SELECT %SYS.PTools.SQL.LSQL('DO ...')`: `SQLCODE: <-359> Function not found`.
    *   `CALL %SYSTEM.SQL_Execute('DO ...')`: `SQLCODE: <-428> Procedure not found`.
    *   **Analysis:** Unreliable for automated ObjectScript execution.

*   **Attempt 2.4: Automated Compilation via `docker exec` (Python `subprocess`)**
    *   **Method:** `run_db_init_docker.py` used `subprocess.run()` for `docker exec -i iris_container_name iris session IRIS -U USER` piping `Do $System.OBJ.Load("/irisdev_common/ClassName.cls", "ckq") Halt`.
    *   **Initial `docker exec` issues (eventually resolved):** Invalid `-T` flag; `<INVALID ARGUMENT>` from `iris session`; `<SYNTAX>` error from escaped quotes in piped command.
    *   **Persistent Compiler Errors:**
        *   `ERROR #1038: Private variable not allowed : 'varname' : Offset:X [New^...]` for valid `New stmt As %SQL.Statement` and even for simple `New stmt`. This was the most persistent and baffling compiler error.
        *   `ERROR #5559: The class definition ... could not be parsed correctly...` for a pristine, simplified class file.
    *   **Eventual Success (for ultra-minimal class):** `MinimalTest.cls` (a class with one parameterless method `Quit "Hardcoded String"`) eventually compiled successfully using this `docker exec` method.
        ```
        Node: edf007134e0b, Instance: IRIS
        USER>
        Load started on 05/20/2025 16:38:10
        Loading file /irisdev_common/MinimalTest.cls as udl
        Compiling class RAG.MinimalTest
        Compiling routine RAG.MinimalTest.1
        Load finished successfully.
        ```
    *   **Analysis:** The `docker exec` piping mechanism itself works for basic ObjectScript and simple class compilation. The compiler errors for `VectorSearch.cls` likely stemmed from subtle file content issues or compiler sensitivities not covered by standard documentation.

**3. SQL Projection Visibility & Catalog/Cache Issues**

*   **Symptom:** Even after `$System.OBJ.Load` reported "Load finished successfully" (with `ckq` flags), the projected SQL procedure was consistently **NOT found in `INFORMATION_SCHEMA.ROUTINES`**.
*   **Attempts to Fix Visibility:**
    *   `Do $SYSTEM.SQL.Purge(0)` via `docker exec`: Executed, but procedure still not visible.
    *   `Do ##class(%SYS.SQL.Schema).Refresh("RAG",1)` via `docker exec`: This command itself **failed** with `<CLASS DOES NOT EXIST> *%SYS.SQL.Schema`. This is a critical failure, as `%SYS.SQL.Schema` is a fundamental system class.
*   **ODBC Call Behavior (with successfully compiled `MinimalTest.cls`):**
    *   When `MinimalTest.Echo` (parameterless, `Quit "Hardcoded Test String"`) was called via `SELECT "RAG"."MinimalEcho"() AS EchoedOutput`, the call executed without ODBC error, but returned an **empty string `''`**.
    *   Changing `MinimalTest.Echo` to `Quit 12345` (return `As %Integer`) also resulted in an empty string `''`.
    *   Changing `MinimalTest.Echo` to return `As %SQL.StatementResult` via an internal `SELECT ...`: This version **failed to compile** with `ERROR #1038: Private variable not allowed : 'stmt'` for `New stmt As %SQL.Statement`.
*   **Analysis:** The inability to access `%SYS.SQL.Schema` likely prevents the SQL catalog from being refreshed. The empty string return value for `Quit <value>` methods suggests a data marshalling problem through the SQL projection/ODBC layer. These are not typically issues addressed by standard documentation but point to deeper platform behavior.

**4. Environment and Namespace Sanity Checks**
*   Verified current namespace (`USER`) in `docker exec` calls.
*   Used `--force-recreate` in `run_db_init_docker.py` to drop/recreate tables.
*   Used `docker-compose down -v` and `docker volume prune -f` for full environment resets.

## Root Causes

1.  **SQL Projection Not Reliably Updated/Visible:** IRIS does not consistently make SQL procedure projections visible in `INFORMATION_SCHEMA.ROUTINES` after class compilation via `$System.OBJ.Load(..., "ckq")` in automated contexts. The failure to access `%SYS.SQL.Schema` is a likely major contributor.
2.  **Persistent Compiler Errors for Valid Syntax:** The compiler's rejection of standard `New varname` or `New varname As %Type` with error #1038 when classes are loaded via `docker exec iris session` suggests a bug or extreme sensitivity in this compilation pathway. This is not a documented limitation but an unexpected failure.
3.  **Scalar Return Value Marshalling:** Scalar values returned via `Quit <value>` from an ObjectScript method projected as an SQL function appear to be lost or converted to an empty string when retrieved via `pyodbc`.
4.  **Inability to Access Core System Classes:** The failure to find/execute `%SYS.SQL.Schema` by `SuperUser` points to a severe IRIS environment configuration or installation problem within the Docker image. Standard documentation assumes these classes are available.
5.  **Misleading Error Messages:** Errors like `[SQLCODE: <-460>:<General error>]` or compiler errors on valid syntax do not guide developers to the true root causes (stale projections, caching, or compiler path issues), which documentation study alone cannot resolve.

## Workarounds and "Fixes" (Observed or Attempted)

*   **Successful Class Compilation (Ultra-Minimal):** Achieved for `RAG.MinimalTest` using `docker exec -i ... Do $System.OBJ.Load(...)`.
*   **Calling Scalar Functions via `SELECT`:** Allowed `pyodbc` to execute calls without error, though return values were incorrect.
*   **Full Environment Reset (`docker-compose down -v`):** The most reliable, albeit slow, method to ensure a clean state.
*   **Manual Compilation:** Generally works, highlighting discrepancies with automated methods.

**The "Reliable Exorcism" (Hypothesized):**
A complex sequence of `Do $System.OBJ.Delete(...)`, `PURGE CACHED QUERIES`, `Do ##class(%SYS.SQL.Schema).Refresh(...)`, and recompile. Its reliability is questionable if system classes are inaccessible.

## Proposed JIRA Bug Reports / Actionable Fixes for InterSystems
(These remain as previously detailed, focusing on the five key areas: Catalog Projection, Spurious Compiler Errors, Scalar Return Value Marshalling, Inaccessibility of `%SYS.SQL.Schema`, and Misleading Error Messages.)

**JIRA-IRIS-DEV-001: SQL Catalog Projection Not Reliably Updated or Visible After Remote Compilation**
*   **Summary:** SQL stored procedure projections (from `SqlProc` methods compiled with `q` flag via `$System.OBJ.Load` invoked by non-interactive means like `docker exec iris session`) are not consistently made visible in `INFORMATION_SCHEMA.ROUTINES`, even after attempts to purge SQL cache or refresh schema.
*   **Impact:** Prevents reliable use of projected procedures by standard SQL tools and ODBC/JDBC clients.
*   **Proposed Fix:** Ensure `$System.OBJ.Load` with the `q` flag, regardless of invocation context, reliably updates the SQL catalog. Provide a robust, documented, and *accessible* API (callable by `SuperUser` from any namespace) to force a full refresh of SQL catalog information for a given schema or globally. Investigate why `%SYS.SQL.Schema` might be inaccessible.

**JIRA-IRIS-DEV-002: Spurious Compiler Error #1038 ("Private variable not allowed") for Valid `New` Statements**
*   **Summary:** The ObjectScript compiler, when invoked via `$System.OBJ.Load` through `docker exec iris session` with piped input, incorrectly flags standard local variable declarations (`New varname` or `New varname As %Type`) with error #1038.
*   **Impact:** Prevents compilation of valid ObjectScript classes, blocking development.
*   **Repro:** Attempt to compile a class with a method containing `New myVar As %SQL.Statement` or even `New myVar` using the described `docker exec` method.
*   **Proposed Fix:** Investigate and fix the parser/compiler behavior for `New` statements in the `docker exec iris session` input context. Ensure it correctly handles standard local variable declarations.

**JIRA-IRIS-DEV-003: Incorrect Scalar Return Value Marshalling for Projected SQL Functions via ODBC**
*   **Summary:** ObjectScript methods returning scalar types (e.g., `%String`, `%Integer`) via `Quit <value>`, when projected as SQL functions and called via `SELECT "Schema"."FuncName"(params)` through `pyodbc`, return an empty string `''` instead of the actual value.
*   **Impact:** Makes it impossible to retrieve scalar return values from such projected functions.
*   **Proposed Fix:** Ensure correct data type marshalling and value transfer for scalar return values from `SqlProc` methods when called as SQL functions via ODBC.

**JIRA-IRIS-DEV-004: Inaccessibility of Core System Class `%SYS.SQL.Schema`**
*   **Summary:** The system class `%SYS.SQL.Schema` cannot be accessed/called (e.g., `##class(%SYS.SQL.Schema).Refresh()`) by `SuperUser` in the `USER` namespace when the command is issued via `docker exec iris session`. Results in `<CLASS DOES NOT EXIST>`.
*   **Impact:** Prevents programmatic SQL schema management and cache clearing, contributing to catalog visibility issues. Indicates a potentially flawed or misconfigured IRIS environment in Docker.
*   **Proposed Fix:** Ensure all core system classes, especially those in `%SYS` required for SQL and system management, are correctly mapped and accessible by default to privileged users like `SuperUser` across all standard namespaces and invocation contexts (including `docker exec iris session`).

**JIRA-IRIS-DEV-005: Misleading Error Messages for Stale/Failed Projections**
*   **Summary:** ODBC errors like `[SQLCODE: <-460>:<General error>]` are ambiguous and do not clearly indicate whether a procedure was found but failed internally, or if it's related to a stale projection or catalog issue. Compiler errors like #1038 on valid syntax are also highly misleading.
*   **Proposed Fix:** Improve error reporting. If a projected procedure call fails due to issues with the underlying (possibly stale) ObjectScript method or its projection, provide more specific SQLCODEs or error messages that guide the developer towards recompilation, cache purging, or checking projection status.

## Conclusion: A Call for Improved Developer Experience

The journey documented in this report highlights significant friction points in the InterSystems IRIS developer experience, particularly with IRIS 2024.1.2 concerning the reliability of its ObjectScript-to-SQL projection mechanisms in containerized workflows. These are not issues of misunderstanding documented features but rather encounters with unexpected platform behavior. While IRIS is a powerful platform, these problems can lead to excessive time spent on debugging environment and tooling quirks.

The "ghosts" of stale procedures and inexplicable compiler errors prompted the project's shift away from Stored Procedures for RAG logic. While this pivot mitigated these specific issues, the project subsequently encountered a different primary blocker with IRIS 2025.1 related to `TO_VECTOR()` and ODBC driver behavior when trying to load vector embeddings for real-data testing (detailed in [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md)).

We urge Development and Quality Development to investigate SP projection and compilation issues thoroughly for overall platform stability and developer experience. Addressing both these foundational issues and the specific vector function limitations is crucial for IRIS to be a robust platform for modern AI/ML applications.

## References

*   [Defining and Using Stored Procedures - InterSystems IRIS Data Platform Documentation](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_procedures)
*   [Working with Cached Queries - InterSystems IRIS Data Platform Documentation](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GSOC_cachedqueries)
*   [PURGE CACHED QUERIES - InterSystems SQL Reference](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=RSQL_purgecachedqueries)
*   [CREATE PROCEDURE - InterSystems SQL Reference](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=RSQL_createprocedure)
*   [Querying the Database - Using InterSystems SQL](https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_queries)
*   [Optimizing SQL Performance (Cached Queries) - InterSystems IRIS Data Platform Documentation](https://docs.intersystems.com/latest/csp/docbook/DocBook.UI.Page.cls?KEY=GSQLOPT_cachedqueries)
*   [SQL Projection of ObjectScript Properties and Collections - InterSystems IRIS Documentation](https://docs.intersystems.com/irisforhealthlatest/csp/docbook/DocBook.UI.Page.cls?KEY=GOBJ_propcoll_sqlproj)
*   [SQL Error Codes - InterSystems IRIS Error Reference](https://docs.intersystems.com/healthconnectlatest/csp/docbook/DocBook.UI.Page.cls?KEY=RERR_sql)
*   [InterSystems Community: How to delete query cache programmatically?](https://community.intersystems.com/post/how-delete-query-cache-programmatically)
*   [InterSystems Community: Error executing stored procedure](https://community.intersystems.com/post/execute-store-procedure-sql-database-problems)
*   [InterSystems Community: Problems to call procedure and return outputvar](https://community.intersystems.com/post/problems-call-procedure-and-return-outputvar)
*   [InterSystems Community: SQL Query from Stored Procedure not working](https://community.intersystems.com/post/sql-query-stored-procedure-not-working)
*   [InterSystems Community: How to code a stored procedure that will return a result set](https://community.intersystems.com/post/intersystems-reports-logi-report-232-how-code-stored-procedure-will-return-result-set-be-used)
*   [InterSystems Community: How and what delete data after changing configuration](https://community.intersystems.com/post/how-and-what-delete-data-after-changing-configuration-system-explorer-tools-sql-performance)
*   [DBA StackExchange: Aside from explicitly flushing the cache or demanding recompiles, what (else) recompiles an execution plan?](https://dba.stackexchange.com/questions/334375/aside-from-explicitly-flushing-the-cache-or-demanding-recompiles-what-recompile)
*   [InterSystems Product Alert: SQL Queries Returning Wrong Results (Example of caching issues)](https://www.intersystems.com/product-alerts-advisories/alert-sql-queries-returning-wrong-results/)
