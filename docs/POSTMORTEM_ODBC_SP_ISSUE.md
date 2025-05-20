# Postmortem: ODBC Stored Procedure Call and Compilation Issues

This document summarizes the extensive troubleshooting process undertaken to enable a Python application to reliably call an InterSystems IRIS ObjectScript class method projected as an SQL stored procedure, particularly focusing on automated class compilation and SQL projection visibility within a Dockerized environment.

## Initial Problem Statement

The primary goal was to call an SQL stored procedure, `RAG.VSU_SearchDocsV3` (projected from `RAG.VectorSearchUtils.SearchSourceDocuments`), from a Python script using `pyodbc`. The initial attempts resulted in a `[SQLCODE: <-460>:<General error>]` from the ODBC driver, indicating the procedure was likely found by name but failed during preparation or execution.

## Phase 1: Addressing the `-460 Error` - Query Complexity

*   **Hypothesis:** The `-460` error was due to the complexity of the SQL query within the `SearchSourceDocuments` method, specifically the `SELECT TOP K ...` clause where `K` was a procedure parameter, or the `VECTOR_COSINE` function.
*   **Attempts:**
    1.  Modified `VectorSearch.cls` to use an intermediate local variable (`localTopK`) for the `TOP K` value in the dynamically constructed SQL string. This aimed to ensure `TopK` was treated as a literal.
    2.  Simplified `SearchSourceDocuments` in `VectorSearch.cls` to a basic "echo" function (`SELECT ? AS EchoedVectorString`) to remove `TOP K` and `VECTOR_COSINE` as variables.
*   **Findings:** The `-460` error persisted even with the simplified echo logic, suggesting the issue was more fundamental than the query complexity itself, possibly related to how the procedure was compiled, projected, or called.

## Phase 2: Automated Class Compilation and SQL Projection

The focus shifted to ensuring `RAG.VectorSearchUtils.cls` was correctly compiled and its `SearchSourceDocuments` method (with `SqlName="VSU_SearchDocsV3"`) was properly projected to SQL and made visible.

*   **Attempt 2.1: Custom SQL Stored Procedure for File-Based Load (`RAG.CompileClassFromFile`)**
    *   **Method:** Created an SQL SP (`RAG.CompileClassFromFile`) that internally called `$System.OBJ.Load("path/to/class.cls", "ckq")`. This SP was called from `run_db_init_docker.py` via DBAPI.
    *   **Result:** Failed with `[SQLCODE: <-400>:<Fatal error occurred>] [Error: <<ENDOFFILE>LoadUDL+17^%apiOBJ>]`.
    *   **Analysis:** Indicated that the IRIS process (running the SP) could not access/read the `.cls` file from the Docker volume mount (e.g., `/irisdev_common/VectorSearch.cls`). This was suspected to be a Docker volume permission issue for the `irisowner` user or an incorrect path from the IRIS server's perspective.

*   **Attempt 2.2: Custom SQL Stored Procedure for Stream-Based Load (`RAG.LoadClassFromStream`)**
    *   **Method:** Created an SQL SP (`RAG.LoadClassFromStream`) that took the class UDL content as a `LONGVARCHAR`, created a `%Stream.GlobalCharacter`, wrote the content to it, and then called `$System.OBJ.LoadStream(stream, "ckq")`. This was intended to bypass filesystem access issues.
    *   **Result:** Initially failed due to ObjectScript syntax errors in the SP definition (`New var As Type`, `Catch ex As Type`). After correcting these, it still failed with `[SQLCODE: <-400>:<Fatal error occurred>] [Location: <SPFunction>]` when called from `run_db_init_docker.py`, even with a minimal, pristine class content.
    *   **Analysis:** Suggested a severe, untrappable crash within `$System.OBJ.LoadStream` itself or a fundamental issue with passing large string/stream content via DBAPI to the SP in this environment.

*   **Attempt 2.3: Direct DBAPI Execution of ObjectScript System Utilities**
    *   `cursor.execute('DO $System.OBJ.Load(...)')`: Failed with `SQLCODE: <-51>:<SQL statement expected>`. (DBAPI `execute` expects SQL).
    *   `cursor.execute('SELECT ##class(%SYS.PTools.SQL).LSQL(\'DO $System.OBJ.Load(...)\')')`: Failed with `SQLCODE: <-12>:<A term expected... ^SELECT #>`. (Issue with `##class` syntax via this DBAPI path).
    *   `cursor.execute('SELECT %SYS.PTools.SQL.LSQL(\'DO $System.OBJ.Load(...)\')')`: Failed with `SQLCODE: <-359>:<SQL Function ... not found>`. (Issue resolving `PTools.SQL.LSQL`).
    *   `cursor.execute('CALL %SYSTEM.SQL_Execute(\'DO $System.OBJ.Load(...)\')')`: Failed with `SQLCODE: <-428>:<Stored procedure not found>` for `%SYSTEM.SQL_EXECUTE`.
    *   **Analysis:** Direct DBAPI calls to system utilities for arbitrary ObjectScript execution proved unreliable or had syntax/resolution issues.

*   **Attempt 2.4: Automated Compilation via `docker exec` (Python `subprocess`)**
    *   **Method:** Modified `run_db_init_docker.py` to use `subprocess.run()` to execute `docker exec -i iris_odbc_test_db iris session IRIS -U USER` and pipe the `Do $System.OBJ.Load("/irisdev_common/VectorSearch.cls", "ckq") Halt` command.
        *   The `app` container's Dockerfile was updated to install Docker CLI.
        *   `docker-compose.yml` was updated to mount the Docker socket into the `app` container and ensure the `iris` container had the `/irisdev_common` volume mount.
    *   **Initial `docker exec` issues:**
        *   Invalid `-T` flag for `docker exec`. Corrected by removing it.
        *   `<INVALID ARGUMENT>` from `iris session`. Corrected by ensuring the ObjectScript command was piped correctly.
        *   Escaped quotes `\\"` in the piped ObjectScript command caused `<SYNTAX>` error in IRIS. Corrected to use raw quotes `"` for the piped command.
    *   **Key Compiler Error:** After fixing `docker exec` mechanics, `$System.OBJ.Load` consistently failed with `ERROR #1038: Private variable not allowed` for `New stmt` and `New sc` within `VectorSearch.cls` (even the simplified version, and even after simplifying the `Class RAG.VectorSearchUtils [...]` declaration line). This error is incorrect for standard local variable declarations.
    *   **Current Diagnostic:** The last step was to change the `docker exec` command to a simple `Write "Hello..."` to confirm basic ObjectScript execution via this piping mechanism. The results for this are pending.

## Phase 3: SQL Projection Visibility in `INFORMATION_SCHEMA.ROUTINES`

*   **Problem:** Even when `$System.OBJ.Load` reported "Load finished successfully." (e.g., during one of the `docker exec` attempts before the #1038 error became persistent), the projected SQL procedure (`VSU_SearchDocsV3`) was consistently NOT found in `INFORMATION_SCHEMA.ROUTINES`, neither by `SuperUser` in `run_db_init_docker.py` nor by the `test` user in `test_pyodbc_driver.py`.
*   **Attempts to Fix:**
    1.  Added `DO $SYSTEM.SQL.Purge(0)` via `docker exec` after class compilation.
    2.  Changed to `DO ##class(%SYS.SQL.Schema).Refresh("RAG",1)` via `docker exec` (after correcting the class name from `$System` to `$SYS`). This refresh command itself failed with `<CLASS DOES NOT EXIST>` for `%SYS.SQL.Schema`.
*   **Status:** Unresolved. The catalog does not update/reflect the projection.

## Current Status & Remaining Issues

1.  **Primary Blocker: Compiler Error #1038 (`Private variable not allowed`)** when using `docker exec` to call `$System.OBJ.Load` on `VectorSearch.cls` (even a simplified version with correct `New varname` syntax). This prevents the class from compiling.
2.  **SQL Projection Invisibility:** If compilation were to succeed, the issue of the projected procedure not appearing in `INFORMATION_SCHEMA.ROUTINES` would need to be solved. The `%SYS.SQL.Schema` class not being found is also a concern.
3.  **ODBC Call Failure (`-460`):** Dependent on successful compilation and projection. If the procedure becomes callable, this error (likely related to `TOP K` or other query aspects) would be the next to address.

The overall problem points to deep incompatibilities or bugs in how class compilation, SQL projection, and catalog updates behave when initiated through non-interactive, remote (DBAPI or `docker exec`) mechanisms in this specific Dockerized IRIS environment, potentially exacerbated by subtle file content issues or an unstable compiler state.
