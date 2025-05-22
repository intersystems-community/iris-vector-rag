# Development Strategy Evolution

This document outlines the evolution of the development and environment strategy for the RAG Templates project, capturing key challenges and decisions.

## Initial Strategy (Pre May 20, 2025)

The initial development approach involved:
1.  A multi-container Docker setup using `docker-compose.yml`:
    *   One container for the InterSystems IRIS database.
    *   A separate container for the Python application, built from an `app.Dockerfile`.
2.  Python application connecting to IRIS via ODBC (`pyodbc` driver).
3.  An attempt to implement database logic (like vector search) using Stored Procedures (SPs) written directly in SQL scripts (`.sql` files), to be created during database initialization. This included experiments with both `LANGUAGE SQL` and `LANGUAGE OBJECTSCRIPT` for these SPs.

## Challenges Encountered

Several significant challenges were faced with this initial strategy, leading to considerable debugging time and developer frustration:

### 1. Stored Procedure Implementation via SQL Scripts
*   **Result-Set Returning SQL SPs:** Defining `LANGUAGE SQL` Stored Procedures that returned result sets (e.g., via a `SELECT` statement) proved difficult to get working correctly with ODBC. Issues included syntax errors related to `RESULT SETS 1` and procedures not being recognized as returning results.
*   **ObjectScript SPs (in `.sql` files):**
    *   Attempts to define `LANGUAGE OBJECTSCRIPT` SPs (using `&sql()` or `%SQL.Statement`) that returned result sets also faced issues. When `RESULT SETS 1` was omitted, `pyodbc` calls often resulted in "No results. Previous SQL was not a query."
    *   When `RESULT SETS 1` was included for `LANGUAGE OBJECTSCRIPT` SPs, IRIS expected a full "Query Procedure" structure (with `EXECUTE`, `FETCH`, `CLOSE`, `DECLARATION` labels), which is overly complex to define inline within an SQL script processed by `db_init.py`.
*   **Class Compilation:** Defining SPs as methods within an ObjectScript class (`.cls` file) is the standard IRIS approach for robust SPs. However, attempts to reliably compile `.cls` files from within the Python application's Docker container (e.g., via `docker exec iris_container Do $SYSTEM.OBJ.Load(...)`) were also problematic and inconsistent.

### 2. Combined Docker Image (IRIS + Python Application)
A subsequent strategy involved creating a single Docker image containing both IRIS and the full Python application environment (Python 3.11, Poetry, dependencies). While this aimed to simplify inter-process communication, it introduced new complexities:
*   **Persistent Docker Build Context Issues:** The `COPY . .` command in the `Dockerfile` repeatedly failed to pick up the latest versions of committed files from the host machine, even when using `docker build --no-cache` and explicitly removing old images (`docker rmi`). This led to the container running outdated code, causing significant debugging delays (e.g., "No such file or directory" errors, tests running against old code versions).
*   **Python Environment Conflicts:**
    *   Initial attempts to make the project's Python 3.11/3.12 the default `/usr/bin/python3` in the image conflicted with IRIS internal scripts (like `/docker-entrypoint.sh iris-after-start`) which expected a system Python and specific modules like `irissqlcli`. This caused IRIS to fail on startup. This was resolved by not overriding the system `python3` and installing the project's Python version separately.
    *   Ensuring the Poetry virtual environment was correctly created and accessed by the correct user (`irisowner`) within the Docker container when scripts were run via `docker exec poetry run ...` also required careful Dockerfile layering and permission management. The "No module named 'intersystems_iris'" error was a symptom of this until resolved.
*   **Debugging Complexity:** Debugging Python script execution and dependency issues inside the multi-layered Docker container proved to be less direct and more time-consuming than desired.

### 3. IRIS SQL Stored Procedure Projection and Caching Issues

A detailed postmortem analysis revealed several critical issues with InterSystems IRIS's mechanisms for projecting ObjectScript class methods as SQL stored procedures, combined with its query caching and SQL catalog management:

*   **SQL Catalog Projection Inconsistencies:** SQL catalog entries for projected procedures did not reliably update or get removed when the underlying ObjectScript class was recompiled or deleted. This led to "zombie" procedures where IRIS continued to serve old, broken, or non-existent code despite correct source code and repeated recompilations.

*   **Persistent Compiler Errors for Valid Syntax:** When using `docker exec iris session` to compile ObjectScript classes, the compiler incorrectly flagged standard local variable declarations (`New varname` or `New varname As %Type`) with error #1038 ("Private variable not allowed"). This prevented compilation of valid ObjectScript classes.

*   **Inaccessibility of Core System Classes:** Critical system classes like `%SYS.SQL.Schema` could not be accessed by `SuperUser` in the `USER` namespace when commands were issued via `docker exec iris session`. This prevented programmatic SQL schema management and cache clearing.

*   **Misleading Error Messages:** ODBC errors like `[SQLCODE: <-460>:<General error>]` were ambiguous and did not clearly indicate whether a procedure was found but failed internally, or if it was related to a stale projection or catalog issue.

*   **Scalar Return Value Marshalling Issues:** Scalar values returned via `Quit <value>` from an ObjectScript method projected as an SQL function were lost or converted to an empty string when retrieved via `pyodbc`.

*   **Persistent Caching Problems:** Cached queries and routine definitions persisted stubbornly, leading to the execution of obsolete logic even after attempts to purge caches and refresh schemas.

These issues were not isolated bugs but appeared to be systemic problems that created a hostile and inefficient development experience, particularly in automated, Dockerized, or CI/CD environments. The only consistently reliable method to ensure a clean state was often a full Docker volume wipe.

## Pivot to Simplified Strategy (Effective May 20, 2025)

Given the persistent friction and time spent on environmental and tooling issues, a decision was made to pivot to a significantly simplified development strategy to prioritize progress on the core RAG functionality.

**New Approach:**
1.  **Dedicated IRIS Docker Container:** A very simple Docker setup (`docker-compose.iris-only.yml`) is used to run only the InterSystems IRIS database in a dedicated container. This container is started and largely left alone.
2.  **Host-Based Python Development:** All Python application logic (RAG pipelines, data loading scripts, tests) is developed and executed directly on the host machine (macOS in this case).
    *   Python 3.11 is managed on the host.
    *   Initially Poetry managed dependencies, but the project has transitioned to using `uv` (a faster Python package installer and environment manager) for dependency management.
3.  **DB-API for Connectivity:** Python scripts on the host connect to the IRIS Docker container via the network (e.g., `localhost:<mapped_port>`) using the native InterSystems Python DB-API (`intersystems_iris` package). ODBC is no longer used for the primary application-database connection.
4.  **Client-Side SQL:** All database queries, including vector search, are constructed and executed as SQL strings directly from the Python code (no Stored Procedures for RAG logic).
5.  **Vector SQL Utilities:** To address IRIS SQL limitations with vector operations, dedicated utility functions were created in `common/vector_sql_utils.py` to safely construct and execute vector search queries. These utilities include:
    * Input validation to prevent SQL injection
    * Proper formatting of vector search SQL
    * Standardized error handling for vector operations

**Rationale for Pivot:**
*   **Avoid SQL Projection Issues:** By moving away from stored procedures to client-side SQL, we avoid the entire class of problems related to SQL projection, catalog updates, and caching.
*   **Reduce Environmental Complexity:** Eliminates the challenges of building and debugging a combined IRIS+Python Docker image and the associated build context, user context, and Python environment conflicts.
*   **Improve Development Feedback Loop:** Running Python code directly on the host allows for faster iteration, easier debugging with familiar host-based tools, and more direct visibility into script execution.
*   **Focus on Core RAG Logic:** By minimizing time spent on Docker and SP intricacies, development can focus on implementing and testing the RAG pipelines.
*   **Leverage Stable Components:** Uses a stable, simple IRIS Docker container and standard host-based Python development practices.
*   **Mitigate IRIS SQL Limitations:** The vector SQL utilities provide a consistent, safe approach to working with IRIS vector operations in `SELECT` queries despite the platform's limitations with parameter markers in vector functions.
    *   **Note on Current Blocker:** While this client-side SQL strategy effectively addresses challenges with querying, a related platform issue concerning the loading of embedding data (often involving `TO_VECTOR` in `INSERT` or `UPDATE` statements, particularly with ODBC) currently blocks full real-data testing and benchmarking. This specific blocker is detailed further in [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md).

**Lessons Learned:**
*   **Avoid Reliance on SQL Projection:** InterSystems IRIS's SQL projection mechanisms for ObjectScript classes are not reliable enough for automated development workflows, especially in containerized environments.
*   **Prefer Client-Side SQL Construction:** For complex operations like vector search, it's more reliable to construct and validate SQL on the client side than to rely on stored procedures.
*   **Implement Robust Input Validation:** Given the limitations of IRIS SQL (e.g., inability to parameterize vector functions), client-side validation is essential for security.
*   **Standardize Error Handling:** Consistent error handling patterns help mitigate the often cryptic or misleading error messages from the database.
*   **Minimize Container Complexity:** Simpler container setups with clear separation of concerns are more maintainable and less prone to environment-specific issues.

**Potential Future Considerations:**
*   Once the RAG application is mature and stable, Dockerizing the Python application (connecting to an external or containerized IRIS) can be revisited for deployment or packaging purposes.
*   If InterSystems addresses the identified issues in future IRIS releases, the approach to stored procedures and SQL projection could be reconsidered.

**Transition to uv (May 21, 2025):**
*   **Motivation:** Poetry, while powerful, can be slower for dependency resolution and installation. The project adopted `uv` for its significantly faster performance and simplified workflow.
*   **Implementation:**
    * Updated setup instructions in README.md to use `uv` for virtual environment creation and dependency management
    * Maintained the Poetry configuration in `pyproject.toml` for compatibility
    * Added support for exporting Poetry dependencies to requirements.txt for use with `uv`
*   **Benefits:**
    * Faster dependency resolution and installation
    * Simplified environment setup for new developers
    * Improved development workflow with quicker iteration cycles

This documented evolution aims to capture the learning process and justify the strategic shifts to a more streamlined and productive development workflow.
