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

## Pivot to Simplified Strategy (Effective May 20, 2025)

Given the persistent friction and time spent on environmental and tooling issues, a decision was made to pivot to a significantly simplified development strategy to prioritize progress on the core RAG functionality.

**New Approach:**
1.  **Dedicated IRIS Docker Container:** A very simple Docker setup (`docker-compose.iris-only.yml`) is used to run only the InterSystems IRIS database in a dedicated container. This container is started and largely left alone.
2.  **Host-Based Python Development:** All Python application logic (RAG pipelines, data loading scripts, tests) is developed and executed directly on the host machine (macOS in this case).
    *   Python 3.11 is managed on the host.
    *   Poetry manages dependencies for the host Python environment.
3.  **DB-API for Connectivity:** Python scripts on the host connect to the IRIS Docker container via the network (e.g., `localhost:<mapped_port>`) using the native InterSystems Python DB-API (`intersystems_iris` package). ODBC is no longer used for the primary application-database connection.
4.  **Client-Side SQL:** All database queries, including vector search, are constructed and executed as SQL strings directly from the Python code (no Stored Procedures for RAG logic).

**Rationale for Pivot:**
*   **Reduce Environmental Complexity:** Eliminates the challenges of building and debugging a combined IRIS+Python Docker image and the associated build context, user context, and Python environment conflicts.
*   **Improve Development Feedback Loop:** Running Python code directly on the host allows for faster iteration, easier debugging with familiar host-based tools, and more direct visibility into script execution.
*   **Focus on Core RAG Logic:** By minimizing time spent on Docker and SP intricacies, development can focus on implementing and testing the RAG pipelines.
*   **Leverage Stable Components:** Uses a stable, simple IRIS Docker container and standard host-based Python development practices.

**Potential Future Considerations:**
*   Once the RAG application is mature and stable, Dockerizing the Python application (connecting to an external or containerized IRIS) can be revisited for deployment or packaging purposes.
*   Alternative Python packaging tools (e.g., `uv`) could be evaluated if Poetry on the host presents significant issues, though it's expected to work well.

This documented evolution aims to capture the learning process and justify the strategic shift to a more streamlined and productive development workflow.
