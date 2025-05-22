# IRIS Vector Search Implementation: Key Lessons and Findings

This document consolidates the key findings and lessons learned from our experience implementing vector search capabilities with InterSystems IRIS. It draws from the detailed postmortem analyses documented in `IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md` and `POSTMORTEM_ODBC_SP_ISSUE.md`.

## Executive Summary

Our project to develop RAG (Retrieval Augmented Generation) templates using InterSystems IRIS for vector search capabilities encountered significant challenges, primarily related to:

1. SQL stored procedure projection from ObjectScript classes
2. Query and catalog caching behaviors
3. Automated class compilation in Dockerized environments
4. Error reporting and diagnostics
5. IRIS SQL vector operations limitations (both for querying and loading).

These issues led to a strategic pivot away from stored procedures toward client-side SQL construction and execution for querying. While this approach proved more reliable for development and addressed some query-time issues, the project is **currently blocked** by a critical limitation: the ODBC driver's behavior with the `TO_VECTOR()` function prevents loading documents with embeddings. This is the **primary project blocker** for testing with real PMC data.

## Key Technical Findings

### 1. SQL Projection and Catalog Management Issues

* **Zombie Procedures:** SQL catalog entries for projected procedures did not reliably update or get removed when the underlying ObjectScript class was recompiled or deleted.
* **Catalog Refresh Failures:** Attempts to refresh the SQL catalog using `##class(%SYS.SQL.Schema).Refresh()` failed with `<CLASS DOES NOT EXIST>` errors.
* **Persistent Caching:** Cached queries and routine definitions persisted stubbornly, leading to the execution of obsolete logic despite recompilation.

### 2. ObjectScript Compilation Challenges

* **Spurious Compiler Errors:** Valid ObjectScript syntax (e.g., `New varname As %Type`) was incorrectly flagged with error #1038 ("Private variable not allowed") when compiled via `docker exec iris session`.
* **System Class Accessibility:** Critical system classes like `%SYS.SQL.Schema` could not be accessed by `SuperUser` in the `USER` namespace when commands were issued via `docker exec`.
* **Inconsistent Compilation Results:** The same class file would compile successfully in interactive mode but fail when compiled through automated means.

### 3. ODBC and Data Marshalling Issues

* **Cryptic Error Messages:** ODBC calls to stored procedures often failed with generic errors like `[SQLCODE: <-460>:<General error>]` that provided little diagnostic value.
* **Return Value Problems:** Scalar values returned via `Quit <value>` from ObjectScript methods were lost or converted to empty strings when retrieved via `pyodbc`.
* **Parameter Passing Limitations:** Complex parameter types (streams, large strings) could not be reliably passed to stored procedures via ODBC.

### 4. Vector-Specific SQL Limitations

* **Parameter Marker Rejection:** The `TO_VECTOR()` function does not accept parameter markers (?, :param), which are standard in SQL for safe query parameterization.
* **TOP/FETCH Clause Limitations:** The `TOP` and `FETCH FIRST` clauses, essential for limiting results in vector similarity searches, do not accept parameter markers.
* **Client Driver Rewriting:** Python, JDBC, and other client drivers replace embedded literals with :%qpar(n) even when no parameter list is supplied, creating misleading parse errors.
* **ODBC Driver Limitations:** When loading documents with embeddings, the ODBC driver encounters limitations with the TO_VECTOR function, which is currently blocking testing with real data.

## Successful Mitigation Strategies

### 1. Client-Side SQL Construction

We developed a set of utility functions in `common/vector_sql_utils.py` that:
* Validate input parameters to prevent SQL injection
* Safely construct SQL queries with proper vector function syntax
* Provide standardized error handling for vector operations

Example functions include:
* `validate_vector_string()` - Ensures vector strings contain only valid characters
* `validate_top_k()` - Validates that top_k is a positive integer
* `format_vector_search_sql()` - Constructs a SQL query for vector search using string interpolation
* `execute_vector_search()` - Executes a vector search SQL query and handles common errors

### 2. Simplified Docker Strategy

* Moved to a dedicated IRIS Docker container with no application logic
* Developed and ran Python application code on the host machine
* Connected to IRIS via network using the native InterSystems Python DB-API
* Avoided complex Docker builds and environment conflicts

### 3. Standardized Vector Search Implementation

* Implemented consistent vector search functions in `common/db_vector_search.py`
* Used the utility functions from `vector_sql_utils.py` to ensure safe and reliable queries
* Added comprehensive error handling and logging
* Created thorough test coverage to verify functionality

## Recommendations for Future IRIS Vector Projects

1. **Avoid SQL Projection:** Prefer client-side SQL construction over stored procedures for vector operations.
2. **Implement Robust Validation:** Always validate vector strings and other parameters before including them in SQL queries.
3. **Standardize Error Handling:** Create consistent patterns for handling and logging database errors.
4. **Simplify Container Strategy:** Use simple, dedicated containers rather than complex multi-service images.
5. **Test Thoroughly:** Create comprehensive tests for vector search functionality, including edge cases.
6. **Document Workarounds:** Maintain clear documentation of any platform limitations and the workarounds implemented.

## Proposed Improvements for InterSystems IRIS

Based on our experience, we recommend InterSystems consider addressing the following areas, which align with the JIRA issues detailed in [`docs/MANAGEMENT_SUMMARY.md`](docs/MANAGEMENT_SUMMARY.md):

### SQL Engine Enhancements:
1.  **Parameter Support in Vector Functions:** Allow parameter markers in `TO_VECTOR()` (SQL-1) and `TOP`/`FETCH FIRST` clauses (SQL-2). This is crucial for standard SQL practices.
2.  **Reliable SQL Projection & Catalog Management:** Ensure SQL catalog entries for projected ObjectScript classes are consistently updated (SQL-4), and that core system classes like `%SYS.SQL.Schema` are always accessible (SQL-6). Address persistent caching issues.
3.  **Improved Compiler Diagnostics & Behavior:** Fix spurious compiler errors (e.g., #1038 for `New varname` - SQL-5) in automated/`docker exec` contexts. Provide clearer compiler error messages.
4.  **LANGUAGE SQL Stored Procedure Enhancements:** Add support for `DECLARE`/`SET` (SQL-3) for better usability if SPs are to be a viable path.

### Client Driver (ODBC, DB-API, JDBC) Enhancements:
1.  **ODBC `TO_VECTOR()` Handling for Loading:** Critically, fix ODBC driver behavior with `TO_VECTOR()` to allow loading of vector embeddings (DRV-2). This is the primary project blocker.
2.  **Literal Rewriting by Drivers:** Modify drivers to avoid rewriting literals to `:%qpar()` when no parameter list is supplied or when it interferes with functions like `TO_VECTOR()` (DRV-1).
3.  **ODBC Stored Procedure Reliability:** Address issues leading to generic errors like `-460` (DRV-3) and ensure correct marshalling of scalar return values (DRV-5).
4.  **DB-API System Utility Execution:** Improve DB-API support for executing system utilities (DRV-4).
5.  **Clearer Error Reporting:** Provide more specific and actionable error messages across all drivers and the SQL engine (GEN-1).

Addressing these areas would significantly improve the developer experience and robustness of IRIS for modern application development, especially those involving AI/ML and vector capabilities.

## Conclusion

While InterSystems IRIS provides vector search capabilities, working with these features, particularly in automated and Dockerized environments, presented significant challenges related to both Stored Procedure development and core SQL vector function behavior. By pivoting to client-side SQL construction for querying and implementing robust validation, we created a functional, albeit workaround-reliant, solution for vector search queries in our RAG templates.

However, the project remains **critically blocked** by the inability to load new document embeddings due to ODBC driver limitations with the `TO_VECTOR()` function. This prevents full real-data testing and benchmarking. This issue requires urgent attention and resolution, potentially through engagement with InterSystems support.

For a detailed explanation of the current primary blocker and other vector limitations, see [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md).

The lessons learned and strategies developed during this project will be valuable for any team implementing vector search with InterSystems IRIS, particularly in modern development environments that rely heavily on automation and containerization.