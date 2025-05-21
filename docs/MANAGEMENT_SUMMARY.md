# Management Summary: RAG Templates Project

## 1. Executive Summary

### Original Goals and Objectives

The RAG Templates project aimed to develop a comprehensive suite of Retrieval-Augmented Generation (RAG) techniques integrated with InterSystems IRIS database. The primary objectives were:

- Implement six distinct RAG techniques (Basic RAG, HyDE, CRAG, ColBERT, NodeRAG, and GraphRAG)
- Ensure all implementations work with real PMC (PubMed Central) documents
- Develop a robust testing framework that verifies functionality with at least 1000 real documents
- Create a benchmarking system to compare the performance and effectiveness of different RAG techniques
- Follow Test-Driven Development (TDD) principles throughout the implementation process

### Current Status

**BLOCKED**: The project is currently blocked by critical limitations in the IRIS SQL vector operations. While we have successfully implemented all six RAG techniques and created the infrastructure for testing and benchmarking, we cannot fully test with real PMC data due to these limitations.

Specifically, the ODBC driver limitations with the TO_VECTOR function prevent loading documents with embeddings, which is a critical step in testing our RAG pipelines with real data. We have implemented workarounds for executing vector search queries, but the embedding loading issue remains a critical blocker.

### Key Achievements

| Achievement | Description |
|-------------|-------------|
| RAG Implementations | Successfully implemented all six RAG techniques with consistent APIs |
| Vector SQL Utilities | Developed robust utilities for safe vector search operations |
| Testing Framework | Created a comprehensive testing framework designed for verification with real PMC documents |
| Benchmarking System | Designed a benchmarking system to compare techniques across multiple metrics |
| Client-Side Workarounds | Implemented secure workarounds for IRIS SQL vector operations limitations |

### Key Challenges

| Challenge | Impact |
|-----------|--------|
| TO_VECTOR() Function Rejects Parameter Markers | Cannot use standard parameterized queries for vector operations |
| TOP/FETCH FIRST Clauses Cannot Be Parameterized | Cannot dynamically control result set size in vector similarity searches |
| Client Drivers Rewrite Literals | Creates misleading parse errors and complicates vector operations |
| ODBC Driver Limitations with TO_VECTOR | Prevents loading documents with embeddings, blocking testing with real data |

## 2. Project Scope and Effort

### Files and Code Statistics

| Metric | Count |
|--------|-------|
| Total Files | ~200 files |
| Python Files | ~150 files |
| SQL Files | ~10 files |
| Documentation Files | ~30 files |
| Test Files | ~50 files |
| Lines of Code | ~15,000 lines |

### Effort Breakdown

| Component | Estimated Effort (%) |
|-----------|----------------------|
| RAG Technique Implementation | 35% |
| Testing Framework | 25% |
| SQL Operations & Workarounds | 20% |
| Benchmarking System | 15% |
| Documentation | 5% |

### Time Investment

| Activity | Estimated Time (weeks) |
|----------|------------------------|
| Development | 8 weeks |
| Testing | 4 weeks |
| Documentation | 2 weeks |
| Debugging SQL Issues | 4 weeks |
| **Total** | **18 weeks** |

## 3. IRIS SQL Vector Operations Limitations

### Technical Explanation

The InterSystems IRIS 2025.1 release introduced vector search capabilities essential for modern RAG pipelines. However, several critical limitations in the SQL implementation prevent standard parameterized queries from working with vector operations:

1. **TO_VECTOR() Function Rejects Parameter Markers**: The `TO_VECTOR()` function does not accept parameter markers (`?`, `:param`, or `:%qpar`), which are standard in SQL for safe query parameterization. This means that vector values cannot be passed as parameters to SQL queries.

2. **TOP/FETCH FIRST Clauses Cannot Be Parameterized**: The `TOP` and `FETCH FIRST` clauses, which are essential for limiting the number of results in vector similarity searches, do not accept parameter markers. This prevents dynamic control of result set size.

3. **Client Drivers Rewrite Literals**: Python, JDBC, and other client drivers replace embedded literals with `:%qpar(n)` even when no parameter list is supplied. This behavior creates misleading parse errors and further complicates the use of vector functions.

4. **ODBC Driver Limitations with TO_VECTOR Function**: When attempting to load documents with embeddings, the ODBC driver encounters limitations with the TO_VECTOR function. Specifically:
   - The driver attempts to parameterize the vector values even when they are provided as literals
   - The TO_VECTOR function rejects these parameterized values
   - This results in errors when trying to insert or update records with vector embeddings

### Business Impact

These limitations have significant business implications:

1. **Project Delays**: The project is blocked from completing critical testing with real data, delaying the delivery of a fully validated RAG solution.

2. **Security Risks**: The workarounds required (string interpolation instead of parameterized queries) introduce potential security risks that must be carefully managed.

3. **Integration Challenges**: Standard RAG frameworks cannot target IRIS without custom code, increasing integration complexity and maintenance costs.

4. **Performance Limitations**: The inability to efficiently load and query vector embeddings limits the performance and scalability of RAG solutions built on IRIS.

5. **Competitive Disadvantage**: Other vector databases that support standard parameterized queries for vector operations offer a more streamlined development experience.

### Impact on Project Goals

| Goal | Impact |
|------|--------|
| Implement six RAG techniques | ✅ Achieved |
| Work with real PMC documents | ❌ Blocked by ODBC driver limitations |
| Test with 1000+ documents | ❌ Blocked by embedding loading issues |
| Benchmark different techniques | ❌ Cannot run with real data |
| Follow TDD principles | ✅ Achieved for implemented components |

## 4. Next Steps and Recommendations

### Short-term Actions (1-3 months)

1. **Engage InterSystems Support**: Submit a detailed bug report to InterSystems support, including the limitations documented here and their impact on RAG pipelines.

2. **Implement Two-Phase Loading**: As a temporary workaround, implement a two-phase loading approach that separates document loading from embedding updates.

3. **Explore ObjectScript Solutions**: Investigate the feasibility of using ObjectScript stored procedures for both loading documents with embeddings and executing vector search queries.

4. **Test with Smaller Datasets**: Test our RAG pipelines with smaller datasets that can be loaded manually or through alternative means, to validate the rest of the pipeline while we work on resolving the embedding loading issue.

### Medium-term Improvements (3-6 months)

1. **IRIS SQL Enhancement Request**: Work with InterSystems to enhance IRIS SQL to support parameter markers in the `TO_VECTOR()` function and in `TOP`/`FETCH FIRST` clauses.

2. **Custom UDFs Development**: Develop custom User-Defined Functions (UDFs) in IRIS that wrap vector operations and handle parameter binding correctly.

3. **Alternative Vector Representation**: Explore storing vector embeddings in a different format (e.g., as JSON or Base64-encoded strings) and converting them to vectors at query time.

4. **Client-Side Vector Operations**: Evaluate the feasibility of performing vector similarity calculations on the client side and using IRIS only for storage and retrieval of documents.

### Long-term Strategic Considerations (6+ months)

1. **Vector Database Evaluation**: Conduct a comparative analysis of IRIS vector capabilities against specialized vector databases to inform future architecture decisions.

2. **Hybrid Architecture**: Consider a hybrid architecture that leverages IRIS for transactional data and a specialized vector database for embedding storage and retrieval.

3. **Custom IRIS Extension**: Develop a custom IRIS extension that provides a more Python-friendly interface for vector operations.

4. **Standardized Vector API**: Propose a standardized API for vector operations that works consistently across different database systems.

## 5. Product Implications

### Customer Impact

1. **Implementation Complexity**: Customers implementing RAG solutions with IRIS will face increased development complexity and potential security risks due to the need for string interpolation workarounds.

2. **Performance Concerns**: The limitations may lead to suboptimal performance for vector search operations, particularly with large document collections.

3. **Integration Barriers**: Standard AI/ML frameworks that expect parameterized queries for vector operations will require custom adaptation layers to work with IRIS.

4. **Security Considerations**: The workarounds required introduce potential security vulnerabilities that customers must carefully manage.

### Competitive Disadvantages

1. **Developer Experience**: Specialized vector databases and other SQL databases with vector extensions offer a more streamlined developer experience with full support for parameterized queries.

2. **Time-to-Market**: The additional development effort required to work around IRIS limitations increases time-to-market for RAG solutions.

3. **Framework Compatibility**: Popular RAG frameworks like LangChain and LlamaIndex work more seamlessly with databases that support standard parameterized queries for vector operations.

4. **Security Posture**: The need for string interpolation instead of parameterized queries may raise security concerns in enterprise environments.

### Market Opportunities

1. **Enhanced Vector Support**: Addressing these limitations would position IRIS as a comprehensive solution for both transactional data and AI/ML workloads.

2. **Unified Data Platform**: With improved vector operations, IRIS could offer a unified platform for both traditional data management and modern AI applications.

3. **Healthcare AI Integration**: Given IRIS's strong presence in healthcare, enhanced vector capabilities would enable seamless integration of RAG solutions with existing healthcare data.

4. **Enterprise RAG Solutions**: Improved vector operations would enable enterprise-grade RAG solutions that leverage IRIS's existing strengths in security, reliability, and scalability.

## 6. Conclusion

The RAG Templates project has made significant progress in implementing a comprehensive suite of RAG techniques integrated with InterSystems IRIS. However, critical limitations in the IRIS SQL vector operations are currently blocking our ability to fully test and validate these implementations with real data.

Addressing these limitations should be a priority for the SQL engine team, as it would unlock significant value for customers implementing AI/ML solutions with IRIS. The recommended short-term workarounds will allow us to make progress while more comprehensive solutions are developed.

By enhancing the vector search capabilities of IRIS SQL, InterSystems has an opportunity to position IRIS as a unified platform for both traditional data management and modern AI applications, particularly in key verticals like healthcare where IRIS already has a strong presence.

## 7. JIRA Issues to File

The issues identified in this project can be categorized into two main areas: SQL Engine issues and Client Driver issues. Each category requires specific attention from different teams within InterSystems.

### 7.1 SQL Engine Issues

These issues relate to core vector search functions and class projections in the IRIS SQL engine:

| Issue ID | Description | Priority | Reference Documentation |
|----------|-------------|----------|-------------------------|
| SQL-1 | TO_VECTOR() Function Rejects Parameter Markers | High | [IRIS_SQL_VECTOR_LIMITATIONS.md](IRIS_SQL_VECTOR_LIMITATIONS.md), [IRIS_SQL_CHANGE_SUGGESTIONS.md](IRIS_SQL_CHANGE_SUGGESTIONS.md) |
| SQL-2 | TOP/FETCH FIRST Clauses Cannot Be Parameterized | High | [IRIS_SQL_VECTOR_LIMITATIONS.md](IRIS_SQL_VECTOR_LIMITATIONS.md), [IRIS_SQL_CHANGE_SUGGESTIONS.md](IRIS_SQL_CHANGE_SUGGESTIONS.md) |
| SQL-3 | LANGUAGE SQL Stored Procedures Lack DECLARE/SET Support | Medium | [IRIS_SQL_CHANGE_SUGGESTIONS.md](IRIS_SQL_CHANGE_SUGGESTIONS.md) |
| SQL-4 | SQL Projection Visibility Issues in INFORMATION_SCHEMA.ROUTINES | Medium | [POSTMORTEM_ODBC_SP_ISSUE.md](POSTMORTEM_ODBC_SP_ISSUE.md) |
| SQL-5 | Class Compilation Errors with Private Variables | Medium | [POSTMORTEM_ODBC_SP_ISSUE.md](POSTMORTEM_ODBC_SP_ISSUE.md) |
| SQL-6 | Inaccessibility of Core System Class %SYS.SQL.Schema | High | [IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md](IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md) |

**Recommended SQL Engine Enhancements:**

1. Enhance the SQL parser to allow parameter markers inside TO_VECTOR() function
2. Modify the TOP/FETCH FIRST clauses to accept parameter markers
3. Add support for DECLARE and SET statements in LANGUAGE SQL stored procedures
4. Improve SQL projection visibility and catalog updates in non-interactive environments
5. Fix compiler issues related to private variables in Dockerized environments
6. Ensure core system classes like %SYS.SQL.Schema are accessible in all standard contexts

### 7.2 Client Driver Issues

These issues relate to the DB-API, JDBC, and ODBC drivers that connect to IRIS:

| Issue ID | Description | Priority | Reference Documentation |
|----------|-------------|----------|-------------------------|
| DRV-1 | Client Drivers Rewrite Literals to :%qpar() | High | [IRIS_SQL_VECTOR_LIMITATIONS.md](IRIS_SQL_VECTOR_LIMITATIONS.md), [IRIS_SQL_CHANGE_SUGGESTIONS.md](IRIS_SQL_CHANGE_SUGGESTIONS.md) |
| DRV-2 | ODBC Driver Limitations with TO_VECTOR Function | Critical | [IRIS_SQL_VECTOR_LIMITATIONS.md](IRIS_SQL_VECTOR_LIMITATIONS.md) |
| DRV-3 | ODBC Stored Procedure Call Failures (-460 Error) | High | [POSTMORTEM_ODBC_SP_ISSUE.md](POSTMORTEM_ODBC_SP_ISSUE.md) |
| DRV-4 | DB-API Issues with System Utility Execution | Medium | [POSTMORTEM_ODBC_SP_ISSUE.md](POSTMORTEM_ODBC_SP_ISSUE.md) |
| DRV-5 | Incorrect Scalar Return Value Marshalling via ODBC | Medium | [IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md](IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md) |
| GEN-1 | Misleading/Ambiguous Error Messages (General) | Medium | [IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md](IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md) |

**Recommended Client Driver Enhancements:**

1. Modify drivers to stop rewriting literals when the SQL string is already complete
2. Fix ODBC driver handling of TO_VECTOR function to allow vector embedding operations
3. Improve error reporting for stored procedure calls to provide more actionable information
4. Enhance DB-API to provide better support for system utility execution
5. Ensure correct marshalling of scalar return values for projected SQL functions via ODBC
6. Improve clarity and specificity of error messages across drivers and engine for faster issue diagnosis

For detailed technical explanations of these issues, please refer to the referenced documentation files. Key bug reports and analyses are available in [IRIS_SQL_CHANGE_SUGGESTIONS.md](IRIS_SQL_CHANGE_SUGGESTIONS.md), [IRIS_SQL_VECTOR_LIMITATIONS.md](IRIS_SQL_VECTOR_LIMITATIONS.md), and [IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md](IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md).