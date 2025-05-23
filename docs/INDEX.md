# Project Documentation Index

Welcome to the RAG Templates project documentation. This page serves as a central index to help you navigate the various documents available.

## 1. Project Overview & Status

*   **[`README.md`](../README.md)**: The main entry point for the project. Provides an overview, setup instructions, current status, and links to key resources.
*   **[`PLAN_STATUS.md`](../PLAN_STATUS.md)**: Detailed breakdown of project phases, tasks, and their current completion status, including blockers.
*   **[`MANAGEMENT_SUMMARY.md`](MANAGEMENT_SUMMARY.md)**: A high-level summary for project managers, outlining goals, status, challenges, and recommendations, including a list of JIRA issues for InterSystems.
*   **[`PROJECT_COMPLETION_REPORT.md`](PROJECT_COMPLETION_REPORT.md)**: Summarizes project achievements, challenges, and outcomes. (Note: Currently reflects a **BLOCKED/INCOMPLETE** status).

## 2. Development Strategy & Lessons Learned

*   **[`DEVELOPMENT_STRATEGY_EVOLUTION.md`](DEVELOPMENT_STRATEGY_EVOLUTION.md)**: Chronicles the evolution of the project's development environment and database interaction strategy, explaining the pivot to client-side SQL.
*   **[`IRIS_VECTOR_SEARCH_LESSONS.md`](IRIS_VECTOR_SEARCH_LESSONS.md)**: Consolidates key findings and lessons learned from implementing vector search capabilities with InterSystems IRIS.

## 3. RAG Technique Implementations

*   **[`COLBERT_IMPLEMENTATION.md`](COLBERT_IMPLEMENTATION.md)**: Details for the ColBERT RAG technique.
*   **[`NODERAG_IMPLEMENTATION.md`](NODERAG_IMPLEMENTATION.md)**: Details for the NodeRAG technique using SQL CTEs.
*   **[`GRAPHRAG_IMPLEMENTATION.md`](GRAPHRAG_IMPLEMENTATION.md)**: Details for the GraphRAG technique using SQL CTEs.
*   **[`CONTEXT_REDUCTION_STRATEGY.md`](CONTEXT_REDUCTION_STRATEGY.md)**: Outlines various strategies for reducing document context size for LLMs.

*(Note: BasicRAG, HyDE, and CRAG implementations are included in the codebase but may not have separate detailed design documents in this `docs/` folder; refer to their respective pipeline scripts and general project plans.)*

## 4. IRIS Platform Issues & Workarounds

### 4.1. Vector SQL Limitations & Suggestions
*   **[`VECTOR_SEARCH_DOCUMENTATION_INDEX.md`](VECTOR_SEARCH_DOCUMENTATION_INDEX.md)**: A central index for all vector search documentation, providing an overview of implementation approaches and their relationships.
*   **[`VECTOR_SEARCH_CONFLUENCE_PAGE.md`](VECTOR_SEARCH_CONFLUENCE_PAGE.md)**: A comprehensive Confluence page for Quality Development and Development teams, with technical details, JIRAs, and clear recommendations.
*   **[`IRIS_SQL_VECTOR_LIMITATIONS.md`](IRIS_SQL_VECTOR_LIMITATIONS.md)**: Primary technical explanation of IRIS SQL vector operations limitations, focusing on the current `TO_VECTOR`/ODBC embedding load blocker.
*   **[`VECTOR_SEARCH_TECHNICAL_DETAILS.md`](VECTOR_SEARCH_TECHNICAL_DETAILS.md)**: Comprehensive technical details about vector search implementation, including environment information, client library behavior, and code examples.
*   **[`VECTOR_SEARCH_ALTERNATIVES.md`](VECTOR_SEARCH_ALTERNATIVES.md)**: Investigation findings on alternative approaches to vector search in IRIS, focusing on solutions from langchain-iris and llama-iris.
*   **[`HNSW_INDEXING_RECOMMENDATIONS.md`](HNSW_INDEXING_RECOMMENDATIONS.md)**: Recommendations for implementing HNSW indexing with InterSystems IRIS for high-performance vector search with large document collections.
*   **[`HNSW_VIEW_TEST_RESULTS.md`](HNSW_VIEW_TEST_RESULTS.md)**: Results of testing view-based approach for HNSW indexing with IRIS 2025.1, confirming that it doesn't work.
*   **[`IRIS_SQL_VECTOR_OPERATIONS.md`](IRIS_SQL_VECTOR_OPERATIONS.md)**: Details the identified limitations and the implemented client-side SQL workarounds for *querying* vector data.
*   **[`IRIS_SQL_CHANGE_SUGGESTIONS.md`](IRIS_SQL_CHANGE_SUGGESTIONS.md)**: A comprehensive bug report and enhancement request document formatted for submission to InterSystems, detailing issues and proposed fixes.
*   **[`iris_sql_vector_limitations_bug_report.md`](iris_sql_vector_limitations_bug_report.md)**: (Original root-level bug report) A concise summary of the vector limitations, suitable for quick reference or external sharing.

### 4.2. Stored Procedure & Compilation Postmortems
*   **[`IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md`](IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md)**: Detailed postmortem on challenges with IRIS SQL Stored Procedure projection, caching, and automated ObjectScript class compilation (primarily based on IRIS 2024.1.2 experiences).
*   **[`POSTMORTEM_ODBC_SP_ISSUE.md`](POSTMORTEM_ODBC_SP_ISSUE.md)**: Focused postmortem on specific ODBC Stored Procedure call and compilation issues encountered (primarily with IRIS 2024.1.2).

## 5. Testing Guides

*   **[`TESTING.md`](TESTING.md)**: The primary and most up-to-date guide for all testing procedures, including unit, E2E, and real-data testing strategies and current limitations.
*   **[`REAL_DATA_TESTING_PLAN.md`](REAL_DATA_TESTING_PLAN.md)**: A detailed plan for testing with real PMC data and a real LLM (execution currently blocked).
*   **[`REAL_DATA_INTEGRATION.md`](REAL_DATA_INTEGRATION.md)**: Supplementary guide on integrating real PMC data, focusing on the `scripts_to_review/load_pmc_data.py` script.
*   **[`CONTEXT_REDUCTION_TESTING.md`](CONTEXT_REDUCTION_TESTING.md)**: Supplementary guide on a Testcontainer-based approach for testing context reduction strategies.
*   **[`REAL_DATA_TESTING.md`](REAL_DATA_TESTING.md)**: Supplementary guide on a Testcontainer-based approach for "real data" (text or mock/pre-loaded embeddings) testing.
*   **[`1000_DOCUMENT_TESTING.md`](1000_DOCUMENT_TESTING.md)**: Supplementary guide detailing specific scripts and `make` targets related to 1000+ document testing.
*   **[`REAL_PMC_1000_TESTING.md`](REAL_PMC_1000_TESTING.md)**: Supplementary guide, similar to above, focusing on the `run_with_real_pmc_data.sh` script for 1000+ document testing.
*   **[`LARGE_SCALE_TESTING.md`](LARGE_SCALE_TESTING.md)**: (Marked as potentially outdated) Describes approaches for testing with 1000+ up to 92,000+ documents.
*   **[`REAL_DATA_TESTING_DEBUG.md`](REAL_DATA_TESTING_DEBUG.md)**: Report on debugging and improvements made to the real data testing framework.

## 6. Benchmarking Guides

*   **[`BENCHMARK_SETUP.md`](BENCHMARK_SETUP.md)**: Instructions for setting up the environment for RAG benchmarks.
*   **[`BENCHMARK_EXECUTION_PLAN.md`](BENCHMARK_EXECUTION_PLAN.md)**: Step-by-step process for executing benchmarks.
*   **[`BENCHMARK_DATASETS.md`](BENCHMARK_DATASETS.md)**: Outlines key datasets and published results for reference.

## 7. Results & Reports

*   **[`E2E_TEST_RESULTS.md`](E2E_TEST_RESULTS.md)**: (Placeholder) Expected structure for end-to-end test results.
*   **[`REAL_DATA_TEST_RESULTS.md`](REAL_DATA_TEST_RESULTS.md)**: Report from an actual (failed) attempt to run tests with real data, highlighting the current blocker.
*   **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)**: (Placeholder) Expected structure for benchmark results.

## 8. Implementation Plans

*   **[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)**: The overall initial implementation plan for the project.
*   **[`DETAILED_IMPLEMENTATION_PLAN.md`](DETAILED_IMPLEMENTATION_PLAN.md)**: A more granular breakdown of implementation tasks and details.

## 9. Archival / Outdated

*   **[`NEXT_TECHNIQUES_SUMMARY.md`](NEXT_TECHNIQUES_SUMMARY.md)**: (Marked as ARCHIVAL - OUTDATED) Early plan for RAG techniques that have since been implemented.