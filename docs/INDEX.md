# Project Documentation Index

Welcome to the RAG Templates project documentation. This page serves as a central index to help you navigate the various documents available.

## 1. Project Overview & Status

*   **[`README.md`](../README.md:1)**: The main entry point for the project. Provides an overview, setup instructions, current status, and links to key resources.
*   **[`PLAN_STATUS.md`](../PLAN_STATUS.md:1)**: Detailed breakdown of project phases, tasks, and their current completion status, including blockers.
*   **[`MANAGEMENT_SUMMARY.md`](MANAGEMENT_SUMMARY.md:1)**: A high-level summary for project managers, outlining goals, status, challenges, and recommendations, including a list of JIRA issues for InterSystems.
*   **[`PROJECT_COMPLETION_REPORT.md`](PROJECT_COMPLETION_REPORT.md:1)**: Summarizes project achievements, challenges, and outcomes. (Note: Currently reflects a **BLOCKED/INCOMPLETE** status).

## 2. Development Strategy & Lessons Learned

*   **[`DEVELOPMENT_STRATEGY_EVOLUTION.md`](DEVELOPMENT_STRATEGY_EVOLUTION.md:1)**: Chronicles the evolution of the project's development environment and database interaction strategy, explaining the pivot to client-side SQL.
*   **[`IRIS_VECTOR_SEARCH_LESSONS.md`](IRIS_VECTOR_SEARCH_LESSONS.md:1)**: Consolidates key findings and lessons learned from implementing vector search capabilities with InterSystems IRIS.

## 3. RAG Technique Implementations

*   **[`COLBERT_IMPLEMENTATION.md`](COLBERT_IMPLEMENTATION.md:1)**: Details for the ColBERT RAG technique.
*   **[`NODERAG_IMPLEMENTATION.md`](NODERAG_IMPLEMENTATION.md:1)**: Details for the NodeRAG technique using SQL CTEs.
*   **[`GRAPHRAG_IMPLEMENTATION.md`](GRAPHRAG_IMPLEMENTATION.md:1)**: Details for the GraphRAG technique using SQL CTEs.
*   **[`CONTEXT_REDUCTION_STRATEGY.md`](CONTEXT_REDUCTION_STRATEGY.md:1)**: Outlines various strategies for reducing document context size for LLMs.

*(Note: BasicRAG, HyDE, and CRAG implementations are included in the codebase but may not have separate detailed design documents in this `docs/` folder; refer to their respective pipeline scripts and general project plans.)*

## 4. IRIS Platform Issues & Workarounds

### 4.1. Vector SQL Limitations & Suggestions
*   **[`IRIS_SQL_VECTOR_LIMITATIONS.md`](IRIS_SQL_VECTOR_LIMITATIONS.md:1)**: Primary technical explanation of IRIS SQL vector operations limitations, focusing on the current `TO_VECTOR`/ODBC embedding load blocker.
*   **[`IRIS_SQL_VECTOR_OPERATIONS.md`](IRIS_SQL_VECTOR_OPERATIONS.md:1)**: Details the identified limitations and the implemented client-side SQL workarounds for *querying* vector data.
*   **[`IRIS_SQL_CHANGE_SUGGESTIONS.md`](IRIS_SQL_CHANGE_SUGGESTIONS.md:1)**: A comprehensive bug report and enhancement request document formatted for submission to InterSystems, detailing issues and proposed fixes.

### 4.2. Stored Procedure & Compilation Postmortems
*   **[`IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md`](IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md:1)**: Detailed postmortem on challenges with IRIS SQL Stored Procedure projection, caching, and automated ObjectScript class compilation (primarily based on IRIS 2024.1.2 experiences).
*   **[`POSTMORTEM_ODBC_SP_ISSUE.md`](POSTMORTEM_ODBC_SP_ISSUE.md:1)**: Focused postmortem on specific ODBC Stored Procedure call and compilation issues encountered (primarily with IRIS 2024.1.2).

## 5. Testing Guides

*   **[`TESTING.md`](TESTING.md:1)**: The primary and most up-to-date guide for all testing procedures, including unit, E2E, and real-data testing strategies and current limitations.
*   **[`REAL_DATA_TESTING_PLAN.md`](REAL_DATA_TESTING_PLAN.md:1)**: A detailed plan for testing with real PMC data and a real LLM (execution currently blocked).
*   **[`REAL_DATA_INTEGRATION.md`](REAL_DATA_INTEGRATION.md:1)**: Supplementary guide on integrating real PMC data, focusing on the `scripts_to_review/load_pmc_data.py` script.
*   **[`CONTEXT_REDUCTION_TESTING.md`](CONTEXT_REDUCTION_TESTING.md:1)**: Supplementary guide on a Testcontainer-based approach for testing context reduction strategies.
*   **[`REAL_DATA_TESTING.md`](REAL_DATA_TESTING.md:1)**: Supplementary guide on a Testcontainer-based approach for "real data" (text or mock/pre-loaded embeddings) testing.
*   **[`1000_DOCUMENT_TESTING.md`](1000_DOCUMENT_TESTING.md:1)**: Supplementary guide detailing specific scripts and `make` targets related to 1000+ document testing.
*   **[`REAL_PMC_1000_TESTING.md`](REAL_PMC_1000_TESTING.md:1)**: Supplementary guide, similar to above, focusing on the `run_with_real_pmc_data.sh` script for 1000+ document testing.
*   **[`LARGE_SCALE_TESTING.md`](LARGE_SCALE_TESTING.md:1)**: (Marked as potentially outdated) Describes approaches for testing with 1000+ up to 92,000+ documents.
*   **[`REAL_DATA_TESTING_DEBUG.md`](REAL_DATA_TESTING_DEBUG.md:1)**: Report on debugging and improvements made to the real data testing framework.

## 6. Benchmarking Guides

*   **[`BENCHMARK_SETUP.md`](BENCHMARK_SETUP.md:1)**: Instructions for setting up the environment for RAG benchmarks.
*   **[`BENCHMARK_EXECUTION_PLAN.md`](BENCHMARK_EXECUTION_PLAN.md:1)**: Step-by-step process for executing benchmarks.
*   **[`BENCHMARK_DATASETS.md`](BENCHMARK_DATASETS.md:1)**: Outlines key datasets and published results for reference.

## 7. Results & Reports

*   **[`E2E_TEST_RESULTS.md`](E2E_TEST_RESULTS.md:1)**: (Placeholder) Expected structure for end-to-end test results.
*   **[`REAL_DATA_TEST_RESULTS.md`](REAL_DATA_TEST_RESULTS.md:1)**: Report from an actual (failed) attempt to run tests with real data, highlighting the current blocker.
*   **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md:1)**: (Placeholder) Expected structure for benchmark results.

## 8. Implementation Plans

*   **[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md:1)**: The overall initial implementation plan for the project.
*   **[`DETAILED_IMPLEMENTATION_PLAN.md`](DETAILED_IMPLEMENTATION_PLAN.md:1)**: A more granular breakdown of implementation tasks and details.

## 9. Archival / Outdated

*   **[`NEXT_TECHNIQUES_SUMMARY.md`](NEXT_TECHNIQUES_SUMMARY.md:1)**: (Marked as ARCHIVAL - OUTDATED) Early plan for RAG techniques that have since been implemented.