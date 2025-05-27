# Project Documentation Index

Welcome to the RAG Templates project documentation. This page serves as a central index to help you navigate the various documents available.

## 1. Project Overview & Status

*   **[`README.md`](../README.md)**: The main entry point for the project. Provides an overview, setup instructions, current status, and links to key resources.
*   **[`PLAN_STATUS.md`](../PLAN_STATUS.md)**: Detailed breakdown of project phases, tasks, and their current completion status.
*   **[`MANAGEMENT_SUMMARY.md`](MANAGEMENT_SUMMARY.md)**: ✅ **UPDATED** - High-level summary for project managers, outlining successful completion, achievements, and current capabilities.
*   **[`PROJECT_COMPLETION_REPORT.md`](PROJECT_COMPLETION_REPORT.md)**: ✅ **UPDATED** - Comprehensive report documenting successful project completion with real PMC data validation.
*   **[`REAL_DATA_VECTOR_SUCCESS_REPORT.md`](../REAL_DATA_VECTOR_SUCCESS_REPORT.md)**: ✅ **NEW** - Success report documenting achievement of real PMC data integration with vector operations.

## 2. Development Strategy & Lessons Learned

*   **[`DEVELOPMENT_STRATEGY_EVOLUTION.md`](DEVELOPMENT_STRATEGY_EVOLUTION.md)**: Chronicles the evolution of the project's development environment and database interaction strategy, explaining the pivot to client-side SQL.
*   **[`IRIS_VECTOR_SEARCH_LESSONS.md`](IRIS_VECTOR_SEARCH_LESSONS.md)**: Consolidates key findings and lessons learned from implementing vector search capabilities with InterSystems IRIS.

## 3. RAG Technique Implementations

*   **[`COLBERT_IMPLEMENTATION.md`](COLBERT_IMPLEMENTATION.md)**: Details for the ColBERT RAG technique.
*   **[`NODERAG_IMPLEMENTATION.md`](NODERAG_IMPLEMENTATION.md)**: Details for the NodeRAG technique using SQL CTEs.
*   **[`GRAPHRAG_IMPLEMENTATION.md`](GRAPHRAG_IMPLEMENTATION.md)**: Details for the GraphRAG technique using SQL CTEs.
*   **[`CONTEXT_REDUCTION_STRATEGY.md`](CONTEXT_REDUCTION_STRATEGY.md)**: Outlines various strategies for reducing document context size for LLMs.

*(Note: BasicRAG, HyDE, and CRAG implementations are included in the codebase but may not have separate detailed design documents in this `docs/` folder; refer to their respective pipeline scripts and general project plans.)*

## 4. Vector Search Implementation & Lessons

### 4.1. Current Working Solutions
*   **[`VECTOR_SEARCH_CONFLUENCE_PAGE.md`](VECTOR_SEARCH_CONFLUENCE_PAGE.md)**: ✅ **UPDATED** - Comprehensive guide showing current working solutions, achievements, and technical lessons learned.
*   **[`HNSW_INDEXING_RECOMMENDATIONS.md`](HNSW_INDEXING_RECOMMENDATIONS.md)**: Production scaling recommendations with HNSW indexing for Enterprise Edition (14x performance improvement).
*   **[`HNSW_VIEW_TEST_RESULTS.md`](HNSW_VIEW_TEST_RESULTS.md)**: Test results confirming view-based approach limitations, validating dual-table architecture recommendation.
*   **[`IRIS_VERSION_MIGRATION_2025.md`](IRIS_VERSION_MIGRATION_2025.md)**: Migration guide and version-specific improvements in IRIS 2025.1.
*   **[`VECTOR_SEARCH_COMMUNITY_VS_LICENSED_COMPARISON_REPORT.md`](VECTOR_SEARCH_COMMUNITY_VS_LICENSED_COMPARISON_REPORT.md)**: ✅ **NEW** - Comprehensive comparison between Community and Licensed editions for vector search capabilities.
*   **[`DATABASE_STATE_ASSESSMENT_REPORT.md`](DATABASE_STATE_ASSESSMENT_REPORT.md)**: ✅ **NEW** - Current database state assessment and infrastructure validation report.

### 4.2. Technical Implementation Details
*   **[`VECTOR_SEARCH_TECHNICAL_DETAILS.md`](VECTOR_SEARCH_TECHNICAL_DETAILS.md)**: Comprehensive technical details about vector search implementation, including environment information and code examples.
*   **[`IRIS_VECTOR_SEARCH_LESSONS.md`](IRIS_VECTOR_SEARCH_LESSONS.md)**: Key lessons learned and best practices for IRIS vector search implementation.
*   **[`VECTOR_SEARCH_DOCUMENTATION_INDEX.md`](VECTOR_SEARCH_DOCUMENTATION_INDEX.md)**: Central index for all vector search documentation and implementation approaches.
*   **[`VECTOR_SEARCH_DOCUMENTATION_PLAN.md`](VECTOR_SEARCH_DOCUMENTATION_PLAN.md)**: ✅ **NEW** - Strategic plan for organizing and maintaining vector search documentation.
*   **[`VECTOR_SEARCH_JIRA_IMPROVEMENTS.md`](VECTOR_SEARCH_JIRA_IMPROVEMENTS.md)**: ✅ **NEW** - Recommendations for IRIS vector search improvements and feature requests.

### 4.3. Historical Context & Limitations (Preserved for Reference)
*   **[`IRIS_SQL_VECTOR_LIMITATIONS.md`](IRIS_SQL_VECTOR_LIMITATIONS.md)**: Historical documentation of limitations (largely resolved with current approach).
*   **[`VECTOR_SEARCH_ALTERNATIVES.md`](VECTOR_SEARCH_ALTERNATIVES.md)**: Investigation findings on alternative approaches and workarounds.
*   **[`IRIS_SQL_VECTOR_OPERATIONS.md`](IRIS_SQL_VECTOR_OPERATIONS.md)**: Details on client-side SQL workarounds for vector operations.
*   **[`IRIS_SQL_CHANGE_SUGGESTIONS.md`](IRIS_SQL_CHANGE_SUGGESTIONS.md)**: Enhancement requests for InterSystems (historical context).
*   **[`iris_sql_vector_limitations_bug_report.md`](iris_sql_vector_limitations_bug_report.md)**: Original bug report (historical reference).

### 4.4. Historical Issues (IRIS 2024.1.2 - Resolved)
*   **[`IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md`](IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md)**: Historical postmortem on IRIS 2024.1.2 stored procedure issues (resolved with current approach).
*   **[`POSTMORTEM_ODBC_SP_ISSUE.md`](POSTMORTEM_ODBC_SP_ISSUE.md)**: Historical ODBC stored procedure issues (resolved with client-side approach).

## 5. Testing & Validation

### 5.1. Current Testing Status
*   **[`TESTING.md`](TESTING.md)**: Primary testing guide covering all procedures, including successful real-data testing strategies.
*   **[`REAL_DATA_TESTING_PLAN.md`](REAL_DATA_TESTING_PLAN.md)**: ✅ **EXECUTED** - Plan for testing with real PMC data (successfully completed).
*   **[`1000_DOCUMENT_TESTING.md`](1000_DOCUMENT_TESTING.md)**: ✅ **VALIDATED** - Guide for 1000+ document testing (successfully executed).
*   **[`REAL_PMC_1000_TESTING.md`](REAL_PMC_1000_TESTING.md)**: ✅ **COMPLETED** - Real PMC data testing with 1000+ documents.

### 5.2. Implementation Guides
*   **[`REAL_DATA_INTEGRATION.md`](REAL_DATA_INTEGRATION.md)**: Guide on integrating real PMC data with the current working approach.
*   **[`CONTEXT_REDUCTION_TESTING.md`](CONTEXT_REDUCTION_TESTING.md)**: Testing strategies for context reduction approaches.
*   **[`REAL_DATA_TESTING.md`](REAL_DATA_TESTING.md)**: Real data testing methodologies and best practices.
*   **[`LARGE_SCALE_TESTING.md`](LARGE_SCALE_TESTING.md)**: Approaches for testing with large document collections (1K-92K+ documents).

## 6. Benchmarking & Performance

### 6.1. Benchmarking Framework
*   **[`BENCHMARK_SETUP.md`](BENCHMARK_SETUP.md)**: Instructions for setting up the environment for RAG benchmarks.
*   **[`BENCHMARK_EXECUTION_PLAN.md`](BENCHMARK_EXECUTION_PLAN.md)**: Step-by-step process for executing benchmarks.
*   **[`BENCHMARK_DATASETS.md`](BENCHMARK_DATASETS.md)**: Key datasets and published results for reference.

### 6.2. Performance Results
*   **[`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md)**: ✅ **READY** - Framework ready for full benchmark execution with real LLM integration.

## 7. Results & Reports

### 7.1. Success Reports
*   **[`REAL_DATA_VECTOR_SUCCESS_REPORT.md`](../REAL_DATA_VECTOR_SUCCESS_REPORT.md)**: ✅ **COMPLETE** - Comprehensive success report with real PMC data validation.
*   **[`E2E_TEST_RESULTS.md`](E2E_TEST_RESULTS.md)**: ✅ **VALIDATED** - End-to-end test results with real data.

### 7.2. Historical Reports
*   **[`REAL_DATA_TEST_RESULTS.md`](REAL_DATA_TEST_RESULTS.md)**: Historical report from early testing attempts (superseded by success report).

## 8. Implementation Plans

*   **[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)**: The overall initial implementation plan for the project.
*   **[`DETAILED_IMPLEMENTATION_PLAN.md`](DETAILED_IMPLEMENTATION_PLAN.md)**: A more granular breakdown of implementation tasks and details.

## 9. Archival / Outdated

*   **[`NEXT_TECHNIQUES_SUMMARY.md`](NEXT_TECHNIQUES_SUMMARY.md)**: (Marked as ARCHIVAL - OUTDATED) Early plan for RAG techniques that have since been implemented.