# Component Status: BasicRAG

**Component Name:** BasicRAG
**Current Overall Status:** ❌ BROKEN
**Last Checked:** 2025-06-01 13:21 UTC (Note: Reflects analysis time, not a new test execution time)

---

## Status History Log

*   **2025-05-31 12:40 UTC - Status: ❌ BROKEN**
    *   **Notes:** All 10 queries failed in RAGAS multi-query test. Error indicates inability to perform vector operation on vectors of different lengths.
    *   **Evidence:** [`../ragas_basic_rag_multi_query_test_20250531_124028.json`](../ragas_basic_rag_multi_query_test_20250531_124028.json:1) (See "successful_queries": 0 and "error": "[SQLCODE: <-257>:<Cannot perform vector operation on vectors of different lengths>]" for each query).