# Known Issues

## 1. SQLCODE: <-104> Error During Token Embedding Insertion in ReconciliationController

**Date Identified:** 2025-06-11

**Affected Component:** [`iris_rag/controllers/reconciliation.py`](iris_rag/controllers/reconciliation.py) - specifically the `_process_single_document_embeddings` method.

**Affected Tests:** [`tests/test_reconciliation_contamination_scenarios.py::test_scenario_incomplete_embeddings`](tests/test_reconciliation_contamination_scenarios.py) (and potentially other tests relying on token embedding regeneration by the `ReconciliationController`).

**Symptoms:**
- The `ragctl run` command, when attempting to remediate "low_diversity" or "incomplete_token_embeddings" by regenerating and inserting token embeddings, fails with an exit code 2.
- The `stderr` log shows an `SQLCODE: <-104>` error: `[%msg: <Field 'RAG.DocumentTokenEmbeddings.token_embedding' (value 'HASH_VALUE@$vector') failed validation>]`.
- This error occurs during the `INSERT INTO RAG.DocumentTokenEmbeddings ...` SQL execution.
- As a result, documents that have their embeddings cleared by the reconciliation process do not get new embeddings inserted, leading to a state of "missing_embeddings".

**Problem Description:**
The InterSystems IRIS database throws an `SQLCODE: <-104>` (Field validation failed) error when the `ReconciliationController` attempts to insert newly generated token embeddings into the `RAG.DocumentTokenEmbeddings` table. The error message suggests that IRIS is attempting to validate an internal hash representation of a vector (e.g., `57D7D0FEAD1206E7425DEAF14BD39DD0@$vector`) rather than the string literal intended for conversion by functions like `TO_VECTOR(?)` or `TO_VECTOR($LISTFROMSTRING(?, ','))`.

**Troubleshooting Steps Taken:**
Several SQL syntax variations for vector insertion were attempted within the `_process_single_document_embeddings` method, all resulting in the same error:
1.  Parameterized `TO_VECTOR(?)` with a bracketed string (e.g., `VALUES (?, ?, ?, TO_VECTOR(?))` with `embedding_str = f"[{','.join(map(str, embedding))}]"`). This pattern works in other parts of the codebase (e.g., [`iris_rag/validation/orchestrator.py`](iris_rag/validation/orchestrator.py)).
2.  Parameterized `TO_VECTOR($LISTFROMSTRING(?, ','))` with a plain comma-separated string (e.g., `VALUES (?, ?, ?, TO_VECTOR($LISTFROMSTRING(?, ',')))` with `embedding_str = ','.join(map(str, embedding))`).
3.  Direct f-string construction of `TO_VECTOR('[f1,f2,...]')` into the SQL query.

The persistence of this error across these methods suggests a subtle issue, possibly within the IRIS DBAPI driver's handling of string parameters that resemble vector literals, especially in the looped execution context of this specific controller method. The driver might be pre-processing the Python string into an IRIS internal vector type *before* the SQL conversion function is called. These functions then fail because they receive this internal type instead of the expected string.

**Current Status:**
- The reconciliation framework can correctly identify data drift issues (including "low_diversity" and "incomplete_token_embeddings").
- The framework can successfully clear existing problematic embeddings.
- The remediation step of inserting new token embeddings is blocked by this `SQLCODE: <-104>` error.

**Recommendation:**
This issue is added to the backlog for further investigation. This may require:
- Deeper debugging of the InterSystems IRIS DBAPI driver behavior.
- Consultation regarding specific IRIS version nuances with vector type handling in parameterized queries, particularly within looped execution contexts.
- Exploring alternative, potentially lower-level, methods for vector insertion if the standard SQL conversion functions prove problematic in this context.