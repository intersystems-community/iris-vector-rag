
# IRIS SQL – Vector‑Search Limitations Blocking RAG Pipelines  
*(expanded bug report with documentation references)*  

## Summary  
Modern RAG pipelines (ColBERT, HyDE, GraphRAG, etc.) need to pass an **embedding vector** and a **top‑k limit** to SQL at runtime.   
In InterSystems IRIS 2025.1 the natural query

```sql
SELECT doc_id,
       VECTOR_COSINE(embedding,
         TO_VECTOR(:embedding, 'DOUBLE', 768)
       ) AS score
FROM  SourceDocuments
ORDER BY score DESC
FETCH FIRST :top_k ROWS ONLY;
```

only works if the vector and limit are **string‑concatenated as literals**, defeating safe parameter binding.  

---

## Context  

* Vector functions were introduced in 2025.1 and showcased in “Using Vector Search” [^3].  
* `VECTOR_COSINE` and `TO_VECTOR` are documented separately [^1][^2].

---

## Issue 1 – `TO_VECTOR()` rejects parameter markers   

| Symptom | Example | References |
|---------|---------|------------|
| `SQLCODE -1, ") expected, : found"` whenever `?`, `:vec`, or `:%qpar` is used inside `TO_VECTOR()` | `VECTOR_COSINE(embedding, TO_VECTOR(:vec,'DOUBLE',768))` | The **TO_VECTOR** docs show only literal strings and note that *input data will always be of type **string**, unless you are using the function in Dynamic SQL* [^1]. |

Attempts (`cursor.execute(sql,[vector])`, named params, or none) all fail: the driver injects `:%qpar(1)`.

---

## Issue 2 – `TOP` / `FETCH FIRST` cannot be parameterised  

| Symptom | Example | References |
|---------|---------|------------|
| `SQLCODE -1, "Expression expected, : found"` | `SELECT TOP :top_k …` | **TOP** reference explains IRIS internally converts the integer to a cached parameter and shows only literals [^4]. |
| Same error with ANSI limit | `… FETCH FIRST :top_k ROWS ONLY` | Row‑limit syntaxes never illustrate parameter markers [^5]. |
| Community threads show devs revert to literals | | `:%qpar` stack‑trace in community post [^8]. |

---

## Issue 3 – `LANGUAGE SQL` stored procedures offer no escape hatch  

| Symptom | Example | References |
|---------|---------|------------|
| SPs fail with `TO_VECTOR(:x)` **or** `TOP :n` | `CREATE PROCEDURE … BEGIN SELECT TOP :n … END` | Stored‑proc guide omits `DECLARE` / `SET` and any vector examples; same “) expected” error results [^7]. |

---

## Issue 4 – Client drivers rewrite literals to `:%qpar()`  

Python and JDBC drivers replace embedded literals with `:%qpar(n)` even when **no** param list is supplied, creating misleading parse errors (see stack trace [^8]). A separate JDBC thread on SQL quoting highlights similar driver behaviour [^9]. The gateway article offers no mitigation [^10].

---

## Resulting Impact  

* Cannot build safe, parameterised server‑side vector search from Python, JDBC, or stored procedures.  
* Forces brittle string‑templating, blocking multi‑tenant APIs and RBAC‑secured queries.  
* Without a fix, RAG frameworks cannot target IRIS without custom code.

---

## Key Learnings  

1. **`TOP n` is implicitly parameterised inside IRIS**, so external bind variables are disallowed [^4].  
2. **`TO_VECTOR()` only accepts literal input** outside ObjectScript Dynamic SQL [^1].  
3. **`LANGUAGE SQL` blocks lack `DECLARE`/`SET`**, preventing run‑time composition [^7].  
4. **`%SQL.Statement` in ObjectScript *does* accept `?` parameters** for vectors [^6], but that relief is unavailable over DBAPI/JDBC.  
5. Driver‑side literal rewriting (`:%qpar`) is an old gotcha, resurfacing in community posts [^8].

---

## Proposed updates to *docs.intersystems.com*

| Page | Required change |
|------|-----------------|
| **TO_VECTOR (SQL)** | Add a *WARNING* box: “`TO_VECTOR()` does **not** accept host variables in client SQL.” |
| **TOP (SQL)** | Clarify that external bind variables are not supported because IRIS already caches the integer. |
| **FETCH (SQL)** | Note that the row‑limit argument must be a literal. |
| **Using Vector Search** | Provide end‑to‑end example using `%SQL.Statement` with bound vector + limit. |
| **Stored Procedures** | Document absence of `DECLARE` / `SET` in `LANGUAGE SQL`; show wrapper pattern. |
| **Driver Guides** | Add troubleshooting note on `:%qpar` rewriting and how to avoid it. |

---

## Priority fixes for engineering  

1. **Parser**: allow `?` / named parameters inside `TO_VECTOR()` and in `TOP`/`FETCH` clauses.  
2. **Driver**: stop rewriting literals when the SQL string is already complete.  
3. **`LANGUAGE SQL`**: support simple `DECLARE` & `SET`, or expose `%Prepare()` from within SPs.

---

## References  

[^1]: *TO_VECTOR (SQL)* – InterSystems SQL Reference.  
[^2]: *VECTOR_COSINE (SQL)* – InterSystems SQL Reference.  
[^3]: *Using Vector Search* – InterSystems SQL Guide.  
[^4]: *TOP (SQL)* – InterSystems SQL Reference.  
[^5]: *FETCH (SQL)* – InterSystems SQL Reference.  
[^6]: *%SQL.Statement* class reference – InterSystems IRIS Documentation.  
[^7]: *Defining and Using Stored Procedures* – InterSystems SQL Guide.  
[^8]: **Entity Framework Provider issue when "Support Delimited Identifiers" is turned off** – InterSystems Developer Community.  
[^9]: **HELP — SQL statements containing double quotes cannot be executed via JDBC** – InterSystems Developer Community.  
[^10]: **Mastering the JDBC SQL Gateway** – InterSystems Developer Community.  
