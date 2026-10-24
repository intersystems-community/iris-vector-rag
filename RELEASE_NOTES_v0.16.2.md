# Release Notes: iris-vector-rag v0.16.2

Released 2026-10-24 as `v0.16.2`.

A correctness and efficiency release for vector search. The query vector is now bound as a
parameter, `TO_VECTOR(?, <type>, <dim>)`, instead of being written into the SQL text. The
results are the same; each query reuses one prepared statement instead of compiling a new
one.

The old reason for writing the vector in, "`TO_VECTOR()` cannot take a parameter marker",
does not hold on IRIS 2026.1 (Build 234U) or 2026.3 with intersystems-irispython 5.3.1 and
5.4.1. That was measured on FLOAT and DOUBLE columns, including the cached second run. Two
forms still fail, and the library avoids both:

- `TO_VECTOR(?, ?, ?)`: type and dimension as parameters, SQLCODE -262;
- the untyped `TO_VECTOR(?)` compared with a DOUBLE column, SQLCODE -259.

## What changed

- **Vector store and CRAG:** `IRISVectorStore` vector search and CRAG's chunk search bind the
  vector.
- **ColBERT:** the in-database MaxSim (`MaxSimInDB.bulk_fetch_maxsim`, `indb_maxsim`) binds
  each query-token vector, so every token reuses one statement.
- **SQL helpers:** `vector_sql_utils.build_bound_vector_dot_sql` is new, and
  `execute_safe_vector_search` takes parameters. The literal-form `build_safe_vector_dot_sql`
  stays for outside callers.
- **Docs:** the two documents that carried the old claim have a dated correction.

## Upgrading

Nothing to do. Code that builds its own vector SQL can switch to the bound form; the
literal form keeps working.
