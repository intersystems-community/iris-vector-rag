# IRIS iris.dbapi: Use SELECT not CALL to invoke stored procedures

## Symptoms
```
iris.dbapi.ProgrammingError: <SQL ERROR>; Details: [SQLCODE: <-51>:<SQL statement expected>]
[%msg: < An SQL statement expected, IDENTIFIER found>]
```
When calling `cursor.execute("CALL pkg.ProcedureName(?,?,?)", [...])` from the Python `iris.dbapi` driver.

## Root Cause
IRIS `iris.dbapi` does not support the `CALL` statement syntax for invoking SQL stored procedures.
`CALL` is valid in ODBC/JDBC but not in the IRIS DBAPI Python driver.

## Solution
Use `SELECT` instead of `CALL`:
```python
# ❌ WRONG — raises SQLCODE -51
cursor.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", [q_json, top_k, n_probe])

# ✅ CORRECT — works with iris.dbapi
cursor.execute("SELECT RAG.ColBERTSearch_Search(?, ?, ?)", [q_json, top_k, n_probe])
row = cursor.fetchone()
result = row[0]  # single-column result
```

## Additional Quirk: Read row[0] BEFORE cursor.close()
IRIS DataRow values become inaccessible after the cursor is closed.
Always read all values from a row before calling `cursor.close()`:
```python
cur = conn.cursor()
try:
    cur.execute("SELECT RAG.ColBERTSearch_Search(?, ?, ?)", [q_json, top_k, n_probe])
    row = cur.fetchone()
    value = row[0]   # ← read HERE, inside the try block
finally:
    cur.close()
# Use `value` here — safe
```

## Tags
iris, iris.dbapi, stored-procedure, sql, CALL, SELECT, SQLCODE-51
