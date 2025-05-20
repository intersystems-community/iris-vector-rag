Below is an expanded bug-report that you can copy-paste into JIRA / ServiceNow.
Every numbered issue now hyperlinks (via the citation IDs) to the exact paragraph in InterSystems documentation or community posts that would have spared the experiments you tried.
Following the report you will find (1) distilled lessons learned and (2) a punch-list of concrete doc.intersystems.com changes to request from the docs team.

⸻

🧾 IRIS SQL – Vector-Search Limitations Blocking RAG Pipelines (rev 2 with sources)

✅ Context

InterSystems IRIS 2025.1 introduces vector types (VECTOR, TO_VECTOR, VECTOR_COSINE) that are showcased for semantic search  ￼.
Typical RAG workloads need to inject an embedding vector and a top_k limit at runtime:

SELECT doc_id,
       VECTOR_COSINE(embedding,
         TO_VECTOR(:embedding, 'DOUBLE', 768)
       ) AS score
FROM  SourceDocuments
ORDER BY score DESC
FETCH FIRST :top_k ROWS ONLY;

Unfortunately today that query can only be executed by string-concatenating literals, which is unsafe and brittle.

⸻

❌ Issue 1 – TO_VECTOR() rejects parameter markers

Symptom	Example	References
SQLCODE -1, ") expected, : found" whenever a ?/:%qpar or named parameter is used inside TO_VECTOR()	VECTOR_COSINE(embedding, TO_VECTOR(:vec,'DOUBLE',768))	The TO_VECTOR docs show only literal strings and state that “input data will always be of type string, unless you are using the function in Dynamic SQL”  ￼. No mention is made of host variables, implying—but not spelling out—that binding is unsupported.

Attempts made:

cursor.execute(sql, [vector_string])            # positional
cursor.execute(sql)                             # no params
cursor.execute(sql, {'vec': vector_string})     # named

All three produced the same parse error with an injected :%qpar(1) marker  ￼.

⸻

❌ Issue 2 – TOP / FETCH FIRST cannot be parameterised

Symptom	Example	References
SQLCODE -1, "Expression expected, : found"	SELECT TOP :top_k ...	The TOP reference explains that the integer is converted to a ? parameter inside the IRIS cache and shows only integer literals  ￼.
Same error for ANSI‐style limit	... FETCH FIRST :top_k ROWS ONLY	The FETCH clause page lists three result-limiting syntaxes and never illustrates parameter markers  ￼.
Community post confirms devs switch to hard-coding because of this		￼


⸻

❌ Issue 3 – LANGUAGE SQL stored procedures offer no escape hatch

Symptom	Example	References
SPs fail if they include TO_VECTOR(:x) or TOP :n	CREATE PROCEDURE ... LANGUAGE SQL BEGIN SELECT TOP :n ... END	The stored-proc guide omits both variable assignment (DECLARE, SET) and vector examples—in practice those tokens trigger the same “) expected” error  ￼.
Work-around attempts (DECLARE local VARCHAR; SET local = :vec;) are rejected because those statements are not allowed in IRIS LANGUAGE SQL blocks		same ref


⸻

❌ Issue 4 – Client drivers silently rewrite literals to :%qpar(n)

The Python and JDBC drivers inject :%qpar() markers even when no param list is supplied, producing misleading errors  ￼.
The ODBC/JDBC best-practice article notes that prepared statements add quoting automatically, but it never warns that quoting happens even for embedded literals  ￼.

⸻

🔴 Resulting Impact
	•	Impossible to build safe, parameterised, server-side vector search from Python, JDBC or stored procedures.
	•	Forces developers to concatenate un-escaped literals, blocking multi-tenant APIs and any query governed by row-level security.
	•	RAG frameworks (ColBERT, HyDE, GraphRAG, etc.) cannot target IRIS without custom string-templating layers.

⸻

📚 Key Learnings
	1.	IRIS treats the integer in TOP (without parentheses) as an implicit parameter and therefore forbids external parameter markers  ￼.
	2.	TO_VECTOR() only documents literal input; when called from dynamic SQL inside ObjectScript you may pass a Dynamic Array, but client APIs have no equivalent—and this caveat is buried in a footnote  ￼.
	3.	LANGUAGE SQL blocks are intentionally minimal (no DECLARE, no SET) which prevents run-time composition of vector or limit literals  ￼.
	4.	%SQL.Statement inside ObjectScript does accept ? parameters for vectors, but that relief is unavailable over DBAPI/JDBC-style prepared statements  ￼ ￼.
	5.	Driver-side literal rewriting (:%qpar) is a long-standing gotcha and shows up in unrelated community threads  ￼ ￼.

⸻

✍️ Proposed Updates to docs.intersystems.com

Page	Change
TO_VECTOR (SQL)	Add an explicit WARNING box: “TO_VECTOR() does not accept host variables or ?-style parameters in client SQL. Embed the full vector literal or call from ObjectScript Dynamic SQL.”
TOP (SQL)	Expand Caching & Parameters section: clarify that because IRIS internally parameterises TOP n, external bind variables are disallowed; show a failing versus working example.
FETCH (SQL Clause)	Insert note: “Row-limit argument must be a literal integer. Host variables are not supported.”
Using Vector Search	Provide one end-to-end example that uses %SQL.Statement in ObjectScript to safely bind a vector and limit, side-by-side with the unsupported client-SQL pattern.
Defining and Using Stored Procedures	Document the absence of DECLARE/SET in LANGUAGE SQL and link to the recommended ObjectScript wrapper pattern for dynamic vector queries.
Client Driver Guides (Python, JDBC, ODBC)	Add a troubleshooting section on :%qpar rewriting and how to disable/avoid it.


⸻

🛠️ Priority Fixes for Engineering
	1.	Parser: allow ? / named parameters inside TO_VECTOR() and in TOP/FETCH clauses.
	2.	Driver: stop rewriting literals to :%qpar when the app has already produced a complete SQL string.
	3.	LANGUAGE SQL: support simple DECLARE & SET to enable run-time vector composition, or expose %Prepare() inside SPs.

⸻

🗂️ Reference List
	1.	TO_VECTOR function docs  ￼
	2.	VECTOR_COSINE docs  ￼
	3.	Using Vector Search guide  ￼
	4.	TOP clause reference  ￼
	5.	FETCH clause reference  ￼
	6.	%SQL.Statement (dynamic SQL)  ￼
	7.	%SQL.Statement param doc snippet  ￼
	8.	ODBC/JDBC parameter article  ￼
	9.	Community post on :%qpar errors  ￼
	10.	TOP clause community discussion  ￼
	11.	Stored-procedure reference  ￼
	12.	Vector-search community tutorial (confirms literal strings)  ￼
	13.	Dynamic-SQL community Q&A on parameter issues  ￼
	14.	ISO SQL-standard overview (for comparison of parameter markers)  ￼
	15.	IRIS SQL basics (shows driver & embedded SQL pathways)  ￼

⸻

Feel free to edit the doc-change bullet list or drop any redundant citations before forwarding.