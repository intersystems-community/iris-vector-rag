# Quickstart: scoped SDK search

## As a consumer

```objectscript
// Point the SDK at your table, including its metadata column.
Do ##class(RAG.SDK.Bridge).SetDefaultTable("MyApp.Memories", "mem_id", "body", "embedding", "attrs")

// Ask for the nearest 5 rows this agent is allowed to see.
Set scope = "agent_id = 'agent-7' AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)"
Set json = ##class(RAG.SDK.Search).VectorSearch(queryVec, 5, scope)

If $Extract(json, 1) '= "[" {
    Set err = {}.%FromJSON(json)
    // err.error names what went wrong: a bad predicate, an unresolvable vector type,
    // or a SQL failure. It is never an empty array pretending to be "no matches".
}

Set hits = [].%FromJSON(json)
Set n = hits.%Size()
For i = 0:1:(n-1) {
    Set hit = hits.%Get(i)
    // hit.id, hit.text, hit.score, and hit.metadata when the table has a metadata column
}
```

Check a table before attaching it:

```objectscript
Write ##class(RAG.SDK.Search).GetVectorType("MyApp.Memories", "embedding")  // FLOAT
```

## Running the tests

No source is mounted into the container, so the classes must be copied in and compiled on the
server before the tests mean anything.

```bash
CT=iris-vector-rag-iris
docker exec "$CT" bash -lc 'rm -rf /tmp/sdk085 && mkdir -p /tmp/sdk085/src /tmp/sdk085/test'
docker cp iris_src/src/RAG "$CT":/tmp/sdk085/src/
docker cp tests/objectscript/RAG "$CT":/tmp/sdk085/test/

printf 'zn "USER"\ndo $SYSTEM.OBJ.LoadDir("/tmp/sdk085",,,1)\ndo ##class(%%UnitTest.Manager).RunTest("RAG.SDK.Test.SearchTest","/loadudl/nodelete")\nhalt\n' \
  | docker exec -i "$CT" iris session IRIS
```

`bash .specify/scripts/bash/check-container.sh` first — it parses the container name and port
out of `CLAUDE.md`, so it is the one place that knows which instance is correct.

## What this feature does not do

- It does not change `RAG.SDK.Schema.Initialize`, which still creates `VECTOR(DOUBLE, N)`
  tables. Search now works against those *and* against the `FLOAT` tables the Python paths
  create.
- It does not touch the consumer repository.
