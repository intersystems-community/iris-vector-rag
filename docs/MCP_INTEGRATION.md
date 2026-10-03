# Model Context Protocol (MCP) Integration

iris-vector-rag ships an MCP server that lets an MCP client (Claude Desktop, Claude
Code, any MCP-capable agent) run the RAG pipelines as tools. It is a Python server
built on the official [`mcp` SDK](https://github.com/modelcontextprotocol/python-sdk)
and speaks **stdio**: the client starts the process and talks to it over stdin and
stdout. The design is recorded in
[ADR 0002](adr/0002-mcp-server-is-python-over-stdio.md).

## Install and start

```bash
pip install "iris-vector-rag[mcp]"
iris-vector-rag-mcp            # or: python -m iris_vector_rag.mcp
```

Both commands do the same thing: start the stdio server and wait for a client. They
take no arguments and have no subcommands. Run by hand they print nothing and wait
for protocol messages on stdin; logs go to stderr.

The previous Node.js server, `Dockerfile.mcp` and the HTTP bridge are retired. They
could not work (the bridge they called was never served). The last commit that
contains them is tagged `archive/mcp-node-v1`.

## Connect to IRIS

The server reads the IRIS connection from the same resolver as the rest of the
library, in this order: environment, `.env` in the working directory, the container
named by `IRIS_CONTAINER_NAME`, the `database.iris` section of the configuration
file, then defaults.

| Variable              | Meaning                                      | Default     |
| --------------------- | -------------------------------------------- | ----------- |
| `IRIS_HOST`           | IRIS host                                    | `localhost` |
| `IRIS_PORT`           | IRIS superserver port                        | `1972`      |
| `IRIS_NAMESPACE`      | Namespace                                    | `USER`      |
| `IRIS_USERNAME`       | User (`IRIS_USER` is a deprecated alias)     | `_SYSTEM`   |
| `IRIS_PASSWORD`       | Password                                     | `SYS`       |
| `IRIS_CONTAINER_NAME` | Docker container whose published port to use | unset       |

A connection that cannot be made is an error returned to the client, not a
substitute answer.

Answer generation uses the LLM the pipelines are configured for (by default OpenAI,
which needs `OPENAI_API_KEY`). MCP clients start the server with a minimal
environment, so put that key in the client's `env` block too.

## Claude Desktop

Add the server to `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "iris-vector-rag": {
      "command": "iris-vector-rag-mcp",
      "env": {
        "IRIS_HOST": "localhost",
        "IRIS_PORT": "1972",
        "IRIS_NAMESPACE": "USER",
        "IRIS_USERNAME": "_SYSTEM",
        "IRIS_PASSWORD": "SYS"
      }
    }
  }
}
```

If the command is not on the client's `PATH`, use the absolute path of the
interpreter instead: `"command": "/path/to/venv/bin/python", "args": ["-m", "iris_vector_rag.mcp"]`.
Restart Claude Desktop after editing the file.

## Tools

Six tools. Every input schema rejects unknown keys, so a client cannot pass extra
arguments through to a pipeline. Every tool returns its result as one JSON text
block.

| Tool                 | Pipeline                | Arguments (besides `query`)                                                                                        |
| -------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `rag_basic`          | BasicRAG                | `top_k`, `include_sources`, `include_metadata`                                                                     |
| `rag_basic_rerank`   | BasicRAG with reranking | `top_k`, `rerank_top_k`, `reranker_model`, `include_sources`, `include_metadata`                                   |
| `rag_crag`           | Corrective RAG          | `top_k`, `confidence_threshold`, `correction_strategy`, `enable_web_search`, `include_sources`, `include_metadata` |
| `rag_graphrag`       | HybridGraphRAG          | `top_k`, `search_method`, `graph_traversal_depth`, `rrf_k`, `include_sources`, `include_metadata`                  |
| `rag_pylate_colbert` | PyLate ColBERT          | `top_k`, `max_query_length`, `interaction_threshold`, `compression_ratio`, `include_sources`, `include_metadata`   |
| `rag_health_check`   | (the server itself)     | `include_details`                                                                                                  |

The ranges, defaults and enumerations are in
`iris_vector_rag/mcp/mcp_tool_schema.json`; the client receives them in the tool
listing.

A pipeline result has `answer`, `retrieved_documents`, `sources`, `metadata` and
`performance` (timings the pipeline reported).

`rag_health_check` reports only what it measured: IRIS reachable (with the round
trip time of `SELECT 1`), each pipeline constructible, embedder loadable.
`status` is `healthy`, `degraded` (a pipeline or the embedder is broken) or
`unavailable` (IRIS cannot be reached). Failures carry the exception type and
message. `pipelines` is included when `include_details` is true or the status is not
healthy.

Dropped from the earlier design: `rag_metrics` (there were no counters behind it)
and `rag_iris_global_graphrag` (it depends on an external application). Either can
return when it can be made sound.

## Errors

- Invalid arguments (unknown key, wrong type, out of range, missing `query`) return
  a tool error naming the problem.
- A pipeline that fails returns a tool error that names the exception type
  (`RuntimeError`, `ImportError`, ...) and points to the server log on stderr. The
  message, paths and credentials stay in the log.
- A pipeline that cannot be imported (for example PyLate ColBERT without its
  dependencies) fails its own tool call and shows in `rag_health_check`; the other
  tools keep working.

## Security

Stdio means the process boundary is the trust boundary: whoever can start the
process, or write to its stdin, can run queries with the IRIS credentials in its
environment. There is no authentication layer because there is no network listener.
Give the server an IRIS user with only the access the pipelines need. A remote
(HTTP) MCP server with real authentication is a separate, undecided piece of work.

## What is tested

- The tool list, schemas, argument rejection, error paths and the health check run
  through the SDK's in-memory client in `tests/unit/test_mcp_server.py`, without IRIS.
- The real process over real stdio (handshake and tool listing) is in
  `tests/unit/test_mcp_stdio_process.py`.
- `tests/integration/test_mcp_*.py` and `tests/e2e/test_mcp_stdio_e2e.py` drive the
  server against a live IRIS (an ephemeral container unless `IRIS_CONTAINER_NAME`
  pins one); run them with `pytest -m mcp`.

Not tested: a real Claude Desktop session.

## Troubleshooting

- **The client shows no tools.** Run `iris-vector-rag-mcp` in a terminal. It should
  wait silently. A message about the MCP SDK means the extra is missing:
  `pip install "iris-vector-rag[mcp]"`.
- **Tools return errors mentioning the database.** Call `rag_health_check`. If
  `database.connected` is false, check the `IRIS_*` variables above; the client may
  not pass your shell environment, so set them in the client's `env` block.
- **Empty results.** The knowledge base is empty or in another namespace; load
  documents first (see the README).
- **Slow first call.** Pipelines and the embedding model load on first use.
