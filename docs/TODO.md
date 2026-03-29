# RAG-Templates TODO

**Last Updated**: 2025-10-05

## Immediate (This Session)

### Infrastructure Fixes - DONE ✅
- [x] Fix password reset infinite loop
- [x] Fix schema column mismatches
- [x] Fix embedding generation
- [x] Implement similarity_search_with_score
- [x] Mark JSON filtering tests as xfail
- [x] Fix pytest-randomly/thinc incompatibility
- [x] Add requires_llm_api marker
- [x] Mark GraphRAG tests as slow

### Current Focus - DONE ✅
- [x] Fix CRAG DocumentChunks table creation
- [x] Fix CRAG vector datatype mismatch (FLOAT→DOUBLE)
- [x] Add ColBERT/PyLate to main pipeline factory
- [x] Feature 028 session 3 complete & committed

### Next Steps
- [ ] Fix remaining CRAG test assertion strictness (5 tests)
- [ ] Address PyLate metadata preservation test (1 test)
- [ ] Coverage improvements (current: 10%, target: 95%)
- [ ] Merge ColBERT factory branch to main

## Short Term (This Week)

### Testing
- [ ] Investigate GraphRAG E2E test failures
- [ ] Add more unit tests for coverage
- [ ] Contract tests for all pipelines

### Infrastructure
- [ ] IRIS JSON metadata filtering (research needed)
- [ ] Test data cleanup automation
- [ ] Pre-flight check improvements

## Medium Term (Next 2 Weeks)

### iris-devtools Package
- [ ] Extract connection management to iris-devtools
- [ ] Extract password reset utilities
- [ ] Extract testcontainers wrapper
- [ ] Publish to PyPI
- [ ] Migrate rag-templates to use iris-devtools

### Documentation
- [ ] Update architecture docs
- [ ] Document IRIS JSON limitations
- [ ] Add troubleshooting guide

### Performance
- [ ] GraphRAG optimization
- [ ] Query performance benchmarking
- [ ] Memory usage profiling

## Long Term (This Month)

### Production Readiness
- [ ] 95%+ test coverage
- [ ] All E2E tests passing (or xfailed with reasons)
- [ ] Performance benchmarks established
- [ ] Documentation complete

### New Features
- [ ] Advanced metadata filtering (IRIS-specific)
- [ ] Query optimization
- [ ] Hybrid search improvements

## Deferred (Future)

### RLM search_context — IRIS native vector search (tier 3)

Current `RLMToolSet.search_context` uses keyword frequency scoring. There is a researched
and ready design to replace it with IRIS native `EMBEDDING()` SQL function using a local
SentenceTransformers model (e.g. all-MiniLM-L6-v2, 384-dim, no GPU, runs via Embedded Python).

**How it works:**
1. Register model once in `%Embedding.Config`:
   ```sql
   INSERT INTO %Embedding.Config (Name, Configuration, EmbeddingClass, Description)
     VALUES ('minilm',
             '{"modelName":"sentence-transformers/all-MiniLM-L6-v2",
               "hfCachePath":"/path/to/hf_cache"}',
             '%Embedding.SentenceTransformers', 'MiniLM 384-dim')
   ```
2. Create session-scoped context chunk table with `EMBEDDING`-typed column → IRIS
   auto-computes embeddings on INSERT.
3. Each `search_context(query)` call becomes a single parameterized SQL:
   ```sql
   SELECT TOP 5 content FROM RLM.ContextChunks
   WHERE session_id = ?
   ORDER BY VECTOR_DOT_PRODUCT(embedding_col, EMBEDDING(?, 'minilm')) DESC
   ```
4. No Python embedding lifecycle in the tool — IRIS Embedded Python handles it.
5. No HNSW index needed for session-scoped small tables (brute-force scan is fine).

**Prereqs to verify before building:**
- `sentence-transformers` installed in IRIS Embedded Python environment
- HuggingFace model cache populated on the IRIS host
- Connecting user has `%USE_EMBEDDING` privilege

**Three-tier design** (keyword → in-memory numpy → IRIS native) was discussed
2026-02-21. See conversation context for full analysis.

### Agents = Actors = IRIS Interoperability — Architecture Synthesis (2026-02-21)

**The convergence thesis:** Three independent inventions of the same pattern:
- 1986: Erlang/OTP Actor model — built for telecom (long-lived stateful connections, fault tolerance)
- 2000s: IRIS Interoperability/Productions — built for healthcare (same requirements)
- 2024-2026: LLM agent frameworks — independently rediscovering the same pattern

George Guimarães' post (Feb 16, 2026): "Your Agent Framework Is Just a Bad Clone of Elixir"
https://georgeguimaraes.com/your-agent-orchestrator-is-just-a-bad-clone-of-elixir/
Key quote: *"The actor model that Erlang introduced in 1986 is the agent model that AI is
rediscovering in 2026."* Evidence: Langroid explicitly cites Actor Framework; AutoGen v0.4
independently rebuilt itself as "event-driven actor framework"; LangGraph = state machines.

**Plaza** is ISC's internal proof: https://plaza.iscinternal.com (Beta, in production)
- Developer: Tim Leavitt (Senior Development Manager), team contact: plazadev@intersystems.com
- Source: //custom_ccrs/us/ISCX/Plaza/BASE/ in Perforce
- ISC's internal AI chat platform — custom agents with Jira/Confluence tools, on-prem + cloud LLMs
- Built entirely on IRIS Interoperability/Productions (`pkg.isc.genai`, `pkg.isc.mcp` packages)

**Plaza's actual architecture (inspected from source):**
```
PassthroughService (EnsLib.REST.GenericService)
    ↓
ModelProviderRouter (Ens.BusinessProcess)      ← routes by model/provider
    ↓
OpaqueToolCallRouter (Ens.BusinessProcess)     ← the ReAct loop (For {} in ObjectScript)
    ├─→ ModelProviderRouter (LLM calls)
    └─→ pkg.isc.mcp.ToolCallRouter             ← tool dispatch
              ↓
        NativeOperation / StdioClientOperation / SSEClientOperation / PythonStdioClientOperation
              ↓
        pkg.isc.mcp.NativeTool subclasses      ← one ObjectScript class per tool
```

Tools are `pkg.isc.mcp.NativeTool` subclasses:
- `Parameter NAME` = tool name
- `/// Description` comment → LLM tool description (auto-extracted)
- `XData inputSchema` → JSON Schema
- `ClassMethod OnRequest()` → implementation
External MCP servers connect via Stdio, SSE, or StreamableHTTP `ClientOperation` classes.

LLM access: currently direct HTTP via `ModelProviderRouter` (OpenAI/Azure/on-prem Qwen).
**Tim wants to replace the LLM layer with aicore calls** — this is the planned IVR/Plaza bridge.

**IVR/IVG → Plaza connection:**
IVR RAG pipelines can be registered as `pkg.isc.mcp.NativeTool` subclasses within Plaza,
making vector search / GraphRAG / hybrid search available as first-class tools to any Plaza
agent without changing Plaza's architecture. When aicore replaces the LLM layer, the same
tool dispatch path continues to work unchanged.

**The three-layer stack for READY 2026:**
```
IRIS Interoperability (Productions/BPL)         ← Actor runtime, supervision, visual trace
    pkg.isc.genai + pkg.isc.mcp (Plaza packages) ← agent + tool framework, MCP integration
aicore / %AI.Agent / %AI.ToolSet                ← LLM substrate, massive concurrency (planned)
IVR / IVG (iris-vector-rag)                     ← RAG as NativeTool, GraphRAG for FHIR
```

**READY 2026** (April 27, National Harbor MD): Theme is "Building AI Agents with InterSystems"
- Mini hackathon on pre-conference day (50 attendees, first-come)
- "upcoming InterSystems capabilities" — signals %AI.Agent / aicore native support incoming
- FHIR + AI was READY 2025 theme; agents + FHIR + Interoperability is the 2026 story

**FHIR angle:** IRIS FHIR Repository + Productions as agent substrate:
- Business Service: receives FHIR event (ADT, Task, ClinicalImpression)
- OpaqueToolCallRouter (or equivalent): orchestrates agent ReAct loop
- IVR NativeTool: vector search over FHIR document corpus for RAG context
- Business Operation: writes agent decision back as FHIR resource

**grongier.pex** (Guillaume Rongier, `iris_pex_embedded_python` on PyPI):
Python-native path to write BusinessService/BusinessProcess/BusinessOperation as pure Python.
Relevant for writing IVR tools as Python-native Interoperability components if needed.

**dc-mais** (Henry Pereira, community.intersystems.com Parts 1-3, Jan-Feb 2026):
Community proof-of-concept implementing the same Double Loop / ReAct / BPL orchestration
pattern as Plaza, independently. Published on Open Exchange. Validates the architecture
is accessible to community developers, not just ISC internal teams.

**Next actions when ready to pursue:**
1. Register IVR RAG pipelines as `pkg.isc.mcp.NativeTool` subclasses for Plaza
2. Work with Tim on aicore-as-LLM-backend for ModelProviderRouter
3. Define %Agent subclass pattern using grongier.pex BusinessProcess
4. Build FHIR+agent demo for READY 2026 using Plaza + IVR + IRIS FHIR Repository

### Nice to Have
- [ ] Multi-pipeline comparison UI
- [ ] Advanced visualization
- [ ] Real-time monitoring dashboard
- [ ] A/B testing framework

## Known Issues

### Critical
None currently

### High Priority
- IRIS JSON metadata filtering (5 xfailed tests)
- GraphRAG E2E test failures
- Low test coverage (10%)

### Medium Priority
- Performance optimization needed
- Documentation gaps

### Low Priority
- Minor UI improvements
- Additional pipeline variants

## Notes
- Focus on test stability and coverage
- iris-devtools will improve reusability
- Document all IRIS-specific learnings
