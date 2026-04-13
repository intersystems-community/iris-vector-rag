# Research: ObjectScript SDK for IVR

**Date**: 2026-04-13 | **Status**: Complete

## Findings

No unknowns — all building blocks are existing IRIS SQL features.

### Decision 1: SQL pattern for VectorSearch

- **Decision**: Use `VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE, N))` with `%SQL.Statement` dynamic queries
- **Rationale**: Same SQL Python IVR uses. `%SQL.Statement` allows parameterized dimension. Avoids embedded SQL limitations with dynamic column names.
- **Alternatives considered**: `&SQL()` embedded SQL — rejected because table/column names aren't compile-time constants (AttachTable supports arbitrary tables). `$vectorop` — rejected per spec constraint (pure SQL, no direct vector ops).

### Decision 2: JSON serialization pattern

- **Decision**: Use `%DynamicArray` and `%DynamicObject` internally, return `.%ToJSON()` strings from all ClassMethods
- **Rationale**: FR-003 requires JSON string signatures. `%DynamicArray` is the standard IRIS JSON builder. Callers get a string they can parse with `{}.%FromJSON()` or pass directly to REST/MCP.
- **Alternatives considered**: Return `%DynamicObject` directly — rejected because it's not wire-friendly and couples callers to IRIS object model.

### Decision 3: Shared DDL file format

- **Decision**: `sql/schema.sql` contains one `CREATE TABLE IF NOT EXISTS` statement per table, separated by `GO` delimiters. ObjectScript `Schema.Initialize()` reads the file and executes each statement. Python IVR's schema module is refactored to read the same file.
- **Rationale**: Single source of truth. `GO` delimiter is standard IRIS SQL script separator. `IF NOT EXISTS` handles idempotency.
- **Alternatives considered**: YAML/JSON schema definition — rejected as overengineering for 3-4 CREATE TABLE statements.

### Decision 4: HybridSearch fusion strategy

- **Decision**: RRF (Reciprocal Rank Fusion) as default, `strategy` parameter for future alternatives.
- **Rationale**: RRF is parameter-free, already used by IVG's `Graph.KG.Service.HybridSearch`. Implementation: run VectorSearch and TextSearch separately, merge results by `1/(k+rank_v) + 1/(k+rank_t)`, sort descending.
- **Alternatives considered**: Weighted linear combination — available as `strategy="linear"` but not default (requires alpha tuning).

### Decision 5: Embedded Python for AddDocumentWithEmbed

- **Decision**: `Language=python` ClassMethod that imports `sentence_transformers`, loads `all-MiniLM-L6-v2`, encodes text, returns comma-separated vector string. The ObjectScript caller then inserts via `TO_VECTOR()`.
- **Rationale**: Keeps the Python boundary minimal — one function that takes text and returns a vector string. The insert is still pure SQL.
- **Alternatives considered**: Call Python via `$system.Python.Run()` — rejected because `Language=python` is cleaner and avoids string escaping issues.
