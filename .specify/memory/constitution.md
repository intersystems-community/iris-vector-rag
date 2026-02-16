# IRIS Vector RAG Constitution

## Core Principles

### Principle 1: IRIS-First Integration Testing
- Tests must run against a live InterSystems IRIS instance; never skip IRIS tests by default.
- Use iris-devtester for container lifecycle and port management; do not hardcode IRIS ports.

### Principle 2: VECTOR Client Limitation (TO_VECTOR required)
- VECTOR columns cannot be inserted directly via SQL; this is a client limitation, not a server limitation.
- All VECTOR inserts must use the TO_VECTOR() function (e.g., `TO_VECTOR(?, FLOAT, 384)`), and the limitation must be documented when relevant.

### Principle 3: .DAT Fixture-First Performance Rule
- Prefer .DAT fixtures for IRIS test data: they are 100x–200x faster than JSON in IRIS load paths.
- Use iris-devtools for .DAT fixture workflows; choose JSON or programmatic fixtures only when .DAT is not feasible.
- Fixture documentation lives in tests/fixtures/README.md.

### Principle 4: Test Isolation by Database State
- Fixtures must provide isolated database states per test.
- Checksum validation ensures reproducibility, and version compatibility checks are required for fixtures.

### Principle 5: Embedding Generation Standards
- Default embedding dimension is 384 (sentence-transformers/all-MiniLM-L6-v2).
- Sentence-transformers integration must be supported and documented.
- NULL or missing embeddings must be handled with zero vectors and explicit logging.

### Principle 6: Configuration & Secrets Hygiene
- Configuration must be environment-driven and validated early; errors should be actionable.
- Never log secrets; passwords and API keys must be masked in logs and diagnostics.

### Principle 7: Backend Mode Awareness (Feature 035)
- The system must be explicit about backend mode (Community Edition vs Enterprise Edition).
- Connection pooling behavior differs by mode; document limits for Community Edition and unlimited pools for Enterprise Edition.

## Development Workflow
- Contract tests define required behaviors and must pass before merge.
- Prefer small, composable changes; keep interfaces stable and document breaking changes.

## Examples

```sql
-- Vector insert must use TO_VECTOR (client limitation)
INSERT INTO RAG.DocumentChunks (chunk_id, embedding)
VALUES (?, TO_VECTOR(?, FLOAT, 384));
```

## Governance
- This constitution supersedes other practices; amendments require documentation and review.

**Version**: 1.0.0 | **Ratified**: 2026-02-08 | **Last Amended**: 2026-02-08
