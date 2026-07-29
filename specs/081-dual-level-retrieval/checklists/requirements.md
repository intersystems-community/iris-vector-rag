# Specification Quality Checklist: Dual-Level (Global/Mix) Retrieval

**Purpose**: Validate specification completeness and quality before `/speckit.clarify` / `/speckit.plan`
**Created**: 2026-07-29
**Feature**: [spec.md](../spec.md) · Analysis: [lightrag-comparison.md](../lightrag-comparison.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs) beyond necessary context
- [x] Focused on user/developer value and business needs
- [x] Written so stakeholders can follow the intent
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain (open decisions deferred to `/speckit.clarify`, listed below)
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded (Out of Scope section)
- [x] Dependencies and assumptions identified (Feature 065 dependency explicit)

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No unresolved implementation leakage that blocks planning

## Resolved decisions (implementation complete 2026-07-29)

- **Empty-relation-embedding fallback for `global`**: ✅ **B — graceful degradation** (FR-009).
  Empty index → `metadata["degraded"]=True` with `degradation_reason`, no exception raised.
  KG tables absent → hard `RetrievalPrerequisiteError` (FR-008). Both paths implemented.
- **SC-001 metric**: ✅ **Recall@K** via `TestRecallBenchmark` in `tests/e2e/test_dual_level_retrieval_e2e.py`.
  Asserts `recall_global >= recall_vector` on labeled thematic queries (xfail until labeled data populated).
- **Keyword extraction**: ✅ **LightRAG JSON format** — `{"high_level_keywords":[...],"low_level_keywords":[...]}`.
  `KeywordExtractor` uses this format; `parse_keywords()` strips markdown fences.
- **`mix` default fusion**: ✅ **A — RRF** (`metadata["fusion_method"]=="rrf"` when no `weights=` given).
  Pass `weights={"relation":0.6,"vector":0.4}` to switch to `"weighted_score"`.

## Implementation status

- ✅ Phase 2: QueryOptions extended (`high/low_level_keywords`, `global`/`mix` accepted by `_FUSION_MODES`)
- ✅ Phase 3: RelationEmbeddingStore (schema, embed_and_store, search, count_embedded)
- ✅ Phase 4: KeywordExtractor + parse_keywords
- ✅ Phase 5: `global` mode — RetrievalEngine dispatch, modes registry, check_prerequisites
- ✅ Phase 6: `mix` mode — RRF/weighted fusion, per-source metadata
- ✅ Phase 7: US3 tunability — pre-supplied keywords skip extraction, injection of cheap_llm
- ✅ Phase 8: Logging, CI, CHANGELOG

## Notes

- Depends on Feature 065 (composable-retrieval) plumbing; merged in v0.12.1.
- Adopts LightRAG's _technique_, not its code/storage — iris keeps its unified IRIS backend (constitution Principle V).
