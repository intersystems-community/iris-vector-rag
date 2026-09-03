# Requirements Checklist: Feature 062

- [X] FR-001: delete_node calls iris_engine.delete_node
- [X] FR-002: delete_node calls vector_store.delete_documents
- [X] FR-003: Idempotent for non-existent nodes
- [X] FR-004: BM25 deletion skipped (no per-doc API)
- [X] FR-005: ValueError for None node_id
- [X] FR-006: ValueError for empty string node_id
- [X] FR-007: Partial failure logs WARNING and propagates
- [X] FR-008: Returns None
- [X] Contract tests: 8/8 passing
- [X] Integration tests: 2 written (skip due to venv incompatibility)
