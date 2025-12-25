# Tasks: LLM Cache and Connection Optimizations

**Input**: Design documents from `/specs/064-llm-cache-disk/`
**Prerequisites**: plan.md, spec.md

**Tests**: TDD approach following Constitution Principle III.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and package structure

- [x] T001 Create `iris_vector_rag/evaluation/__init__.py` module structure
- [x] T002 [P] Update `iris_vector_rag/common/llm_cache_config.py` to support `disk` backend type

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [x] T003 [P] Implement `SHA-256` prompt hashing utility in `iris_vector_rag/common/utils.py`
- [x] T004 Define `BenchmarkQuery` and `CacheEntry` schemas in `iris_vector_rag/core/models.py`

---

## Phase 3: User Story 1 - Disk-Based LLM Caching (Priority: P1) 🎯 MVP

**Goal**: Store LLM responses in persistent JSON files

- [x] T005 [P] [US1] Create contract test for `DiskCacheBackend` in `tests/contract/test_llm_cache_disk.py`
- [x] T006 [US1] Implement `DiskCacheBackend` in `iris_vector_rag/common/llm_cache_disk.py`
- [x] T007 [US1] Register `disk` backend in `LangchainCacheManager.setup_cache` within `iris_vector_rag/common/llm_cache_manager.py`

---

## Phase 4: User Story 2 - Fast-Path Connection Hardening (Priority: P1)

**Goal**: Automatic IRIS password reset and hardening bypass

- [x] T008 [US2] Implement `_hard_fix_iris_passwords` routine in `iris_vector_rag/common/iris_connection.py` using Docker exec
- [x] T009 [US2] Update `get_iris_connection` to trigger hard-fix on `SQLCODE -108` or similar errors

---

## Phase 5: User Story 3 - Unified Evaluation Framework (Priority: P2)

**Goal**: Standard multi-hop metrics and dataset loaders

- [x] T010 [US3] Port `DatasetLoader` from HippoRAG2 to `iris_vector_rag/evaluation/datasets.py`
- [x] T011 [US3] Port `MetricsCalculator` with `norm_id` support to `iris_vector_rag/evaluation/metrics.py`
- [x] T012 [P] [US3] Integration test for evaluation framework in `tests/integration/test_evaluation.py`

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T013 Update `default_config.yaml` with caching and connection parameters
- [x] T014 [P] Update documentation in `README.md` for the new evaluation package
- [x] T015 Run quickstart.md validation

