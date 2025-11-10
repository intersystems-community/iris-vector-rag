# Tasks: Documentation Review and README Optimization

**Input**: Design documents from `/Users/tdyar/ws/iris-vector-rag-private/specs/055-perform-top-to/`
**Prerequisites**: plan.md (✓), research.md (✓), data-model.md (✓), contracts/ (✓)

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✅ Loaded - Documentation-only feature with validation approach
   → ✅ Extract: Python 3.11+, pytest, requests, markdown processing
2. Load optional design documents:
   → ✅ data-model.md: 5 entities (DocumentationFile, CodeExample, Link, ValidationResult, ValidationError)
   → ✅ contracts/: 3 contract test files (link, code example, README structure validation)
   → ✅ research.md: Documentation tools, README optimization strategy, validation approach
3. Generate tasks by category:
   → Setup: Validation scripts, baseline report
   → Tests: 3 contract test files (16 total tests)
   → Core: README optimization, new guides, documentation index
   → Integration: Link fixes, example updates
   → Polish: Final validation, commit preparation
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file (README.md) = sequential
   → Validation before and after updates
5. Number tasks sequentially (T001-T020)
6. ✅ Generated dependency graph
7. ✅ Created parallel execution examples
8. Validate task completeness:
   → ✅ All 3 contracts have test execution tasks
   → ✅ All 23 functional requirements covered
   → ✅ README optimization complete
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- All paths are absolute from repository root: `/Users/tdyar/ws/iris-vector-rag-private/`

## Phase 3.1: Setup & Baseline Validation

- [ ] **T001** [P] Run baseline contract test for link validation
  - **File**: `specs/055-perform-top-to/contracts/link_validation_contract.py`
  - **Command**: `pytest specs/055-perform-top-to/contracts/link_validation_contract.py -v --tb=short > specs/055-perform-top-to/baseline_link_validation.txt 2>&1 || true`
  - **Purpose**: Establish current state of links before fixes
  - **Expected**: Tests will FAIL (broken links exist per FR-005, FR-006)

- [ ] **T002** [P] Run baseline contract test for code examples
  - **File**: `specs/055-perform-top-to/contracts/code_example_contract.py`
  - **Command**: `pytest specs/055-perform-top-to/contracts/code_example_contract.py -v --tb=short > specs/055-perform-top-to/baseline_code_examples.txt 2>&1 || true`
  - **Purpose**: Identify examples using old module names (iris_rag)
  - **Expected**: Tests will FAIL (FR-001, FR-019 violations exist)

- [ ] **T003** [P] Run baseline contract test for README structure
  - **File**: `specs/055-perform-top-to/contracts/readme_structure_contract.py`
  - **Command**: `pytest specs/055-perform-top-to/contracts/readme_structure_contract.py -v --tb=short > specs/055-perform-top-to/baseline_readme_structure.txt 2>&1 || true`
  - **Purpose**: Verify README exceeds 400 lines
  - **Expected**: Tests will FAIL (README is 518 lines per FR-013)

- [ ] **T004** Generate consolidated baseline report
  - **File**: `specs/055-perform-top-to/baseline_report.md`
  - **Action**: Create markdown report summarizing all validation failures
  - **Include**:
    - Total broken links found
    - Code examples using old module names (line numbers)
    - README line count (current vs target)
    - Sections to move (IRIS EMBEDDING, MCP, Architecture)
  - **Purpose**: Clear view of work needed before starting fixes

## Phase 3.2: README Optimization (Sequential - all edit README.md)

- [ ] **T005** Create IRIS EMBEDDING detailed guide
  - **File**: `docs/IRIS_EMBEDDING_GUIDE.md`
  - **Action**: Extract IRIS EMBEDDING section from README (lines 303-389, 87 lines)
  - **Content**: Complete IRIS EMBEDDING documentation with:
    - Model caching architecture
    - Performance benchmarks (346x speedup)
    - Configuration examples
    - Multi-field vectorization
    - When to use IRIS EMBEDDING
  - **Purpose**: Move detailed content out of README (FR-008)

- [ ] **T006** Create or enhance Pipeline comparison guide
  - **File**: `docs/PIPELINE_GUIDE.md`
  - **Action**: Create comprehensive pipeline comparison guide
  - **Content**:
    - When to use each pipeline (basic, basic_rerank, crag, graphrag, multi_query_rrf, pylate_colbert)
    - Performance characteristics
    - Use case examples
    - Configuration templates
  - **Purpose**: Consolidate pipeline selection guidance

- [ ] **T007** Enhance MCP integration documentation
  - **File**: `docs/MCP_INTEGRATION.md`
  - **Action**: Move MCP section details from README (lines 390-423, 34 lines)
  - **Content**:
    - MCP server setup
    - Claude Desktop configuration
    - Available tools
    - Testing MCP integration
  - **Purpose**: Reduce README clutter (FR-008)

- [ ] **T008** Update README IRIS EMBEDDING section with link
  - **File**: `README.md`
  - **Action**: Replace lines 303-389 with concise summary (10-15 lines) + link to `docs/IRIS_EMBEDDING_GUIDE.md`
  - **New content**:
    ```markdown
    ## IRIS EMBEDDING: Auto-Vectorization

    **346x faster auto-vectorization with model caching** - automatic embedding
    generation without repeated model loading overhead.

    **Key Benefits**:
    - ⚡ 346x speedup - 1,746 documents in 3.5 seconds vs 20 minutes
    - 🎯 95% cache hit rate - models stay in memory
    - 🚀 50ms average latency

    📖 **[Complete IRIS EMBEDDING Guide →](docs/IRIS_EMBEDDING_GUIDE.md)**
    ```
  - **Reduction**: 87 lines → 15 lines (saves 72 lines)

- [ ] **T009** Update README MCP section with link
  - **File**: `README.md`
  - **Action**: Replace lines 390-423 with concise summary (8-10 lines) + link to `docs/MCP_INTEGRATION.md`
  - **Reduction**: 34 lines → 10 lines (saves 24 lines)

- [ ] **T010** Update README Architecture section with link
  - **File**: `README.md`
  - **Action**: Replace lines 426-445 with brief overview + link to `docs/architecture/COMPREHENSIVE_ARCHITECTURE_OVERVIEW.md`
  - **Reduction**: 20 lines → 8 lines (saves 12 lines)

- [ ] **T011** Fix old module name references in README
  - **File**: `README.md`
  - **Action**: Replace ALL occurrences of `iris_rag` with `iris_vector_rag`
  - **Known locations**: Lines 84, 119 (from spec FR-019)
  - **Find/Replace**:
    - `from iris_rag import` → `from iris_vector_rag import`
    - `import iris_rag` → `import iris_vector_rag`
  - **Verification**: Run grep to confirm zero matches

- [ ] **T012** Verify README line count ≤400
  - **File**: `README.md`
  - **Action**: Check line count after optimizations
  - **Command**: `wc -l README.md`
  - **Expected**: ≤400 lines (started at 518, saved 72+24+12=108 lines → ~410 lines)
  - **If still over**: Identify additional sections to condense or move

## Phase 3.3: Documentation Structure (Parallel - different files)

- [ ] **T013** [P] Create documentation index
  - **File**: `docs/README.md`
  - **Action**: Create comprehensive documentation navigation index
  - **Structure**:
    - ## Getting Started (README, USER_GUIDE, API_REFERENCE)
    - ## Advanced Topics (IRIS_EMBEDDING_GUIDE, PIPELINE_GUIDE, MCP_INTEGRATION)
    - ## Development & Contributing (CONTRIBUTING, DEVELOPMENT, testing guides)
    - ## Architecture & Design (architecture/* files)
  - **Purpose**: Single entry point for all documentation (FR-015, FR-018)

- [ ] **T014** [P] Archive obsolete documents
  - **Directory**: `docs/archived/`
  - **Action**: Identify and move obsolete documents
  - **Candidates** (from research):
    - Completed migration guides
    - Outdated roadmaps
    - Superseded specifications
  - **Command**: `mkdir -p docs/archived/ && git mv <obsolete-doc> docs/archived/`
  - **Purpose**: Reduce clutter (FR-017)

- [ ] **T015** [P] Update all docs to use correct module name
  - **Files**: All `docs/**/*.md` files
  - **Action**: Replace `iris_rag` with `iris_vector_rag` across all documentation
  - **Command**: `find docs/ -name "*.md" -exec sed -i '' 's/from iris_rag import/from iris_vector_rag import/g' {} \;`
  - **Verification**: `find docs/ -name "*.md" -exec grep -Hn "from iris_rag import\|import iris_rag" {} \;` should return nothing
  - **Purpose**: Consistency across all docs (FR-001, FR-019)

## Phase 3.4: Link and Example Validation & Fixes (Sequential - iterative)

- [ ] **T016** Fix broken internal links
  - **Files**: Various documentation files
  - **Action**: Based on baseline report, fix all broken internal links
  - **Process**:
    1. Review `baseline_link_validation.txt` for broken internal links
    2. For each broken link, either:
       - Create missing target file
       - Update link to correct target
       - Remove link if target no longer relevant
  - **Verification**: Re-run link validation contract test

- [ ] **T017** Verify external links resolve
  - **Files**: Various documentation files
  - **Action**: Check external links and update/remove broken ones
  - **Note**: Some external links may be temporarily unavailable - use judgment
  - **Verification**: Re-run link validation contract test

- [ ] **T018** Verify all code examples execute
  - **Files**: README.md and docs/**/*.md files with Python examples
  - **Action**: Test execution of code examples identified in baseline
  - **Process**:
    1. Extract Python code blocks
    2. Attempt compilation with `ast.parse()`
    3. Fix syntax errors
    4. Verify imports reference existing modules
  - **Verification**: Re-run code example contract test

## Phase 3.5: Final Validation & Polish

- [ ] **T019** Run final contract test validation
  - **Files**: All contract tests in `specs/055-perform-top-to/contracts/`
  - **Command**: `pytest specs/055-perform-top-to/contracts/ -v`
  - **Expected**: ALL tests PASS
  - **If failures**: Return to T016-T018 and fix remaining issues

- [ ] **T020** Create final validation report
  - **File**: `specs/055-perform-top-to/final_validation_report.md`
  - **Content**:
    - ✅ README line count: <current> / 400 (PASS/FAIL)
    - ✅ Broken links: 0
    - ✅ Old module names: 0 occurrences
    - ✅ Code examples: All valid syntax
    - ✅ Documentation index: Created
    - ✅ New guides: 3 created (IRIS_EMBEDDING, PIPELINE, enhanced MCP)
  - **Purpose**: Proof all 23 functional requirements met

## Dependencies

### Critical Path
```
T001-T003 (Baseline) → T004 (Report) → T005-T012 (README Optimization) → T013-T015 (Structure) [P] → T016-T018 (Fixes) → T019 (Validation) → T020 (Report)
```

### Parallel Execution Groups
**Group 1: Baseline Validation** (can run together):
- T001 (link validation)
- T002 (code examples)
- T003 (README structure)

**Group 2: New Documentation Files** (can run together):
- T005 (IRIS_EMBEDDING_GUIDE.md)
- T006 (PIPELINE_GUIDE.md)
- T007 (MCP_INTEGRATION.md update)

**Group 3: Documentation Structure** (can run together):
- T013 (docs/README.md index)
- T014 (archive obsolete docs)
- T015 (fix module names in docs/)

**Sequential Tasks** (must run in order):
- T008-T012: All edit README.md, must be sequential
- T016-T018: Iterative fixes based on validation results
- T019-T020: Final validation after all fixes complete

## Parallel Example
```bash
# Launch Group 1 (Baseline Validation) in parallel:
pytest specs/055-perform-top-to/contracts/link_validation_contract.py -v --tb=short > specs/055-perform-top-to/baseline_link_validation.txt 2>&1 &
pytest specs/055-perform-top-to/contracts/code_example_contract.py -v --tb=short > specs/055-perform-top-to/baseline_code_examples.txt 2>&1 &
pytest specs/055-perform-top-to/contracts/readme_structure_contract.py -v --tb=short > specs/055-perform-top-to/baseline_readme_structure.txt 2>&1 &
wait

# Launch Group 2 (New Documentation) in parallel:
# (Create IRIS_EMBEDDING_GUIDE.md)
# (Create PIPELINE_GUIDE.md)
# (Update MCP_INTEGRATION.md)
# These can be done concurrently as they edit different files

# Launch Group 3 (Structure) in parallel:
# (Create docs/README.md)
# (Archive obsolete docs)
# (Fix module names in docs/)
# These can be done concurrently as they edit different files
```

## Notes
- **[P] tasks**: Different files, no dependencies - can run in parallel
- **Sequential tasks** (T008-T012): All edit same file (README.md) - must be sequential
- **TDD approach**: Baseline tests (T001-T003) establish failures before fixes
- **Validation gates**: T019 must PASS before considering feature complete
- **Commit strategy**: Commit after each phase (baseline, optimization, structure, validation)
- **Expected total time**: 3-4 hours (1 hour setup/baseline, 2 hours updates, 1 hour validation/polish)

## Task Generation Rules Applied
✅ 3 contract files → 3 baseline test tasks (T001-T003)
✅ 3 new guides (entities in plan) → 3 creation tasks (T005-T007)
✅ 1 README file → sequential optimization tasks (T008-T012)
✅ Documentation structure → parallel structure tasks (T013-T015)
✅ Validation requirements → iterative fix tasks (T016-T018)
✅ Quality gates → final validation tasks (T019-T020)

**Total Tasks**: 20 (2 less than estimated 18-22, well scoped)

## Success Criteria
After completing all tasks, the following MUST be true:
- ✅ All contract tests pass (pytest specs/055-perform-top-to/contracts/ -v)
- ✅ README.md ≤ 400 lines
- ✅ 0 broken links
- ✅ 0 occurrences of old module name (iris_rag)
- ✅ Documentation index exists (docs/README.md)
- ✅ 3 new detailed guides created
- ✅ All code examples have valid syntax
- ✅ Final validation report confirms all 23 functional requirements met
