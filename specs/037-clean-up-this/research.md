# Phase 0: Research - Repository Cleanup

**Feature**: Repository Cleanup and Organization
**Date**: 2025-10-08
**Status**: Complete

## Overview

This research document addresses technical unknowns and establishes best practices for the repository cleanup feature. Since this is a file organization/maintenance task rather than a code development feature, minimal technical research is required.

## Research Items

### 1. Python Project Structure Standards

**Question**: Should the repository follow a specific Python project structure convention (src-layout vs flat-layout)?

**Decision**: Maintain existing flat-layout structure with `iris_rag/` at repository root

**Rationale**:
- Project is already established with flat-layout pattern
- All imports currently reference `from iris_rag.*`
- Changing to src-layout would require:
  - Moving iris_rag/ to src/iris_rag/
  - Updating all import statements across codebase
  - Modifying pyproject.toml, Makefile, docker configs
  - Risk of breaking existing integrations
- No functional benefit for a mature project
- User clarified: follow existing structure, no migration needed

**Alternatives Considered**:
- **src-layout** (`src/iris_rag/`): Common in modern Python projects, better for distribution
  - Rejected: Too disruptive for established project, no migration justification
- **Namespace packages**: More complex, unnecessary for single package
  - Rejected: Adds complexity without benefit

**References**:
- [Python Packaging Guide - src layout vs flat layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/)
- Existing codebase structure analysis

---

### 2. File Classification Strategy

**Question**: How to systematically categorize files as Essential, Relocatable, or Removable?

**Decision**: Three-tier classification with pattern matching and manual review

**Classification Rules**:

**Essential Files** (Keep at root):
- README.md, USER_GUIDE.md, CLAUDE.md - Primary documentation
- DOCUMENTATION_INDEX.md, DOCUMENTATION_AUDIT_REPORT.md - Documentation management
- docker-compose*.yml - Container orchestration
- Makefile - Build automation
- pyproject.toml, setup.py, requirements*.txt - Package management
- pytest.ini, .coveragerc - Testing configuration
- .gitignore, .env.example - Git and environment config
- LICENSE - Legal

**Relocatable Files** (Move to docs/):
- docs/docs/STATUS.md, docs/docs/PROGRESS.md, docs/docs/TODO.md, docs/docs/docs/CHANGELOG.md - Status tracking
- Any historical technical documentation not already in docs/

**Removable Files**:
- Temporary/cache: `*.pyc`, `__pycache__/`, `.pytest_cache/`, `.coverage`
- Old evaluation reports: `outputs/reports/ragas_evaluations/*.{html,json}`
- Historical tracking: `*_COMPLETION_SUMMARY.md`, `*_SESSION_NOTES.md`, `*_REPORT.md` (except DOCUMENTATION_AUDIT_REPORT.md)
- Duplicate documentation: Keep newest based on modification time
- Old verification outputs: `outputs/pipeline_verification_*.json`

**Rationale**:
- Maps directly to FR-001, FR-002, FR-003, FR-004 requirements
- Pattern-based rules enable automated classification
- Manual review catches edge cases (e.g., DOCUMENTATION_AUDIT_REPORT.md is recent, not historical)
- User clarified removal categories a, b, c, f in spec

**Implementation Approach**:
```python
def classify_file(path: Path) -> FileCategory:
    if path.name in ESSENTIAL_FILES:
        return FileCategory.ESSENTIAL
    if matches_removable_pattern(path):
        return FileCategory.REMOVABLE
    if is_status_file(path):
        return FileCategory.RELOCATABLE
    # Default: manual review needed
    return FileCategory.REVIEW_REQUIRED
```

**Alternatives Considered**:
- **Size-based classification**: Remove files > X MB
  - Rejected: Size doesn't correlate with necessity
- **Age-based classification**: Remove files older than X months
  - Rejected: Critical docs may be old, recent files may be temporary
- **Content-based classification**: Parse file contents
  - Rejected: Overcomplicated, filename patterns sufficient

---

### 3. Test Validation Approach

**Question**: How to ensure 100% test pass rate is maintained during cleanup?

**Decision**: Incremental cleanup with test validation after each batch

**Validation Strategy**:
1. **Baseline**: Run full test suite before any changes, record results
2. **Batch operations**: Group file operations by type (remove, move, update)
3. **Test after each batch**: Run pytest after each operation batch
4. **Compare results**: Verify test count and pass rate unchanged
5. **Rollback on failure**: Use git staging to revert failed batch

**Implementation**:
```bash
# Baseline
pytest --tb=short > baseline_tests.txt
baseline_count=$(grep "passed" baseline_tests.txt | awk '{print $1}')

# After each batch
pytest --tb=short > current_tests.txt
current_count=$(grep "passed" current_tests.txt | awk '{print $1}')

if [ "$current_count" != "$baseline_count" ]; then
    git restore --staged .
    git restore .
    echo "ERROR: Test count changed. Rolling back."
    exit 1
fi
```

**Rationale**:
- Satisfies FR-009, FR-010 (run tests, verify 100% pass rate)
- Satisfies FR-011 (rollback on failure)
- Incremental approach limits blast radius
- Git staging enables atomic rollback

**Batch Grouping**:
1. Remove temporary/cache files (low risk)
2. Remove old evaluation reports (low risk)
3. Remove historical tracking files (medium risk)
4. Move status files to docs/ (medium risk - may have hardcoded paths)
5. Consolidate duplicate docs (medium risk)
6. Update documentation links (high risk - verify no broken links)

**Alternatives Considered**:
- **All-at-once cleanup**: Faster but riskier
  - Rejected: Violates incremental validation requirement
- **Pre-scan for references**: Check all files for imports/links before removing
  - Partial adoption: Used for documentation link validation (FR-013)
- **Mock test run**: Run tests without actual file changes
  - Rejected: Doesn't validate real-world impact

---

### 4. Documentation Link Validation

**Question**: How to detect and fix broken documentation links after file moves?

**Decision**: Two-phase approach - automated detection + manual fixes

**Detection Tools**:
- `grep -r "path/to/moved/file.md"` - Find hardcoded paths
- Custom script to parse Markdown links: `[text](path.md)`
- Check DOCUMENTATION_INDEX.md for stale references

**Update Strategy**:
1. Before moving files: Scan for references
2. After moving files: Update references automatically where possible
3. Run link validation tool
4. Manual review for complex cases

**Link Validation**:
```python
def validate_markdown_links(root_dir: Path) -> List[BrokenLink]:
    broken = []
    for md_file in root_dir.rglob("*.md"):
        for link in extract_links(md_file):
            if is_relative_path(link):
                target = (md_file.parent / link).resolve()
                if not target.exists():
                    broken.append(BrokenLink(md_file, link, target))
    return broken
```

**Rationale**:
- Satisfies FR-013 (check broken links)
- Satisfies FR-016, FR-017 (update documentation)
- Automated detection prevents manual oversight

**Alternatives Considered**:
- **markdown-link-check** npm package: Good but adds Node.js dependency
  - Rejected: Keep tooling Python-only
- **Manual review only**: Error-prone
  - Rejected: Automate where possible

---

## Summary of Decisions

| Research Item | Decision | Impact on Implementation |
|---------------|----------|--------------------------|
| Python structure | Flat-layout (existing) | No code migration, focus on file org only |
| Classification | 3-tier with patterns | Enables automated + manual review |
| Test validation | Incremental batches | Requires batch grouping, git staging |
| Link validation | Automated detection | Custom Python script needed |

## Next Steps

Phase 1 will use these research findings to create:
- data-model.md: File classification data structures
- contracts/: Cleanup operation interfaces
- quickstart.md: Step-by-step execution guide
- Contract tests: Validation logic

---

**Research Complete**: 2025-10-08
