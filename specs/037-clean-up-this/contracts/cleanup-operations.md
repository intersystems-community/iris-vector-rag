# Cleanup Operations Contract

**Feature**: Repository Cleanup and Organization
**Date**: 2025-10-08
**Phase**: Phase 1 - Design

## Overview

This document defines the contracts (interfaces) for all cleanup operations. These operations are the core actions that implement the functional requirements.

## Core Operations

### 1. scan_repository()

Scan repository and inventory all files.

**Contract**:
```python
def scan_repository(repo_root: Path, exclude_dirs: List[str] = None) -> FileInventory:
    """
    Scan repository and create file inventory.

    Args:
        repo_root: Absolute path to repository root
        exclude_dirs: Directories to skip (e.g., ['.git', '.venv', 'node_modules'])

    Returns:
        FileInventory with all scanned files

    Raises:
        ValueError: If repo_root doesn't exist or isn't a directory
        PermissionError: If cannot read directory

    Implements: FR-001 (identify all files at repository root)
    """
```

**Postconditions**:
- FileInventory.total_files > 0
- All files have valid paths
- All file sizes and timestamps are populated

---

### 2. classify_files()

Classify files into categories.

**Contract**:
```python
def classify_files(inventory: FileInventory) -> ClassificationReport:
    """
    Classify all files in inventory.

    Args:
        inventory: FileInventory from scan_repository()

    Returns:
        ClassificationReport with classification decisions

    Raises:
        ValueError: If inventory is empty or invalid

    Implements: FR-002 (categorize files into essential/relocatable/removable)
    """
```

**Classification Logic**:
- Essential: Matches ESSENTIAL_FILES list
- Relocatable: Status files (docs/docs/STATUS.md, etc.), current documentation
- Removable: Temporary files, old outputs, historical docs, duplicates
- Review Required: Anything not matching patterns

**Postconditions**:
- Every file has a category
- Every file has a classification reason
- essential_count + relocatable_count + removable_count + review_required_count == total_files

---

### 3. remove_files()

Remove files marked as removable.

**Contract**:
```python
def remove_files(
    removable_files: List[RepositoryFile],
    dry_run: bool = False
) -> RemovalReport:
    """
    Remove files from filesystem.

    Args:
        removable_files: Files marked FileCategory.REMOVABLE
        dry_run: If True, don't actually delete files

    Returns:
        RemovalReport with removal results

    Raises:
        PermissionError: If cannot delete file
        FileNotFoundError: If file doesn't exist (warning, not failure)

    Implements: FR-004, FR-007 (remove unnecessary files)
    """
```

**Safety Checks**:
- Verify file is not tracked by git (or is staged for deletion)
- Check file is not referenced by any essential file
- Confirm file matches removable patterns

**Postconditions**:
- All successfully removed files no longer exist
- Failed removals are recorded in RemovalReport.files_failed
- bytes_freed is sum of removed file sizes

---

### 4. move_files()

Move files to new locations.

**Contract**:
```python
def move_files(
    relocatable_files: List[RepositoryFile],
    target_map: Dict[Path, Path],
    dry_run: bool = False
) -> MoveReport:
    """
    Move files to new locations.

    Args:
        relocatable_files: Files marked FileCategory.RELOCATABLE
        target_map: Mapping of source path to destination path
        dry_run: If True, don't actually move files

    Returns:
        MoveReport with move results

    Raises:
        PermissionError: If cannot move file
        FileExistsError: If destination already exists

    Implements: FR-003, FR-008 (move relocatable files to appropriate subdirectories)
    """
```

**Move Logic**:
- Create target directory if doesn't exist
- Move file using shutil.move() or git mv
- Update git index if file is tracked

**Postconditions**:
- All successfully moved files exist at new location
- Original locations no longer have files
- directories_created lists new dirs
- files_moved contains (source, destination) tuples

---

### 5. consolidate_duplicates()

Find and remove duplicate files, keeping newest.

**Contract**:
```python
def consolidate_duplicates(
    doc_files: List[DocumentationFile],
    dry_run: bool = False
) -> RemovalReport:
    """
    Find duplicate documentation files and remove older versions.

    Args:
        doc_files: All documentation files
        dry_run: If True, don't actually delete files

    Returns:
        RemovalReport with duplicate removal results

    Implements: FR-006 (consolidate duplicate files, keep newest)
    """
```

**Duplicate Detection**:
1. Group files by base name (stem)
2. For each group with multiple files:
   - Sort by modification time (newest first)
   - Mark first as canonical
   - Mark rest as duplicates for removal

**Postconditions**:
- For each base name, only one file remains
- Kept file is the most recently modified
- All removed files are older versions

---

### 6. validate_tests()

Run test suite and compare to baseline.

**Contract**:
```python
def validate_tests(
    baseline_report: Optional[TestReport] = None,
    test_command: str = "pytest --tb=short"
) -> TestReport:
    """
    Run test suite and validate results.

    Args:
        baseline_report: Baseline test results for comparison (None = establish baseline)
        test_command: pytest command to execute

    Returns:
        TestReport with test execution results

    Raises:
        RuntimeError: If tests cannot be executed

    Implements: FR-009, FR-010 (run tests, verify 100% pass rate)
    """
```

**Validation Logic**:
- Execute test command
- Parse output for test counts
- Compare to baseline if provided
- Return TestReport with comparison results

**Postconditions**:
- TestReport.total_tests > 0
- TestReport.passed_tests + TestReport.failed_tests == TestReport.total_tests
- If baseline provided: TestReport.matches_baseline() checked

---

### 7. rollback_changes()

Rollback file changes using git.

**Contract**:
```python
def rollback_changes(
    operation_type: str,
    affected_paths: List[Path]
) -> RollbackReport:
    """
    Rollback file changes using git restore.

    Args:
        operation_type: Type of operation to rollback ("remove" | "move" | "update")
        affected_paths: Paths affected by operation

    Returns:
        RollbackReport with rollback results

    Raises:
        GitError: If git restore fails

    Implements: FR-011 (rollback changes if tests fail)
    """
```

**Rollback Strategy**:
- Use `git restore --staged .` to unstage changes
- Use `git restore .` to revert working directory
- Verify all affected paths are reverted

**Postconditions**:
- All affected paths are restored to pre-operation state
- Git index is clean (no staged changes)

---

### 8. check_broken_links()

Find broken links in documentation.

**Contract**:
```python
def check_broken_links(
    doc_root: Path,
    path_mapping: Dict[Path, Path] = None
) -> DocumentationUpdateReport:
    """
    Find broken links in Markdown documentation.

    Args:
        doc_root: Root directory to scan for .md files
        path_mapping: Map of old paths to new paths (from MoveReport)

    Returns:
        DocumentationUpdateReport with broken links

    Implements: FR-013 (check for broken documentation links)
    """
```

**Link Detection**:
- Parse all `.md` files for Markdown links: `[text](path)`
- Check if relative path targets exist
- Report broken links with source file and line number

**Postconditions**:
- All .md files scanned
- All relative links validated
- broken_links_found contains full context

---

### 9. update_documentation()

Update documentation links after file moves.

**Contract**:
```python
def update_documentation(
    path_mapping: Dict[Path, Path],
    doc_root: Path,
    dry_run: bool = False
) -> DocumentationUpdateReport:
    """
    Update documentation links after file moves.

    Args:
        path_mapping: Map of old paths to new paths (from MoveReport)
        doc_root: Root directory containing .md files
        dry_run: If True, don't actually update files

    Returns:
        DocumentationUpdateReport with update results

    Implements: FR-016 (update documentation that references moved files)
    """
```

**Update Logic**:
1. Scan all .md files for links
2. For each link matching old path in mapping:
   - Calculate new relative path from source to destination
   - Replace link in file
3. Track updates and broken links

**Postconditions**:
- All links to moved files are updated
- DOCUMENTATION_INDEX.md reflects new locations
- No broken links remain (unless they were already broken)

---

### 10. update_documentation_index()

Update DOCUMENTATION_INDEX.md with new file locations.

**Contract**:
```python
def update_documentation_index(
    index_file: Path,
    path_mapping: Dict[Path, Path],
    dry_run: bool = False
) -> bool:
    """
    Update DOCUMENTATION_INDEX.md with new file locations.

    Args:
        index_file: Path to DOCUMENTATION_INDEX.md
        path_mapping: Map of old paths to new paths
        dry_run: If True, don't actually update file

    Returns:
        True if updated successfully

    Implements: FR-017 (update DOCUMENTATION_INDEX.md)
    """
```

**Update Logic**:
- Parse DOCUMENTATION_INDEX.md
- Find all Markdown links
- Update links matching old paths
- Preserve formatting and structure

**Postconditions**:
- All links in index are valid
- Table structure preserved
- Quick Navigation section updated

---

## Operation Sequencing

Operations must be executed in specific order:

```
1. scan_repository()
   ↓
2. classify_files()
   ↓
3. validate_tests() [baseline]
   ↓
4. BATCH 1: remove_files(temporary/cache files)
   ↓ validate_tests() → ROLLBACK if fail
   ↓
5. BATCH 2: remove_files(old evaluation reports)
   ↓ validate_tests() → ROLLBACK if fail
   ↓
6. BATCH 3: consolidate_duplicates()
   ↓ validate_tests() → ROLLBACK if fail
   ↓
7. BATCH 4: remove_files(historical tracking)
   ↓ validate_tests() → ROLLBACK if fail
   ↓
8. BATCH 5: move_files(status files to docs/)
   ↓ validate_tests() → ROLLBACK if fail
   ↓
9. BATCH 6: update_documentation()
   ↓ validate_tests() → ROLLBACK if fail
   ↓
10. check_broken_links()
    ↓
11. update_documentation_index()
    ↓
12. validate_tests() [final]
```

## Error Handling

All operations follow Constitutional Principle VI (Explicit Error Handling):

**Error Categories**:
1. **Validation Errors**: Invalid inputs (ValueError)
2. **Permission Errors**: Cannot read/write files (PermissionError)
3. **Git Errors**: Git operations fail (GitError)
4. **Test Failures**: Tests don't pass (triggers rollback)

**Error Messages** must include:
- What failed
- Why it failed
- What file/path was involved
- How to fix it

**Example**:
```
Error: Cannot remove file
Path: /path/to/file.txt
Reason: File is referenced by tests/test_example.py:42
Fix: Remove reference from test before deleting file, or reclassify as ESSENTIAL
```

## Contract Test Requirements

Each operation requires a contract test:

1. `test_scan_repository_contract.py`
2. `test_classify_files_contract.py`
3. `test_remove_files_contract.py`
4. `test_move_files_contract.py`
5. `test_consolidate_duplicates_contract.py`
6. `test_validate_tests_contract.py`
7. `test_rollback_changes_contract.py`
8. `test_check_broken_links_contract.py`
9. `test_update_documentation_contract.py`
10. `test_update_documentation_index_contract.py`

Tests must verify:
- Preconditions (valid inputs)
- Postconditions (correct outputs)
- Error handling (invalid inputs raise correct exceptions)
- Dry run mode works (no actual changes)

---

**Phase 1 Contracts Complete**: 2025-10-08
