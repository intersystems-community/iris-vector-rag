# Data Model: Repository Cleanup

**Feature**: Repository Cleanup and Organization
**Date**: 2025-10-08
**Phase**: Phase 1 - Design

## Overview

This document defines the data structures used for repository cleanup operations. These models represent files, classification decisions, and cleanup operation results.

## Core Entities

### 1. FileCategory (Enum)

Categorization of files based on cleanup action.

```python
from enum import Enum

class FileCategory(Enum):
    """File classification for cleanup operations."""
    ESSENTIAL = "essential"        # Keep at repository root
    RELOCATABLE = "relocatable"    # Move to appropriate directory
    REMOVABLE = "removable"        # Delete during cleanup
    REVIEW_REQUIRED = "review"     # Manual review needed
```

**Usage**: Maps to FR-002 requirement for three-tier classification.

---

### 2. RepositoryFile

Base representation of any file in the repository.

```python
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Optional

@dataclass
class RepositoryFile:
    """Base class for repository files."""
    path: Path                          # Absolute path to file
    relative_path: Path                 # Path relative to repo root
    size_bytes: int                     # File size
    modified_time: datetime             # Last modification time
    category: FileCategory              # Classification
    reason: str                         # Justification for classification

    def is_tracked_by_git(self) -> bool:
        """Check if file is tracked by git."""
        pass

    def get_references(self) -> List[Path]:
        """Find files that reference this file."""
        pass
```

**Validation Rules** (from FR-012):
- `path` must exist at time of classification
- `category` must not be None
- `reason` must explain classification decision

**Relationships**:
- Subclassed by specific file types (DocumentationFile, GeneratedOutput, etc.)

---

### 3. TopLevelFile

File located at repository root.

```python
@dataclass
class TopLevelFile(RepositoryFile):
    """File at repository root level."""
    is_config: bool                     # Configuration file (docker-compose, .gitignore)
    is_documentation: bool              # Documentation file (README, guides)
    is_essential: bool                  # Essential for project function

    @staticmethod
    def classify(path: Path) -> FileCategory:
        """Classify top-level file based on FR-001, FR-002."""
        if path.name in ESSENTIAL_FILES:
            return FileCategory.ESSENTIAL
        if path.suffix in ['.md', '.txt'] and not is_historical(path):
            return FileCategory.RELOCATABLE
        if matches_removable_pattern(path):
            return FileCategory.REMOVABLE
        return FileCategory.REVIEW_REQUIRED
```

**Essential Files** (from research.md):
- README.md, USER_GUIDE.md, CLAUDE.md
- DOCUMENTATION_INDEX.md
- docker-compose*.yml, Makefile
- pyproject.toml, pytest.ini
- .gitignore, LICENSE

---

### 4. DocumentationFile

Markdown or text documentation.

```python
@dataclass
class DocumentationFile(RepositoryFile):
    """Documentation file (Markdown, text)."""
    purpose: str                        # What this doc describes
    target_location: Optional[Path]     # Where to move (if relocatable)
    is_duplicate: bool                  # True if duplicate exists
    duplicate_of: Optional[Path]        # Path to canonical version
    is_current: bool                    # True if actively maintained

    def find_duplicates(self, all_files: List['DocumentationFile']) -> List[Path]:
        """Find files with same base name."""
        basename = self.path.stem
        return [f.path for f in all_files
                if f.path.stem == basename and f.path != self.path]

    def is_newer_than(self, other: 'DocumentationFile') -> bool:
        """Compare modification times (for FR-006)."""
        return self.modified_time > other.modified_time
```

**Duplicate Handling** (FR-006):
- When duplicates found, keep most recently modified
- Delete older versions
- Example: Multiple SUMMARY.md files → keep newest

---

### 5. GeneratedOutput

Files created by tests, evaluations, or scripts.

```python
@dataclass
class GeneratedOutput(RepositoryFile):
    """Generated output from tests/evaluations."""
    generation_date: datetime           # When file was created
    generator: str                      # What created it (RAGAS, pytest, etc.)
    is_outdated: bool                   # True if regeneratable/old

    def is_removable(self, cutoff_date: datetime) -> bool:
        """Determine if output is old enough to remove (FR-007)."""
        return self.is_outdated or self.generation_date < cutoff_date
```

**Removable Generated Outputs** (FR-004, FR-007):
- `outputs/reports/ragas_evaluations/*.html`
- `outputs/reports/ragas_evaluations/*.json`
- `outputs/pipeline_verification_*.json`
- `.pytest_cache/`, `.coverage` files

---

### 6. StatusFile

Project status tracking files.

```python
@dataclass
class StatusFile(RepositoryFile):
    """Status tracking file (docs/docs/STATUS.md, docs/docs/TODO.md, etc.)."""
    is_current: bool                    # True if actively maintained
    target_location: Path               # Where to move (docs/)

    @staticmethod
    def should_move_to_docs(path: Path) -> bool:
        """Determine if status file should move to docs/ (FR-008)."""
        current_status_files = [
            'docs/docs/STATUS.md', 'docs/docs/PROGRESS.md', 'docs/docs/TODO.md', 'docs/docs/docs/CHANGELOG.md'
        ]
        return path.name in current_status_files
```

**Current Status Files** (FR-008):
- Move to docs/: docs/docs/STATUS.md, docs/docs/PROGRESS.md, docs/docs/TODO.md, docs/docs/docs/CHANGELOG.md

**Historical Status Files** (FR-004):
- Delete: `*_COMPLETION_SUMMARY.md`, `*_SESSION_NOTES.md`

---

### 7. TemporaryFile

Cache, build artifacts, temporary files.

```python
@dataclass
class TemporaryFile(RepositoryFile):
    """Temporary or cache file."""
    file_type: str                      # Type (cache, build, temp)

    @staticmethod
    def is_temporary(path: Path) -> bool:
        """Check if file is temporary/cache (FR-004a)."""
        temp_patterns = [
            '*.pyc', '__pycache__', '.pytest_cache',
            '.coverage', '*.egg-info', 'dist/', 'build/'
        ]
        return any(path.match(pattern) for pattern in temp_patterns)
```

---

## Operation Results

### 8. FileInventory

Complete repository scan results.

```python
@dataclass
class FileInventory:
    """Results of repository scan."""
    repo_root: Path
    scan_time: datetime
    total_files: int
    files_by_category: Dict[FileCategory, List[RepositoryFile]]

    def get_top_level_files(self) -> List[TopLevelFile]:
        """Get all files at repository root (FR-001)."""
        return [f for f in self.all_files if f.relative_path.parent == Path('.')]

    def get_removable_files(self) -> List[RepositoryFile]:
        """Get all files marked for removal (FR-004)."""
        return self.files_by_category.get(FileCategory.REMOVABLE, [])

    def get_relocatable_files(self) -> List[RepositoryFile]:
        """Get all files to move (FR-003)."""
        return self.files_by_category.get(FileCategory.RELOCATABLE, [])
```

---

### 9. ClassificationReport

File classification results.

```python
@dataclass
class ClassificationReport:
    """Results of file classification."""
    total_files: int
    essential_count: int
    relocatable_count: int
    removable_count: int
    review_required_count: int
    classification_map: Dict[Path, FileCategory]
    classification_reasons: Dict[Path, str]

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Classification Summary:
- Essential: {self.essential_count}
- Relocatable: {self.relocatable_count}
- Removable: {self.removable_count}
- Review Required: {self.review_required_count}
Total: {self.total_files}
"""
```

---

### 10. RemovalReport

File removal operation results.

```python
@dataclass
class RemovalReport:
    """Results of file removal operation."""
    files_removed: List[Path]
    files_failed: List[Tuple[Path, str]]  # (path, error_message)
    bytes_freed: int
    removal_time: datetime

    def rollback_required(self) -> bool:
        """Check if any removals failed (FR-011)."""
        return len(self.files_failed) > 0
```

---

### 11. MoveReport

File relocation operation results.

```python
@dataclass
class MoveReport:
    """Results of file move operation."""
    files_moved: List[Tuple[Path, Path]]  # (source, destination)
    files_failed: List[Tuple[Path, str]]   # (source, error_message)
    directories_created: List[Path]
    move_time: datetime

    def get_path_mapping(self) -> Dict[Path, Path]:
        """Map old paths to new paths for doc updates (FR-016)."""
        return {src: dst for src, dst in self.files_moved}

    def rollback_required(self) -> bool:
        """Check if any moves failed (FR-011)."""
        return len(self.files_failed) > 0
```

---

### 12. TestReport

Test suite execution results.

```python
@dataclass
class TestReport:
    """Results of test suite execution."""
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    baseline_test_count: Optional[int]  # For comparison

    def matches_baseline(self) -> bool:
        """Check if test count matches baseline (FR-010)."""
        if self.baseline_test_count is None:
            return True
        return self.passed_tests == self.baseline_test_count

    def all_passed(self) -> bool:
        """Check if 100% tests passed (FR-009)."""
        return self.failed_tests == 0 and self.passed_tests > 0
```

---

### 13. DocumentationUpdateReport

Documentation link update results.

```python
@dataclass
class BrokenLink:
    """A broken documentation link."""
    source_file: Path
    link_text: str
    target_path: Path
    line_number: int

@dataclass
class DocumentationUpdateReport:
    """Results of documentation link updates."""
    files_updated: List[Path]
    links_updated: int
    broken_links_found: List[BrokenLink]
    update_time: datetime

    def has_broken_links(self) -> bool:
        """Check if broken links remain (FR-013)."""
        return len(self.broken_links_found) > 0
```

---

## Validation Rules

### File Classification Validation

From FR-001 through FR-008:

```python
def validate_classification(file: RepositoryFile) -> List[str]:
    """Validate file classification follows requirements."""
    errors = []

    # FR-002: Must have category
    if file.category is None:
        errors.append(f"{file.path}: Missing category")

    # FR-002: Must have reason
    if not file.reason:
        errors.append(f"{file.path}: Missing classification reason")

    # FR-003: Relocatable must have target
    if file.category == FileCategory.RELOCATABLE:
        if isinstance(file, (DocumentationFile, StatusFile)):
            if not file.target_location:
                errors.append(f"{file.path}: Relocatable but no target location")

    # FR-006: Duplicates must be handled
    if isinstance(file, DocumentationFile) and file.is_duplicate:
        if not file.duplicate_of:
            errors.append(f"{file.path}: Marked duplicate but no canonical version")

    return errors
```

### Test Validation

From FR-009, FR-010:

```python
def validate_test_results(baseline: TestReport, current: TestReport) -> List[str]:
    """Validate test results match baseline."""
    errors = []

    # FR-009: All tests must pass
    if not current.all_passed():
        errors.append(f"Test failures detected: {current.failed_tests} failed")

    # FR-010: Test count must match baseline
    if not current.matches_baseline():
        errors.append(
            f"Test count mismatch: baseline={baseline.passed_tests}, "
            f"current={current.passed_tests}"
        )

    return errors
```

---

## Entity Relationships

```
FileInventory
├── TopLevelFile (Essential)
├── TopLevelFile (Relocatable)
│   ├── DocumentationFile
│   │   └── duplicate_of → DocumentationFile
│   └── StatusFile
│       └── target_location: docs/
├── GeneratedOutput (Removable)
├── TemporaryFile (Removable)
└── DocumentationFile (Review Required)

ClassificationReport
└── classification_map → RepositoryFile instances

RemovalReport
└── files_removed → RepositoryFile.path

MoveReport
└── files_moved → (RepositoryFile.path, target_location)

TestReport
└── baseline_test_count for comparison

DocumentationUpdateReport
└── broken_links_found → BrokenLink instances
```

---

## State Transitions

### File Lifecycle During Cleanup

```
1. SCAN PHASE
   Unscanned → RepositoryFile(category=REVIEW_REQUIRED)

2. CLASSIFICATION PHASE
   REVIEW_REQUIRED → ESSENTIAL | RELOCATABLE | REMOVABLE

3. OPERATION PHASE
   ESSENTIAL → No action
   RELOCATABLE → Move to target_location
   REMOVABLE → Delete

4. VALIDATION PHASE
   Run tests, check links
   If failed → ROLLBACK
   If passed → COMMIT
```

### Test Validation State Machine

```
START
  ↓
RUN_BASELINE_TESTS
  ↓
EXECUTE_CLEANUP_BATCH
  ↓
RUN_CURRENT_TESTS
  ↓
COMPARE_RESULTS
  ├─ Match → COMMIT_BATCH
  └─ Mismatch → ROLLBACK_BATCH → REPORT_ERROR
```

---

## Summary

This data model supports all functional requirements:
- **FR-001, FR-002**: TopLevelFile classification
- **FR-003**: MoveReport with target locations
- **FR-004**: TemporaryFile, GeneratedOutput removal
- **FR-005, FR-006**: DocumentationFile with duplicate handling
- **FR-007**: GeneratedOutput is_removable logic
- **FR-008**: StatusFile with target_location
- **FR-009, FR-010**: TestReport validation
- **FR-011**: RemovalReport/MoveReport rollback detection
- **FR-012**: RepositoryFile.get_references()
- **FR-013**: DocumentationUpdateReport with BrokenLink
- **FR-016, FR-017**: MoveReport.get_path_mapping() for doc updates

---

**Phase 1 Data Model Complete**: 2025-10-08
