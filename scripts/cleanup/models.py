"""
Data models for repository cleanup operations.

Implements the data model from specs/037-clean-up-this/data-model.md
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class FileCategory(Enum):
    """File classification for cleanup operations."""
    ESSENTIAL = "essential"        # Keep at repository root
    RELOCATABLE = "relocatable"    # Move to appropriate directory
    REMOVABLE = "removable"        # Delete during cleanup
    REVIEW_REQUIRED = "review"     # Manual review needed


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
        """Check if file is tracked by git (simplified - would need git module)."""
        # Placeholder - actual implementation would use GitPython
        return not any(part.startswith('.') for part in self.path.parts)

    def get_references(self) -> List[Path]:
        """Find files that reference this file (simplified)."""
        # Placeholder - actual implementation would grep for references
        return []


@dataclass
class TopLevelFile(RepositoryFile):
    """File at repository root level."""
    is_config: bool = False             # Configuration file
    is_documentation: bool = False      # Documentation file
    is_essential: bool = False          # Essential for project function


@dataclass
class DocumentationFile(RepositoryFile):
    """Documentation file (Markdown, text)."""
    purpose: str = ""                   # What this doc describes
    target_location: Optional[Path] = None  # Where to move (if relocatable)
    is_duplicate: bool = False          # True if duplicate exists
    duplicate_of: Optional[Path] = None  # Path to canonical version
    is_current: bool = True             # True if actively maintained


@dataclass
class GeneratedOutput(RepositoryFile):
    """Generated output from tests/evaluations."""
    generation_date: datetime = field(default_factory=datetime.now)
    generator: str = ""                 # What created it
    is_outdated: bool = True            # True if regeneratable/old


@dataclass
class StatusFile(RepositoryFile):
    """Status tracking file (STATUS.md, TODO.md, etc.)."""
    is_current: bool = True             # True if actively maintained
    target_location: Optional[Path] = None  # Where to move


@dataclass
class TemporaryFile(RepositoryFile):
    """Temporary or cache file."""
    file_type: str = "cache"            # Type (cache, build, temp)


@dataclass
class FileInventory:
    """Results of repository scan."""
    repo_root: Path
    scan_time: datetime
    total_files: int
    files_by_category: Dict[FileCategory, List[RepositoryFile]] = field(default_factory=dict)
    all_files: List[RepositoryFile] = field(default_factory=list)

    def get_top_level_files(self) -> List[RepositoryFile]:
        """Get all files at repository root."""
        return [f for f in self.all_files
                if len(f.relative_path.parts) == 1]

    def get_removable_files(self) -> List[RepositoryFile]:
        """Get all files marked for removal."""
        return self.files_by_category.get(FileCategory.REMOVABLE, [])

    def get_relocatable_files(self) -> List[RepositoryFile]:
        """Get all files to move."""
        return self.files_by_category.get(FileCategory.RELOCATABLE, [])


@dataclass
class ClassificationReport:
    """Results of file classification."""
    total_files: int
    essential_count: int
    relocatable_count: int
    removable_count: int
    review_required_count: int
    classification_map: Dict[Path, FileCategory] = field(default_factory=dict)
    classification_reasons: Dict[Path, str] = field(default_factory=dict)

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


@dataclass
class RemovalReport:
    """Results of file removal operation."""
    files_removed: List[Path] = field(default_factory=list)
    files_failed: List[Tuple[Path, str]] = field(default_factory=list)
    bytes_freed: int = 0
    removal_time: datetime = field(default_factory=datetime.now)

    def rollback_required(self) -> bool:
        """Check if any removals failed."""
        return len(self.files_failed) > 0


@dataclass
class MoveReport:
    """Results of file move operation."""
    files_moved: List[Tuple[Path, Path]] = field(default_factory=list)
    files_failed: List[Tuple[Path, str]] = field(default_factory=list)
    directories_created: List[Path] = field(default_factory=list)
    move_time: datetime = field(default_factory=datetime.now)

    def get_path_mapping(self) -> Dict[Path, Path]:
        """Map old paths to new paths for doc updates."""
        return {src: dst for src, dst in self.files_moved}

    def rollback_required(self) -> bool:
        """Check if any moves failed."""
        return len(self.files_failed) > 0


@dataclass
class TestReport:
    """Results of test suite execution."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    execution_time: float = 0.0
    baseline_test_count: Optional[int] = None

    def matches_baseline(self) -> bool:
        """Check if test count matches baseline."""
        if self.baseline_test_count is None:
            return True
        return self.passed_tests == self.baseline_test_count

    def all_passed(self) -> bool:
        """Check if 100% tests passed."""
        return self.failed_tests == 0 and self.passed_tests > 0


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
    files_updated: List[Path] = field(default_factory=list)
    links_updated: int = 0
    broken_links_found: List[BrokenLink] = field(default_factory=list)
    update_time: datetime = field(default_factory=datetime.now)

    def has_broken_links(self) -> bool:
        """Check if broken links remain."""
        return len(self.broken_links_found) > 0


@dataclass
class RollbackReport:
    """Results of rollback operation."""
    operation_type: str
    affected_paths: List[Path] = field(default_factory=list)
    rollback_successful: bool = False
    error_message: str = ""
