"""
Cleanup operations implementation.

Implements operations from specs/037-clean-up-this/contracts/cleanup-operations.md
"""

import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    BrokenLink,
    ClassificationReport,
    DocumentationFile,
    DocumentationUpdateReport,
    FileCategory,
    FileInventory,
    GeneratedOutput,
    MoveReport,
    RemovalReport,
    RepositoryFile,
    RollbackReport,
    StatusFile,
    TemporaryFile,
    TestReport,
    TopLevelFile,
)

# Essential files that must remain at repository root
ESSENTIAL_FILES = {
    'README.md', 'USER_GUIDE.md', 'CLAUDE.md',
    'DOCUMENTATION_INDEX.md', 'DOCUMENTATION_AUDIT_REPORT.md',
    'LICENSE', 'Makefile', 'pyproject.toml', 'pytest.ini',
    '.gitignore', '.env.example', 'setup.py',
}

# Docker and config files
ESSENTIAL_PATTERNS = [
    'docker-compose*.yml',
    'requirements*.txt',
]

# Status files to move to docs/
STATUS_FILES = {
    'STATUS.md', 'PROGRESS.md', 'TODO.md', 'CHANGELOG.md'
}

# Patterns for temporary/cache files
TEMP_PATTERNS = [
    '*.pyc', '__pycache__', '.pytest_cache',
    '.coverage', '*.egg-info', 'dist/', 'build/',
    '.mypy_cache', '.ruff_cache'
]

# Patterns for old generated outputs
OLD_OUTPUT_PATTERNS = [
    'outputs/pipeline_verification_*.json',
    'outputs/reports/ragas_evaluations/*.html',
    'outputs/reports/ragas_evaluations/*.json',
]

# Patterns for historical tracking files
HISTORICAL_PATTERNS = [
    '*_COMPLETION_SUMMARY.md',
    '*_SESSION_NOTES.md',
    'MORNING_BRIEFING.md',
]


def scan_repository(
    repo_root: Path,
    exclude_dirs: Optional[List[str]] = None
) -> FileInventory:
    """
    Scan repository and create file inventory.

    Args:
        repo_root: Absolute path to repository root
        exclude_dirs: Directories to skip

    Returns:
        FileInventory with all scanned files
    """
    if not repo_root.exists() or not repo_root.is_dir():
        raise ValueError(f"Repository root does not exist or is not a directory: {repo_root}")

    if exclude_dirs is None:
        exclude_dirs = ['.git', '.venv', 'node_modules', '__pycache__']

    all_files = []
    scan_time = datetime.now()

    for root, dirs, files in os.walk(repo_root):
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            relative_path = file_path.relative_to(repo_root)

            try:
                stat = file_path.stat()
                repo_file = RepositoryFile(
                    path=file_path,
                    relative_path=relative_path,
                    size_bytes=stat.st_size,
                    modified_time=datetime.fromtimestamp(stat.st_mtime),
                    category=FileCategory.REVIEW_REQUIRED,
                    reason="Not yet classified"
                )
                all_files.append(repo_file)
            except (OSError, PermissionError) as e:
                print(f"Warning: Could not access {file_path}: {e}")
                continue

    return FileInventory(
        repo_root=repo_root,
        scan_time=scan_time,
        total_files=len(all_files),
        all_files=all_files,
        files_by_category={}
    )


def classify_files(inventory: FileInventory) -> ClassificationReport:
    """
    Classify all files in inventory.

    Args:
        inventory: FileInventory from scan_repository()

    Returns:
        ClassificationReport with classification decisions
    """
    if not inventory.all_files:
        raise ValueError("Inventory is empty - no files to classify")

    classification_map = {}
    classification_reasons = {}
    counts = {cat: 0 for cat in FileCategory}

    for file in inventory.all_files:
        category, reason = _classify_single_file(file, inventory.repo_root)
        file.category = category
        file.reason = reason
        classification_map[file.path] = category
        classification_reasons[file.path] = reason
        counts[category] += 1

    # Group files by category
    inventory.files_by_category = {
        cat: [f for f in inventory.all_files if f.category == cat]
        for cat in FileCategory
    }

    return ClassificationReport(
        total_files=inventory.total_files,
        essential_count=counts[FileCategory.ESSENTIAL],
        relocatable_count=counts[FileCategory.RELOCATABLE],
        removable_count=counts[FileCategory.REMOVABLE],
        review_required_count=counts[FileCategory.REVIEW_REQUIRED],
        classification_map=classification_map,
        classification_reasons=classification_reasons
    )


def _classify_single_file(file: RepositoryFile, repo_root: Path) -> tuple[FileCategory, str]:
    """Classify a single file."""
    filename = file.path.name
    relative = str(file.relative_path)

    # Essential files
    if filename in ESSENTIAL_FILES:
        return FileCategory.ESSENTIAL, f"Essential file: {filename}"

    # Essential patterns (docker-compose*, requirements*)
    for pattern in ESSENTIAL_PATTERNS:
        if file.path.match(pattern):
            return FileCategory.ESSENTIAL, f"Essential pattern: {pattern}"

    # Status files (move to docs/)
    if filename in STATUS_FILES:
        return FileCategory.RELOCATABLE, f"Status file - move to docs/"

    # Temporary/cache files
    for pattern in TEMP_PATTERNS:
        if file.path.match(f"**/{pattern}") or filename.endswith(pattern.strip('*')):
            return FileCategory.REMOVABLE, f"Temporary/cache file: {pattern}"

    # Old generated outputs
    for pattern in OLD_OUTPUT_PATTERNS:
        if file.path.match(pattern):
            return FileCategory.REMOVABLE, f"Old generated output: {pattern}"

    # Historical tracking files
    for pattern in HISTORICAL_PATTERNS:
        if file.path.match(pattern):
            return FileCategory.REMOVABLE, f"Historical tracking file: {pattern}"

    # Duplicate detection (simplified - check for multiple files with same stem)
    if file.path.suffix == '.md' and 'SUMMARY' in filename.upper():
        return FileCategory.REVIEW_REQUIRED, "Potential duplicate - manual review needed"

    # Keep source code, tests, docs structure
    if any(part in relative for part in ['iris_rag/', 'tests/', 'docs/', 'scripts/', 'specs/']):
        return FileCategory.ESSENTIAL, "Part of source code/tests/docs structure"

    # Default: review required
    return FileCategory.REVIEW_REQUIRED, "Requires manual classification"


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
    """
    report = RemovalReport()

    for file in removable_files:
        try:
            if dry_run:
                print(f"[DRY RUN] Would remove: {file.path}")
                report.files_removed.append(file.path)
                report.bytes_freed += file.size_bytes
            else:
                if file.path.is_file():
                    report.bytes_freed += file.size_bytes
                    file.path.unlink()
                    report.files_removed.append(file.path)
                    print(f"Removed: {file.path}")
                elif file.path.is_dir():
                    shutil.rmtree(file.path)
                    report.files_removed.append(file.path)
                    print(f"Removed directory: {file.path}")
        except Exception as e:
            report.files_failed.append((file.path, str(e)))
            print(f"Failed to remove {file.path}: {e}")

    return report


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
    """
    report = MoveReport()

    for file in relocatable_files:
        if file.path not in target_map:
            continue

        dest = target_map[file.path]

        try:
            if dry_run:
                print(f"[DRY RUN] Would move: {file.path} -> {dest}")
                report.files_moved.append((file.path, dest))
            else:
                # Create target directory if needed
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.parent not in report.directories_created:
                    report.directories_created.append(dest.parent)

                # Move file
                shutil.move(str(file.path), str(dest))
                report.files_moved.append((file.path, dest))
                print(f"Moved: {file.path} -> {dest}")
        except Exception as e:
            report.files_failed.append((file.path, str(e)))
            print(f"Failed to move {file.path}: {e}")

    return report


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
    """
    # Group by basename
    groups: Dict[str, List[DocumentationFile]] = {}
    for doc in doc_files:
        stem = doc.path.stem
        if stem not in groups:
            groups[stem] = []
        groups[stem].append(doc)

    removable = []
    for stem, files in groups.items():
        if len(files) > 1:
            # Sort by modification time (newest first)
            files.sort(key=lambda f: f.modified_time, reverse=True)
            # Keep first (newest), remove rest
            for old_file in files[1:]:
                old_file.category = FileCategory.REMOVABLE
                old_file.reason = f"Duplicate of {files[0].path.name} (older version)"
                removable.append(old_file)

    return remove_files(removable, dry_run=dry_run)


def validate_tests(
    baseline_report: Optional[TestReport] = None,
    test_command: str = "pytest --tb=short"
) -> TestReport:
    """
    Run test suite and validate results.

    Args:
        baseline_report: Baseline test results for comparison
        test_command: pytest command to execute

    Returns:
        TestReport with test execution results
    """
    try:
        result = subprocess.run(
            test_command.split(),
            capture_output=True,
            text=True,
            timeout=300
        )
        output = result.stdout + result.stderr

        # Parse pytest output (simplified)
        passed = failed = 0
        for line in output.split('\n'):
            if ' passed' in line:
                match = re.search(r'(\d+) passed', line)
                if match:
                    passed = int(match.group(1))
            if ' failed' in line:
                match = re.search(r'(\d+) failed', line)
                if match:
                    failed = int(match.group(1))

        return TestReport(
            total_tests=passed + failed,
            passed_tests=passed,
            failed_tests=failed,
            execution_time=0.0,
            baseline_test_count=baseline_report.passed_tests if baseline_report else None
        )
    except Exception as e:
        print(f"Test execution failed: {e}")
        return TestReport(
            total_tests=0,
            passed_tests=0,
            failed_tests=1,
            baseline_test_count=baseline_report.passed_tests if baseline_report else None
        )


def rollback_changes(
    operation_type: str,
    affected_paths: List[Path]
) -> RollbackReport:
    """
    Rollback file changes using git restore.

    Args:
        operation_type: Type of operation to rollback
        affected_paths: Paths affected by operation

    Returns:
        RollbackReport with rollback results
    """
    report = RollbackReport(
        operation_type=operation_type,
        affected_paths=affected_paths
    )

    try:
        # Restore staged changes
        subprocess.run(['git', 'restore', '--staged', '.'], check=True)
        # Restore working directory
        subprocess.run(['git', 'restore', '.'], check=True)

        report.rollback_successful = True
        print(f"Rollback successful for {operation_type}")
    except Exception as e:
        report.rollback_successful = False
        report.error_message = str(e)
        print(f"Rollback failed: {e}")

    return report


def check_broken_links(
    doc_root: Path,
    path_mapping: Optional[Dict[Path, Path]] = None
) -> DocumentationUpdateReport:
    """
    Find broken links in Markdown documentation.

    Args:
        doc_root: Root directory to scan for .md files
        path_mapping: Map of old paths to new paths

    Returns:
        DocumentationUpdateReport with broken links
    """
    report = DocumentationUpdateReport()

    for md_file in doc_root.rglob('*.md'):
        try:
            content = md_file.read_text()
            for line_no, line in enumerate(content.split('\n'), 1):
                # Find Markdown links: [text](path)
                for match in re.finditer(r'\[([^\]]+)\]\(([^\)]+)\)', line):
                    link_text = match.group(1)
                    link_path = match.group(2)

                    # Skip external links
                    if link_path.startswith(('http://', 'https://', '#')):
                        continue

                    # Check if relative path exists
                    target = (md_file.parent / link_path).resolve()
                    if not target.exists():
                        report.broken_links_found.append(BrokenLink(
                            source_file=md_file,
                            link_text=link_text,
                            target_path=target,
                            line_number=line_no
                        ))
        except Exception as e:
            print(f"Error checking links in {md_file}: {e}")

    return report


def update_documentation(
    path_mapping: Dict[Path, Path],
    doc_root: Path,
    dry_run: bool = False
) -> DocumentationUpdateReport:
    """
    Update documentation links after file moves.

    Args:
        path_mapping: Map of old paths to new paths
        doc_root: Root directory containing .md files
        dry_run: If True, don't actually update files

    Returns:
        DocumentationUpdateReport with update results
    """
    report = DocumentationUpdateReport()

    for md_file in doc_root.rglob('*.md'):
        try:
            content = md_file.read_text()
            updated_content = content
            links_updated_in_file = 0

            # Replace old paths with new paths in links
            for old_path, new_path in path_mapping.items():
                old_relative = str(old_path.name)  # Simplified
                new_relative = str(new_path.relative_to(doc_root) if new_path.is_relative_to(doc_root) else new_path.name)

                if old_relative in updated_content:
                    updated_content = updated_content.replace(old_relative, new_relative)
                    links_updated_in_file += 1

            if links_updated_in_file > 0:
                if not dry_run:
                    md_file.write_text(updated_content)
                report.files_updated.append(md_file)
                report.links_updated += links_updated_in_file
                print(f"Updated {links_updated_in_file} links in {md_file}")
        except Exception as e:
            print(f"Error updating {md_file}: {e}")

    return report


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
    """
    if not index_file.exists():
        print(f"Index file not found: {index_file}")
        return False

    try:
        content = index_file.read_text()
        updated_content = content

        # Update links in index
        for old_path, new_path in path_mapping.items():
            old_link = f"]({old_path.name})"
            new_link = f"]({new_path})"
            if old_link in updated_content:
                updated_content = updated_content.replace(old_link, new_link)

        if updated_content != content:
            if not dry_run:
                index_file.write_text(updated_content)
            print(f"Updated DOCUMENTATION_INDEX.md")
            return True
        else:
            print("No updates needed in DOCUMENTATION_INDEX.md")
            return True
    except Exception as e:
        print(f"Error updating index: {e}")
        return False
