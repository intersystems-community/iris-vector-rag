#!/usr/bin/env python3
"""
Corrected Repository Synchronization Script

HARDCODED CORRECT DIRECTION: FROM internal GitLab repository TO public GitHub repository

This script handles:
1. Copying updated files and directories from internal GitLab repo to public GitHub repo
2. Filtering out internal/private content using exclude patterns
3. Staging and committing changes to public repository
4. Pushing to public GitHub repository

Note: Some files like uv.lock are excluded from sync because they are ignored by the target
repository's .gitignore file, which would cause git staging failures during the sync process.

Usage:
    python scripts/sync_to_public.py --sync-all
    python scripts/sync_to_public.py --sync-all --push
    python scripts/sync_to_public.py --validate-sync
"""

import argparse
import fnmatch
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SyncConfig:
    """Configuration for repository synchronization."""

    source_repo_path: str  # Internal GitLab repo (source)
    target_repo_path: str  # Public GitHub repo (target)
    files_to_sync: List[Dict[str, str]]
    directories_to_sync: List[Dict[str, any]]
    git_branch: str = "main"
    commit_message_template: str = "sync: update from internal repository"


@dataclass
class SyncResult:
    """Result of a synchronization operation."""

    success: bool
    files_synced: List[str]
    directories_synced: List[str]
    commit_hash: Optional[str] = None
    error_message: Optional[str] = None
    changes_detected: bool = False


class PublicRepositorySynchronizer:
    """Handles synchronization FROM internal GitLab repo TO public GitHub repo."""

    def __init__(self, config: SyncConfig):
        self.config = config
        self.source_repo = Path(config.source_repo_path)  # Internal GitLab repo
        self.target_repo = Path(config.target_repo_path)  # Public GitHub repo

        # Validate paths
        if not self.source_repo.exists():
            raise ValueError(
                f"Source repository path does not exist: {config.source_repo_path}"
            )
        if not self.target_repo.exists():
            raise ValueError(
                f"Target repository path does not exist: {config.target_repo_path}"
            )

    def sync_all_content(self, dry_run: bool = False) -> SyncResult:
        """
        Synchronize all content FROM internal repo TO public repo.

        Args:
            dry_run: If True, only show what would be synced without making changes

        Returns:
            SyncResult with details of the synchronization
        """
        logger.info(
            "🔄 Starting synchronization FROM internal GitLab TO public GitHub..."
        )

        files_synced = []
        directories_synced = []
        changes_detected = False

        try:
            # Sync individual files
            file_result = self._sync_files(dry_run)
            files_synced.extend(file_result.files_synced)
            changes_detected = changes_detected or file_result.changes_detected

            # Sync directories
            dir_result = self._sync_directories(dry_run)
            directories_synced.extend(dir_result.directories_synced)
            changes_detected = changes_detected or dir_result.changes_detected

            if dry_run:
                return SyncResult(
                    success=True,
                    files_synced=[],
                    directories_synced=[],
                    changes_detected=changes_detected,
                )

            if not files_synced and not directories_synced:
                logger.info("✅ No content needed synchronization")
                return SyncResult(
                    success=True,
                    files_synced=[],
                    directories_synced=[],
                    changes_detected=False,
                )

            return SyncResult(
                success=True,
                files_synced=files_synced,
                directories_synced=directories_synced,
                changes_detected=True,
            )

        except Exception as e:
            logger.error(f"❌ Synchronization failed: {e}")
            return SyncResult(
                success=False,
                files_synced=[],
                directories_synced=[],
                error_message=str(e),
            )

    def _sync_files(self, dry_run: bool = False) -> SyncResult:
        """Synchronize individual files FROM source TO target."""
        logger.info("📄 Synchronizing individual files...")

        files_synced = []
        changes_detected = False

        for file_mapping in self.config.files_to_sync:
            source_path = self.source_repo / file_mapping["source"]
            target_path = self.target_repo / file_mapping["target"]

            if not source_path.exists():
                logger.warning(f"⚠️ Source file not found: {source_path}")
                continue

            # Check if files are different
            if self._files_differ(source_path, target_path):
                changes_detected = True

                if dry_run:
                    logger.info(
                        f"📝 Would sync: {file_mapping['source']} → {file_mapping['target']}"
                    )
                else:
                    # Ensure target directory exists
                    target_path.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file FROM source TO target
                    shutil.copy2(source_path, target_path)
                    logger.info(
                        f"✅ Synced: {file_mapping['source']} → {file_mapping['target']}"
                    )
                    files_synced.append(file_mapping["target"])
            else:
                logger.debug(f"📄 No changes: {file_mapping['source']}")

        return SyncResult(
            success=True,
            files_synced=files_synced,
            directories_synced=[],
            changes_detected=changes_detected,
        )

    def _sync_directories(self, dry_run: bool = False) -> SyncResult:
        """Synchronize directories FROM source TO target with filtering."""
        logger.info("📁 Synchronizing directories...")

        directories_synced = []
        changes_detected = False

        for dir_mapping in self.config.directories_to_sync:
            source_dir = self.source_repo / dir_mapping["source"]
            target_dir = self.target_repo / dir_mapping["target"]
            exclude_patterns = dir_mapping.get("exclude_patterns", [])

            if not source_dir.exists():
                logger.warning(f"⚠️ Source directory not found: {source_dir}")
                continue

            # Check if directory sync is needed
            if self._directory_sync_needed(source_dir, target_dir, exclude_patterns):
                changes_detected = True

                if dry_run:
                    logger.info(
                        f"📝 Would sync directory: {dir_mapping['source']} → {dir_mapping['target']}"
                    )
                    self._preview_directory_changes(
                        source_dir, target_dir, exclude_patterns
                    )
                else:
                    # Sync directory with filtering FROM source TO target
                    synced = self._sync_directory_with_filtering(
                        source_dir, target_dir, exclude_patterns
                    )
                    if synced:
                        logger.info(
                            f"✅ Synced directory: {dir_mapping['source']} → {dir_mapping['target']}"
                        )
                        directories_synced.append(dir_mapping["target"])
            else:
                logger.debug(f"📁 No changes: {dir_mapping['source']}")

        return SyncResult(
            success=True,
            files_synced=[],
            directories_synced=directories_synced,
            changes_detected=changes_detected,
        )

    def _directory_sync_needed(
        self, source_dir: Path, target_dir: Path, exclude_patterns: List[str]
    ) -> bool:
        """Check if directory synchronization is needed."""
        if not target_dir.exists():
            return True

        # Get filtered file lists
        source_files = self._get_filtered_files(source_dir, exclude_patterns)
        target_files = self._get_filtered_files(target_dir, exclude_patterns)

        # Check if file sets are different
        if source_files != target_files:
            return True

        # Check if any files have different content
        for rel_path in source_files:
            source_file = source_dir / rel_path
            target_file = target_dir / rel_path

            if self._files_differ(source_file, target_file):
                return True

        return False

    def _get_filtered_files(
        self, directory: Path, exclude_patterns: List[str]
    ) -> Set[str]:
        """Get set of relative file paths in directory, excluding patterns."""
        files = set()

        if not directory.exists():
            return files

        for file_path in directory.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(directory)
                rel_path_str = str(rel_path)

                # Check if file should be excluded
                excluded = False
                for pattern in exclude_patterns:
                    if fnmatch.fnmatch(rel_path_str, pattern) or fnmatch.fnmatch(
                        file_path.name, pattern
                    ):
                        excluded = True
                        break

                if not excluded:
                    files.add(rel_path_str)

        return files

    def _sync_directory_with_filtering(
        self, source_dir: Path, target_dir: Path, exclude_patterns: List[str]
    ) -> bool:
        """Synchronize directory with filtering FROM source TO target, returns True if changes were made."""
        changes_made = False

        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)

        # Get source files (filtered)
        source_files = self._get_filtered_files(source_dir, exclude_patterns)

        # Copy/update files FROM source TO target
        for rel_path_str in source_files:
            source_file = source_dir / rel_path_str
            target_file = target_dir / rel_path_str

            # Ensure target subdirectory exists
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Copy if different or missing
            if not target_file.exists() or self._files_differ(source_file, target_file):
                shutil.copy2(source_file, target_file)
                changes_made = True
                logger.debug(f"  📄 Copied: {rel_path_str}")

        # Remove files in target that don't exist in source (filtered)
        if target_dir.exists():
            target_files = self._get_filtered_files(target_dir, exclude_patterns)
            for rel_path_str in target_files:
                if rel_path_str not in source_files:
                    target_file = target_dir / rel_path_str
                    target_file.unlink()
                    changes_made = True
                    logger.debug(f"  🗑️ Removed: {rel_path_str}")

            # Remove empty directories
            self._remove_empty_directories(target_dir)

        return changes_made

    def _preview_directory_changes(
        self, source_dir: Path, target_dir: Path, exclude_patterns: List[str]
    ):
        """Preview what changes would be made to a directory."""
        source_files = self._get_filtered_files(source_dir, exclude_patterns)
        target_files = (
            self._get_filtered_files(target_dir, exclude_patterns)
            if target_dir.exists()
            else set()
        )

        # Files to add/update
        for rel_path_str in source_files:
            source_file = source_dir / rel_path_str
            target_file = target_dir / rel_path_str

            if not target_file.exists():
                logger.info(f"    ➕ Would add: {rel_path_str}")
            elif self._files_differ(source_file, target_file):
                logger.info(f"    📝 Would update: {rel_path_str}")

        # Files to remove
        for rel_path_str in target_files:
            if rel_path_str not in source_files:
                logger.info(f"    ➖ Would remove: {rel_path_str}")

    def _remove_empty_directories(self, directory: Path):
        """Remove empty directories recursively."""
        for subdir in directory.rglob("*"):
            if subdir.is_dir() and not any(subdir.iterdir()):
                try:
                    subdir.rmdir()
                    logger.debug(
                        f"  🗑️ Removed empty directory: {subdir.relative_to(directory)}"
                    )
                except OSError:
                    pass  # Directory not empty or permission issue

    def commit_and_push(
        self, sync_result: SyncResult, push: bool = False
    ) -> SyncResult:
        """
        Commit synced content to public repo and optionally push to GitHub.

        Args:
            sync_result: Result from sync_all_content
            push: Whether to push to GitHub

        Returns:
            Updated SyncResult with commit information
        """
        if not sync_result.success or (
            not sync_result.files_synced and not sync_result.directories_synced
        ):
            logger.info("📝 No changes to commit")
            return sync_result

        try:
            # Change to target repository directory (public GitHub repo)
            original_cwd = os.getcwd()
            os.chdir(self.target_repo)

            # Stage all changes
            all_changes = sync_result.files_synced + sync_result.directories_synced
            for change in all_changes:
                result = subprocess.run(
                    ["git", "add", change], capture_output=True, text=True, check=True
                )
                logger.info(f"📋 Staged: {change}")

            # Create commit message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = (
                f"{self.config.commit_message_template}\n\nContent updated:\n"
            )

            if sync_result.files_synced:
                commit_message += "\nFiles:\n"
                for file_path in sync_result.files_synced:
                    commit_message += f"- {file_path}\n"

            if sync_result.directories_synced:
                commit_message += "\nDirectories:\n"
                for dir_path in sync_result.directories_synced:
                    commit_message += f"- {dir_path}\n"

            commit_message += f"\nSynced at: {timestamp}"

            # Commit changes
            result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                capture_output=True,
                text=True,
                check=True,
            )

            # Extract commit hash
            commit_hash = result.stdout.strip().split()[1] if result.stdout else None
            logger.info(f"✅ Committed changes to public repo: {commit_hash}")

            # Push if requested
            if push:
                result = subprocess.run(
                    ["git", "push", "origin", self.config.git_branch],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logger.info(
                    f"🚀 Pushed to public GitHub repository on {self.config.git_branch}"
                )

            sync_result.commit_hash = commit_hash
            return sync_result

        except subprocess.CalledProcessError as e:
            error_msg = f"Git operation failed: {e.stderr}"
            logger.error(f"❌ {error_msg}")
            sync_result.success = False
            sync_result.error_message = error_msg
            return sync_result
        except Exception as e:
            error_msg = f"Commit/push failed: {e}"
            logger.error(f"❌ {error_msg}")
            sync_result.success = False
            sync_result.error_message = error_msg
            return sync_result
        finally:
            os.chdir(original_cwd)

    def _files_differ(self, file1: Path, file2: Path) -> bool:
        """Check if two files have different content."""
        if not file2.exists():
            return True

        try:
            # For binary files, compare file sizes first
            if file1.stat().st_size != file2.stat().st_size:
                return True

            # For text files, compare content
            if file1.suffix in [
                ".py",
                ".md",
                ".txt",
                ".yaml",
                ".yml",
                ".json",
                ".js",
                ".ts",
                ".css",
                ".html",
            ]:
                with open(file1, "r", encoding="utf-8", errors="ignore") as f1, open(
                    file2, "r", encoding="utf-8", errors="ignore"
                ) as f2:
                    return f1.read() != f2.read()
            else:
                # For binary files, compare byte by byte
                with open(file1, "rb") as f1, open(file2, "rb") as f2:
                    return f1.read() != f2.read()
        except Exception:
            # If we can't read the files, assume they're different
            return True


def get_default_config() -> SyncConfig:
    """Get default synchronization configuration with HARDCODED CORRECT DIRECTION."""
    script_dir = Path(__file__).parent.parent  # Go up from scripts/ to project root

    # HARDCODED CORRECT DIRECTION:
    # SOURCE: Internal GitLab repository (current directory)
    # TARGET: Public GitHub repository (../rag-templates-sanitized)
    source_repo_path = str(script_dir)  # Internal GitLab repo
    target_repo_path = str(
        script_dir.parent / "rag-templates-sanitized"
    )  # Public GitHub repo

    # Define files to synchronize FROM internal TO public
    files_to_sync = [
        # Core project files
        {"source": "README.md", "target": "README.md"},
        {"source": "ROADMAP.md", "target": "ROADMAP.md"},  # Public roadmap
        {"source": "pyproject.toml", "target": "pyproject.toml"},
        {"source": "setup.py", "target": "setup.py"},
        {"source": "requirements.txt", "target": "requirements.txt"},
        # Critical build and dependency files
        {"source": "Makefile", "target": "Makefile"},
        {"source": "pytest.ini", "target": "pytest.ini"},
        # Note: uv.lock is excluded because it's ignored by the target repository's .gitignore
        # This prevents git staging failures during the sync process
        # Docker and container files
        {"source": "docker-compose.yml", "target": "docker-compose.yml"},
        {"source": ".dockerignore", "target": ".dockerignore"},
        # Git configuration
        {"source": ".gitignore", "target": ".gitignore"},
        {"source": ".gitattributes", "target": ".gitattributes"},
        # Environment and setup scripts
        {"source": "activate_env.sh", "target": "activate_env.sh"},
        {"source": "module.xml", "target": "module.xml"},
        # Release and development files
        {"source": "CHANGELOG.md", "target": "CHANGELOG.md"},
        # Note: CLAUDE.md and .clinerules are internal files - NEVER sync to public
    ]

    # Define directories to synchronize FROM internal TO public
    directories_to_sync = [
        # Core library directories
        {
            "source": "common/",
            "target": "common/",
            "exclude_patterns": [
                "*.pyc",
                "__pycache__/",
                "*.log",
                "CLEANUP_SUMMARY.md",
                "temp_*",
            ],
        },
        {
            "source": "iris_rag/",
            "target": "iris_rag/",
            "exclude_patterns": ["*.pyc", "__pycache__/", "*.log", "temp_*"],
        },
        {
            "source": "rag_templates/",
            "target": "rag_templates/",
            "exclude_patterns": ["*.pyc", "__pycache__/", "*.log", "temp_*"],
        },
        # Configuration and setup
        {
            "source": "config/",
            "target": "config/",
            "exclude_patterns": [
                "*.pyc",
                "__pycache__/",
                "monitoring.json",
                "sync_config.yaml",
                "*.key",
                "*.secret",
            ],
        },
        # Complete documentation directory (instead of just two files)
        {
            "source": "docs/",
            "target": "docs/",
            "exclude_patterns": [
                "*.pyc",
                "__pycache__/",
                "*.log",
                "temp_*",
                "*.tmp",
                "INTERNAL_ROADMAP.md",  # Exclude internal roadmap from public sync
                "CRITICAL_SECURITY_AUDIT_REPORT.md",  # Contains sensitive API keys - NEVER sync
                "OBJECTSCRIPT_SYNTAX_LEARNING_REPORT.md",  # Internal ObjectScript learning notes
                "internal_*",  # Exclude any internal-prefixed docs
                "private_*",  # Exclude any private-prefixed docs
            ],
        },
        # Critical missing directories
        {
            "source": "quick_start/",
            "target": "quick_start/",
            "exclude_patterns": ["*.pyc", "__pycache__/", "*.log", "temp_*", "*.tmp"],
        },
        {
            "source": "tools/",
            "target": "tools/",
            "exclude_patterns": ["*.pyc", "__pycache__/", "*.log", "temp_*", "*.tmp"],
        },
        {
            "source": "benchmarks/",
            "target": "benchmarks/",
            "exclude_patterns": [
                "*.pyc",
                "__pycache__/",
                "*.log",
                "temp_*",
                "results_*",
                "*.tmp",
            ],
        },
        {
            "source": "examples/",
            "target": "examples/",
            "exclude_patterns": ["*.pyc", "__pycache__/", "*.log", "temp_*", "*.tmp"],
        },
        # Scripts directory with comprehensive filtering
        {
            "source": "scripts/",
            "target": "scripts/",
            "exclude_patterns": [
                "*.pyc",
                "__pycache__/",
                "*.log",
                "temp_*",
                "*.tmp",
                "sync_to_public.py",  # Don't sync the sync script itself
                "internal_*",  # Exclude internal-only scripts
                "private_*",  # Exclude private scripts
                "*.secret",  # Exclude secret files
                "*.key",  # Exclude key files
            ],
        },
        # Data directory with better filtering
        {
            "source": "data/",
            "target": "data/",
            "exclude_patterns": [
                "*.pyc",
                "__pycache__/",
                "*.log",
                "temp_*",
                "*.tmp",
                "pmc_oas_downloaded/",
                "pmc_100k_downloaded/",
                "pmc_enterprise_download.log",
                "*.key",
                "*.secret",
                "test_pmc_downloads/",  # Exclude test downloads
            ],
        },
        # Evaluation directory
        {
            "source": "eval/",
            "target": "eval/",
            "exclude_patterns": ["*.pyc", "__pycache__/", "*.log", "temp_*", "*.tmp"],
        },
        # Node.js components
        {
            "source": "nodejs/",
            "target": "nodejs/",
            "exclude_patterns": [
                "node_modules/",
                "*.log",
                "coverage/",
                "temp_*",
                "*.tmp",
            ],
        },
        # ObjectScript components
        {
            "source": "objectscript/",
            "target": "objectscript/",
            "exclude_patterns": ["*.pyc", "__pycache__/", "*.log", "temp_*", "*.tmp"],
        },
        # Tests with comprehensive filtering
        {
            "source": "tests/",
            "target": "tests/",
            "exclude_patterns": [
                "*.pyc",
                "__pycache__/",
                "*.log",
                "temp_*",
                "*.tmp",
                "reports/",
                "working/",
                "validation/",
                "test_output/",
                "*.secret",
                "*.key",
                "experimental/*/temp_*",  # Exclude temp files in experimental
            ],
        },
    ]

    return SyncConfig(
        source_repo_path=source_repo_path,
        target_repo_path=target_repo_path,
        files_to_sync=files_to_sync,
        directories_to_sync=directories_to_sync,
        git_branch="master",
        commit_message_template="sync: update from internal GitLab repository",
    )


def main():
    """Main entry point for the corrected synchronization script."""
    parser = argparse.ArgumentParser(
        description="Synchronization FROM internal GitLab repository TO public GitHub repository"
    )
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Synchronize all content FROM internal TO public",
    )
    parser.add_argument(
        "--push", action="store_true", help="Push changes to GitHub after committing"
    )
    parser.add_argument(
        "--validate-sync",
        action="store_true",
        help="Validate current synchronization status",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without making changes",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default="master",
        help="Git branch to push to (default: master)",
    )

    args = parser.parse_args()

    if not any([args.sync_all, args.validate_sync]):
        parser.print_help()
        sys.exit(1)

    try:
        # Load configuration with HARDCODED CORRECT DIRECTION
        config = get_default_config()

        # Override branch if specified
        if hasattr(args, "branch") and args.branch:
            config.git_branch = args.branch

        # Initialize synchronizer
        synchronizer = PublicRepositorySynchronizer(config)

        if args.sync_all:
            # Perform synchronization FROM internal TO public
            sync_result = synchronizer.sync_all_content(dry_run=args.dry_run)

            if not sync_result.success:
                logger.error(f"❌ Synchronization failed: {sync_result.error_message}")
                sys.exit(1)

            if args.dry_run:
                if sync_result.changes_detected:
                    logger.info("📝 Dry run completed - changes would be made")
                    sys.exit(1)  # Exit code 1 indicates changes needed
                else:
                    logger.info("✅ Dry run completed - no changes needed")
                    sys.exit(0)

            if sync_result.files_synced or sync_result.directories_synced:
                # Commit and optionally push to GitHub
                final_result = synchronizer.commit_and_push(sync_result, push=args.push)

                if final_result.success:
                    logger.info(
                        "🎉 Synchronization FROM internal TO public completed successfully!"
                    )
                    if final_result.commit_hash:
                        logger.info(f"📝 Commit: {final_result.commit_hash}")

                    # Show summary
                    if sync_result.files_synced:
                        logger.info(f"📄 Files synced: {len(sync_result.files_synced)}")
                    if sync_result.directories_synced:
                        logger.info(
                            f"📁 Directories synced: {len(sync_result.directories_synced)}"
                        )

                    sys.exit(0)
                else:
                    logger.error(f"❌ Commit/push failed: {final_result.error_message}")
                    sys.exit(1)
            else:
                logger.info("✅ No synchronization needed - all content up to date")
                sys.exit(0)

    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
