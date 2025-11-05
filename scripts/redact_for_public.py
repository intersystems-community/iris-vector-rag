#!/usr/bin/env python3
"""
Redact Internal References for Public Repository

This script performs comprehensive redaction of internal references,
preparing the codebase for public release.

Features:
- Pattern-based redaction with configurable rules
- Detailed logging of all changes
- Dry-run mode for preview
- Backup creation
- Git integration
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepositoryRedactor:
    """Redact internal references from repository for public release."""

    def __init__(self, repo_root: Path, dry_run: bool = False):
        self.repo_root = repo_root
        self.dry_run = dry_run
        self.changes_log: List[Dict] = []

        # Redaction rules
        self.redaction_rules = {
            # Internal GitLab → Public GitHub
            'github.com/intersystems-community': 'github.com/intersystems-community',
            'https://github.com/intersystems-community/iris-rag-templates': 'https://github.com/intersystems-community/iris-rag-templates',
            'git@github.com:intersystems-community/iris-rag-templates.git': 'git@github.com:intersystems-community/iris-rag-templates.git',

            # Internal Docker registry → Public Docker Hub
            'intersystemsdc/iris-community': 'intersystemsdc/iris-community',

            # Internal pull requests → Pull requests
            'https://github.com/intersystems-community/iris-rag-templates/-/merge_requests': 'https://github.com/intersystems-community/iris-rag-templates/pulls',
            'pull request': 'pull request',
            'Pull Request': 'Pull Request',
            'PR #': 'PR #',

            # Internal email references
            'maintainer@example.com': 'maintainer@example.com',

            # Internal documentation references
            '/intersystems-community/': '/intersystems-community/',
        }

        # File patterns to skip
        self.skip_patterns = [
            r'\.git/',
            r'\.venv/',
            r'__pycache__/',
            r'\.pyc$',
            r'\.so$',
            r'\.dylib$',
            r'\.dat$',
            r'\.DAT$',
            r'node_modules/',
            r'\.egg-info/',
            r'dist/',
            r'build/',
        ]

        # File patterns to process (text files only)
        self.process_extensions = {
            '.py', '.md', '.txt', '.yml', '.yaml', '.json', '.toml',
            '.sh', '.bash', '.cfg', '.ini', '.env.example', '.gitignore',
            '.dockerignore', '.editorconfig', '.flake8', '.coveragerc'
        }

    def is_text_file(self, file_path: Path) -> bool:
        """Check if file is a text file that should be processed."""
        # Skip by pattern
        for pattern in self.skip_patterns:
            if re.search(pattern, str(file_path)):
                return False

        # Check extension
        if file_path.suffix in self.process_extensions:
            return True

        # Check files without extension (like Makefile, Dockerfile)
        if not file_path.suffix and file_path.name in {'Makefile', 'Dockerfile', 'README', 'LICENSE'}:
            return True

        return False

    def redact_file(self, file_path: Path) -> Tuple[int, List[str]]:
        """
        Redact internal references in a single file.

        Returns:
            Tuple of (number_of_changes, list_of_change_descriptions)
        """
        if not self.is_text_file(file_path):
            return 0, []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError) as e:
            logger.warning(f"Skipping {file_path}: {e}")
            return 0, []

        original_content = content
        changes = []
        total_replacements = 0

        # Apply redaction rules
        for pattern, replacement in self.redaction_rules.items():
            if pattern in content:
                count = content.count(pattern)
                content = content.replace(pattern, replacement)
                total_replacements += count
                changes.append(f"{pattern} → {replacement} ({count} occurrences)")

        # Write changes if not dry run
        if content != original_content:
            if not self.dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            # Log changes
            self.changes_log.append({
                'file': str(file_path.relative_to(self.repo_root)),
                'replacements': total_replacements,
                'changes': changes
            })

            return total_replacements, changes

        return 0, []

    def redact_repository(self) -> Dict:
        """
        Redact all files in repository.

        Returns:
            Summary dictionary with statistics
        """
        logger.info(f"Starting redaction of repository: {self.repo_root}")
        logger.info(f"Dry run: {self.dry_run}")

        files_processed = 0
        files_modified = 0
        total_replacements = 0

        # Walk through all files
        for file_path in self.repo_root.rglob('*'):
            if not file_path.is_file():
                continue

            files_processed += 1

            count, changes = self.redact_file(file_path)
            if count > 0:
                files_modified += 1
                total_replacements += count
                rel_path = file_path.relative_to(self.repo_root)
                logger.info(f"✓ {rel_path}: {count} replacements")
                for change in changes:
                    logger.debug(f"  - {change}")

        summary = {
            'files_processed': files_processed,
            'files_modified': files_modified,
            'total_replacements': total_replacements,
            'dry_run': self.dry_run,
            'changes_log': self.changes_log
        }

        return summary

    def save_changes_log(self, output_file: Path):
        """Save detailed changes log to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.changes_log, f, indent=2)
        logger.info(f"Changes log saved to: {output_file}")

    def create_backup(self, backup_dir: Path):
        """Create backup of repository before redaction."""
        logger.info(f"Creating backup at: {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(self.repo_root, backup_dir, ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__'))
        logger.info("Backup created successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Redact internal references for public repository release'
    )
    parser.add_argument(
        '--repo-root',
        type=Path,
        default=Path.cwd(),
        help='Repository root directory (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup before redaction'
    )
    parser.add_argument(
        '--backup-dir',
        type=Path,
        default=Path('/tmp/rag-templates-backup'),
        help='Backup directory location'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        default=Path('redaction_changes.json'),
        help='Output file for detailed changes log'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Validate repository
    if not (args.repo_root / '.git').exists():
        logger.error(f"Not a git repository: {args.repo_root}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Repository Redaction for Public Release")
    logger.info("=" * 60)

    # Create backup if requested
    if args.backup and not args.dry_run:
        redactor = RepositoryRedactor(args.repo_root, dry_run=True)
        redactor.create_backup(args.backup_dir)

    # Perform redaction
    redactor = RepositoryRedactor(args.repo_root, dry_run=args.dry_run)
    summary = redactor.redact_repository()

    # Save changes log
    if summary['files_modified'] > 0:
        redactor.save_changes_log(args.log_file)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Redaction Summary")
    logger.info("=" * 60)
    logger.info(f"Files processed: {summary['files_processed']}")
    logger.info(f"Files modified: {summary['files_modified']}")
    logger.info(f"Total replacements: {summary['total_replacements']}")
    logger.info(f"Dry run: {summary['dry_run']}")
    logger.info("")

    if args.dry_run:
        logger.info("This was a DRY RUN - no files were modified")
        logger.info("Run without --dry-run to apply changes")
    else:
        logger.info("✅ Redaction complete!")
        logger.info(f"Detailed log: {args.log_file}")

        if args.backup:
            logger.info(f"Backup location: {args.backup_dir}")

    logger.info("")

    return 0 if summary['total_replacements'] >= 0 else 1


if __name__ == '__main__':
    sys.exit(main())
