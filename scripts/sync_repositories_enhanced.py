#!/usr/bin/env python3
"""
Enhanced Repository Synchronization Script

Automates the synchronization of source code, documentation, and selected files between
the internal GitLab repository and the public GitHub repository.

This script handles:
1. Copying updated files and directories from sanitized repository
2. Filtering out internal/private content using exclude patterns
3. Staging and committing changes to internal repository
4. Pushing to internal GitLab repository
5. Comprehensive validation of sync status

Usage:
    python scripts/sync_repositories_enhanced.py --sync-all
    python scripts/sync_repositories_enhanced.py --sync-all --push
    python scripts/sync_repositories_enhanced.py --validate-sync
"""

import os
import sys
import argparse
import subprocess
import shutil
import logging
import yaml
import fnmatch
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SyncConfig:
    """Configuration for repository synchronization."""
    internal_repo_path: str
    sanitized_repo_path: str
    files_to_sync: List[Dict[str, str]]
    directories_to_sync: List[Dict[str, any]]
    git_branch: str = "feature/enterprise-rag-system-complete"
    commit_message_template: str = "sync: update source code and documentation from sanitized repository"
    excluded_scripts: List[str] = None
    included_scripts: List[Dict[str, str]] = None

@dataclass
class SyncResult:
    """Result of a synchronization operation."""
    success: bool
    files_synced: List[str]
    directories_synced: List[str]
    commit_hash: Optional[str] = None
    error_message: Optional[str] = None
    changes_detected: bool = False

class EnhancedRepositorySynchronizer:
    """Handles comprehensive synchronization between internal and sanitized repositories."""
    
    def __init__(self, config: SyncConfig):
        self.config = config
        self.internal_repo = Path(config.internal_repo_path)
        self.sanitized_repo = Path(config.sanitized_repo_path)
        
        # Validate paths
        if not self.internal_repo.exists():
            raise ValueError(f"Internal repository path does not exist: {config.internal_repo_path}")
        if not self.sanitized_repo.exists():
            raise ValueError(f"Sanitized repository path does not exist: {config.sanitized_repo_path}")
    
    def sync_all_content(self, dry_run: bool = False) -> SyncResult:
        """
        Synchronize all content (files and directories) from sanitized to internal repository.
        
        Args:
            dry_run: If True, only show what would be synced without making changes
            
        Returns:
            SyncResult with details of the synchronization
        """
        logger.info("🔄 Starting comprehensive content synchronization...")
        
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
            
            # Sync included scripts
            script_result = self._sync_included_scripts(dry_run)
            directories_synced.extend(script_result.directories_synced)
            changes_detected = changes_detected or script_result.changes_detected
            
            if dry_run:
                return SyncResult(
                    success=True,
                    files_synced=[],
                    directories_synced=[],
                    changes_detected=changes_detected
                )
            
            if not files_synced and not directories_synced:
                logger.info("✅ No content needed synchronization")
                return SyncResult(
                    success=True,
                    files_synced=[],
                    directories_synced=[],
                    changes_detected=False
                )
            
            return SyncResult(
                success=True,
                files_synced=files_synced,
                directories_synced=directories_synced,
                changes_detected=True
            )
            
        except Exception as e:
            logger.error(f"❌ Synchronization failed: {e}")
            return SyncResult(
                success=False,
                files_synced=[],
                directories_synced=[],
                error_message=str(e)
            )
    
    def _sync_files(self, dry_run: bool = False) -> SyncResult:
        """Synchronize individual files."""
        logger.info("📄 Synchronizing individual files...")
        
        files_synced = []
        changes_detected = False
        
        for file_mapping in self.config.files_to_sync:
            source_path = self.sanitized_repo / file_mapping["source"]
            target_path = self.internal_repo / file_mapping["target"]
            
            if not source_path.exists():
                logger.warning(f"⚠️ Source file not found: {source_path}")
                continue
            
            # Check if files are different
            if self._files_differ(source_path, target_path):
                changes_detected = True
                
                if dry_run:
                    logger.info(f"📝 Would sync: {file_mapping['source']} → {file_mapping['target']}")
                else:
                    # Ensure target directory exists
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file
                    shutil.copy2(source_path, target_path)
                    logger.info(f"✅ Synced: {file_mapping['source']} → {file_mapping['target']}")
                    files_synced.append(file_mapping["target"])
            else:
                logger.debug(f"📄 No changes: {file_mapping['source']}")
        
        return SyncResult(
            success=True,
            files_synced=files_synced,
            directories_synced=[],
            changes_detected=changes_detected
        )
    
    def _sync_directories(self, dry_run: bool = False) -> SyncResult:
        """Synchronize directories with filtering."""
        logger.info("📁 Synchronizing directories...")
        
        directories_synced = []
        changes_detected = False
        
        for dir_mapping in self.config.directories_to_sync:
            source_dir = self.sanitized_repo / dir_mapping["source"]
            target_dir = self.internal_repo / dir_mapping["target"]
            exclude_patterns = dir_mapping.get("exclude_patterns", [])
            
            if not source_dir.exists():
                logger.warning(f"⚠️ Source directory not found: {source_dir}")
                continue
            
            # Check if directory sync is needed
            if self._directory_sync_needed(source_dir, target_dir, exclude_patterns):
                changes_detected = True
                
                if dry_run:
                    logger.info(f"📝 Would sync directory: {dir_mapping['source']} → {dir_mapping['target']}")
                    self._preview_directory_changes(source_dir, target_dir, exclude_patterns)
                else:
                    # Sync directory with filtering
                    synced = self._sync_directory_with_filtering(source_dir, target_dir, exclude_patterns)
                    if synced:
                        logger.info(f"✅ Synced directory: {dir_mapping['source']} → {dir_mapping['target']}")
                        directories_synced.append(dir_mapping["target"])
            else:
                logger.debug(f"📁 No changes: {dir_mapping['source']}")
        
        return SyncResult(
            success=True,
            files_synced=[],
            directories_synced=directories_synced,
            changes_detected=changes_detected
        )
    
    def _sync_included_scripts(self, dry_run: bool = False) -> SyncResult:
        """Synchronize included script directories."""
        if not self.config.included_scripts:
            return SyncResult(success=True, files_synced=[], directories_synced=[], changes_detected=False)
        
        logger.info("📜 Synchronizing included scripts...")
        
        directories_synced = []
        changes_detected = False
        
        for script_mapping in self.config.included_scripts:
            source_dir = self.sanitized_repo / script_mapping["source"]
            target_dir = self.internal_repo / script_mapping["target"]
            
            if not source_dir.exists():
                logger.warning(f"⚠️ Source script directory not found: {source_dir}")
                continue
            
            # Check if directory sync is needed
            if self._directory_sync_needed(source_dir, target_dir, ["*.pyc", "__pycache__/"]):
                changes_detected = True
                
                if dry_run:
                    logger.info(f"📝 Would sync scripts: {script_mapping['source']} → {script_mapping['target']}")
                else:
                    # Sync script directory
                    synced = self._sync_directory_with_filtering(source_dir, target_dir, ["*.pyc", "__pycache__/"])
                    if synced:
                        logger.info(f"✅ Synced scripts: {script_mapping['source']} → {script_mapping['target']}")
                        directories_synced.append(script_mapping["target"])
            else:
                logger.debug(f"📜 No changes: {script_mapping['source']}")
        
        return SyncResult(
            success=True,
            files_synced=[],
            directories_synced=directories_synced,
            changes_detected=changes_detected
        )
    
    def _directory_sync_needed(self, source_dir: Path, target_dir: Path, exclude_patterns: List[str]) -> bool:
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
    
    def _get_filtered_files(self, directory: Path, exclude_patterns: List[str]) -> Set[str]:
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
                    if fnmatch.fnmatch(rel_path_str, pattern) or fnmatch.fnmatch(file_path.name, pattern):
                        excluded = True
                        break
                
                if not excluded:
                    files.add(rel_path_str)
        
        return files
    
    def _sync_directory_with_filtering(self, source_dir: Path, target_dir: Path, exclude_patterns: List[str]) -> bool:
        """Synchronize directory with filtering, returns True if changes were made."""
        changes_made = False
        
        # Ensure target directory exists
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Get source files (filtered)
        source_files = self._get_filtered_files(source_dir, exclude_patterns)
        
        # Copy/update files from source
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
    
    def _preview_directory_changes(self, source_dir: Path, target_dir: Path, exclude_patterns: List[str]):
        """Preview what changes would be made to a directory."""
        source_files = self._get_filtered_files(source_dir, exclude_patterns)
        target_files = self._get_filtered_files(target_dir, exclude_patterns) if target_dir.exists() else set()
        
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
                    logger.debug(f"  🗑️ Removed empty directory: {subdir.relative_to(directory)}")
                except OSError:
                    pass  # Directory not empty or permission issue
    
    def commit_and_push(self, sync_result: SyncResult, push: bool = False) -> SyncResult:
        """
        Commit synced content and optionally push to remote.
        
        Args:
            sync_result: Result from sync_all_content
            push: Whether to push to remote repository
            
        Returns:
            Updated SyncResult with commit information
        """
        if not sync_result.success or (not sync_result.files_synced and not sync_result.directories_synced):
            logger.info("📝 No changes to commit")
            return sync_result
        
        try:
            # Change to internal repository directory
            original_cwd = os.getcwd()
            os.chdir(self.internal_repo)
            
            # Stage all changes
            all_changes = sync_result.files_synced + sync_result.directories_synced
            for change in all_changes:
                result = subprocess.run(
                    ["git", "add", change],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"📋 Staged: {change}")
            
            # Create commit message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            commit_message = f"{self.config.commit_message_template}\n\nContent updated:\n"
            
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
                check=True
            )
            
            # Extract commit hash
            commit_hash = result.stdout.strip().split()[1] if result.stdout else None
            logger.info(f"✅ Committed changes: {commit_hash}")
            
            # Push if requested
            if push:
                result = subprocess.run(
                    ["git", "push", "origin", self.config.git_branch],
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.info(f"🚀 Pushed to {self.config.git_branch}")
            
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
    
    def validate_sync_status(self) -> Dict[str, any]:
        """
        Validate the current synchronization status between repositories.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("🔍 Validating comprehensive synchronization status...")
        
        validation_results = {
            "files_in_sync": [],
            "files_out_of_sync": [],
            "directories_in_sync": [],
            "directories_out_of_sync": [],
            "missing_content": [],
            "total_files": len(self.config.files_to_sync),
            "total_directories": len(self.config.directories_to_sync),
            "sync_percentage": 0.0
        }
        
        # Validate individual files
        for file_mapping in self.config.files_to_sync:
            source_path = self.sanitized_repo / file_mapping["source"]
            target_path = self.internal_repo / file_mapping["target"]
            
            if not source_path.exists():
                validation_results["missing_content"].append({
                    "type": "file",
                    "item": file_mapping["source"],
                    "location": "sanitized_repo"
                })
                continue
            
            if not target_path.exists():
                validation_results["missing_content"].append({
                    "type": "file",
                    "item": file_mapping["target"],
                    "location": "internal_repo"
                })
                continue
            
            if self._files_differ(source_path, target_path):
                validation_results["files_out_of_sync"].append({
                    "source": file_mapping["source"],
                    "target": file_mapping["target"]
                })
            else:
                validation_results["files_in_sync"].append({
                    "source": file_mapping["source"],
                    "target": file_mapping["target"]
                })
        
        # Validate directories
        for dir_mapping in self.config.directories_to_sync:
            source_dir = self.sanitized_repo / dir_mapping["source"]
            target_dir = self.internal_repo / dir_mapping["target"]
            exclude_patterns = dir_mapping.get("exclude_patterns", [])
            
            if not source_dir.exists():
                validation_results["missing_content"].append({
                    "type": "directory",
                    "item": dir_mapping["source"],
                    "location": "sanitized_repo"
                })
                continue
            
            if not target_dir.exists():
                validation_results["missing_content"].append({
                    "type": "directory",
                    "item": dir_mapping["target"],
                    "location": "internal_repo"
                })
                continue
            
            if self._directory_sync_needed(source_dir, target_dir, exclude_patterns):
                validation_results["directories_out_of_sync"].append({
                    "source": dir_mapping["source"],
                    "target": dir_mapping["target"]
                })
            else:
                validation_results["directories_in_sync"].append({
                    "source": dir_mapping["source"],
                    "target": dir_mapping["target"]
                })
        
        # Calculate sync percentage
        total_items = validation_results["total_files"] + validation_results["total_directories"]
        in_sync_items = len(validation_results["files_in_sync"]) + len(validation_results["directories_in_sync"])
        missing_items = len(validation_results["missing_content"])
        
        if total_items > 0:
            valid_items = total_items - missing_items
            if valid_items > 0:
                validation_results["sync_percentage"] = (in_sync_items / valid_items) * 100
        
        # Log results
        logger.info(f"📊 Sync Status: {validation_results['sync_percentage']:.1f}%")
        logger.info(f"   ✅ Files in sync: {len(validation_results['files_in_sync'])}")
        logger.info(f"   ⚠️ Files out of sync: {len(validation_results['files_out_of_sync'])}")
        logger.info(f"   ✅ Directories in sync: {len(validation_results['directories_in_sync'])}")
        logger.info(f"   ⚠️ Directories out of sync: {len(validation_results['directories_out_of_sync'])}")
        logger.info(f"   ❌ Missing content: {len(validation_results['missing_content'])}")
        
        return validation_results
    
    def _files_differ(self, file1: Path, file2: Path) -> bool:
        """Check if two files have different content."""
        if not file2.exists():
            return True
        
        try:
            # For binary files, compare file sizes first
            if file1.stat().st_size != file2.stat().st_size:
                return True
            
            # For text files, compare content
            if file1.suffix in ['.py', '.md', '.txt', '.yaml', '.yml', '.json', '.js', '.ts', '.css', '.html']:
                with open(file1, 'r', encoding='utf-8', errors='ignore') as f1, \
                     open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                    return f1.read() != f2.read()
            else:
                # For binary files, compare byte by byte
                with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
                    return f1.read() != f2.read()
        except Exception:
            # If we can't read the files, assume they're different
            return True

def load_enhanced_config_from_yaml(config_path: str) -> SyncConfig:
    """Load enhanced synchronization configuration from YAML file."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Determine paths relative to script location
        script_dir = Path(__file__).parent.parent  # Go up from scripts/ to project root
        
        # Resolve repository paths
        internal_repo_path = str(script_dir / config_data['repositories']['internal_repo_path'])
        sanitized_repo_path = str(script_dir / config_data['repositories']['sanitized_repo_path'])
        
        # Extract git configuration
        git_config = config_data.get('git', {})
        branch = git_config.get('branch', 'feature/enterprise-rag-system-complete')
        commit_message_template = git_config.get('commit_message_template', 'sync: update source code and documentation from sanitized repository')
        
        # Extract files and directories to sync
        files_to_sync = config_data.get('files_to_sync', [])
        directories_to_sync = config_data.get('directories_to_sync', [])
        excluded_scripts = config_data.get('excluded_scripts', [])
        included_scripts = config_data.get('included_scripts', [])
        
        return SyncConfig(
            internal_repo_path=internal_repo_path,
            sanitized_repo_path=sanitized_repo_path,
            files_to_sync=files_to_sync,
            directories_to_sync=directories_to_sync,
            git_branch=branch,
            commit_message_template=commit_message_template,
            excluded_scripts=excluded_scripts,
            included_scripts=included_scripts
        )
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")

def get_enhanced_default_config() -> SyncConfig:
    """Get enhanced default synchronization configuration."""
    # Try to load from default config file first
    script_dir = Path(__file__).parent.parent  # Go up from scripts/ to project root
    default_config_path = script_dir / "config" / "sync_config.yaml"
    
    if default_config_path.exists():
        try:
            return load_enhanced_config_from_yaml(str(default_config_path))
        except Exception as e:
            logger.warning(f"Failed to load default config file: {e}")
            logger.info("Falling back to minimal defaults")
    
    # Fallback to minimal defaults
    internal_repo_path = str(script_dir)
    sanitized_repo_path = str(script_dir.parent / "rag-templates-sanitized")
    
    # Define minimal files to synchronize
    files_to_sync = [
        {"source": "README.md", "target": "README.md"},
        {"source": "docs/README.md", "target": "docs/README.md"},
        {"source": "rag_templates/README.md", "target": "rag_templates/README.md"}
    ]
    
    return SyncConfig(
        internal_repo_path=internal_repo_path,
        sanitized_repo_path=sanitized_repo_path,
        files_to_sync=files_to_sync,
        directories_to_sync=[]
    )

def main():
    """Main entry point for the enhanced synchronization script."""
    parser = argparse.ArgumentParser(
        description="Enhanced synchronization between internal and sanitized repositories"
    )
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Synchronize all content (files and directories)"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push changes to remote repository after committing"
    )
    parser.add_argument(
        "--validate-sync",
        action="store_true",
        help="Validate current synchronization status"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without making changes"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to custom configuration file (YAML format)"
    )
    
    args = parser.parse_args()
    
    if not any([args.sync_all, args.validate_sync]):
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        if args.config_file:
            config = load_enhanced_config_from_yaml(args.config_file)
        else:
            config = get_enhanced_default_config()
        
        # Initialize synchronizer
        synchronizer = EnhancedRepositorySynchronizer(config)
        
        if args.validate_sync:
            validation_results = synchronizer.validate_sync_status()
            
            if validation_results["sync_percentage"] >= 95.0:
                logger.info("🎉 Repository is well synchronized!")
                sys.exit(0)
            else:
                logger.warning(f"⚠️ Synchronization needed: {validation_results['sync_percentage']:.1f}% in sync")
                sys.exit(1)
        
        if args.sync_all:
            # Perform comprehensive synchronization
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
                # Commit and optionally push
                final_result = synchronizer.commit_and_push(sync_result, push=args.push)
                
                if final_result.success:
                    logger.info("🎉 Comprehensive synchronization completed successfully!")
                    if final_result.commit_hash:
                        logger.info(f"📝 Commit: {final_result.commit_hash}")
                    
                    # Show summary
                    if sync_result.files_synced:
                        logger.info(f"📄 Files synced: {len(sync_result.files_synced)}")
                    if sync_result.directories_synced:
                        logger.info(f"📁 Directories synced: {len(sync_result.directories_synced)}")
                    
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