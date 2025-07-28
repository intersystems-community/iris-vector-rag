"""
Test-driven refactoring for ColBERT encoder cleanup.

This test file drives the refactoring process to:
1. Consolidate multiple loader files into a unified loader
2. Remove duplicate mock ColBERT encoder functions

Following TDD principles: these tests will initially fail (red phase),
then we'll implement the refactoring to make them pass (green phase).
"""

import os
import glob
import re
from pathlib import Path


def test_unified_loader_exists_and_old_loaders_are_gone():
    """
    Test that the new unified loader exists and old loader files are deleted.
    
    This test enforces the refactoring requirement to:
    - Create a new unified loader at data/unified_loader.py
    - Remove the five old loader files that contain duplicate functionality
    
    This test will initially fail because the refactoring hasn't been done yet.
    """
    # Check that the new unified loader exists
    unified_loader_path = Path("data/unified_loader.py")
    assert unified_loader_path.exists(), (
        f"Unified loader file {unified_loader_path} does not exist. "
        "The refactoring should create this file to consolidate loader functionality."
    )
    
    # Check that the old loader files have been deleted
    old_loader_files = [
        "data/loader_fixed.py",
        "data/loader_vector_fixed.py", 
        "data/loader_varchar_fixed.py",
        "data/loader_optimized_performance.py",
        "data/loader_conservative_optimized.py"
    ]
    
    for old_file in old_loader_files:
        old_file_path = Path(old_file)
        assert not old_file_path.exists(), (
            f"Old loader file {old_file} still exists. "
            f"The refactoring should remove this file as its functionality "
            f"should be consolidated into the unified loader."
        )


def test_no_duplicate_mock_encoders():
    """
    Test that duplicate mock ColBERT encoder functions have been removed.
    
    This test searches for patterns like 'create_mock_colbert_encoder' or 
    'mock_colbert_encoder' across scripts/ and tests/ directories to ensure
    duplicate implementations have been cleaned up.
    
    This test will initially fail because there are currently multiple
    duplicate mock encoder functions scattered across the codebase.
    """
    # Search patterns for mock encoder functions
    patterns = [
        r'def\s+create_mock_colbert_encoder',
        r'def\s+.*mock_colbert_encoder',
        r'create_mock_colbert_encoder\s*=',
        r'mock_colbert_encoder\s*='
    ]
    
    # Directories to search
    search_dirs = ["scripts", "tests"]
    
    # File extensions to check
    file_extensions = ["*.py"]
    
    total_occurrences = 0
    found_files = []
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for ext in file_extensions:
            pattern_path = os.path.join(search_dir, "**", ext)
            for file_path in glob.glob(pattern_path, recursive=True):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.MULTILINE)
                        if matches:
                            total_occurrences += len(matches)
                            found_files.append({
                                'file': file_path,
                                'pattern': pattern,
                                'matches': len(matches)
                            })
                            
                except (UnicodeDecodeError, PermissionError):
                    # Skip files that can't be read
                    continue
    
    # Assert that no duplicate mock encoder functions exist
    assert total_occurrences == 0, (
        f"Found {total_occurrences} duplicate mock ColBERT encoder function(s) "
        f"across {len(found_files)} file(s). The refactoring should consolidate "
        f"these into a single, reusable implementation. "
        f"Files with duplicates: {[f['file'] for f in found_files]}"
    )


def test_unified_loader_has_required_functionality():
    """
    Test that the unified loader contains the essential functionality
    from the old loaders (this test will be implemented after the basic
    refactoring is complete).
    
    This is a placeholder test that will be expanded once the unified
    loader is created.
    """
    # This test will be implemented in the green phase
    # after the unified loader is created
    unified_loader_path = Path("data/unified_loader.py")
    
    if unified_loader_path.exists():
        # Check that the unified loader has essential functions
        with open(unified_loader_path, 'r') as f:
            content = f.read()
            
        # Look for key function signatures that should be present
        essential_functions = [
            "def load_documents",
            "def process_document", 
            "def insert_document"
        ]
        
        for func in essential_functions:
            assert func in content, (
                f"Unified loader missing essential function: {func}. "
                f"The unified loader should consolidate functionality from old loaders."
            )