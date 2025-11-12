#!/usr/bin/env python3
"""
Update all imports from 'common.' to 'iris_vector_rag.common.'

This script updates 40+ import statements across the codebase after moving
the common module inside iris_vector_rag package.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Directories to update
DIRS_TO_UPDATE = [
    "iris_vector_rag",
    "tests",
    "evaluation_framework",
    "adapters",
]

# Patterns to replace
PATTERNS = [
    (r'from common\.', 'from iris_vector_rag.common.'),
    (r'import common\.', 'import iris_vector_rag.common.'),
]


def update_file(file_path: Path) -> Tuple[bool, int]:
    """
    Update imports in a single file.

    Returns:
        (changed, replacement_count)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        replacement_count = 0

        for pattern, replacement in PATTERNS:
            new_content, count = re.subn(pattern, replacement, content)
            content = new_content
            replacement_count += count

        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, replacement_count

        return False, 0

    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False, 0


def main():
    updated_files: List[Tuple[Path, int]] = []
    skipped_files = 0
    total_replacements = 0

    for dir_name in DIRS_TO_UPDATE:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            print(f"⏭️  Skipping {dir_name} (doesn't exist)")
            continue

        print(f"\n📁 Processing {dir_name}/...")

        for py_file in dir_path.rglob("*.py"):
            changed, count = update_file(py_file)

            if changed:
                updated_files.append((py_file, count))
                total_replacements += count
                print(f"  ✅ {py_file}: {count} replacement(s)")
            else:
                skipped_files += 1

    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"✅ Updated files: {len(updated_files)}")
    print(f"⏭️  Skipped files: {skipped_files}")
    print(f"🔄 Total replacements: {total_replacements}")

    if updated_files:
        print("\n📝 Files modified:")
        for file_path, count in sorted(updated_files):
            print(f"  - {file_path} ({count} replacements)")

    print("\n✨ Import update complete!")
    return len(updated_files), total_replacements


if __name__ == '__main__':
    updated, total = main()
    exit(0 if updated > 0 else 1)
