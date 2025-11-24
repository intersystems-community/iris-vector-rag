#!/usr/bin/env python3
"""
Import Verification Script for Legacy Package Cleanup
Scans codebase for imports of specified packages before removal.
Usage: python check_imports.py package1 package2 package3 [> output.txt]
"""

import ast
import pathlib
import sys
from typing import List, Tuple

def find_imports(file_path: pathlib.Path, target_packages: List[str]) -> List[Tuple[str, int, str]]:
    """Parse Python file and find imports of target packages.

    Args:
        file_path: Path to Python file
        target_packages: List of package names to search for

    Returns:
        List of tuples (file_path, line_number, import_name)
    """
    try:
        with open(file_path) as f:
            tree = ast.parse(f.read(), filename=str(file_path))

        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(pkg) for pkg in target_packages):
                        imports.append((str(file_path), node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(pkg) for pkg in target_packages):
                    imports.append((str(file_path), node.lineno, node.module))

        return imports
    except Exception as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        return []


def scan_directory(directory: pathlib.Path, target_packages: List[str]) -> List[Tuple[str, int, str]]:
    """Recursively scan directory for Python files with target imports.

    Args:
        directory: Root directory to scan
        target_packages: List of package names to search for

    Returns:
        List of all found imports across all files
    """
    all_imports = []

    for py_file in directory.rglob("*.py"):
        # Skip virtual environments and build directories
        if any(part in py_file.parts for part in [".venv", "venv", "__pycache__", "build", "dist", ".eggs"]):
            continue

        imports = find_imports(py_file, target_packages)
        all_imports.extend(imports)

    return all_imports


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_imports.py package1 package2 package3 ...", file=sys.stderr)
        print("Example: python check_imports.py iris_rag rag_templates common", file=sys.stderr)
        sys.exit(1)

    target_packages = sys.argv[1:]
    project_root = pathlib.Path.cwd()

    print(f"🔍 Scanning {project_root} for imports of: {', '.join(target_packages)}")
    print("=" * 80)

    imports = scan_directory(project_root, target_packages)

    if imports:
        print(f"\n❌ Found {len(imports)} imports:\n")
        for file_path, line_no, import_name in sorted(imports):
            print(f"{file_path}:{line_no} → import {import_name}")
        print("\n⚠️  WARNING: Do not remove these packages - they are still in use!")
        sys.exit(1)
    else:
        print(f"\n✅ No imports found for packages: {', '.join(target_packages)}")
        print("✅ Safe to remove these package directories")
        sys.exit(0)


if __name__ == "__main__":
    main()
