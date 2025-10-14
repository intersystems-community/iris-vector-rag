#!/usr/bin/env python
"""Extract available pipeline types from iris_rag factory.

This helper script queries the iris_rag factory to get the list of
available pipeline types, outputting them as a comma-separated string
for consumption by Makefile targets.

Usage:
    python scripts/utils/get_pipeline_types.py

Output:
    basic,basic_rerank,crag,graphrag,pylate_colbert

Exit Codes:
    0: Success
    1: Error (import failure, no pipelines found, extraction failure)
"""
import sys
import inspect
import re

try:
    from iris_rag import _create_pipeline_legacy
except ImportError:
    print("ERROR: Cannot import iris_rag. Is the package installed?", file=sys.stderr)
    print("       Ensure you have run: uv sync or pip install -e .", file=sys.stderr)
    sys.exit(1)

try:
    # Get source code of factory function
    source = inspect.getsource(_create_pipeline_legacy)

    # Extract available_types list
    # Pattern: available_types = ["type1", "type2", ...]
    match = re.search(r'available_types\s*=\s*\[([\s\S]*?)\]', source)
    if not match:
        raise ValueError("Cannot find available_types list in factory source")

    # Parse list items (extract strings between quotes)
    list_content = match.group(1)
    types = re.findall(r'"([^"]+)"', list_content)

    if not types:
        print("ERROR: No pipeline types available from factory", file=sys.stderr)
        print("       This indicates a bug in iris_rag factory - please report", file=sys.stderr)
        sys.exit(1)

    # Output comma-separated list
    print(','.join(types))

except Exception as e:
    print(f"ERROR: Cannot extract pipeline types: {e}", file=sys.stderr)
    print("       Factory source may have changed - please update helper script", file=sys.stderr)
    sys.exit(1)
