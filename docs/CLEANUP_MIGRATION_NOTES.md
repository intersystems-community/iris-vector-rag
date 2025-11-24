# Root Directory Cleanup - Migration Notes

**Related**: Feature 052 - Root Directory Cleanup

## File Relocations

This document provides reference information for files and directories relocated during the root directory cleanup (November 2024).

### Adapter Files

**Old Location**: `adapters/rag_templates_bridge.py`
**New Location**: `iris_vector_rag/adapters/rag_templates_bridge.py`

The RAG templates bridge adapter has been moved into the main `iris_vector_rag` package structure for better organization.

### Removed Directories

The following directories were removed as they contained temporary/output files that are not part of the core repository:

#### outputs/
- **Status**: Removed (gitignored)
- **Purpose**: Runtime output files, test results, pipeline validation reports
- **Note**: Generated dynamically by tests and evaluation scripts
- **References**: Some historical documentation may reference files in this directory

#### validation_results/
- **Status**: Removed (already gitignored)
- **Purpose**: Historical validation and test reports
- **Note**: These were temporary outputs from validation runs

#### mem0_integration/, mem0-mcp-server/, supabase-mcp-memory-server/
- **Status**: Removed (orphaned experimental code)
- **Purpose**: Old memory integration experiments
- **Note**: Replaced by current memory management system

### Analysis Files

**Old Location**: `analysis/` (root level)
**New Location**: `docs/analysis/`

Analysis scripts and reports moved to documentation structure.

### Future Test Files

**Old Location**: `future_tests_not_ready/` (root level)
**New Location**: `tests/future_tests_not_ready/`

Incomplete/future test files moved into the tests directory structure.

## Documentation Link Updates

Documentation files with broken links due to these relocations may need updating. Key affected areas:

1. **Integration guides** - adapter path references updated
2. **Testing documentation** - some references to removed output directories
3. **Historical reports** - may reference removed validation_results/

## Import Path Changes

**No import path changes were required.** All Python imports continue to work:

```python
# Adapter import - works the same
from iris_vector_rag.adapters import rag_templates_bridge

# Legacy packages - still work
import iris_rag
import rag_templates
import common
```

## For Developers

If you encounter a broken link in documentation:

1. Check if the file was relocated (see above)
2. Update the link to the new location
3. If the file was in outputs/ or validation_results/, it was a temporary file - update documentation to note it's generated dynamically

## Rollback

A backup tag `pre-cleanup-backup` was created before cleanup. To rollback:

```bash
git checkout pre-cleanup-backup
```

## Questions

See the complete cleanup report: `specs/052-root-directory-cleanup/CLEANUP_COMPLETE.md`
