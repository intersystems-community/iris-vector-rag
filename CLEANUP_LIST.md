# Files to Clean Up

## Outdated/Duplicate Scripts
- Multiple evaluation scripts with similar functionality
- Many test scripts that seem to be one-offs or experiments
- Scripts with "temp_" prefix that should be removed
- Multiple validation scripts that could be consolidated

## Specific Files Identified
- Scripts with overlapping functionality in scripts/utilities/evaluation/
- Temporary scripts in scripts/utilities/
- Old migration scripts that may no longer be needed
- Test harness scripts that duplicate functionality

## To Be Reviewed
- scripts/utilities/temp_*.py
- scripts/utilities/test_*.py (many seem experimental)
- scripts/utilities/validation/* (consolidate into fewer scripts)
- Multiple "ultimate" and "master" demo scripts

## New Issues Found During Debugging
- `scripts/load_data_with_embeddings.py` - temporary script for loading embeddings
- `scripts/debug_embedding_validation_with_fix.py` - debugging script
- `common/db_vector_search.py.pre_v2_update` - old backup
- `common/db_vector_search.py.pre_table_fix` - old backup
- `scripts/utilities/investigate_vector_indexing_reality.py.pre_v2_update` - old backup
- `scripts/utilities/investigate_vector_indexing_reality.py` - empty file

## Architectural Issues to Fix
- Multiple vector search implementations need consolidation:
  - `common/db_vector_search.py` - uses RAG.SourceDocuments with dynamic dimension
  - `iris_rag/storage/vector_store_iris.py` - expects different column names (id, text_content, metadata)
- Hardcoded dimensions in various places instead of using config
- Table structure mismatches between implementations
- IRIS module import inconsistencies (iris vs intersystems_iris)