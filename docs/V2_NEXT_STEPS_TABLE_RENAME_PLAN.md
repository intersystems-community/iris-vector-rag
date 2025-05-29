# V2 Pipeline Next Steps: Table Rename Plan

## 🎯 CRITICAL NEXT STEP: Table Renaming Strategy

### Current State
- **Original tables** (SourceDocuments, DocumentChunks, etc.) have VARCHAR embedding columns
- **V2 tables** (SourceDocuments_V2, etc.) have native VECTOR columns with 99,990 documents
- **V2 pipelines** currently reference the _V2 table names

### The Plan: Swap Tables and Update Pipelines

#### Phase 1: Rename Tables (Database Side)
```sql
-- Step 1: Rename original tables to _OLD suffix
ALTER TABLE RAG.SourceDocuments RENAME TO SourceDocuments_OLD;
ALTER TABLE RAG.DocumentChunks RENAME TO DocumentChunks_OLD;
ALTER TABLE RAG.DocumentTokenEmbeddings RENAME TO DocumentTokenEmbeddings_OLD;

-- Step 2: Rename V2 tables to original names
ALTER TABLE RAG.SourceDocuments_V2 RENAME TO SourceDocuments;
ALTER TABLE RAG.DocumentChunks_V2 RENAME TO DocumentChunks;
ALTER TABLE RAG.DocumentTokenEmbeddings_V2 RENAME TO DocumentTokenEmbeddings;
```

#### Phase 2: Update V2 Pipelines (Code Side)
Update all V2 pipeline files to use original table names:
- `basic_rag/pipeline_v2.py` → Change `SourceDocuments_V2` to `SourceDocuments`
- `crag/pipeline_v2.py` → Change `DocumentChunks_V2` to `DocumentChunks`
- `hyde/pipeline_v2.py` → Update all table references
- `noderag/pipeline_v2.py` → Update all table references
- `graphrag/pipeline_v2.py` → Update all table references
- `hybrid_ifind_rag/pipeline_v2.py` → Update all table references

#### Phase 3: Make V2 the Default
1. Update imports in `__init__.py` files to export V2 as the default
2. Keep original pipelines available as `pipeline_legacy.py` for compatibility
3. Update documentation to reflect V2 as the standard

### Benefits of This Approach
1. **Zero code changes needed in existing applications** - they'll automatically get the performance benefits
2. **V2 becomes the default** without breaking anything
3. **Legacy pipelines remain available** if needed for debugging
4. **Clean migration path** - no complex data movement

### Implementation Script Needed
Create `scripts/finalize_v2_migration.py`:
```python
"""
Finalize V2 migration by renaming tables and updating pipelines
"""
# 1. Rename database tables
# 2. Update all V2 pipeline files to use original table names
# 3. Update __init__.py files to export V2 as default
# 4. Run validation tests
```

### Validation Steps
1. Verify all tables renamed correctly
2. Test each pipeline with original table names
3. Confirm performance improvements maintained
4. Document the migration completion

## 🚨 IMPORTANT REMINDERS

1. **This is the CRITICAL next step** - without this, V2 improvements aren't accessible by default
2. **Must update BOTH database tables AND code** - they need to stay in sync
3. **Keep legacy pipelines available** - some users might need the fallback behavior
4. **Test thoroughly** - this affects all RAG operations

## Timeline
- **Immediate**: Create the migration script
- **Next**: Execute table renames in database
- **Then**: Update all V2 pipeline code
- **Finally**: Update documentation and defaults

This plan ensures the V2 performance improvements become the standard while maintaining backward compatibility.