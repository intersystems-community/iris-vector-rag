# Project Structure Refinement Specification

**Document Version**: 1.0  
**Date**: 2025-06-11  
**Author**: RAG Templates Team  

## Executive Summary

This specification defines a cleaner, more logical, and maintainable directory structure for the RAG Templates project. The current structure has evolved organically and contains significant redundancy, unclear naming conventions, and scattered outputs that make it difficult for new team members to navigate and contribute effectively.

## Current State Analysis

### Problems with Current Structure

1. **Archive Proliferation**: Multiple archive directories (`archive/`, `archived_pipelines/`, `deprecated/`, `migration_backup_*/`) create confusion about where old code resides
2. **RAG Technique Fragmentation**: RAG implementations are scattered across top-level directories (`basic_rag/`, `colbert/`, `crag/`, `graphrag/`, `hyde/`, `noderag/`) while the primary development occurs in `iris_rag/pipelines/`
3. **Output Chaos**: Generated outputs are scattered across multiple directories (`benchmark_results/`, `dev_ragas_results_local/`, `logs/`, `project_status_logs/`, `reports/`, `test_results/`)
4. **Script Confusion**: Unclear distinction between `core_scripts/` and `scripts/`
5. **Source Code Ambiguity**: Both `src/` and `iris_rag/` contain source code, creating confusion about the primary codebase location
6. **Redundant Directories**: `rag_templates/` appears to duplicate functionality in `iris_rag/`

### Current Directory Count
- **Total top-level directories**: 35+
- **Archive-related directories**: 8
- **RAG technique directories**: 6
- **Output directories**: 6
- **Script directories**: 2

## Proposed Target Structure

```
rag-templates/
├── iris_rag/                    # Primary application code (UNCHANGED - already well-organized)
│   ├── adapters/
│   ├── cli/
│   ├── config/
│   ├── controllers/
│   ├── core/
│   ├── embeddings/
│   ├── llm/
│   ├── monitoring/
│   ├── pipelines/               # All RAG technique implementations
│   ├── services/
│   ├── storage/
│   ├── utils/
│   └── validation/
├── common/                      # Shared utilities and database functions (UNCHANGED)
├── data/                        # Data processing and ingestion (UNCHANGED)
├── tests/                       # All test files (UNCHANGED)
├── config/                      # Configuration files (UNCHANGED)
├── docs/                        # Documentation (UNCHANGED)
├── objectscript/                # ObjectScript integration (UNCHANGED)
├── outputs/                     # NEW: Consolidated output directory
│   ├── benchmarks/              # Benchmark results (from benchmark_results/)
│   ├── logs/                    # Application logs (from logs/)
│   ├── reports/                 # Generated reports (from reports/)
│   ├── test_results/            # Test outputs (from test_results/)
│   └── dev_results/             # Development results (from dev_ragas_results_local/)
├── scripts/                     # NEW: Consolidated scripts directory
│   ├── core/                    # Essential scripts (from core_scripts/)
│   ├── evaluation/              # Evaluation scripts (from eval/)
│   ├── utilities/               # Utility scripts (from scripts/)
│   └── examples/                # Example usage (from examples/)
├── tools/                       # NEW: Development and build tools
│   ├── bin/                     # Executable tools (from bin/)
│   ├── chunking/                # Chunking utilities (from chunking/)
│   └── lib/                     # Libraries (from lib/)
├── archive/                     # NEW: Single consolidated archive
│   ├── deprecated/              # All deprecated code
│   ├── legacy_pipelines/        # Old RAG implementations
│   ├── migration_backups/       # All migration backups
│   └── historical_reports/      # Old reports and logs
├── dev/                         # Development environment setup (UNCHANGED)
└── specs/                       # Project specifications (UNCHANGED)
```

### Eliminated Directories

The following top-level directories will be **removed** through consolidation:

- `archived_pipelines/` → `archive/legacy_pipelines/`
- `basic_rag/` → `archive/legacy_pipelines/basic_rag/`
- `benchmark_results/` → `outputs/benchmarks/`
- `bug_reproductions/` → `archive/deprecated/bug_reproductions/`
- `colbert/` → `archive/legacy_pipelines/colbert/`
- `core_scripts/` → `scripts/core/`
- `crag/` → `archive/legacy_pipelines/crag/`
- `deprecated/` → `archive/deprecated/`
- `dev_ragas_results_local/` → `outputs/dev_results/`
- `eval/` → `scripts/evaluation/`
- `examples/` → `scripts/examples/`
- `graphrag/` → `archive/legacy_pipelines/graphrag/`
- `hyde/` → `archive/legacy_pipelines/hyde/`
- `jdbc_exploration/` → `archive/deprecated/jdbc_exploration/`
- `logs/` → `outputs/logs/`
- `migration_backup_*/` → `archive/migration_backups/`
- `noderag/` → `archive/legacy_pipelines/noderag/`
- `project_status_logs/` → `outputs/logs/project_status/`
- `rag_templates/` → `archive/deprecated/rag_templates/`
- `reports/` → `outputs/reports/`
- `src/` → `archive/deprecated/src/`
- `test_results/` → `outputs/test_results/`

## Rationale for Changes

### 1. Single Archive Strategy

**Problem**: Multiple archive directories create confusion about where to find old code.

**Solution**: Consolidate all archived content into a single `archive/` directory with clear subdirectories:
- `deprecated/`: Code that is no longer maintained
- `legacy_pipelines/`: Old RAG implementations superseded by `iris_rag/pipelines/`
- `migration_backups/`: All timestamped migration backups
- `historical_reports/`: Old reports and status logs

**Benefits**:
- Single location for all historical content
- Clear categorization of archived material
- Easier cleanup and maintenance

### 2. RAG Technique Consolidation

**Problem**: RAG implementations scattered across top-level directories while active development happens in `iris_rag/pipelines/`.

**Solution**: Move all legacy RAG directories to `archive/legacy_pipelines/` since the codebase search shows that active development uses `iris_rag.pipelines.*` modules.

**Benefits**:
- Clear indication that `iris_rag/` is the primary codebase
- Eliminates confusion about which implementations are current
- Maintains historical implementations for reference

### 3. Output Standardization

**Problem**: Generated outputs scattered across 6+ directories with inconsistent naming.

**Solution**: Create single `outputs/` directory with standardized subdirectories:
- `benchmarks/`: All benchmark results and analysis
- `logs/`: Application and system logs
- `reports/`: Generated reports and summaries
- `test_results/`: Test outputs and coverage reports
- `dev_results/`: Development and experimental results

**Benefits**:
- Predictable location for all generated content
- Easier to add to `.gitignore` patterns
- Simplified backup and cleanup procedures

### 4. Script Organization

**Problem**: Unclear distinction between `core_scripts/` and `scripts/`, plus evaluation scripts in separate `eval/` directory.

**Solution**: Consolidate into single `scripts/` directory with clear categorization:
- `core/`: Essential operational scripts
- `evaluation/`: All evaluation and benchmarking scripts
- `utilities/`: Helper and maintenance scripts
- `examples/`: Usage examples and demos

**Benefits**:
- Single location for all executable scripts
- Clear categorization by purpose
- Easier script discovery and maintenance

### 5. Development Tools Organization

**Problem**: Development tools scattered across `bin/`, `chunking/`, `lib/` directories.

**Solution**: Create `tools/` directory to house all development utilities:
- `bin/`: Executable tools and binaries
- `chunking/`: Text chunking utilities
- `lib/`: Shared libraries and dependencies

**Benefits**:
- Clear separation of development tools from application code
- Easier tool discovery and management
- Consistent with common project conventions

## Migration Plan Outline

### Phase 1: Archive Consolidation (Low Risk)
1. Create `archive/` directory structure
2. Move `deprecated/` → `archive/deprecated/`
3. Move `migration_backup_*/` → `archive/migration_backups/`
4. Move legacy RAG directories → `archive/legacy_pipelines/`
5. Update `.gitignore` to exclude `archive/` from normal operations

### Phase 2: Output Reorganization (Medium Risk)
1. Create `outputs/` directory structure
2. Move output directories to `outputs/` subdirectories
3. Update scripts and configuration files to use new paths
4. Update documentation and README files

### Phase 3: Script Consolidation (Medium Risk)
1. Create `scripts/` directory structure
2. Move and reorganize script directories
3. Update any hardcoded script paths in configuration
4. Update CLI tools and automation scripts

### Phase 4: Tool Organization (Low Risk)
1. Create `tools/` directory structure
2. Move development tools to appropriate subdirectories
3. Update build scripts and documentation

### Phase 5: Cleanup and Validation (Low Risk)
1. Remove empty directories
2. Update all documentation
3. Validate all tests still pass
4. Update CI/CD configurations

## Future Guidelines

### Directory Naming Conventions
- Use lowercase with underscores for multi-word directories
- Prefer descriptive names over abbreviations
- Group related functionality under common parent directories

### New Content Placement Rules

1. **RAG Pipeline Development**: All new RAG techniques go in `iris_rag/pipelines/<technique_name>/`
2. **Generated Outputs**: All generated content goes in `outputs/<category>/`
3. **Scripts**: All executable scripts go in `scripts/<category>/`
4. **Development Tools**: All development utilities go in `tools/<category>/`
5. **Deprecated Code**: All deprecated code goes in `archive/deprecated/`

### Maintenance Guidelines

1. **Monthly Archive Review**: Review `archive/` contents monthly and remove truly obsolete material
2. **Output Cleanup**: Implement automated cleanup of old outputs (>30 days for dev results, >90 days for logs)
3. **Script Organization**: Maintain clear README files in each script category explaining purpose and usage
4. **Documentation Updates**: Update all documentation when adding new directories or moving content

### Access Control Recommendations

1. **Archive Directory**: Consider making `archive/` read-only to prevent accidental modifications
2. **Output Directory**: Ensure `outputs/` is writable by all development processes
3. **Script Directory**: Maintain executable permissions on scripts in `scripts/` subdirectories

## Implementation Checklist

- [ ] Create new directory structure
- [ ] Move archived content to `archive/`
- [ ] Consolidate outputs to `outputs/`
- [ ] Reorganize scripts to `scripts/`
- [ ] Move tools to `tools/`
- [ ] Update configuration files
- [ ] Update documentation
- [ ] Update CI/CD pipelines
- [ ] Validate all tests pass
- [ ] Update team onboarding documentation

## Success Metrics

1. **Reduced Directory Count**: From 35+ top-level directories to ~12
2. **Improved Discoverability**: New team members can locate relevant code within 5 minutes
3. **Simplified Maintenance**: Archive cleanup takes <30 minutes monthly
4. **Clear Ownership**: Each directory has a clear purpose and owner
5. **Consistent Patterns**: All similar content follows the same organizational pattern

## Risk Mitigation

1. **Backup Strategy**: Create full project backup before starting migration
2. **Incremental Approach**: Implement changes in phases with validation between each
3. **Rollback Plan**: Maintain ability to revert changes if issues arise
4. **Team Communication**: Notify all team members before each migration phase
5. **Documentation**: Update all relevant documentation immediately after each change

---

**Next Steps**: Review this specification with the team and obtain approval before beginning Phase 1 implementation.