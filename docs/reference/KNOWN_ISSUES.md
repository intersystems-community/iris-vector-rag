# Known Issues

**Last Updated:** June 13, 2025  
**Project Status:** Post-Enterprise Refactoring (100% Success Rate Achieved)

## Overview

This document tracks known issues, their current status, and available workarounds for the RAG Templates project. The project has recently achieved 100% success rate for all 7 RAG pipeline implementations (as of December 2025), but some historical and potential issues are documented here for reference.

## Status Legend

- 🚨 **CRITICAL** - Blocks core functionality, requires immediate attention
- ⚠️ **HIGH** - Significant impact on functionality or performance
- 📋 **MEDIUM** - Moderate impact, should be addressed in next sprint
- 💡 **LOW** - Minor issue, can be addressed during maintenance
- ✅ **RESOLVED** - Issue has been fixed and verified
- 🧊 **ON HOLD** - Issue acknowledged but not actively being worked on

---

## Active Issues

### 📋 Benchmark Metrics Collection Incomplete
**Status:** 📋 **MEDIUM**  
**Component:** Benchmarking System  
**First Reported:** June 9, 2025  

**Description:**
Recent benchmark reports show "N/A" values for retrieval quality and answer quality metrics, with only performance metrics (throughput) being collected successfully.

**Impact:**
- Incomplete performance analysis
- Cannot compare RAG techniques on quality metrics
- Limits ability to make informed technique selection decisions

**Evidence:**
```
| Technique | Context Recall | Precision At 5 | Precision At 10 |
| --- | --- | --- | --- |
| basic_rag | N/A | N/A | N/A |
| hyde | N/A | N/A | N/A |
| colbert | N/A | N/A | N/A |
```

**Workaround:**
- Use throughput metrics for performance comparison
- Manually run RAGAS evaluations for quality assessment

**Related Files:**
- [`outputs/reports/benchmarks/runs/benchmark_20250609_123034/reports/benchmark_report.md`](outputs/reports/benchmarks/runs/benchmark_20250609_123034/reports/benchmark_report.md)

---

## Recently Resolved Issues (Archive)

### ✅ ColBERT Vector Handling Issues - RESOLVED
**Status:** ✅ **RESOLVED** (June 8, 2025)  
**Component:** ColBERT Pipeline  
**Severity:** 🚨 **CRITICAL**  

**Description:**
ColBERT pipeline was failing due to vector format incompatibilities and missing token embeddings, causing `SQLCODE: <-104>` errors during vector insertion operations.

**Resolution:**
- Implemented [`common.db_vector_utils.insert_vector()`](common/db_vector_utils.py) utility for consistent vector handling
- Fixed vector data type handling and TO_VECTOR() syntax
- Achieved 99.4% performance improvement (from ~6-9 seconds to ~0.039 seconds per document)
- ColBERT now production-ready with enterprise-grade performance

**Performance Impact:**
- Database queries reduced from O(Number of Documents) to O(1)
- Processing time improved by ~99.4%
- Transformed from I/O-bound to compute-bound behavior

### ✅ Pipeline Architecture Inconsistencies - RESOLVED
**Status:** ✅ **RESOLVED** (June 11, 2025)  
**Component:** Core Architecture  
**Severity:** 🚨 **CRITICAL**  

**Description:**
Legacy pipeline implementations had inconsistent APIs, parameter naming, and error handling, leading to a 28.6% success rate across RAG techniques.

**Resolution:**
- Complete enterprise refactoring implemented
- Unified [`iris_rag`](iris_rag/) package with modular architecture
- Standardized parameter naming (`iris_connector`, `embedding_func`, `llm_func`)
- Achieved 100% success rate (7/7 pipelines operational)
- Reduced main reconciliation controller from 1064 to 311 lines (70% reduction)

**Components Fixed:**
- BasicRAG, ColBERT, HyDE, CRAG, NodeRAG, GraphRAG, HybridIFind pipelines
- Database connection management
- Configuration system
- Error handling and logging

### ✅ Vector Index Creation Failures - RESOLVED
**Status:** ✅ **RESOLVED** (June 2025)  
**Component:** Database Schema  
**Severity:** ⚠️ **HIGH**  

**Description:**
Vector index creation was failing with SQL syntax errors: `[SQLCODE: <-1>:<Invalid SQL statement>] [%msg: < ON expected, NOT found ^ CREATE INDEX IF NOT>]`

**Resolution:**
- Fixed SQL syntax for IRIS database compatibility
- Implemented proper vector index creation procedures
- Updated schema management system to handle IRIS-specific syntax

**Workaround (Historical):**
- Manual index creation using correct IRIS SQL syntax
- Use `SELECT TOP n` instead of `LIMIT n` for IRIS compatibility

### ✅ Embedding Coverage Issues - RESOLVED
**Status:** ✅ **RESOLVED** (June 2025)  
**Component:** Data Population  
**Severity:** 🚨 **CRITICAL**  

**Description:**
Only 6 out of 1006 documents had embeddings generated (0.6% coverage), severely limiting vector search effectiveness.

**Resolution:**
- Fixed data loader to generate embeddings for all documents
- Implemented comprehensive embedding generation pipeline
- Achieved 100% embedding coverage for 1000+ PMC documents
- Added validation to ensure embedding completeness

**Impact Resolution:**
- Vector search now functional across entire document corpus
- All RAG techniques can retrieve relevant documents effectively
- Performance metrics show consistent document retrieval

---

## Monitoring and Prevention

### Automated Issue Detection

The project includes several automated systems to prevent and detect issues:

1. **Pre-condition Validation System**
   - Validates database tables, embeddings, and dependencies
   - Prevents runtime failures with clear setup guidance
   - Covers all 7 pipeline types with specific validation rules

2. **Comprehensive Test Coverage**
   - TDD workflow with pytest framework
   - Real end-to-end tests with 1000+ PMC documents
   - Automated validation reports generated regularly

3. **Performance Monitoring**
   - Benchmark results tracked in [`outputs/reports/benchmarks/`](outputs/reports/benchmarks/)
   - RAGAS evaluation results in [`outputs/reports/ragas_evaluations/`](outputs/reports/ragas_evaluations/)
   - Validation reports in [`outputs/reports/validation/`](outputs/reports/validation/)

### Issue Reporting Guidelines

When reporting new issues:

1. **Check Recent Reports**: Review latest validation and benchmark reports
2. **Provide Context**: Include pipeline type, configuration, and environment details
3. **Include Logs**: Attach relevant error messages and stack traces
4. **Test Isolation**: Verify issue occurs in clean environment
5. **Performance Impact**: Document any performance degradation

### Regular Maintenance

**Monthly Tasks:**
- Review benchmark results for performance regressions
- Check validation reports for new failure patterns
- Update dependency versions and security patches
- Archive resolved issues and update documentation

**Quarterly Tasks:**
- Comprehensive system health assessment
- Performance benchmarking with full dataset
- Security review and vulnerability assessment
- Technical debt evaluation and planning

---

## Future Considerations

### Planned Enhancements
The following items are tracked in [`BACKLOG.md`](../project_governance/BACKLOG.md) and may introduce new considerations:


1. **SQL RAG Library Initiative** - Direct SQL stored procedure access
2. **ColBERT `pylate` Integration** - 128-dimensional embeddings
3. **VectorStore Interface Implementation** - Pythonic database interactions

### Potential Risk Areas

Based on project history and planned changes:

1. **Vector Dimension Changes** - Migration from 768-dim to 128-dim embeddings
2. **API Compatibility** - New SQL interfaces may require API updates
3. **Performance Scaling** - Testing with larger datasets (10K+ documents)
4. **Dependency Updates** - New ML/AI library versions may introduce breaking changes

---

## Support and Resources

### Documentation
- **User Guide**: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- **Developer Guide**: [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md)
- **Configuration**: [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md)
- **API Reference**: [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)

### Testing Commands
- **Comprehensive Testing**: `make test-1000`
- **Performance Testing**: `make test-tdd-comprehensive-ragas`
- **Reconciliation Testing**: `make test-reconciliation`
- **Documentation Validation**: `make docs-build-check`

### Project Governance
- **Backlog Management**: [`BACKLOG.md`](../project_governance/BACKLOG.md)
- **Project Rules**: [`.clinerules`](../../.clinerules)
- **Governance Notes**: [`docs/project_governance/`](docs/project_governance/)

---

**For questions about specific issues or to report new problems, please refer to the project documentation or reach out to the development team.**

**Next Review:** July 13, 2025