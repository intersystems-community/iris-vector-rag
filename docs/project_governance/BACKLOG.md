# Project Backlog

**Last Updated:** June 13, 2025  
**Current Project Phase:** Post-Enterprise Refactoring  

## Overview

This backlog tracks planned work, in-progress tasks, and completed milestones for the RAG Templates project. The project has recently completed a major enterprise refactoring milestone (June 11, 2025) and is now focused on advanced features and optimizations.

## Status Legend

- ✅ **COMPLETED** - Task is finished and verified
- 🚧 **IN PROGRESS** - Currently being worked on
- 📋 **PLANNED** - Scheduled for future implementation
- 🔄 **UNDER REVIEW** - Awaiting review or approval
- ❌ **BLOCKED** - Cannot proceed due to dependencies
- 🧊 **ON HOLD** - Temporarily paused

---

## Current Sprint (June 2025)

### High Priority Items

#### 📋 SQL RAG Library Initiative - Phase 1
**Status:** 📋 **PLANNED**  
**Priority:** High  
**Estimated Effort:** 3-4 weeks  

**Context:**
Make RAG techniques accessible directly via SQL stored procedures within InterSystems IRIS, leveraging native `EMBEDDING` data type and Embedded Python for core logic.

**Key Objectives:**
- Validate core architecture for SQL Stored Procedure to Embedded Python interaction
- Implement [`RAG.BasicSearch`](objectscript/) stored procedure with SQL interface
- Implement [`RAG.HyDESearch`](objectscript/) stored procedure for HyDE technique
- Develop basic IRIS SQL configuration management
- Create foundational Embedded Python utility functions

**Success Criteria:**
- Successfully execute `RAG.BasicSearch` and `RAG.HyDESearch` via SQL
- Demonstrate retrieval of relevant documents based on query text
- Configuration for LLMs and pipeline parameters manageable via IRIS
- Core interaction patterns established and documented

**Detailed Plan:** See [`docs/SQL_RAG_LIBRARY_PLAN.md`](docs/SQL_RAG_LIBRARY_PLAN.md)

#### 📋 ColBERT `pylate` Integration Investigation
**Status:** 📋 **PLANNED**  
**Priority:** Medium-High  
**Estimated Effort:** 2-3 weeks  

**Context:**
Investigate [`pylate`](https://github.com/lightonai/pylate) library for ColBERT re-ranking and 128-dimensional embeddings to improve storage efficiency and performance.

**Potential Benefits:**
- **128-dim Embeddings**: Reduce storage from 768-dim to 128-dim vectors
- **Faster Similarity Calculations**: Improved performance during MaxSim stage
- **Re-ranking Capabilities**: Leverage `pylate.rank.rerank` function for enhanced retrieval

**Action Items:**
- [ ] Add `pylate` as project dependency
- [ ] Test loading `fjmgAI/reason-colBERT-150M-GTE-ModernColBERT` via `pylate.models.ColBERT`
- [ ] Verify 128-dim embedding output
- [ ] Refactor [`scripts/populate_colbert_token_embeddings_native_vector.py`](scripts/) for `pylate`
- [ ] Update [`RAG.DocumentTokenEmbeddings`](iris_rag/storage/) schema to 128 dimensions
- [ ] Evaluate `pylate.rank.rerank` performance vs existing [`_calculate_maxsim`](iris_rag/pipelines/colbert.py) logic

#### 📋 VectorStore Interface Implementation
**Status:** 📋 **PLANNED**  
**Priority:** Medium  
**Estimated Effort:** 2 weeks  

**Context:**
Implement comprehensive `VectorStore` abstract base class to standardize Python interactions with InterSystems IRIS vector database.

**Goals:**
- Create Pythonic `VectorStore` interface similar to LangChain/LlamaIndex patterns
- Implement `IRISVectorStore` class with core methods (`add`, `delete`, `search`)
- Encapsulate IRIS-specific SQL for vector operations
- Support HNSW indexes and IRIS-specific vector features

**Benefits:**
- Improved code organization and reusability
- Easier integration with Python-centric RAG frameworks
- Cleaner Python code with encapsulated database operations

---

## Recently Completed (June 2025)

### ✅ Enterprise Refactoring Milestone - COMPLETED
**Date Completed:** June 11, 2025 ✅ **COMPLETED**

**Major Achievements:**

#### ✅ Generalized Reconciliation Architecture - COMPLETED
- [x] **Modular Refactoring**: Reduced main controller from 1064 to 311 lines (70% reduction) ✅ **COMPLETED**
- [x] **Component Architecture**: 7 specialized modules in [`iris_rag/controllers/reconciliation_components/`](iris_rag/controllers/reconciliation_components/) ✅ **COMPLETED**
- [x] **Critical Bug Resolution**: Resolved `SQLCODE: <-104>` vector insertion errors ✅ **COMPLETED**
- [x] **Vector Standardization**: Implemented [`common.db_vector_utils.insert_vector()`](common/db_vector_utils.py) utility ✅ **COMPLETED**
- [x] **100% Test Coverage**: All 5 contamination scenarios passing in [`tests/test_reconciliation_contamination_scenarios.py`](tests/test_reconciliation_contamination_scenarios.py) ✅ **COMPLETED**
- [x] **Daemon Mode**: Full daemon mode with signal handling and CLI integration ✅ **COMPLETED**

#### ✅ Project Structure Refinement - COMPLETED
- [x] **Archive Consolidation**: Single [`archive/`](archive/) directory with clear categorization ✅ **COMPLETED**
- [x] **Output Standardization**: Unified [`outputs/`](outputs/) directory structure ✅ **COMPLETED**
- [x] **Directory Reduction**: From 35+ top-level directories to ~12 ✅ **COMPLETED**
- [x] **Script Organization**: Consolidated [`scripts/`](scripts/) directory structure ✅ **COMPLETED**

#### ✅ Documentation Refinement - COMPLETED
- [x] **File Reduction**: From 100+ files to ~14 essential documents in [`docs/`](docs/) ✅ **COMPLETED**
- [x] **Archive Migration**: Historical documentation moved to [`archive/archived_documentation/`](archive/archived_documentation/) ✅ **COMPLETED**
- [x] **Configuration Consolidation**: Unified [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) guide ✅ **COMPLETED**
- [x] **Improved Navigation**: Clear structure with [`docs/guides/`](docs/guides/) and [`docs/reference/`](docs/reference/) ✅ **COMPLETED**

#### ✅ Archive Pruning - COMPLETED
- [x] **Size Reduction**: 70-80% reduction in archive size while preserving essential context ✅ **COMPLETED**
- [x] **Content Organization**: Clear categorization in [`archive/archived_documentation/`](archive/archived_documentation/) ✅ **COMPLETED**
- [x] **Documentation**: Comprehensive [`archive/README.md`](archive/README.md) ✅ **COMPLETED**

### ✅ ColBERT Performance Optimization - COMPLETED
**Date Completed:** June 8, 2025 ✅ **COMPLETED**

**Performance Impact Achieved:**
- **Database Queries**: Reduced from O(Number of Documents) to O(1) for token embeddings
- **Processing Time**: Improved from ~6-9 seconds to ~0.039 seconds per document (~99.4% reduction)
- **Behavioral Shift**: Transformed from I/O-bound to compute-bound behavior
- **Production Readiness**: ColBERT now viable for enterprise production use

### ✅ LLM Caching System - COMPLETED
**Date Completed:** June 8, 2025 ✅ **COMPLETED**

**Key Features Implemented:**
- **IRIS Backend**: Persistent cache storage using existing database infrastructure
- **Langchain Integration**: Seamless integration with Langchain's caching system
- **Automatic Integration**: All pipelines automatically use cached LLM functions
- **Performance Monitoring**: Built-in hit/miss tracking and performance metrics
- **TTL Support**: Automatic expiration of cached responses

### ✅ TDD+RAGAS Performance Testing Integration - COMPLETED
**Date Completed:** June 8, 2025 ✅ **COMPLETED**

**Key Features Implemented:**
- **RAGAS Quality Metrics**: Answer relevancy, context precision, faithfulness, context recall
- **Performance Benchmarking**: Response time, success rate, documents retrieved metrics
- **Scalability Testing**: Testing across different document corpus sizes
- **Automated Reporting**: Comprehensive Markdown reports with analysis

### ✅ Database Schema Management System - Phase 1 - COMPLETED
**Date Completed:** June 8, 2025 ✅ **COMPLETED**

**Core Components Implemented:**
- [x] **SchemaManager**: Central orchestrator for schema operations ✅ **COMPLETED**
- [x] **ConfigDetector**: Configuration analysis and mismatch detection ✅ **COMPLETED**
- [x] **MigrationEngine**: Safe schema migrations with rollback support ✅ **COMPLETED**
- [x] **ValidationValidator**: Schema integrity validation ✅ **COMPLETED**

---

## Future Enhancements

### Database Schema Management System - Phase 2
**Status:** 📋 **PLANNED**  
**Priority:** Medium  

**Context:**
Extend schema management system to support stored procedure-based operations for enhanced performance and database-side processing capabilities.

**Proposed Architecture:**
- **StoredProcInterface**: Plugin for stored procedure operations
- **StoredProcManager**: Manage procedure lifecycle and versions
- **StoredProcRegistry**: Track available procedures and signatures

**Key Stored Procedures to Implement:**
- `RAG_ENSURE_SCHEMA_COMPATIBILITY`: Schema validation and migration planning
- `RAG_MIGRATE_VECTOR_DIMENSIONS`: Vector dimension migrations
- `RAG_CREATE_SCHEMA_BACKUP`: Backup creation before migrations
- `RAG_ROLLBACK_SCHEMA`: Schema rollback operations

### Database Schema Management System - Phase 3
**Status:** 📋 **PLANNED**  
**Priority:** Medium-High  

**Context:**
Support systems where users have existing content in database and RAG system creates views to link to its own tables, enabling integration without data migration.

**Integration Scenarios:**
- **Customer Support Documents**: Link existing support ticket tables
- **Knowledge Base Articles**: Integrate existing documentation systems
- **Enterprise Content**: Connect to existing document management systems

### RAG Overlay Architecture for Existing IRIS Content
**Status:** 📋 **PLANNED**  
**Priority:** Medium-High  
**Estimated Effort:** 2-3 weeks  

**Context:**
Enable adding RAG capabilities on top of existing IRIS servers with content, using views and minimal data duplication. This supports enterprise scenarios where customers already have document content in IRIS and want to add RAG search without migrating data.

**Key Objectives:**
- **View-Based Integration**: Create views that map existing tables to RAG schema
- **Minimal Data Duplication**: Only duplicate what's necessary (embeddings, IFind indexes)
- **Non-Invasive Setup**: Don't modify existing customer tables
- **Configurable Mapping**: Support different source table schemas
- **IFind Overlay**: Create IFind indexes on existing text content via views

**Technical Approach:**
- **Schema Discovery**: Automatically detect existing text content tables
- **Mapping Configuration**: YAML/JSON config for field mapping (title, content, id fields)
- **View Generation**: Auto-generate views that expose existing data in RAG format
- **Embedding Pipeline**: Generate and store embeddings separately, link via doc_id
- **IFind Integration**: Create minimal IFind tables that reference existing content

**Success Criteria:**
- Successfully add RAG to existing IRIS instance without data migration
- Support multiple source table schemas through configuration
- Achieve same RAG functionality as fresh installation
- Minimal storage overhead (only embeddings + IFind indexes)

**Example Use Cases:**
- Hospital system with existing patient document tables
- Legal firm with case document database  
- Manufacturing company with technical documentation
- Educational institution with course content database

### TDD+RAGAS CI/CD Integration
**Status:** 📋 **PLANNED**  
**Priority:** Medium  

**Proposed Features:**
- **CI/CD Pipeline Integration**: Automated TDD+RAGAS tests in GitLab CI/CD
- **Performance Trend Analysis**: Historical performance tracking and regression detection
- **Quality Regression Alerts**: Automated alerts when RAGAS metrics fall below thresholds
- **Comparative Analysis**: Cross-pipeline performance comparison dashboards

---

## Maintenance & Operations

### Regular Maintenance Tasks

#### Monthly Reviews
- [ ] **Archive Review**: Review [`archive/`](archive/) contents and remove obsolete material
- [ ] **Documentation Review**: Quarterly assessment of documentation relevance
- [ ] **Performance Monitoring**: Review benchmark results and performance trends
- [ ] **Dependency Updates**: Update project dependencies and security patches

#### Quarterly Tasks
- [ ] **Backlog Grooming**: Review and prioritize backlog items
- [ ] **Technical Debt Assessment**: Identify and plan technical debt reduction
- [ ] **Security Review**: Comprehensive security assessment and updates
- [ ] **Performance Benchmarking**: Full performance benchmark suite execution

### Ongoing Monitoring
- **Test Coverage**: Maintain >95% test coverage across all components
- **Performance Metrics**: Monitor RAG pipeline performance and optimization opportunities
- **Documentation Quality**: Ensure documentation stays current with code changes
- **Security Compliance**: Regular security scans and vulnerability assessments

---

## Guidelines for Junior Developers

### Getting Started
1. **Read Documentation**: Start with [`docs/README.md`](docs/README.md) for navigation
2. **Developer Guide**: Follow [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) for onboarding
3. **Configuration**: Review [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for setup
4. **Testing**: Understand testing approach in [`docs/guides/`](docs/guides/) directory

### Understanding the Codebase
- **Primary Code**: All active development is in [`iris_rag/`](iris_rag/) directory
- **RAG Pipelines**: Implementations are in [`iris_rag/pipelines/`](iris_rag/pipelines/)
- **Common Utilities**: Shared functions in [`common/`](common/) directory
- **Tests**: Comprehensive test suite in [`tests/`](tests/) directory

### Contributing Guidelines
- **TDD Workflow**: Always write tests first (see [`.clinerules`](.clinerules))
- **Real Data Testing**: Use 1000+ documents for meaningful tests
- **Vector Operations**: Always use [`common.db_vector_utils.insert_vector()`](common/db_vector_utils.py)
- **Documentation**: Update documentation for any new features

### Common Tasks
- **Running Tests**: Use `make test-1000` for comprehensive testing
- **Performance Testing**: Use `make test-tdd-comprehensive-ragas`
- **Reconciliation Testing**: Use `make test-reconciliation`
- **Documentation Building**: Use `make docs-build-check`

---

## Backlog Management

### Prioritization Criteria
1. **Critical Bugs**: Immediate priority
2. **Performance Issues**: High priority
3. **Feature Requests**: Medium priority based on user impact
4. **Technical Debt**: Scheduled during maintenance windows
5. **Documentation**: Ongoing priority to maintain quality

### Status Updates
- **Weekly**: Update task statuses and progress
- **Sprint Planning**: Review and prioritize upcoming work
- **Milestone Reviews**: Assess completed work and plan next phases
- **Quarterly**: Major backlog review and strategic planning

### Communication
- All major changes should be documented in [`docs/project_governance/`](docs/project_governance/)
- Completion notes should be created for significant milestones
- Regular status updates should be shared with the team

---

**For questions about this backlog or specific tasks, please refer to the project documentation or reach out to the development team.**